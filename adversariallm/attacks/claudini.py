"""Single-file implementation of Claudini-style attacks.

@article{panfilov2026claudini,
  title = {Claudini: Autoresearch Discovers State-of-the-Art Adversarial Attack Algorithms for LLMs},
  author = {Alexander Panfilov and Peter Romov and Igor Shilov and Yves-Alexandre de Montjoye and Jonas Geiping and Maksym Andriushchenko},
  journal = {arXiv preprint},
  eprint = {2603.24511},
  archivePrefix = {arXiv},
  year = {2026},
  url = {https://arxiv.org/abs/2603.24511},
}

Implementations are based on https://github.com/romovpa/claudini

Supported variants:
- ``claude_v63``: decoupled ADC + LSGM
- ``claude_v82``: v63 core with v82 defaults
- ``claude_oss_v53``: DPTO + MAC coarse-to-fine replacement
"""

import copy
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..dataset import PromptDataset
from ..lm_utils import TokenMergeError, generate_ragged_batched, get_disallowed_ids, get_flops, prepare_conversation
from ..types import Conversation
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult

_NORM_PATTERNS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    ".ln_1",
    ".ln_2",
)
_SUPPORTED_VARIANTS = {"claude_v63", "claude_v82", "claude_oss_v53"}
_VARIANT_DEFAULTS = {
    "claude_v63": {
        "lr": 10.0,
        "momentum": 0.99,
        "ema_alpha": 0.01,
        "num_starts": 6,
        "lsgm_gamma": 0.85,
        "allow_non_ascii": False,
    },
    "claude_v82": {
        "lr": 12.0,
        "momentum": 0.99,
        "ema_alpha": 0.01,
        "num_starts": 8,
        "lsgm_gamma": 0.70,
        "allow_non_ascii": False,
    },
    "claude_oss_v53": {
        "lr": 10.0,
        "momentum": 0.908,
        "ema_alpha": 0.01,
        "num_starts": 6,
        "lsgm_gamma": 0.85,
        "allow_non_ascii": True,
    },
}


@dataclass
class ClaudiniConfig:
    name: str = "claudini"
    type: str = "discrete"
    version: str = "0.0.1"
    variant: str = "claude_oss_v53"
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)

    seed: int = 0
    num_steps: int = 250
    n_tokens_adv: int = 20
    # Optional completion-count override for the final optimization step only.
    # If None, all steps use generation_config.num_return_sequences.
    last_step_num_return_sequences: int | None = None
    init_mode: Literal["manual", "random_allowed"] = "manual"
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    # ADC-style defaults (v63/v82)
    lr: float | None = None
    momentum: float | None = None
    ema_alpha: float | None = None
    num_starts: int | None = None
    lsgm_gamma: float | None = None
    # DPTO-style defaults (oss_v53)
    num_candidates: int = 80
    topk_per_position: int = 300
    dpto_temperature: float = 0.4
    n_replace: int = 2
    switch_fraction: float = 0.8

    allow_non_ascii: bool | None = None
    allow_special: bool = False
    log_progress: bool = True
    progress_log_step_interval: int = 100


class ClaudiniAttack(Attack):
    def __init__(self, config: ClaudiniConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        if self.config.variant not in _SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported claudini variant '{self.config.variant}'. "
                "Currently supported: ['claude_v63', 'claude_v82', 'claude_oss_v53']"
            )
        self._normalized_variant = self.config.variant
        self._apply_version_defaults()

        self.disallowed_ids: torch.Tensor | None = None

    @staticmethod
    def _validate_single_turn_conversation(conversation: Conversation) -> None:
        assert len(conversation) == 2, "claudini attack currently supports two-turn conversations only"
        assert conversation[0]["role"] == "user" and conversation[1]["role"] == "assistant"

    def _apply_version_defaults(self) -> None:
        variant_defaults = _VARIANT_DEFAULTS[self._normalized_variant]
        for key, default_value in variant_defaults.items():
            if getattr(self.config, key) is None:
                setattr(self.config, key, default_value)

    def run(
        self,
        target,
        dataset: PromptDataset,
    ) -> AttackResult:
        model = target.model
        tokenizer = target.tokenizer
        self.disallowed_ids = get_disallowed_ids(
            tokenizer,
            allow_non_ascii=self.config.allow_non_ascii,
            allow_special=self.config.allow_special,
        ).to(model.device)
        vocab_size = model.get_input_embeddings().weight.size(0)
        self.disallowed_ids = self.disallowed_ids[self.disallowed_ids < vocab_size]

        runs: list[SingleAttackRunResult] = []
        for conversation in dataset:
            runs.append(self._attack_single_conversation(model, tokenizer, conversation))
        return AttackResult(runs=runs)

    def _resolve_init_string(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> str:
        if self.config.init_mode == "manual":
            return self.config.optim_str_init

        assert self.disallowed_ids is not None
        vocab_size = model.get_input_embeddings().weight.size(0)
        allowed = torch.ones(vocab_size, dtype=torch.bool, device=model.device)
        allowed[self.disallowed_ids] = False
        allowed_ids = torch.nonzero(allowed, as_tuple=True)[0]
        sampled = allowed_ids[torch.randint(0, allowed_ids.numel(), (self.config.n_tokens_adv,), device=model.device)]
        return tokenizer.decode(sampled.tolist(), skip_special_tokens=False)

    def _register_lsgm_hooks(self, model: PreTrainedModel) -> list:
        handles = []
        gamma = self.config.lsgm_gamma
        for name, module in model.named_modules():
            if any(p in name for p in _NORM_PATTERNS):

                def hook(_m, grad_input, _grad_output, _gamma=gamma):
                    if grad_input and grad_input[0] is not None:
                        grad_input[0].data *= _gamma

                handles.append(module.register_full_backward_hook(hook))
        return handles

    @staticmethod
    def _remove_hooks(handles: list) -> None:
        for h in handles:
            h.remove()
        handles.clear()

    @contextmanager
    def _lsgm_hooks(self, model: PreTrainedModel):
        handles = self._register_lsgm_hooks(model)
        try:
            yield
        finally:
            self._remove_hooks(handles)

    @torch.no_grad()
    def _make_sparse_batched(self, z: torch.Tensor, sparsities: torch.Tensor) -> torch.Tensor:
        # Port of claudini unrolled v63 sparsification.
        k_restarts, seq_len, vocab_size = z.shape
        result = z.clone()

        for k in range(k_restarts):
            s_float = sparsities[k].item()
            s_floor = int(s_float)
            s_frac = s_float - s_floor

            if s_floor >= vocab_size:
                result[k] = result[k].relu() + 1e-6
                result[k] /= result[k].sum(dim=-1, keepdim=True)
                continue

            n_higher = max(int(s_frac * seq_len), min(5, seq_len))
            perm = torch.randperm(seq_len, device=z.device)

            for j in range(seq_len):
                pos = perm[j].item()
                s = (s_floor + 1) if j < n_higher else s_floor
                s = max(s, 1)

                if s >= vocab_size:
                    result[k, pos] = result[k, pos].relu() + 1e-6
                else:
                    _, topk_idx = result[k, pos].topk(s)
                    new_vals = torch.zeros_like(result[k, pos])
                    new_vals[topk_idx] = result[k, pos, topk_idx].relu() + 1e-6
                    result[k, pos] = new_vals

                result[k, pos] /= result[k, pos].sum()

        return result

    def _tokenize_with_suffix(
        self,
        tokenizer: PreTrainedTokenizerBase,
        conversation: Conversation,
        suffix: str,
    ) -> tuple[
        Conversation,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Tokenize conversation with suffix, retrying with one extra space on merge errors."""
        attack_conversation = [
            {"role": "user", "content": conversation[0]["content"] + suffix},
            {"role": "assistant", "content": conversation[1]["content"]},
        ]
        try:
            return attack_conversation, prepare_conversation(
                tokenizer=tokenizer, conversation=conversation, conversation_opt=attack_conversation
            )[0]
        except TokenMergeError:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + " " + suffix},
                {"role": "assistant", "content": conversation[1]["content"]},
            ]
            return attack_conversation, prepare_conversation(
                tokenizer=tokenizer, conversation=conversation, conversation_opt=attack_conversation
            )[0]

    def _build_step_prompt_artifacts(
        self,
        tokenizer: PreTrainedTokenizerBase,
        conversation: Conversation,
        step_suffix_ids: list[torch.Tensor],
        *,
        device: torch.device,
    ) -> tuple[list[Conversation], list[list[int]], list[torch.Tensor]]:
        step_model_inputs: list[Conversation] = []
        step_model_input_tokens: list[list[int]] = []
        token_prompts: list[torch.Tensor] = []

        for suffix_ids in step_suffix_ids:
            suffix_str = tokenizer.decode(suffix_ids.tolist(), skip_special_tokens=False)
            conv_step: Conversation = [
                {"role": "user", "content": conversation[0]["content"] + suffix_str},
                {"role": "assistant", "content": ""},
            ]
            step_model_inputs.append(copy.deepcopy(conv_step))

            _conv_step, parts = self._tokenize_with_suffix(tokenizer, conversation, suffix_str)
            pre, attack_prefix, prompt, attack_suffix, post, _target = parts
            prompt_tokens = torch.cat([pre, attack_prefix, prompt, attack_suffix, post]).to(device)
            token_prompts.append(prompt_tokens)
            step_model_input_tokens.append(prompt_tokens.tolist())

        return step_model_inputs, step_model_input_tokens, token_prompts

    @staticmethod
    def _estimate_step_flops(
        model: PreTrainedModel,
        *,
        n_tokens_in: int,
        n_tokens_out: int,
        k_restarts: int,
    ) -> int:
        return int(
            get_flops(
                model,
                n_tokens_in=n_tokens_in * k_restarts,
                n_tokens_out=n_tokens_out * k_restarts,
                type="forward_and_backward",
            )
        )

    def _attack_single_conversation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        conversation,
    ) -> SingleAttackRunResult:
        if self._normalized_variant in {"claude_v63", "claude_v82"}:
            return self._attack_single_conversation_adc(model, tokenizer, conversation)
        if self._normalized_variant == "claude_oss_v53":
            return self._attack_single_conversation_oss_v53(model, tokenizer, conversation)
        raise ValueError(f"Unsupported claudini variant '{self.config.variant}'")

    def _attack_single_conversation_adc(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        conversation,
    ) -> SingleAttackRunResult:
        self._validate_single_turn_conversation(conversation)

        start = time.time()
        init_adv = self._resolve_init_string(model, tokenizer)

        _, parts = self._tokenize_with_suffix(tokenizer, conversation, init_adv)
        pre, attack_prefix, prompt, attack_suffix, post, target = parts

        before_ids = torch.cat([pre, attack_prefix, prompt]).unsqueeze(0).to(model.device)
        after_ids = post.unsqueeze(0).to(model.device)
        target_ids = target.unsqueeze(0).to(model.device)

        embed = model.get_input_embeddings()
        before_embeds = embed(before_ids).detach()
        after_embeds = embed(after_ids).detach()
        target_embeds = embed(target_ids).detach()

        adv_len = attack_suffix.numel()
        if adv_len == 0:
            raise ValueError("claudini attack requires a non-empty suffix initialization")

        k_restarts = self.config.num_starts
        vocab_size = embed.weight.size(0)
        z = torch.randn(k_restarts, adv_len, vocab_size, device=model.device)
        if self.disallowed_ids is not None and self.disallowed_ids.numel() > 0:
            z[:, :, self.disallowed_ids] = -1e10
        z = z.softmax(dim=-1)

        soft_opt = torch.nn.Parameter(z)
        optimizer = torch.optim.SGD([soft_opt], lr=self.config.lr, momentum=self.config.momentum)

        running_wrong: torch.Tensor | None = None
        global_best_loss = float("inf")
        global_best_ids: torch.Tensor | None = None

        step_suffix_ids: list[torch.Tensor] = []
        step_losses: list[float] = []
        step_soft_losses: list[float] = []
        step_times: list[float] = []
        step_flops: list[int] = []

        with self._lsgm_hooks(model):
            if self.config.log_progress:
                self.logger.info(
                    "Claudini v63 start: steps=%d restarts=%d adv_tokens=%d",
                    self.config.num_steps,
                    self.config.num_starts,
                    adv_len,
                )
            for _step in range(self.config.num_steps):
                step_start = time.time()
                optimizer.zero_grad(set_to_none=True)

                soft_embeds = torch.matmul(soft_opt.to(torch.float32), embed.weight.detach().to(torch.float32)).to(
                    before_embeds.dtype
                )
                input_embeds = torch.cat(
                    [
                        before_embeds.expand(k_restarts, -1, -1),
                        soft_embeds,
                        after_embeds.expand(k_restarts, -1, -1),
                        target_embeds.expand(k_restarts, -1, -1),
                    ],
                    dim=1,
                )

                logits = model(inputs_embeds=input_embeds).logits
                shift = input_embeds.shape[1] - target_ids.shape[1]
                target_len = target_ids.shape[1]
                shift_logits = logits[..., shift - 1 : shift - 1 + target_len, :].contiguous()

                target_expanded = target_ids.expand(k_restarts, -1)
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    target_expanded.reshape(-1),
                    reduction="none",
                )
                loss_per_restart = loss_per_token.view(k_restarts, target_len).mean(dim=1)
                soft_loss = loss_per_restart.sum()
                soft_loss_val = float(soft_loss.item() / k_restarts)

                with torch.no_grad():
                    preds = shift_logits.argmax(dim=-1)
                    wrong_counts = (preds != target_expanded).float().sum(dim=1)

                soft_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    if running_wrong is None:
                        running_wrong = wrong_counts.clone()
                    else:
                        running_wrong += (wrong_counts - running_wrong) * self.config.ema_alpha

                    sparsities = (2.0**running_wrong).clamp(max=vocab_size / 2)
                    if self.disallowed_ids is not None and self.disallowed_ids.numel() > 0:
                        soft_opt.data[:, :, self.disallowed_ids] = -1000.0

                    pre_sparse = soft_opt.data.clone()
                    soft_opt.data.copy_(self._make_sparse_batched(soft_opt.data, sparsities))

                    all_ids = pre_sparse.argmax(dim=-1)  # [K, L]
                    all_input_ids = torch.cat(
                        [
                            before_ids.expand(k_restarts, -1),
                            all_ids,
                            after_ids.expand(k_restarts, -1),
                            target_ids.expand(k_restarts, -1),
                        ],
                        dim=1,
                    )
                    all_logits = model(input_ids=all_input_ids).logits
                    all_shift = all_input_ids.shape[1] - target_ids.shape[1]
                    all_shift_logits = all_logits[..., all_shift - 1 : all_shift - 1 + target_len, :].contiguous()
                    disc_loss_per_tok = F.cross_entropy(
                        all_shift_logits.view(-1, all_shift_logits.size(-1)),
                        target_expanded.reshape(-1),
                        reduction="none",
                    )
                    discrete_losses = disc_loss_per_tok.view(k_restarts, target_len).mean(dim=1)

                    best_k = int(discrete_losses.argmin().item())
                    step_best_loss = float(discrete_losses[best_k].item())
                    if step_best_loss < global_best_loss:
                        global_best_loss = step_best_loss
                        global_best_ids = all_ids[best_k].clone()

                    assert global_best_ids is not None
                    step_suffix_ids.append(global_best_ids.clone())
                    step_losses.append(step_best_loss)
                    step_soft_losses.append(soft_loss_val)
                    step_times.append(time.time() - step_start)

                    n_tokens_in = int(before_ids.size(1) + adv_len + after_ids.size(1))
                    step_flops.append(
                        self._estimate_step_flops(
                            model,
                            n_tokens_in=n_tokens_in,
                            n_tokens_out=int(target_len),
                            k_restarts=k_restarts,
                        )
                    )
                    should_log = self.config.log_progress and (
                        _step == 0
                        or _step == self.config.num_steps - 1
                        or (
                            self.config.progress_log_step_interval > 0
                            and (_step + 1) % self.config.progress_log_step_interval == 0
                        )
                    )
                    if should_log:
                        self.logger.info(
                            "Claudini v63 step %d/%d: best_loss=%.6f step_loss=%.6f soft_loss=%.6f",
                            _step + 1,
                            self.config.num_steps,
                            global_best_loss,
                            step_best_loss,
                            soft_loss_val,
                        )

        assert global_best_ids is not None

        step_model_inputs, step_model_input_tokens, token_prompts = self._build_step_prompt_artifacts(
            tokenizer,
            conversation,
            step_suffix_ids,
            device=model.device,
        )

        completions = self._generate_step_completions(model, tokenizer, token_prompts)

        steps: list[AttackStepResult] = []
        for idx in range(len(step_suffix_ids)):
            scores = {"claudini": {"soft_loss": [step_soft_losses[idx]]}}
            steps.append(
                AttackStepResult(
                    step=idx,
                    model_completions=completions[idx],
                    scores=scores,
                    time_taken=step_times[idx],
                    flops=step_flops[idx],
                    loss=step_losses[idx],
                    model_input=step_model_inputs[idx],
                    model_input_tokens=step_model_input_tokens[idx],
                )
            )

        return SingleAttackRunResult(
            original_prompt=copy.deepcopy(conversation),
            steps=steps,
            total_time=time.time() - start,
        )

    def _compute_target_loss(
        self,
        model: PreTrainedModel,
        *,
        input_embeds: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = model(inputs_embeds=input_embeds).logits
        shift = input_embeds.shape[1] - target_ids.shape[1]
        target_len = target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : shift - 1 + target_len, :].contiguous()
        target_expanded = target_ids.expand(input_embeds.size(0), -1)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_expanded.reshape(-1),
            reduction="none",
        )
        per_instance = per_token_loss.view(input_embeds.size(0), target_len).mean(dim=1)
        return per_instance, shift_logits

    def _generate_step_completions(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        token_prompts: list[torch.Tensor],
    ) -> list[list[str]]:
        base_n = self.config.generation_config.num_return_sequences
        last_n = self.config.last_step_num_return_sequences
        if last_n is None:
            last_n = base_n

        if last_n <= 0:
            raise ValueError("last_step_num_return_sequences must be > 0 when set")

        if len(token_prompts) <= 1 or last_n == base_n:
            return generate_ragged_batched(
                model,
                tokenizer,
                token_list=token_prompts,
                max_new_tokens=self.config.generation_config.max_new_tokens,
                temperature=self.config.generation_config.temperature,
                top_p=self.config.generation_config.top_p,
                top_k=self.config.generation_config.top_k,
                num_return_sequences=base_n,
                initial_batch_size=len(token_prompts),
            )

        # Generate all non-final steps with the baseline return count,
        # then oversample only the final step.
        prefix = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_prompts[:-1],
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=base_n,
            initial_batch_size=len(token_prompts) - 1,
        )
        last = generate_ragged_batched(
            model,
            tokenizer,
            token_list=[token_prompts[-1]],
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=last_n,
            initial_batch_size=1,
        )
        return prefix + last

    def _dpto_sample(
        self,
        *,
        control_toks: torch.Tensor,
        optim_embeds: torch.Tensor,
        grad: torch.Tensor,
        embed_weights: torch.Tensor,
        n_replace: int,
    ) -> torch.Tensor:
        eps = 1e-12
        l_pos = optim_embeds.shape[0]
        device = grad.device

        grad_norm = grad / (grad.norm(dim=-1, keepdim=True) + eps)
        topk = min(self.config.topk_per_position, embed_weights.shape[0])
        top_indices = torch.empty(l_pos, topk, device=device, dtype=torch.long)

        for pos in range(l_pos):
            dir_pos = optim_embeds[pos] - embed_weights
            dir_norm_pos = dir_pos / (dir_pos.norm(dim=-1, keepdim=True) + eps)
            cos_pos = grad_norm[pos] @ dir_norm_pos.T
            if self.disallowed_ids is not None and self.disallowed_ids.numel() > 0:
                cos_pos[self.disallowed_ids.to(device)] = -float("inf")
            cos_pos[control_toks[pos]] = -float("inf")
            _, top_indices[pos] = cos_pos.topk(topk)

        candidate_embeds = embed_weights[top_indices]
        candidate_dirs = optim_embeds.unsqueeze(1) - candidate_embeds
        dot_scores = torch.einsum("ld,lkd->lk", grad, candidate_dirs)
        probs = torch.softmax(dot_scores / max(self.config.dpto_temperature, eps), dim=1)

        bsz = self.config.num_candidates
        sampled = control_toks.repeat(bsz, 1)
        if n_replace <= 1:
            samples_per_pos = bsz // l_pos
            remainder = bsz % l_pos
            all_positions: list[int] = []
            all_tokens: list[torch.Tensor] = []
            for pos in range(l_pos):
                n = samples_per_pos + (1 if pos < remainder else 0)
                if n == 0:
                    continue
                token_indices = torch.multinomial(probs[pos], n, replacement=True)
                token_ids = top_indices[pos][token_indices]
                all_positions.extend([pos] * n)
                all_tokens.append(token_ids)
            positions = torch.tensor(all_positions, device=device, dtype=torch.long)
            tokens = torch.cat(all_tokens, dim=0)
            sampled[torch.arange(bsz, device=device), positions] = tokens
            return sampled

        for b in range(bsz):
            pos_perm = torch.randperm(l_pos, device=device)[:n_replace]
            for pos in pos_perm:
                tok_idx = torch.multinomial(probs[pos], 1).item()
                sampled[b, pos] = top_indices[pos, tok_idx]
        return sampled

    def _attack_single_conversation_oss_v53(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        conversation,
    ) -> SingleAttackRunResult:
        self._validate_single_turn_conversation(conversation)

        start = time.time()
        init_adv = self._resolve_init_string(model, tokenizer)
        _, parts = self._tokenize_with_suffix(tokenizer, conversation, init_adv)
        pre, attack_prefix, prompt, attack_suffix, post, target = parts

        before_ids = torch.cat([pre, attack_prefix, prompt]).unsqueeze(0).to(model.device)
        after_ids = post.unsqueeze(0).to(model.device)
        target_ids = target.unsqueeze(0).to(model.device)
        current_ids = attack_suffix.unsqueeze(0).to(model.device)
        if current_ids.numel() == 0:
            raise ValueError("claudini attack requires a non-empty suffix initialization")

        embed = model.get_input_embeddings()
        embed_weights = embed.weight.detach()
        before_embeds = embed(before_ids).detach()
        after_embeds = embed(after_ids).detach()
        target_embeds = embed(target_ids).detach()

        momentum_grad: torch.Tensor | None = None
        step_suffix_ids: list[torch.Tensor] = []
        step_losses: list[float] = []
        step_times: list[float] = []
        step_flops: list[int] = []

        for step_num in range(self.config.num_steps):
            step_start = time.time()
            optim_ids = current_ids
            one_hot = F.one_hot(optim_ids, num_classes=embed.num_embeddings).to(model.device, embed_weights.dtype)
            optim_embeds = (one_hot @ embed_weights).detach().clone()
            optim_embeds.requires_grad_()

            input_embeds = torch.cat([before_embeds, optim_embeds, after_embeds, target_embeds], dim=1)
            losses, _ = self._compute_target_loss(model, input_embeds=input_embeds, target_ids=target_ids)
            loss = losses.mean()
            grad = torch.autograd.grad(outputs=[loss], inputs=[optim_embeds])[0]

            with torch.no_grad():
                if momentum_grad is None:
                    momentum_grad = grad.clone()
                else:
                    momentum_grad = self.config.momentum * momentum_grad + (1 - self.config.momentum) * grad

                switch_step = int(self.config.num_steps * self.config.switch_fraction)
                current_n_replace = 1 if step_num >= switch_step else self.config.n_replace
                sampled_ids = self._dpto_sample(
                    control_toks=current_ids.squeeze(0),
                    optim_embeds=optim_embeds.squeeze(0).detach(),
                    grad=momentum_grad.squeeze(0).detach(),
                    embed_weights=embed_weights,
                    n_replace=current_n_replace,
                )
                actual_b = sampled_ids.shape[0]
                eval_embeds = torch.cat(
                    [
                        before_embeds.expand(actual_b, -1, -1),
                        embed(sampled_ids),
                        after_embeds.expand(actual_b, -1, -1),
                        target_embeds.expand(actual_b, -1, -1),
                    ],
                    dim=1,
                )
                batch_losses, _ = self._compute_target_loss(model, input_embeds=eval_embeds, target_ids=target_ids)
                best_idx = int(batch_losses.argmin().item())
                best_loss = float(batch_losses[best_idx].item())
                current_ids = sampled_ids[best_idx].unsqueeze(0)
                step_suffix_ids.append(current_ids.squeeze(0).clone())
                step_losses.append(best_loss)
                step_times.append(time.time() - step_start)
                n_tokens_in = int(before_ids.size(1) + current_ids.size(1) + after_ids.size(1))
                step_flops.append(
                    self._estimate_step_flops(
                        model,
                        n_tokens_in=n_tokens_in,
                        n_tokens_out=int(target_ids.size(1)),
                        k_restarts=max(1, 1 + actual_b),
                    )
                )
                should_log = self.config.log_progress and (
                    step_num == 0
                    or step_num == self.config.num_steps - 1
                    or (
                        self.config.progress_log_step_interval > 0
                        and (step_num + 1) % self.config.progress_log_step_interval == 0
                    )
                )
                if should_log:
                    self.logger.info(
                        "Claudini oss_v53 step %d/%d: best_loss=%.6f n_replace=%d",
                        step_num + 1,
                        self.config.num_steps,
                        best_loss,
                        current_n_replace,
                    )

        step_model_inputs, step_model_input_tokens, token_prompts = self._build_step_prompt_artifacts(
            tokenizer,
            conversation,
            step_suffix_ids,
            device=model.device,
        )

        completions = self._generate_step_completions(model, tokenizer, token_prompts)

        steps: list[AttackStepResult] = []
        for idx in range(len(step_suffix_ids)):
            steps.append(
                AttackStepResult(
                    step=idx,
                    model_completions=completions[idx],
                    scores={},
                    time_taken=step_times[idx],
                    flops=step_flops[idx],
                    loss=step_losses[idx],
                    model_input=step_model_inputs[idx],
                    model_input_tokens=step_model_input_tokens[idx],
                )
            )

        return SingleAttackRunResult(
            original_prompt=copy.deepcopy(conversation),
            steps=steps,
            total_time=time.time() - start,
        )
