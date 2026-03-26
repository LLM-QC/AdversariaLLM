"""Claudini-inspired attacks implemented in local repo attack format.

Currently supports:
- ``claude_v63``: Decoupled ADC + LSGM (ported from claudini unrolled v63)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from ..dataset import PromptDataset
from ..lm_utils import TokenMergeError, generate_ragged_batched, get_disallowed_ids, get_flops, prepare_conversation
from ..types import Conversation

_NORM_PATTERNS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    ".ln_1",
    ".ln_2",
)


@dataclass
class ClaudiniConfig:
    name: str = "claudini"
    type: str = "discrete"
    version: str = "claude_v63"
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)

    seed: int = 0
    num_steps: int = 250
    n_tokens_adv: int = 20
    init_mode: Literal["manual", "random_allowed"] = "manual"
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    # v63 defaults
    lr: float = 10.0
    momentum: float = 0.99
    ema_alpha: float = 0.01
    num_starts: int = 6
    lsgm_gamma: float = 0.85

    allow_non_ascii: bool = False
    allow_special: bool = False


class ClaudiniAttack(Attack):
    def __init__(self, config: ClaudiniConfig):
        super().__init__(config)
        if self.config.version != "claude_v63":
            raise ValueError(
                f"Unsupported claudini version '{self.config.version}'. "
                "Currently supported: ['claude_v63']"
            )

        self.disallowed_ids: torch.Tensor | None = None

    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttackResult:
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

    @staticmethod
    def _estimate_step_flops(
        model: PreTrainedModel,
        *,
        n_tokens_in: int,
        n_tokens_out: int,
        k_restarts: int,
    ) -> int:
        try:
            return int(
                get_flops(
                    model,
                    n_tokens_in=n_tokens_in * k_restarts,
                    n_tokens_out=n_tokens_out * k_restarts,
                    type="forward_and_backward",
                )
            )
        except Exception:
            return 0

    def _attack_single_conversation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        conversation,
    ) -> SingleAttackRunResult:
        assert len(conversation) == 2, "claudini attack currently supports two-turn conversations only"
        assert conversation[0]["role"] == "user" and conversation[1]["role"] == "assistant"

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

        lsgm_handles = self._register_lsgm_hooks(model)
        step_suffix_ids: list[torch.Tensor] = []
        step_losses: list[float] = []
        step_soft_losses: list[float] = []
        step_times: list[float] = []
        step_flops: list[int] = []

        try:
            for _step in range(self.config.num_steps):
                step_start = time.time()
                optimizer.zero_grad(set_to_none=True)

                soft_embeds = torch.matmul(soft_opt.to(torch.float32), embed.weight.detach().to(torch.float32)).to(before_embeds.dtype)
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

                    sparsities = (2.0 ** running_wrong).clamp(max=vocab_size / 2)
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
        finally:
            self._remove_hooks(lsgm_handles)

        assert global_best_ids is not None

        # Build per-step conversations and model-input tokens for generation.
        token_prompts: list[torch.Tensor] = []
        step_model_inputs = []
        step_model_input_tokens = []
        for suffix_ids in step_suffix_ids:
            suffix_str = tokenizer.decode(suffix_ids.tolist(), skip_special_tokens=False)
            conv_step = [
                {"role": "user", "content": conversation[0]["content"] + suffix_str},
                {"role": "assistant", "content": ""},
            ]
            step_model_inputs.append(copy.deepcopy(conv_step))

            _attack_conversation_i, parts_i = self._tokenize_with_suffix(tokenizer, conversation, suffix_str)
            pre_i, attack_prefix_i, prompt_i, attack_suffix_i, post_i, _target_i = parts_i

            prompt_tokens = torch.cat([pre_i, attack_prefix_i, prompt_i, attack_suffix_i, post_i]).to(model.device)
            token_prompts.append(prompt_tokens)
            step_model_input_tokens.append(prompt_tokens.tolist())

        completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_prompts,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=len(token_prompts),
        )

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
