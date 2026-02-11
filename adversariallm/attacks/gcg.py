"""Single-file implementation of the GCG attack with additional options.

@article{zou2023universal,
  title={Universal and transferable adversarial attacks on aligned language models},
  author={Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and Kolter, J Zico and Fredrikson, Matt},
  journal={arXiv preprint arXiv:2307.15043},
  year={2023}
}

Extensively tested against a variety of models, including:
    cais/zephyr_7b_r2d2
    ContinuousAT/Llama-2-7B-CAT
    ContinuousAT/Phi-CAT
    ContinuousAT/Zephyr-CAT
    google/gemma-2-2b-it
    GraySwanAI/Llama-3-8B-Instruct-RR
    GraySwanAI/Mistral-7B-Instruct-RR
    HuggingFaceH4/zephyr-7b-beta
    meta-llama/Llama-2-7b-chat-hf
    meta-llama/Meta-Llama-3.1-8B-Instruct
    microsoft/Phi-3-mini-4k-instruct
    mistralai/Mistral-7B-Instruct-v0.3
    qwen/Qwen2-7B-Instruct

The implementation is inspired by nanoGCG, but fixes several issues in nanoGCG,
mostly related to tokenization.
"""
import gc
import logging
import math
import random
import string
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, Optional, cast

import torch
import transformers
from torch import Tensor
from tqdm import trange
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from .attack import (Attack, AttackResult, AttackStepResult, GenerationConfig,
                     SingleAttackRunResult)
from ..dataset import PromptDataset
from ..lm_utils import (TokenMergeError, filter_suffix, generate_ragged_batched,
                        get_disallowed_ids, get_flops, prepare_conversation,
                        with_max_batchsize)


@dataclass
class GCGConfig:
    name: str = "gcg"
    type: str = "discrete"
    version: str = ""
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    num_steps: int = 250
    seed: int = 0
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    loss: Literal["mellowmax", "cw", "ce"] = "ce"
    use_constrained_gradient: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    allow_special: bool = False
    filter_ids: bool = True
    verbosity: str = "WARNING"
    token_selection: str = "default"
    grow_target: bool = False
    grad_smoothing: int = 1  # 1 = no smoothing, 2 = smooth over 2 tokens, etc.
    grad_momentum: float = 0.0  # momentum over steps


def compute_loss(shift_logits: Tensor, shift_labels: Tensor, loss_type: str, disallowed_ids: Tensor, mellowmax_alpha: float = 1.0, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> Tensor:
    """Computes the loss based on the specified loss type.

    Args:
        shift_logits: Tensor of shape (batch_size, seq_len, vocab_size)
        shift_labels: Tensor of shape (batch_size, seq_len)
        loss_type: Type of loss to compute ('mellowmax', 'cw', 'ce', 'entropy')
        mellowmax_alpha: Alpha parameter for mellowmax loss

    Returns:
        loss: Tensor of shape (batch_size,)

    Raises:
        NotImplementedError: If the loss type is not implemented
    """
    if loss_type == "ce":
        # Standard cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_logits.shape[0], -1).mean(dim=-1)
    elif loss_type == "mellowmax":
        label_logits = torch.gather(
            shift_logits, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        def mellowmax(t: Tensor, alpha=1.0, dim=-1):
            return (
                1.0 / alpha * (
                    torch.logsumexp(alpha * t, dim=dim)
                    - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
                )
            )
        loss = mellowmax(-label_logits, alpha=mellowmax_alpha, dim=-1)
    elif loss_type == "cw":
        # Get logits for target tokens
        target_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
        tmp_logits = shift_logits.clone()
        tmp_logits.scatter_(-1, shift_labels.unsqueeze(-1), -1e3)
        max_other_logits = tmp_logits.max(dim=-1).values  # (B, T, D) -> (B, T)

        loss = max_other_logits - target_logits  # (B, T)
        loss = loss.clamp_min(-1e-3).mean(dim=-1)  # (B, T) -> (B,)
    # label-free objectives:
    elif loss_type == "entropy":
        # Maximize entropy of predicted logits
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
        D = shift_logits.size(-1)
        # We want to maximize entropy, so we negate it to make it a loss to minimize
        loss = -entropy.mean(dim=-1) + math.log(D)  # (B, T) -> (B,)
    elif loss_type == "entropy_no_disallowed":
        # Maximize entropy of predicted logits, while excluding disallowed ids
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        B, T, D = probs.shape
        mask = torch.zeros((1,1,D), device=probs.device, dtype=torch.bool)
        mask[0, 0, disallowed_ids] = True
        disallowed_probs = probs[mask.expand(B, T, -1)]
        disallowed_loss = disallowed_probs.mean(dim=-1)

        allowed_probs = probs * ~mask
        entropy = -(allowed_probs * log_probs).sum(dim=-1)  # (B, T)
        D = shift_logits.size(-1)
        entropy_loss = -entropy.mean(dim=-1) + math.log(D)
        # We want to maximize entropy, so we negate it to make it a loss to minimize
        loss = entropy_loss + disallowed_loss  # (B, T) -> (B,)
    elif loss_type == "entropy_first_token":
        # Maximize entropy of first token
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
        # We want to maximize entropy, so we negate it to make it a loss to minimize
        D = shift_logits.size(-1)
        loss = -entropy[:, 0] + math.log(D) # (B, T) -> (B,)
    elif loss_type == "entropy_first_token_high_then_low":
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
        D = shift_logits.size(-1)
        # Maximize entropy of first token
        # Minimize entropy of all other tokens
        loss = -entropy[:, 0] + math.log(D) + entropy[:, 1:].mean(dim=-1)  # (B, T) -> (B,)
    elif loss_type == "entropy_adaptive":
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
        # Maximize entropy of first token
        # Minimize entropy of all other tokens
        mask = (probs[:, 0].max(dim=-1).values < 0.7).any()  # (B, D) -> (,)
        D = shift_logits.size(-1)
        loss = -entropy[:, 0] + math.log(D) + mask * entropy[:, 1:].mean(dim=-1)  # (B, T) -> (B,)
    elif loss_type == "smallmax":
        max_logits = shift_logits.max(dim=-1).values
        loss = max_logits.mean(dim=-1)
    elif loss_type == "smallmax_first_token":
        max_logits = shift_logits.max(dim=-1).values # (B, T, D) -> (B, T)
        loss = max_logits[:, 0] # (B, T) -> (B,)
    elif loss_type == "smallmax_prob":
        probs = torch.nn.functional.softmax(shift_logits, dim=-1)
        max_logits = probs.max(dim=-1).values
        loss = max_logits.mean(dim=-1)
    elif loss_type == "smallmax_prob_first_token":
        probs = torch.nn.functional.softmax(shift_logits[:, 0, :], dim=-1) # (B, T, D) -> (B, D)
        loss = probs.max(dim=-1).values # (B, D) -> (B,)
    elif loss_type == "kl_allowed":
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        B, T, D = log_probs.shape
        N_valid = D - len(disallowed_ids)
        tgt_dist = torch.full((1, 1, D), device=log_probs.device, fill_value=1 / N_valid)
        tgt_dist[0, 0, disallowed_ids] = 0
        loss = torch.nn.functional.kl_div(log_probs, tgt_dist.expand(B, T, -1), reduction="none").sum(dim=-1) # (B, T, D) -> (B, T)
        loss = loss[:, 0] # (B, T) -> (B,)
    elif loss_type == "kl_allowed_fwd":
        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)[:, 0]  # (B, T, D) -> (B, D)
        B, V = log_probs.shape
        N_valid = V - len(disallowed_ids)
        tgt_dist = torch.full((1, V), device=log_probs.device, fill_value=1 / N_valid)
        tgt_dist[0, disallowed_ids] = 0
        model_probs = log_probs.exp()
        log_tgt = torch.log(tgt_dist + 1e-30)
        loss = torch.nn.functional.kl_div(
            log_tgt.expand(B, -1),
            model_probs,
            reduction="none"
        )                                 # (B, D)
        loss = loss.sum(dim=-1)  # (B, D) -> (B,)
    elif loss_type == "kl_allowed_fwd_ascii_only":
        assert tokenizer is not None, "tokenizer is required for kl_allowed_fwd_ascii_only loss"
        allowed_toks = (
            string.ascii_letters
            + string.whitespace
            + string.digits
            + string.punctuation
            + tokenizer.convert_ids_to_tokens(tokenizer.encode("a b")[-1:])[0][0]
        )
        new_disallowed_ids = []
        for i in range(len(tokenizer)):
            if i in disallowed_ids:
                new_disallowed_ids.append(i)
            elif any(c not in allowed_toks for c in tokenizer.convert_ids_to_tokens([i])[0]):
                new_disallowed_ids.append(i)

        log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
        B, T, D = log_probs.shape
        N_valid = D - len(new_disallowed_ids)
        tgt_dist = torch.full((1, 1, D), device=log_probs.device, fill_value=1 / N_valid)
        tgt_dist[0, 0, new_disallowed_ids] = 0
        model_probs = log_probs.exp()
        log_tgt = torch.log(tgt_dist + 1e-30) # tiny ε avoids log(0) → -inf-nan
        loss = torch.nn.functional.kl_div(
            log_tgt.expand(B, T, -1),
            model_probs,
            reduction="none"
        )                                 # (B, T, D)
        loss = loss.sum(dim=-1)[:, 0]  # (B, T, D) -> (B,)
    else:
        raise NotImplementedError(f"Loss function {loss_type} not implemented")

    return loss


class GCGAttack(Attack):
    def __init__(self, config: GCGConfig):
        super().__init__(config)
        self.tokenizer = None  # Will be set in run()
        self.logger = logging.getLogger("nanogcg")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, dataset: PromptDataset) -> AttackResult:
        self.tokenizer = tokenizer  # Store tokenizer as instance variable
        self.not_allowed_ids = get_disallowed_ids(tokenizer, self.config.allow_non_ascii, self.config.allow_special).to(model.device)
        # need to have this filter here for models like gemma-3 which add extra tokens that do not have embeddings
        # we cannot filter the ids inside the get_disallowed_ids function because we need
        # the embedding layer weights to see the correct sizes
        embeddings = model.get_input_embeddings().weight
        assert isinstance(embeddings, torch.Tensor), "embeddings are expected to be a tensor"
        num_embeddings = embeddings.size(0)
        self.not_allowed_ids = self.not_allowed_ids[self.not_allowed_ids < num_embeddings]
        runs = []
        for conversation in dataset:
            runs.append(self._attack_single_conversation(model, tokenizer, conversation))
        return AttackResult(runs=runs)

    def _attack_single_conversation(self, model, tokenizer, conversation) -> SingleAttackRunResult:
        t0 = time.time()
        try:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + self.config.optim_str_init},
                {"role": "assistant", "content": conversation[1]["content"]},
            ]
            pre_ids, attack_prefix_ids, prompt_ids, attack_suffix_ids, post_ids, target_ids = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
        except TokenMergeError:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + " " + self.config.optim_str_init},
                {"role": "assistant", "content": conversation[1]["content"]},
            ]
            pre_ids, attack_prefix_ids, prompt_ids, attack_suffix_ids, post_ids, target_ids = prepare_conversation(tokenizer, conversation, attack_conversation)[0]

        pre_ids = pre_ids.unsqueeze(0).to(model.device)
        # attack_prefix_ids = attack_prefix_ids.unsqueeze(0).to(model.device)
        prompt_ids = prompt_ids.unsqueeze(0).to(model.device)
        pre_prompt_ids = torch.cat([pre_ids, prompt_ids], dim=1)
        attack_ids = attack_suffix_ids.unsqueeze(0).to(model.device)
        post_ids = post_ids.unsqueeze(0).to(model.device)
        target_ids = target_ids.unsqueeze(0).to(model.device)

        # Embed everything that doesn't get optimized
        embedding_layer = model.get_input_embeddings()
        pre_prompt_embeds, post_embeds, target_embeds = [
            embedding_layer(ids) for ids in (pre_prompt_ids, post_ids, target_ids)
        ]
        # Compute the KV Cache for tokens that appear before the optimized tokens
        if self.config.use_prefix_cache and "gemma" not in model.name_or_path:
            with torch.no_grad():
                self.prefix_cache = DynamicCache()
                output = model(inputs_embeds=pre_prompt_embeds, past_key_values=self.prefix_cache, use_cache=True)
                self.prefix_cache = output.past_key_values
            flops_prefill = get_flops(model, pre_prompt_embeds.shape[0]*pre_prompt_embeds.shape[1], 0, "forward")
        else:
            self.prefix_cache = None
            flops_prefill = 0

        self.target_ids = target_ids
        self.pre_prompt_embeds = pre_prompt_embeds
        self.post_embeds = post_embeds
        self.target_embeds = target_embeds

        if self.config.grow_target:
            self.target_length = 1
        else:
            self.target_length = target_ids.size(1)
        # Initialize the attack buffer
        buffer, flops_init = self.init_buffer(model, attack_ids)
        optim_ids = buffer.get_best_ids()
        assert self.tokenizer is not None, "Shouldn't happen but at least type hints are happy"
        token_selection = SubstitutionSelectionStrategy(
            self.config,
            self.prefix_cache,
            self.pre_prompt_embeds,
            self.post_embeds,
            self.target_embeds,
            self.target_ids,
            self.not_allowed_ids,
            self.tokenizer
        )
        losses = []
        times = []
        flops = []
        optim_strings = []
        self.stop_flag = False
        current_loss = buffer.get_lowest_loss()

        for i in (pbar := trange(self.config.num_steps, file=sys.stdout)):
            current_loss, time_for_step, optim_ids, optim_str, flops_for_step = self._single_step(model, tokenizer, conversation, token_selection, buffer, optim_ids)
            losses.append(current_loss)
            times.append(time_for_step)
            optim_strings.append(optim_str)
            if i == 0:
                flops.append(flops_for_step + flops_prefill + flops_init)
            else:
                flops.append(flops_for_step)
            pbar.set_postfix({"Loss": current_loss, "# TGT Toks": self.target_length, "Best Attack": optim_str[:80]})

            if self.stop_flag:
                self.logger.info("Early stopping due to finding a perfect match.")
                break

        token_list = []
        attack_conversations = []
        for attack in optim_strings:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + attack},
                {"role": "assistant", "content": ""},
            ]
            tokens = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
            token_list.append(torch.cat(tokens[:5]))
            attack_conversations.append(attack_conversation)
        batch_completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            initial_batch_size=len(token_list),
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )  # (N_steps, N_return_sequences, T)
        steps = []
        t1 = time.time()
        for i in range(len(optim_strings)):
            step = AttackStepResult(
                step=i,
                model_completions=batch_completions[i],
                time_taken=times[i],
                loss=losses[i],
                flops=flops[i],
                model_input=attack_conversations[i],
                model_input_tokens=token_list[i].tolist(),
            )
            steps.append(step)

        run = SingleAttackRunResult(
            original_prompt=conversation,
            steps=steps,
            total_time=t1 - t0,
        )
        return run

    def _single_step(self, model, tokenizer, conversation, token_selection, buffer, optim_ids):
        """
        Single step of the GCG attack.
        One step of the attack is defined as:
        1. Selecting the next token to replace
        2. Generating completions for the selected token


        Args:
            model: The model to attack.
            tokenizer: The tokenizer to use.
            conversation: The conversation to attack.
            token_selection: The token selection strategy to use.
            buffer: The buffer to use.
            optim_ids: The initial optim_ids to use.
        """
        t0a = time.time()

        # Setup target for token selection
        token_selection.target_ids = self.target_ids[:, :self.target_length]
        token_selection.target_embeds = self.target_embeds[:, :self.target_length]

        # Compute the token gradient
        sampled_ids, sampled_ids_pos, grad, flops_select = token_selection(
            optim_ids.squeeze(0),
            model,
            self.config.search_width,
            self.config.topk,
            self.config.n_replace,
        )

        with torch.no_grad():
            # Sample candidate token sequences
            if self.config.filter_ids:
                # We're trying to be as strict as possible here, so we filter
                # the entire prompt, not just the attack sequence in an isolated
                # way. This is because the prompt and attack can affect each
                # other's tokenization in some cases.
                idx = filter_suffix(
                    tokenizer,
                    conversation,
                    [[None, sampled_ids.cpu()]],
                )
                sampled_ids = sampled_ids[idx]
                sampled_ids_pos = sampled_ids_pos[idx]

            # Compute loss on candidates
            compute_loss_fn = partial(self.compute_candidates_loss, model)
            loss, acc, flops_loss = with_max_batchsize(compute_loss_fn, sampled_ids)

            torch.cuda.synchronize()  # Ensure GPU computation is complete
            flops_loss = flops_loss.sum().item()

            # Select best candidate and update buffer
            current_loss = loss.min().item()
            optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
            if self.config.grow_target and acc[loss.argmin()]:
                self.target_length += 1
            # Update the buffer based on the loss
            flops_for_step = (flops_select + flops_loss)
            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)

        # Get best IDs from buffer and decode
        optim_ids = buffer.get_best_ids()
        optim_str = tokenizer.batch_decode(optim_ids)[0]

        return current_loss, time.time() - t0a, optim_ids, optim_str, flops_for_step

    def init_buffer(self, model, init_buffer_ids):
        config = self.config

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)
        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        compute_loss_fn = partial(self.compute_candidates_loss, model)
        init_buffer_losses, _, flops_init = with_max_batchsize(compute_loss_fn, init_buffer_ids)
        flops_init = flops_init.sum()

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
        return buffer, flops_init.item()

    @torch.no_grad()
    def compute_candidates_loss(
        self,
        model: transformers.PreTrainedModel,
        attack_ids: Tensor,
    ) -> tuple[Tensor, torch.BoolTensor, int]:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            model : transformers.PreTrainedModel
                the model to compute the loss with respect to
            attack_ids : Tensor, shape = (B, T)
                the attack token ids to evaluate

        Returns:
            loss : Tensor, shape = (B,)
                the GCG loss on all candidate sequences
            acc : Tensor, shape = (B,)
                the accuracy on all candidate sequences
            flops : int
                Number of floating-point operations performed
        """
        B = attack_ids.shape[0]
        T = self.pre_prompt_embeds.size(1)
        if self.prefix_cache:
            input_embeds = torch.cat(
                [
                    model.get_input_embeddings()(attack_ids),
                    self.post_embeds.repeat(B, 1, 1),
                    self.target_embeds[:, :self.target_length].repeat(B, 1, 1),
                ],
                dim=1,
            )
            for i, layer in enumerate(self.prefix_cache.layers):
                layer.keys = layer.keys[:1, :, :T].expand(B, -1, -1, -1)
                layer.values = layer.values[:1, :, :T].expand(B, -1, -1, -1)
            outputs = model(
                inputs_embeds=input_embeds,
                past_key_values=self.prefix_cache,
                use_cache=True,
            )
            for i, layer in enumerate(self.prefix_cache.layers):
                layer.keys = layer.keys[:1]
                layer.values = layer.values[:1]
            self.prefix_cache.crop(T)
        else:
            input_embeds = torch.cat(
                [
                    self.pre_prompt_embeds.repeat(B, 1, 1),
                    model.get_input_embeddings()(attack_ids),
                    self.post_embeds.repeat(B, 1, 1),
                    self.target_embeds[:, :self.target_length].repeat(B, 1, 1),
                ],
                dim=1,
            )
            outputs = model(inputs_embeds=input_embeds)
        flops = get_flops(model, input_embeds.shape[1], 0, "forward")

        logits = outputs.logits
        tmp = logits.size(1) - self.target_ids[:, :self.target_length].size(1)
        shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
        shift_labels = self.target_ids[:, :self.target_length].repeat(B, 1)

        loss = compute_loss(shift_logits, shift_labels, self.config.loss, self.not_allowed_ids, self.config.mellowmax_alpha, self.tokenizer)  # (B,)

        acc: torch.BoolTensor = (shift_logits.argmax(-1) == shift_labels).all(-1)  # (B, T) -> (B,)

        if self.config.early_stop:
            if acc.any().item():
                self.stop_flag = True

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        return loss, acc, torch.tensor(flops).expand_as(loss)


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]


class SubstitutionSelectionStrategy:
    def __init__(self, config: GCGConfig, prefix_cache: list[tuple[Tensor, Tensor]], pre_prompt_embeds: Tensor, post_embeds: Tensor, target_embeds: Tensor, target_ids: Tensor, not_allowed_ids: Tensor, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.strategy = config.token_selection
        self.prefix_cache = prefix_cache
        self.pre_prompt_embeds = pre_prompt_embeds
        self.post_embeds = post_embeds
        self.target_embeds = target_embeds
        self.target_ids = target_ids
        self.not_allowed_ids = not_allowed_ids
        self.tokenizer = tokenizer
        self.grad_buffer = None

    def __call__(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int,
        n_replace: int,
        *args,
        **kwargs,
    ):
        if self.strategy == "default":
            return self._sample_ids_from_grad(
                ids,
                model,
                search_width,
                topk,
                n_replace,
                *args,
                **kwargs,
            )
        elif self.strategy == "random_overall":
            return self._random_overall(
                ids,
                model,
                search_width,
                topk,
                n_replace,
                *args,
                **kwargs,
            )
        elif self.strategy == "random_per_position":
            return self._random_per_position(
                ids,
                model,
                search_width,
                topk,
                n_replace,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid replacement selection strategy: {self.strategy}")

    def _sample_ids_from_grad(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
    ):
        """Returns `search_width` combinations of token ids based on the token gradient.
        Original GCG does this.

        Args:
            ids : Tensor, shape = (n_optim_ids)
                the sequence of token ids that are being optimized
            grad : Tensor, shape = (n_optim_ids, vocab_size)
                the gradient of the GCG loss computed with respect to the one-hot token embeddings
            search_width : int
                the number of candidate sequences to return
            topk : int
                the topk to be used when sampling from the gradient
            n_replace: int
                the number of token positions to update per sequence
            not_allowed_ids: Tensor, shape = (n_ids)
                the token ids that should not be used in optimization

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
        """
        # Initial gradient computation
        flops = 0
        grad, flops_grad = self.compute_token_gradient(ids.unsqueeze(0), model)
        grad = grad.squeeze(0)  # (n_optim_ids, vocab_size)
        flops += flops_grad

        n_smoothing = self.config.grad_smoothing
        if n_smoothing > 1:
            allowed_ids = [i for i in range(self.target_embeds.size(-1)) if i not in self.not_allowed_ids]

            # Get batch size for gradient smoothing
            batch_size = 64
            total_samples = n_smoothing - 1

            all_grads = grad.clone()

            # Process in batches
            for batch_start in range(0, total_samples, batch_size):
                current_batch_size = min(batch_size, total_samples - batch_start)

                grad_ids_batch = ids.clone().unsqueeze(0).repeat(current_batch_size, 1)  # (batch_size, n_optim_ids)

                random_positions = torch.randint(0, grad_ids_batch.shape[1], (current_batch_size, 1), device=ids.device)
                random_indices = torch.tensor([random.choice(allowed_ids) for _ in range(current_batch_size)],
                                             device=ids.device).unsqueeze(1)
                grad_ids_batch.scatter_(1, random_positions, random_indices)
                batch_grads, flops_grad = self.compute_token_gradient(grad_ids_batch, model)
                batch_grads = batch_grads.detach()
                flops += flops_grad
                all_grads += batch_grads.sum(0)
            grad = all_grads / n_smoothing
        grad_momentum = self.config.grad_momentum
        if grad_momentum > 0.0:
            if self.grad_buffer is None:
                self.grad_buffer = grad
            else:
                self.grad_buffer = grad_momentum * self.grad_buffer + (1 - grad_momentum) * grad
            grad = self.grad_buffer
        n_optim_tokens = len(ids)
        original_ids = ids.repeat(search_width, 1)

        if self.not_allowed_ids is not None:
            grad[:, self.not_allowed_ids.to(grad.device)] = float("inf")
        # (n_optim_ids, topk)
        topk_ids = grad.topk(topk, dim=1, largest=False, sorted=False).indices

        sampled_ids_pos = torch.randint(
            0, n_optim_tokens, (search_width, n_replace), device=grad.device
        )  # (search_width, n_replace)
        sampled_topk_idx = torch.randint(
            0, topk, (search_width, n_replace, 1), device=grad.device
        )

        sampled_ids_val = (
            topk_ids[sampled_ids_pos].gather(2, sampled_topk_idx).squeeze(2)
        )  # (search_width, n_replace)

        new_ids = original_ids.scatter_(
            1, sampled_ids_pos, sampled_ids_val
        )  # (search_width, n_optim_ids)

        return new_ids, sampled_ids_pos, grad, flops

    @torch.no_grad()
    def _random_overall(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
    ):
        """Returns `search_width` random token substitutions.

        Args:
            ids : Tensor, shape = (n_optim_ids,)
                the sequence of token ids that are being optimized
            grad : Tensor, shape = (n_optim_ids, vocab_size)
                the gradient of the GCG loss computed with respect to the one-hot token embeddings
            search_width : int
                the number of candidate sequences to return
            topk : int
                the topk to be used when sampling from the gradient
            n_replace: int
                the number of token positions to update per sequence

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
        """
        vocab_size = model.get_input_embeddings().weight.size(0)
        n_optim_tokens = ids.shape[0]
        original_ids = ids.repeat(search_width, 1)

        # Create valid token mask
        valid_tokens = torch.ones(vocab_size, dtype=torch.bool, device=ids.device)
        if self.not_allowed_ids is not None:
            valid_tokens[self.not_allowed_ids.to(ids.device)] = False

        # Sample positions and token indices
        sampled_ids_pos = torch.randint(0, n_optim_tokens, (search_width, 1), device=ids.device)
        valid_token_indices = torch.nonzero(valid_tokens).squeeze()
        sampled_topk_idx = valid_token_indices[torch.randint(0, valid_token_indices.size(0), (search_width, 1), device=ids.device)]

        # Create new sequences with substitutions
        new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_topk_idx)

        grad = None
        return new_ids, sampled_ids_pos, grad, 0

    @torch.no_grad()
    def _random_per_position(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
    ):
        """Returns `search_width` random token substitutions.

        Args:
            ids : Tensor, shape = (n_optim_ids,)
                the sequence of token ids that are being optimized
            model : transformers.PreTrainedModel
                the model to compute the gradient with respect to
            search_width : int
                the number of candidate sequences to return
            topk : int
                the topk to be used when sampling from the gradient
            n_replace: int
                the number of token positions to update per sequence

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
            sampled_ids_pos : Tensor, shape = (search_width, 1)
                the positions of the sampled token ids
            grad : Tensor, shape = (N, V)
                the gradient of the GCG loss computed with respect to the one-hot token embeddings
            flops : int
                the number of floating-point operations
        """
        # Sample search_width//ids.shape[0] substitutions at each position
        N = ids.shape[0]
        V = model.get_input_embeddings().weight.size(0)
        samples_per_position = search_width // N

        positions = torch.arange(N, device=ids.device) # (N,)
        original_ids = ids.repeat(search_width, 1) # (search_width, N)

        # Get valid ids for each position (all except not_allowed_ids)
        valid_ids = torch.ones((N, V), dtype=torch.bool, device=ids.device)
        if self.not_allowed_ids is not None:
            valid_ids[:, self.not_allowed_ids.to(ids.device)] = False

        # Sample indices for each position in parallel
        sampled_ids = torch.empty((N, samples_per_position), dtype=torch.long, device=ids.device) # (N, samples_per_position)
        rand_perm = torch.argsort(torch.rand_like(valid_ids.float()), dim=1) # (N, V)
        valid_perm = torch.masked_select(rand_perm, valid_ids).reshape(N, -1) # (N, samples_per_position)
        sampled_ids = valid_perm[:, :samples_per_position]

        # Reshape to (total_samples, 1) format
        sampled_topk_idx = sampled_ids.reshape(-1)
        sampled_ids_pos = positions.repeat_interleave(samples_per_position)
        original_ids = original_ids[:samples_per_position * N] # (search_width * N,)
        new_ids = original_ids.scatter_(1, sampled_ids_pos.unsqueeze(1), sampled_topk_idx.unsqueeze(1)) # (search_width * N,) -> (search_width, N)
        return new_ids, sampled_ids_pos, None, 0

    def _lowest_gradient_magnitude(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
    ):
        """Returns `search_width` combinations of token ids with the lowest token gradient.

        Args:
            ids : Tensor, shape = (n_optim_ids)
                the sequence of token ids that are being optimized
            grad : Tensor, shape = (n_optim_ids, vocab_size)
                the gradient of the GCG loss computed with respect to the one-hot token embeddings
            search_width : int
                the number of candidate sequences to return
            topk : int
                the topk to be used when sampling from the gradient
            n_replace: int
                the number of token positions to update per sequence

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
        """
        grad, flops_grad = self.compute_token_gradient(ids.unsqueeze(0), model)
        grad = grad.squeeze(0)  # (n_optim_ids, vocab_size)
        n_optim_ids = len(ids)
        original_ids = ids.repeat(search_width, 1)

        if self.not_allowed_ids is not None:
            grad[:, self.not_allowed_ids.to(grad.device)] = float("inf")

        # We have 32768 * 20 = 655360 substitutions to evaluate
        # Here we crop this down with the smallest gradient heuristic to topk * 20
        topk_ids = (
            grad.abs()
            .view(-1)
            .topk(topk * n_optim_ids, largest=False, sorted=False)
            .indices
        )  # (n_optim_ids, topk)
        topk_ids = torch.randperm(grad.view(-1).shape[0], device=topk_ids.device)[
            : topk * n_optim_ids
        ]  # (n_optim_ids, topk)

        # We then crop again randomly to search_width candidates
        topk_ids = topk_ids[
            torch.randperm(topk_ids.size(0), device=topk_ids.device)[:search_width]
        ].unsqueeze(1)  # (search_width, 1)

        sampled_ids_pos = topk_ids // grad.size(1)
        sampled_topk_idx = topk_ids % grad.size(1)

        new_ids = original_ids.scatter_(
            1, sampled_ids_pos, sampled_topk_idx
        )  # (search_width, n_optim_ids)

        return new_ids, sampled_ids_pos, None, flops_grad

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
        model: transformers.PreTrainedModel,
    ) -> tuple[Tensor, int]:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
        optim_ids : Tensor, shape = (N, n_optim_ids)
            the sequence of token ids that are being optimized
        model : transformers.PreTrainedModel
            the model to compute the gradient with respect to

        Returns:
            grad : Tensor, shape = (N, n_optim_ids, vocab_size)
                the gradient of the GCG loss computed with respect to the one-hot token embeddings
            flops : int
                Number of floating-point operations performed
        """
        assert optim_ids.ndim == 2
        embedding_layer = model.get_input_embeddings()

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings  # type: ignore
        )
        optim_ids_onehot = optim_ids_onehot.to(dtype=model.dtype, device=model.device)
        optim_ids_onehot.requires_grad_()

        embedding_weight = cast(Tensor, embedding_layer.weight)
        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        if self.config.use_constrained_gradient:
            optim_embeds = (
                optim_ids_onehot / optim_ids_onehot.sum(dim=-1, keepdim=True)
            ) @ embedding_weight
        else:
            optim_embeds = optim_ids_onehot @ embedding_weight
        if hasattr(embedding_layer, "embed_scale"):  # For gemma
            optim_embeds = optim_embeds * embedding_layer.embed_scale.to(optim_embeds)

        B = optim_embeds.shape[0]
        if self.prefix_cache:
            T = self.pre_prompt_embeds.shape[1]
            input_embeds = torch.cat(
                [optim_embeds, self.post_embeds.repeat(B, 1, 1), self.target_embeds.repeat(B, 1, 1)], dim=1
            )
            for i, layer in enumerate(self.prefix_cache.layers):
                layer.keys = layer.keys[:1, :, :T].expand(B, -1, -1, -1)
                layer.values = layer.values[:1, :, :T].expand(B, -1, -1, -1)
            output = model(
                inputs_embeds=input_embeds,
                past_key_values=self.prefix_cache,
                use_cache=True,
            )
            for i, layer in enumerate(self.prefix_cache.layers):
                layer.keys = layer.keys[:1]
                layer.values = layer.values[:1]
            self.prefix_cache.crop(T)
        else:
            input_embeds = torch.cat(
                [
                    self.pre_prompt_embeds.repeat(B, 1, 1),
                    optim_embeds,
                    self.post_embeds.repeat(B, 1, 1),
                    self.target_embeds.repeat(B, 1, 1),
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)
        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1 : -1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids.repeat(B, 1)

        loss = compute_loss(shift_logits, shift_labels, self.config.loss, self.not_allowed_ids, self.config.mellowmax_alpha, self.tokenizer)
        loss = loss.mean()

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[loss],
            inputs=[optim_ids_onehot],
            create_graph=False,
            retain_graph=False
        )[0]
        flops = get_flops(model, input_embeds.shape[0]*input_embeds.shape[1], 0, "forward_and_backward")
        return optim_ids_onehot_grad, flops
