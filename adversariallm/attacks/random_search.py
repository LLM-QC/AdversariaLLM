"""
Cleaned-up random search attack inspired by baseline comparisons in the GCG paper.

@article{zou2023universal,
  title={Universal and Transferable Adversarial Attacks on Aligned Language Models},
  author={Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and Kolter, J Zico and Fredrikson, Matt},
  journal={arXiv preprint arXiv:2307.15043},
  year={2023}
}

Adds explicit hyper-parameters
  • `neighborhood_radius`    - max token flips per neighbour
  • `num_steps`              - evaluation budget / generations
  • `candidates_per_generation`  - neighbours evaluated *in parallel* each generation

This revision parallelises candidate evaluation with the `with_max_batchsize` helper so GPU/TPU capacity is saturated regardless of how many neighbours you ask for.
"""
import functools
import time
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from .attack import (Attack, AttackResult, AttackStepResult, GenerationConfig,
                     SingleAttackRunResult)
from ..lm_utils import (generate_ragged_batched, get_disallowed_ids,
                          prepare_conversation, with_max_batchsize)


@dataclass
class RandomSearchConfig:
    name: str = "random_search"
    type: str = "discrete"
    version: str = ""
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)

    # --- Search hyper‑parameters -------------------------------------------
    num_steps: int = 100  # evaluation budget / generations
    candidates_per_generation: int = 8  # number of neighbours tried per generation
    neighborhood_radius: int = 1  # ≤ ‑tokens flipped per neighbour
    # -----------------------------------------------------------------------

    seed: int = 0
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"


class RandomSearchAttack(Attack):
    """Random search in discrete token space with GPU‑efficient batching."""

    def __init__(self, config: RandomSearchConfig):
        super().__init__(config)
        if self.config.placement == "suffix":
            assert self.config.optim_str_init
        elif self.config.placement == "prompt":
            assert not self.config.optim_str_init

    def _get_random_allowed_tokens(
        self,
        model: torch.nn.Module,
        tokenizer,
        k: int,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample *k* allowed token IDs on *model.device*."""
        allowed = torch.ones(len(tokenizer), dtype=torch.bool, device=model.device)
        allowed[self.disallowed_ids] = False
        allowed_idx = torch.nonzero(allowed, as_tuple=True)[0]
        if weights is None:
            probs = torch.ones_like(allowed_idx, dtype=torch.float)
        else:
            probs = weights[allowed_idx].clamp(min=1e-9)
        return allowed_idx[torch.multinomial(probs, k, replacement=True)]

    @torch.no_grad()
    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        self.disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)

        # ---- Tokenise dataset & build masks --------------------------------
        x, y, atk_mask, tgt_mask, attn_mask, original_convs = self._prepare_batch_inputs(tokenizer, dataset)

        # ---- Run attack with automatic sub‑batching ------------------------
        batched_runner = functools.partial(
            self._attack_batch, model, tokenizer
        )
        runs = with_max_batchsize(
            batched_runner,
            original_convs,
            x,
            y,
            attn_mask,
            atk_mask,
            tgt_mask,
        )
        return AttackResult(runs=runs)

    def _prepare_batch_inputs(self, tokenizer, dataset):
        xs, atk_masks, tgt_masks, originals = [], [], [], []
        for conv in dataset:
            assert len(conv) == 2, "Random search attack only supports two‑turn conversations"
            originals.append(conv)

            if self.config.placement == "suffix":
                conv_opt = [
                    {"role": "user", "content": conv[0]["content"] + self.config.optim_str_init},
                    {"role": "assistant", "content": conv[1]["content"]},
                ]
                prep_arg = conv
            else:  # prompt placement
                conv_opt = [
                    {"role": "user", "content": self.config.optim_str_init or ""},
                    {"role": "assistant", "content": conv[1]["content"]},
                ]
                prep_arg = [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": conv[1]["content"]},
                ]
            pre, atk_pre, prm, atk_suf, post, tgt = prepare_conversation(tokenizer, prep_arg, conv_opt)[0]
            seq = torch.cat([pre, atk_pre, prm, atk_suf, post, tgt])
            atk_mask = torch.cat(
                [
                    torch.zeros_like(pre),
                    torch.ones_like(atk_pre),
                    torch.zeros_like(prm),
                    torch.ones_like(atk_suf),
                    torch.zeros_like(post),
                    torch.zeros_like(tgt),
                ]
            )
            tgt_mask = torch.cat(
                [
                    torch.zeros_like(pre),
                    torch.zeros_like(atk_pre),
                    torch.zeros_like(prm),
                    torch.zeros_like(atk_suf),
                    torch.zeros_like(post),
                    torch.ones_like(tgt),
                ]
            ).roll(-1, 0)
            xs.append(seq)
            atk_masks.append(atk_mask)
            tgt_masks.append(tgt_mask)

        x = pad_sequence(xs, batch_first=True, padding_value=tokenizer.pad_token_id)
        atk_mask = pad_sequence(atk_masks, batch_first=True)
        tgt_mask = pad_sequence(tgt_masks, batch_first=True)
        attn_mask = (x != tokenizer.pad_token_id).long()
        y = x.clone()
        y[:, :-1] = x[:, 1:]
        return x, y, atk_mask, tgt_mask, attn_mask, originals

    def _attack_batch(
        self,
        model: torch.nn.Module,
        tokenizer,
        original_convs_batch: List[List[dict]],
        x: torch.Tensor,
        y: torch.Tensor,
        attn_mask: torch.Tensor,
        atk_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> List[SingleAttackRunResult]:
        device = model.device
        x, y, attn_mask, atk_mask, tgt_mask = (
            t.to(device) for t in (x, y, attn_mask, atk_mask, tgt_mask)
        )
        num_examples = x.size(0)
        mut_idx = [torch.nonzero(m, as_tuple=True)[0] for m in atk_mask]

        best_seq = x.clone()
        best_loss = torch.full((num_examples,), float("inf"), device=device)

        step_convs, step_tokens, losses, times = (
            [[] for _ in range(num_examples)] for _ in range(4)
        )
        t0 = time.time()

        def mutate(seq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            """Flip ≤ neighbourhood_radius tokens within *positions*."""
            n = len(positions)
            if n == 0:
                return seq
            k = min(self.config.neighborhood_radius, n)
            sel = positions[torch.randperm(n, device=device)[:k]]
            seq_new = seq.clone()
            seq_new[sel] = self._get_random_allowed_tokens(model, tokenizer, k)
            return seq_new

        pbar = trange(self.config.num_steps, desc="random-search")
        for step in pbar:
            t_step = time.time()

            # Build all neighbours in one dense tensor
            cand_batches = []
            for _ in range(self.config.candidates_per_generation):
                c = best_seq.clone()
                for i in range(num_examples):
                    c[i] = mutate(best_seq[i], mut_idx[i])
                cand_batches.append(c)
            cand_stack = torch.cat(cand_batches, dim=0)  # (k·B, L)

            repeat_factor = self.config.candidates_per_generation
            attn_stack = attn_mask.repeat(repeat_factor, 1)
            y_stack = y.repeat(repeat_factor, 1)
            tgt_stack = tgt_mask.repeat(repeat_factor, 1)

            def _chunk_loss(seqs, am, y_, tgt_):
                logits = model(input_ids=seqs, attention_mask=am).logits
                lp_tok = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y_.view(-1), reduction="none"
                )
                lp_tok = lp_tok * tgt_.view(-1)
                lp_per = lp_tok.view(seqs.size(0), -1)
                tgt_len = tgt_.sum(dim=1)
                return torch.where(tgt_len > 0, lp_per.sum(dim=1) / tgt_len, torch.zeros_like(tgt_len, dtype=torch.float))

            cand_loss: torch.Tensor = with_max_batchsize(
                _chunk_loss, cand_stack, attn_stack, y_stack, tgt_stack
            )  # type: ignore
            cand_loss = cand_loss.view(repeat_factor, num_examples)  # (k, B)

            # ---- Greedy accept if better ---------------------------------
            for k_idx in range(repeat_factor):
                better = cand_loss[k_idx] < best_loss
                if better.any():
                    best_loss = torch.where(better, cand_loss[k_idx], best_loss)
                    best_seq = torch.where(
                        better.unsqueeze(1), cand_batches[k_idx], best_seq
                    )

            for i in range(num_examples):
                atk_tokens = best_seq[i, mut_idx[i]]
                atk_str = tokenizer.decode(atk_tokens, skip_special_tokens=False)
                user_content = (
                    original_convs_batch[i][0]["content"] + atk_str
                    if self.config.placement == "suffix"
                    else atk_str
                )
                conv_rec = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": ""},
                ]
                step_convs[i].append(conv_rec)
                step_tokens[i].append(best_seq[i].clone())
                losses[i].append(best_loss[i].item())
                times[i].append(time.time() - t_step)

            pbar.set_postfix({"avg_loss": f"{best_loss.mean().item():.4f}"})

        # Final generation of model outputs for every stored step
        flat_tokens = [t for ex in step_tokens for t in ex]
        outputs = generate_ragged_batched(
            model,
            tokenizer,
            token_list=flat_tokens,
            initial_batch_size=len(flat_tokens),
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )

        # ---- Collate ------------------------------------------------------
        total_time = time.time() - t0
        runs: List[SingleAttackRunResult] = []
        for i in range(num_examples):
            step_results = []
            for s in range(self.config.num_steps):
                out = outputs[i * self.config.num_steps + s]
                step_results.append(
                    AttackStepResult(
                        step=s,
                        model_completions=out,
                        time_taken=times[i][s],
                        loss=losses[i][s],
                        model_input=step_convs[i][s],
                        model_input_tokens=step_tokens[i][s].tolist(),
                    )
                )
            runs.append(
                SingleAttackRunResult(
                    original_prompt=original_convs_batch[i],
                    steps=step_results,
                    total_time=total_time / num_examples,
                )
            )
        return runs
