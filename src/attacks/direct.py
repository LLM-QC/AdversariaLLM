"""Baseline, just prompts the original prompt."""

import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import transformers

from src.attacks import Attack, AttackResult
from src.lm_utils import generate_ragged, get_batched_losses, prepare_tokens


@dataclass
class DirectConfig:
    name: str = "direct"
    type: str = "discrete"
    placement: Optional[str] = None
    generate_completions: Literal["all", "best", "last"] = "all"
    num_steps: int = 1
    seed: int = 0
    batch_size: int = 4
    max_new_tokens: int = 256


class DirectAttack(Attack):
    def __init__(self, config: DirectConfig):
        super().__init__(config)

    @torch.no_grad
    def run(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        dataset: torch.utils.data.Dataset,
    ) -> AttackResult:
        result = AttackResult([], [], [], [], [])
        token_lists = []
        for msg, target in dataset:
            result.prompts.append(msg)
            result.attacks.append([""])
            token_lists.append(
                prepare_tokens(
                    tokenizer,
                    msg["content"],
                    target,
                    attack="",
                    placement="suffix",
                )
            )

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for chunk in chunks(token_lists, self.config.batch_size):
            t0 = time.time()
            completions = generate_ragged(
                model,
                tokenizer,
                token_list=[torch.cat([a, b, c, d], dim=0) for a, b, c, d, e in chunk],
                max_new_tokens=self.config.max_new_tokens,
            )
            token_list = [torch.cat(tokens, dim=0) for tokens in chunk]
            targets = [t.roll(-1, 0) for t in token_list]
            losses = get_batched_losses(model, targets, token_list=token_list)
            losses = [
                l[-tl:].mean().item()
                for l, tl in zip(losses, [e.size(0) for a, b, c, d, e in chunk])
            ]
            for l, c in zip(losses, completions):
                result.losses.append([l])
                result.completions.append([c])
                result.times.append([time.time() - t0])
        return result
