"""Baseline, just prompts the original prompt."""

import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import transformers

from src.attacks import Attack, AttackResult
from src.lm_utils import generate_ragged_batched, prepare_tokens


@dataclass
class PrefillingConfig:
    name: str = "prefilling"
    type: str = "discrete"
    placement: Optional[str] = None
    generate_completions: Literal["all", "best", "last"] = "all"
    num_steps: int = 1
    seed: int = 0
    batch_size: int = 4
    max_new_tokens: int = 256


class PrefillingAttack(Attack):
    def __init__(self, config: PrefillingConfig):
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
        targets = []
        for msg, target in dataset:
            result.prompts.append(msg)
            result.attacks.append([""])
            targets.append(target)
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
            completions = generate_ragged_batched(
                model,
                tokenizer,
                token_list=[torch.cat(c, dim=0) for c in chunk],
                max_new_tokens=self.config.max_new_tokens,
                initial_batch_size=self.config.batch_size,
            )
            for c, t in zip(completions, targets):
                result.losses.append([None])
                result.completions.append([t + c])
                result.times.append([time.time() - t0])
        return result
