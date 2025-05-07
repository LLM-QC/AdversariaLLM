"""We simply fill the beginning of the model response with an affirmative suffix."""

import time
from dataclasses import dataclass, field

import torch
import transformers

from src.attacks import Attack, AttackResult, GenerationConfig, SingleAttackRunResult, AttackStepResult
from src.lm_utils import generate_ragged_batched, prepare_conversation


@dataclass
class PrefillingConfig:
    name: str = "prefilling"
    type: str = "discrete"
    version: str = ""
    num_steps: int = 1
    seed: int = 0
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


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
        t_start = time.time()
        token_list = []
        # 1. Prepare inputs
        for conversation in dataset:
            toks = prepare_conversation(tokenizer, conversation)
            toks_flat = [t for turn_toks in toks for t in turn_toks]
            token_list.append(torch.cat(toks_flat))

        completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )
        t_end = time.time()
        B = len(completions)
        runs = []
        for i in range(B):
            # We manuall add back the prefilled prefix here since it is part of the model
            # turn
            completions_i = completions[i]
            for j in range(len(completions_i)):
                completions_i[j] = dataset[i][-1]["content"] + completions_i[j]
            run = SingleAttackRunResult(
                original_prompt=dataset[i],
                steps=[
                    AttackStepResult(
                        step=0,
                        model_completions=completions[i],
                        time_taken=(t_end - t_start) / B,
                        loss=None,
                        model_input=dataset[i],
                        model_input_tokens=token_list[i].tolist(),
                    )
                ],
                total_time=(t_end - t_start) / B,
            )
            runs.append(run)

        return AttackResult(runs=runs)
