"""Baseline, just prompts with the original prompt."""

import copy
import logging
import time
from dataclasses import dataclass, field

import torch
import transformers

from ..defenses import TargetSystem
from ..lm_utils import prepare_conversation
from .attack import (Attack, AttackResult, AttackStepResult,
                     GenerationConfig, SingleAttackRunResult)
from ..types import Conversation


@dataclass
class DirectConfig:
    """Config for the Direct attack."""
    name: str = "direct"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0


class DirectAttack(Attack):
    """A baseline attack that simply prompts the model with the original prompt."""
    def __init__(self, config: DirectConfig):
        super().__init__(config)

    @torch.no_grad
    def run(
        self,
        target: TargetSystem,
        dataset: torch.utils.data.Dataset,
    ) -> AttackResult:
        """Run the Direct attack on the given dataset.

        Parameters:
        ----------
            dataset: The dataset to attack.
            target: Runtime interface for target-system generation and loss.

        Returns:
        -------
            AttackResult: The result of the attack
        """
        t0 = time.time()
        # --- 1. Prepare Inputs ---
        original_conversations: list[Conversation] = []
        generation_conversations: list[Conversation] = []
        loss_full_token_tensors_list: list[torch.Tensor] = []
        generation_prompt_token_tensors_list: list[torch.Tensor] = []

        for conversation in dataset:
            # Assuming conversation = [{'role': 'user', ...}, {'role': 'assistant', ...}]
            assert len(conversation) == 2, "Direct attack currently assumes single-turn conversation."
            original_conversations.append(conversation)
            conv_for_generation = copy.deepcopy(conversation)
            conv_for_generation[-1]["content"] = ""
            generation_conversations.append(conv_for_generation)

            token_tensors = prepare_conversation(target.tokenizer, conv_for_generation)
            flat_prompt_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
            generation_prompt_token_tensors_list.append(torch.cat(flat_prompt_tokens, dim=0))

            token_tensors = prepare_conversation(target.tokenizer, conversation)
            flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
            # Concatenate all turns for the full input/target context.
            loss_full_token_tensors_list.append(torch.cat(flat_tokens, dim=0))

        # --- 2. Calculate Losses ---
        B = len(original_conversations)
        loss_time_total = 0.0
        t_start_loss = time.time()
        instance_losses = target.loss(
            loss_full_token_tensors_list,
            generation_prompt_token_tensors_list,
            initial_batch_size=B,
        )
        t_end_loss = time.time()
        loss_time_total = t_end_loss - t_start_loss

        # --- 3. Generate Completions ---
        t_start_gen = time.time()
        generation_result = target.generate(
            generation_conversations,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=B * self.config.generation_config.num_return_sequences,
        )
        completions = generation_result.gen
        generation_input_ids = generation_result.require_input_ids("Direct", expected_len=B)
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen

        t1 = time.time()
        # --- 4. Assemble Results ---
        runs = []
        for i in range(B):
            original_prompt = original_conversations[i]
            model_input = copy.deepcopy(generation_conversations[i])
            model_completions = completions[i]
            loss = instance_losses[i]
            model_input_tokens = generation_input_ids[i]

            # Create the single step result for this direct "attack"
            step_result = AttackStepResult(
                step=0,
                model_completions=model_completions,
                model_completions_raw=generation_result.raw_for(i),
                time_taken=(t1 - t0) / B,
                loss=loss,
                flops=0,
                model_input=model_input,
                model_input_tokens=model_input_tokens,
                defense_metadata=generation_result.defense_metadata_for(i),
            )

            # Create the result for this single run
            run_result = SingleAttackRunResult(
                original_prompt=original_prompt,
                steps=[step_result],
                total_time=t1 - t0,  # Total time for this run is the instance time
            )

            runs.append(run_result)

        logging.info(f"Direct attack run completed. Total Time: {t1 - t0:.2f}s, "
                     f"Generation Time: {gen_time_total:.2f}s, Loss Calc Time: {loss_time_total:.2f}s")

        return AttackResult(runs=runs)
