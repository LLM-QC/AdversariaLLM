"""Baseline, just prompts with the original prompt."""

import copy
import logging
import time
from dataclasses import dataclass, field

import torch
import transformers

from .attack import (Attack, AttackResult, AttackStepResult,
                     GenerationConfig, SingleAttackRunResult)
from ..lm_utils import (generate_ragged_batched, get_losses_batched,
                        prepare_conversation)
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
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        dataset: torch.utils.data.Dataset,
    ) -> AttackResult:
        """Run the Direct attack on the given dataset.

        Parameters:
        ----------
            model: The model to attack.
            tokenizer: The tokenizer to use.
            dataset: The dataset to attack.

        Returns:
        -------
            AttackResult: The result of the attack
        """
        t0 = time.time()

        # --- 1. Prepare Inputs ---
        original_conversations: list[Conversation] = []
        full_token_tensors_list: list[torch.Tensor] = []
        prompt_token_tensors_list: list[torch.Tensor] = []
        target_token_tensors_list: list[torch.Tensor] = []

        for conversation in dataset:
            # Assuming conversation = [{'role': 'user', ...}, {'role': 'assistant', ...}]
            assert len(conversation) == 2, "Direct attack currently assumes single-turn conversation."
            original_conversations.append(conversation)

            token_tensors = prepare_conversation(tokenizer, conversation)
            flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]

            # Concatenate all turns for the full input/target context
            full_token_tensors_list.append(torch.cat(flat_tokens, dim=0))

            # Identify prompt tokens (everything before the target assistant turn)
            prompt_token_tensors_list.append(torch.cat(flat_tokens[:-1]))
            target_token_tensors_list.append(flat_tokens[-1])

        # --- 2. Calculate Losses ---
        B = len(original_conversations)
        t_start_loss = time.time()
        # We need targets shifted by one position for standard next-token prediction loss
        shifted_target_tensors_list = [t.roll(-1, 0) for t in full_token_tensors_list]

        # Calculate loss for the full sequences
        with torch.no_grad():
            all_losses_per_token = get_losses_batched(
                model,
                targets=shifted_target_tensors_list,
                token_list=full_token_tensors_list,
                initial_batch_size=B,
            )

        # Extract average loss *only* over the target tokens for each instance
        instance_losses = []
        for i in range(B):
            full_len = full_token_tensors_list[i].size(0)
            prompt_len = prompt_token_tensors_list[i].size(0)
            # Loss corresponds to predicting token i+1 given tokens 0..i
            # We want loss for predicting target tokens, which start at index `prompt_len`
            # The relevant loss values are at indices `prompt_len-1` to `full_len-2`
            # (inclusive start, exclusive end for slicing)
            target_token_losses = all_losses_per_token[i][prompt_len-1:full_len-1]
            if target_token_losses.numel() > 0:
                avg_loss = target_token_losses.mean().item()
            else:
                avg_loss = None  # Handle cases with empty targets if necessary
            instance_losses.append(avg_loss)

        t_end_loss = time.time()
        loss_time_total = t_end_loss - t_start_loss

        # --- 3. Generate Completions ---
        t_start_gen = time.time()
        completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=prompt_token_tensors_list,  # Generate from the prompt tokens
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=B,
        )
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen

        t1 = time.time()
        # --- 4. Assemble Results ---
        runs = []
        for i in range(B):
            original_prompt = original_conversations[i]
            model_input = copy.deepcopy(original_prompt)
            model_input[-1]["content"] = ""
            model_completions = completions[i]
            loss = instance_losses[i]

            # Get token lists (convert tensors to lists of ints)
            model_input_tokens = prompt_token_tensors_list[i].tolist()

            # Create the single step result for this direct "attack"
            step_result = AttackStepResult(
                step=0,
                model_completions=model_completions,
                time_taken=(t1 - t0) / B,
                loss=loss,
                flops=0,
                model_input=model_input,
                model_input_tokens=model_input_tokens,
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
