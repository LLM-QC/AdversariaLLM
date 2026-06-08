"""Baseline, just prompts with the original prompt."""

import copy
import logging
import time
from dataclasses import dataclass, field

import torch
import transformers

from ..defenses import create_defended_text_generator
from ..lm_utils import LocalTextGenerator, get_losses_batched, prepare_conversation
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
        runtime_defense_cfg = getattr(self.config, "defense", None)
        use_runtime_defense = bool(runtime_defense_cfg and runtime_defense_cfg.get("type", "none") != "none")
        runtime_generator = LocalTextGenerator(model, tokenizer)
        runtime_generator = create_defended_text_generator(runtime_defense_cfg, base_generator=runtime_generator)

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

            token_tensors = prepare_conversation(tokenizer, conv_for_generation)
            flat_prompt_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
            generation_prompt_token_tensors_list.append(torch.cat(flat_prompt_tokens, dim=0))

            if not use_runtime_defense:
                token_tensors = prepare_conversation(tokenizer, conversation)
                flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
                # Concatenate all turns for the full input/target context
                loss_full_token_tensors_list.append(torch.cat(flat_tokens, dim=0))

        # --- 2. Calculate Losses ---
        B = len(original_conversations)
        loss_time_total = 0.0
        if use_runtime_defense:
            logging.info("Runtime defense enabled for direct; skipping internal loss computation.")
            instance_losses = [None] * B
        else:
            t_start_loss = time.time()
            # We need targets shifted by one position for standard next-token prediction loss
            shifted_target_tensors_list = [t.roll(-1, 0) for t in loss_full_token_tensors_list]

            # Calculate loss for the full sequences
            with torch.no_grad():
                all_losses_per_token = get_losses_batched(
                    model,
                    targets=shifted_target_tensors_list,
                    token_list=loss_full_token_tensors_list,
                    initial_batch_size=B,
                )

            # Extract average loss *only* over the target tokens for each instance
            instance_losses = []
            for i in range(B):
                full_len = loss_full_token_tensors_list[i].size(0)
                prompt_len = generation_prompt_token_tensors_list[i].size(0)
                target_token_losses = all_losses_per_token[i][prompt_len-1:full_len-1]
                if target_token_losses.numel() > 0:
                    avg_loss = target_token_losses.mean().item()
                else:
                    avg_loss = None
                instance_losses.append(avg_loss)

            t_end_loss = time.time()
            loss_time_total = t_end_loss - t_start_loss

        # --- 3. Generate Completions ---
        t_start_gen = time.time()
        generation_result = runtime_generator.generate(
            generation_conversations,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=B * self.config.generation_config.num_return_sequences,
        )
        completions = generation_result.gen
        completions_raw = generation_result.raw_gen
        defense_decisions = generation_result.defense_decisions
        generation_input_ids = generation_result.input_ids
        if generation_input_ids is None:
            raise ValueError(
                "Direct requires `generation_result.input_ids` from the runtime generator "
                "(expected for local/defended-local generators)."
            )
        if len(generation_input_ids) != B:
            raise ValueError(
                f"Direct received mismatched input_ids length: expected {B}, got {len(generation_input_ids)}."
            )
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
                model_completions_raw=completions_raw[i] if completions_raw is not None else None,
                time_taken=(t1 - t0) / B,
                loss=loss,
                flops=0,
                model_input=model_input,
                model_input_tokens=model_input_tokens,
                defense_metadata=[d.get("metadata", d) for d in defense_decisions[i]]
                if defense_decisions is not None
                else None,
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
