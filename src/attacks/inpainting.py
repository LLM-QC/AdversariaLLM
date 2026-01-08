"""Inpainting (diffusion) transfer attack.

@article{lüdke2025diffusionllmsnaturaladversaries,
      title={Diffusion LLMs are Natural Adversaries for any LLM},
      author={David Lüdke and Tom Wollschläger and Paul Ungermann and Stephan Günnemann and Leo Schwinn},
      year={2025},
}
"""

import copy
import logging
import time
from dataclasses import dataclass, field

import pandas as pd
import torch
import transformers
from beartype.typing import Optional
from huggingface_hub import hf_hub_download

from ..dataset import PromptDataset
from ..lm_utils import generate_ragged_batched, get_losses_batched, prepare_conversation
from ..types import Conversation
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult


@dataclass
class InpaintingConfig:
    """Config for the Inpainting attack."""

    name: str = "inpainting"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    custom_data_path: Optional[str] = None  # Optional custom data path
    num_samples_per_behavior: int = 1024


class InpaintingAttack(Attack):
    """Basic transfer attack implementation, which loads attack prompts generated with a diffusion language model via inpainting
    with an affirmative target and queries the target model with it.

    Currently only the jbb dataset is supported.
    """

    def __init__(self, config: InpaintingConfig):
        super().__init__(config)
        # Load and sample inpainting attack data into memory for lookup during the attack run.

        if config.custom_data_path is not None:
            data_path = config.custom_data_path
        else:
            # Download the dataset from Hugging Face Hub if not cached.
            data_path = hf_hub_download(
                repo_id="davecasp/inpainting_attack_large",
                filename="inpainting_attack.csv",
                repo_type="dataset",
            )

        self.inpainting_data = pd.read_csv(data_path)
        # Downsample variants per behavior; seed ensures reproducibility.
        if config.num_samples_per_behavior is not None:
            self.inpainting_data = (
                self.inpainting_data.groupby("original_prompt")
                .apply(
                    lambda x: x.sample(n=min(config.num_samples_per_behavior, len(x)), random_state=config.seed),
                    include_groups=False,
                )
                .reset_index(level=0)
            )
        assert all(self.inpainting_data["original_prompt"].value_counts() == config.num_samples_per_behavior), (
            "Not all behaviors have the specified number of inpainting samples after sampling."
        )

    @torch.no_grad
    def run(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataset: PromptDataset,
    ) -> AttackResult:
        """Run the Inpainting attack on the given dataset.
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
        input_conversations: list[Conversation] = []
        original_conversations: list[Conversation] = []
        full_token_tensors_list: list[torch.Tensor] = []
        prompt_token_tensors_list: list[torch.Tensor] = []
        target_token_tensors_list: list[torch.Tensor] = []

        inpainting_prompts_per_prompt = 0
        for conversation in dataset:
            conversations = match_inpainting_prompts(conversation, self.inpainting_data)
            original_conversations.append(copy.deepcopy(conversation))
            if inpainting_prompts_per_prompt == 0:
                inpainting_prompts_per_prompt = len(conversations)
            assert len(conversations) == inpainting_prompts_per_prompt, (
                "All prompts in the dataset must have the same number of inpainting variants."
            )

            for conversation in conversations:
                # Assuming conversation = [{'role': 'user', ...}, {'role': 'assistant', ...}]
                assert len(conversation) == 2, "Inpainting attack currently assumes single-turn conversation."
                input_conversations.append(conversation)

                token_tensors = prepare_conversation(tokenizer, conversation)
                flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]

                # Concatenate all turns for the full input/target context
                full_token_tensors_list.append(torch.cat(flat_tokens, dim=0))

                # Identify prompt tokens (everything before the target assistant turn)
                prompt_token_tensors_list.append(torch.cat(flat_tokens[:-1]))
                target_token_tensors_list.append(flat_tokens[-1])

        # --- 2. Calculate Losses ---
        B = len(input_conversations)
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
                verbose=True,
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
            target_token_losses = all_losses_per_token[i][prompt_len - 1 : full_len - 1]
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
            verbose=True,
        )
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen

        t1 = time.time()
        # --- 4. Assemble Results ---
        num_original_prompts = B // inpainting_prompts_per_prompt

        runs = []
        for i in range(num_original_prompts):
            step_results = []
            original_prompt = original_conversations[i]
            for j in range(inpainting_prompts_per_prompt):
                idx = i * inpainting_prompts_per_prompt + j
                model_input = copy.deepcopy(input_conversations[idx])
                model_input[-1]["content"] = ""
                model_completions = completions[idx]
                loss = instance_losses[idx]

                # Get token lists (convert tensors to lists of ints)
                model_input_tokens = prompt_token_tensors_list[idx].tolist()

                # Create the result for this inpainting attack step
                step_result = AttackStepResult(
                    step=j,
                    model_completions=model_completions,
                    time_taken=(t1 - t0) / B,
                    loss=loss,
                    flops=0,
                    model_input=model_input,
                    model_input_tokens=model_input_tokens,
                )
                step_results.append(step_result)

            # Create the result for this single run
            run_result = SingleAttackRunResult(
                original_prompt=original_prompt,
                steps=step_results,
                total_time=t1 - t0,  # Total time for this run is the instance time
            )

            runs.append(run_result)

        logging.info(
            f"Inpainting attack run completed. Total Time: {t1 - t0:.2f}s, "
            f"Generation Time: {gen_time_total:.2f}s, Loss Calc Time: {loss_time_total:.2f}s"
        )

        return AttackResult(runs=runs)


def match_inpainting_prompts(
    conversation: Conversation, inpainting_data: pd.DataFrame
) -> list[Conversation]:
    """Matches the conversation prompt with inpainting data and returns multiple modified conversations.

    Parameters:
    ----------
        conversation: The original conversation.
        inpainting_data: DataFrame containing inpainting prompts and completions.

    Returns:
    -------
        list: List of modified conversations with inpainted prompts.
    """
    original_prompt = conversation[0]["content"]
    # Find all matching inpainting entries
    matched_rows = inpainting_data[inpainting_data["original_prompt"] == original_prompt]

    if matched_rows.empty:
        raise ValueError(f"No matching inpainting prompt found for: {original_prompt}")

    # Create a conversation for each inpainted prompt
    conversations = []
    for _, row in matched_rows.iterrows():
        new_conv = copy.deepcopy(conversation)
        new_conv[0]["content"] = row["inpainted_prompt"]
        conversations.append(new_conv)

    return conversations