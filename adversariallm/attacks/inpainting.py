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
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import torch
import transformers
from beartype.typing import Optional
from huggingface_hub import hf_hub_download

from ..defenses import create_defended_text_generator
from ..dataset import PromptDataset
from ..lm_utils import LocalTextGenerator, get_losses_batched, prepare_conversation
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

        self.inpainting_data = self._load_inpainting_data(data_path)
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

    @staticmethod
    def _load_inpainting_data(path: str) -> pd.DataFrame:
        suffix = Path(path).suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix in {"", ".csv", ".txt"}:
            return pd.read_csv(path)
        raise ValueError(
            f"Unsupported inpainting data format: {path}. "
            "Supported extensions: .csv, .tsv, .parquet"
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
        runtime_defense_cfg = getattr(self.config, "defense", None)
        use_runtime_defense = bool(runtime_defense_cfg and runtime_defense_cfg.get("type", "none") != "none")

        # --- 1. Prepare Inputs ---
        generation_conversations: list[Conversation] = []
        original_conversations: list[Conversation] = []
        loss_full_token_tensors_list: list[torch.Tensor] = []
        generation_prompt_token_tensors_list: list[torch.Tensor] = []

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
                conv_for_generation = copy.deepcopy(conversation)
                conv_for_generation[-1]["content"] = ""
                generation_conversations.append(conv_for_generation)

                prompt_token_tensors = prepare_conversation(tokenizer, conv_for_generation)
                prompt_flat_tokens = [t for turn_tokens in prompt_token_tensors for t in turn_tokens]
                generation_prompt_token_tensors_list.append(torch.cat(prompt_flat_tokens, dim=0))

                if not use_runtime_defense:
                    token_tensors = prepare_conversation(tokenizer, conversation)
                    flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
                    # Concatenate all turns for the full input/target context
                    loss_full_token_tensors_list.append(torch.cat(flat_tokens, dim=0))

        B = len(generation_conversations)
        runtime_generator = LocalTextGenerator(model, tokenizer)
        runtime_generator = create_defended_text_generator(runtime_defense_cfg, base_generator=runtime_generator)

        # --- 2. Calculate Losses ---
        loss_time_total = 0.0
        if use_runtime_defense:
            logging.info("Runtime defense enabled for inpainting; skipping internal loss computation.")
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
                    verbose=True,
                )

            # Extract average loss *only* over the target tokens for each instance
            instance_losses = []
            for i in range(B):
                full_len = loss_full_token_tensors_list[i].size(0)
                prompt_len = generation_prompt_token_tensors_list[i].size(0)
                target_token_losses = all_losses_per_token[i][prompt_len - 1 : full_len - 1]
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
            initial_batch_size=B,
            verbose=True,
        )
        completions = generation_result.gen
        completions_raw: list[list[str]] | None = generation_result.raw_gen
        defense_decisions: list[list[dict]] | None = generation_result.defense_decisions
        generation_input_ids = generation_result.input_ids
        if generation_input_ids is None:
            raise ValueError(
                "Inpainting requires `generation_result.input_ids` from the runtime generator "
                "(expected for local/defended-local generators)."
            )
        if len(generation_input_ids) != B:
            raise ValueError(
                "Inpainting received mismatched `generation_result.input_ids` length: "
                f"expected {B}, got {len(generation_input_ids)}."
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
                model_input = copy.deepcopy(generation_conversations[idx])
                model_completions = completions[idx]
                loss = instance_losses[idx]
                model_input_tokens = generation_input_ids[idx]

                # Create the result for this inpainting attack step
                step_result = AttackStepResult(
                    step=j,
                    model_completions=model_completions,
                    model_completions_raw=completions_raw[idx] if completions_raw is not None else None,
                    time_taken=(t1 - t0) / B,
                    loss=loss,
                    flops=0,
                    model_input=model_input,
                    model_input_tokens=model_input_tokens,
                    defense_metadata=[d.get("metadata", d) for d in defense_decisions[idx]]
                    if defense_decisions is not None
                    else None,
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
