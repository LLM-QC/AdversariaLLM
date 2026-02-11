"""Best-of-N attack.

@article{hughes2024best,
  title={Best-of-n jailbreaking},
  author={Hughes, John and Price, Sara and Lynch, Aengus and Schaeffer, Rylan and Barez, Fazl and Koyejo, Sanmi and Sleight, Henry and Jones, Erik and Perez, Ethan and Sharma, Mrinank},
  journal={arXiv preprint arXiv:2412.03556},
  year={2024}
}
"""

import copy
import logging
import random
import time
from dataclasses import dataclass, field

import torch
import transformers
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ..dataset import PromptDataset
from ..lm_utils import generate_ragged_batched, get_losses_batched, prepare_conversation
from ..types import Conversation
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult


@dataclass
class BonConfig:
    """Config for the Bon attack."""
    name: str = "bon"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    num_steps: int = 100
    sigma: float = 0.4
    word_scrambling: bool = True
    random_capitalization: bool = True
    ascii_perturbation: bool = True


def apply_word_scrambling(text: str, sigma: float) -> str:
    """
    Scrambles the middle characters of words longer than 3 characters in the input text.
    The probability of scrambling is determined by sigma.

    Example:
    Input: "The quick brown fox jumps"
    Output: "The qiuck bwron fox jpums"
    """
    words = text.split()
    scrambled_words = []
    for word in words:
        if len(word) > 3 and random.random() < sigma ** (1 / 2):
            chars = list(word)
            middle_chars = chars[1:-1]
            random.shuffle(middle_chars)
            scrambled_word = chars[0] + "".join(middle_chars) + chars[-1]
            scrambled_words.append(scrambled_word)
        else:
            scrambled_words.append(word)
    return " ".join(scrambled_words)


def apply_random_capitalization(text: str, sigma: float) -> str:
    """
    Randomly capitalizes letters in the input text.

    Input: "The quick brown fox jumps"
    Output: "The qUick bRoWn fOx jUmps"
    """
    new_text = []
    for c in text:
        if c.isalpha() and random.random() < sigma ** (1 / 2):
            if "a" <= c <= "z":
                new_text.append(chr(ord(c) - 32))  # Convert to uppercase
            elif "A" <= c <= "Z":
                new_text.append(chr(ord(c) + 32))  # Convert to lowercase
        else:
            new_text.append(c)
    return "".join(new_text)


def apply_ascii_noising(text: str, sigma: float) -> str:
    """
    Perturbs the ASCII characters of the input text.

    Example:
    Input: "The quick brown fox jumps"
    Output: "Tge quick brown fox junps"
    """
    new_text = []
    for c in text:
        if c.isprintable() and random.random() < sigma**3:
            perturbation = random.choice([-1, 1])
            new_char_code = ord(c) + perturbation
            # Ensure new character is printable ASCII
            if 32 <= new_char_code <= 126:
                new_text.append(chr(new_char_code))
            else:
                new_text.append(c)
        else:
            new_text.append(c)
    return "".join(new_text)


def process_text_augmentation(
    text: str,
    sigma: float,
    seed: int,
    word_scrambling: bool,
    random_capitalization: bool,
    ascii_perturbation: bool,
):
    if seed is not None:
        random.seed(seed)

    if word_scrambling:
        text = apply_word_scrambling(text, sigma)
    if random_capitalization:
        text = apply_random_capitalization(text, sigma)
    if ascii_perturbation:
        text = apply_ascii_noising(text, sigma)

    return text


class BonAttack(Attack):
    """A baseline attack that simply prompts the model with the original prompt."""
    def __init__(self, config: BonConfig):
        super().__init__(config)

        if self.config.generation_config.temperature != 1.0:
            logging.warning(f"The Best-of-N paper used temperature=1.0, but you have set temperature={self.config.generation_config.temperature}.")

    @torch.no_grad
    def run(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        dataset: PromptDataset,
    ) -> AttackResult:
        """Run the Best-of-N attack on the given dataset.

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
        modified_conversations: list[Conversation] = []
        full_token_tensors_list: list[torch.Tensor] = []
        prompt_token_tensors_list: list[torch.Tensor] = []
        target_token_tensors_list: list[torch.Tensor] = []

        for conversation in tqdm(dataset, desc="Preparing inputs"):
            # Assuming conversation = [{'role': 'user', ...}, {'role': 'assistant', ...}]
            assert len(conversation) == 2, "Best-of-N attack currently assumes single-turn conversation."
            original_conversations.append(conversation)
            for j in range(self.config.num_steps):
                c = copy.deepcopy(conversation)
                c[0]["content"] = process_text_augmentation(
                    c[0]["content"],
                    sigma=self.config.sigma,
                    seed=self.config.seed + j,
                    word_scrambling=self.config.word_scrambling,
                    random_capitalization=self.config.random_capitalization,
                    ascii_perturbation=self.config.ascii_perturbation
                )
                modified_conversations.append(c)
                token_tensors = prepare_conversation(tokenizer, c)
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
        logging.info(f"Calculating losses...")
        # Calculate loss for the full sequences
        with torch.no_grad():
            all_losses_per_token = get_losses_batched(
                model,
                targets=shifted_target_tensors_list,
                token_list=full_token_tensors_list,
                initial_batch_size=max(B, 128),
            )

        # Extract average loss *only* over the target tokens for each instance
        instance_losses = []
        for i in range(B * self.config.num_steps):
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
        logging.info(f"Loss calculation time: {loss_time_total:.2f}s")
        # --- 3. Generate Completions ---
        logging.info(f"Generating completions...")
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
            initial_batch_size=max(B, self.config.num_steps, 1024),
            verbose=True,
        )
        t_end_gen = time.time()
        gen_time_total = t_end_gen - t_start_gen
        logging.info(f"Generation time: {gen_time_total:.2f}s, per instance: {gen_time_total/B:.2f}s, per sequence: {gen_time_total/B/self.config.num_steps:.2f}s")
        t1 = time.time()
        # --- 4. Assemble Results ---
        runs = []
        for i in range(B):
            original_prompt = original_conversations[i]
            model_input = copy.deepcopy(original_prompt)
            model_input[-1]["content"] = ""
            step_results = []
            for j in range(self.config.num_steps):
                model_completions = completions[i * self.config.num_steps + j]
                loss = instance_losses[i * self.config.num_steps + j]
                # Get token lists (convert tensors to lists of ints)
                model_input = modified_conversations[i * self.config.num_steps + j]
                model_input[-1]["content"] = ""
                model_input_tokens = prompt_token_tensors_list[i * self.config.num_steps + j].tolist()

                step_result = AttackStepResult(
                    step=j,
                    model_completions=model_completions,
                    time_taken=(t1 - t0) / B / self.config.num_steps,
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

        logging.info(f"Best-of-N attack run completed. Total Time: {t1 - t0:.2f}s, "
                     f"Generation Time: {gen_time_total:.2f}s, Loss Calc Time: {loss_time_total:.2f}s")

        return AttackResult(runs=runs)
