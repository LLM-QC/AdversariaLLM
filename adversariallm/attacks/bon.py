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

from ..defenses import TargetSystem
from ..dataset import PromptDataset
from ..lm_utils import prepare_conversation
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
        target: TargetSystem,
        dataset: PromptDataset,
    ) -> AttackResult:
        """Run the Best-of-N attack on the given dataset.

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
        prompt_token_tensors_list: list[torch.Tensor] = []

        for conversation in tqdm(dataset, desc="Preparing inputs"):
            # Assuming conversation = [{'role': 'user', ...}, {'role': 'assistant', ...}]
            assert len(conversation) == 2, "Best-of-N attack currently assumes single-turn conversation."
            original_conversations.append(conversation)
            for j in range(self.config.num_steps):
                augmented_user = process_text_augmentation(
                    conversation[0]["content"],
                    sigma=self.config.sigma,
                    seed=self.config.seed + j,
                    word_scrambling=self.config.word_scrambling,
                    random_capitalization=self.config.random_capitalization,
                    ascii_perturbation=self.config.ascii_perturbation
                )

                generation_conv = [
                    {"role": "user", "content": augmented_user},
                    {"role": "assistant", "content": ""},
                ]
                generation_conversations.append(generation_conv)

                loss_conv = [
                    {"role": "user", "content": augmented_user},
                    {"role": "assistant", "content": conversation[1]["content"]},
                ]
                token_tensors = prepare_conversation(target.tokenizer, loss_conv)
                flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
                # Identify prompt tokens (everything before the target assistant turn)
                prompt_token_tensors_list.append(torch.cat(flat_tokens[:-1]))

                # Concatenate all turns for the full input/target context.
                loss_full_token_tensors_list.append(torch.cat(flat_tokens, dim=0))

        # --- 2. Calculate Losses ---
        B = len(original_conversations)
        t_start_loss = time.time()
        logging.info("Calculating losses...")
        instance_losses = target.loss(
            loss_full_token_tensors_list,
            prompt_token_tensors_list,
            initial_batch_size=max(B, 128),
        )
        t_end_loss = time.time()
        loss_time_total = t_end_loss - t_start_loss
        logging.info(f"Loss calculation time: {loss_time_total:.2f}s")
        # --- 3. Generate Completions ---
        logging.info(f"Generating completions...")
        t_start_gen = time.time()
        generation_result = target.generate(
            generation_conversations,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=max(B, self.config.num_steps, 1024),
            verbose=True,
        )
        completions = generation_result.gen
        generation_input_ids = generation_result.require_input_ids(
            "BON",
            expected_len=B * self.config.num_steps,
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
                model_input = copy.deepcopy(generation_conversations[i * self.config.num_steps + j])
                model_input_tokens = generation_input_ids[i * self.config.num_steps + j]

                step_result = AttackStepResult(
                    step=j,
                    model_completions=model_completions,
                    model_completions_raw=generation_result.raw_for(i * self.config.num_steps + j),
                    time_taken=(t1 - t0) / B / self.config.num_steps,
                    loss=loss,
                    flops=0,
                    model_input=model_input,
                    model_input_tokens=model_input_tokens,
                    defense_metadata=generation_result.defense_metadata_for(i * self.config.num_steps + j),
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
