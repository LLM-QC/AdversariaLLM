"""Implementation of the BEAST attack.

Adapted from https://github.com/dreadnode/research/blob/main/notebooks/Mistral%20-%20BEAST%20Beam%20Attack.ipynb

# TODO: add prefix cache to get_perplexity
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from tqdm import trange

from src.lm_utils import generate_ragged_batched, prepare_tokens, with_max_batchsize
from transformers import AutoTokenizer, AutoModelForCausalLM

from .attack import Attack, AttackResult


@dataclass
class BEASTConfig:
    name: str = "beast"
    type: str = "discrete"
    placement: str = "suffix"
    generate_completions: Literal["all", "best", "last"] = "all"
    num_steps: int = 30  # also the suffix length
    seed: int = 0
    batch_size: int = 2
    optim_str_init: str = ""
    k1: float = 10.0
    k2: float = 10.0
    temperature: float = 1.0
    max_new_tokens: int = 256


class BEASTAttack(Attack):
    def __init__(self, config: BEASTConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size
        assert (
            self.config.temperature > 0.0
        ), "Temperature must be greater than 0 for BEAST"

    @torch.no_grad()
    def run(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset) -> AttackResult:
        """
        Runs the BEASTAttack on a given model and dataset.

        Args:
            model (torch.nn.Module): The language model to be attacked.
            tokenizer: Tokenizer compatible with the model.
            dataset: Iterable of (message, target) pairs containing the input prompts
                and the desired target strings.

        Returns:
            AttackResult: Holds all data about the generated attacks, losses, prompts,
                completions, and execution times.
        """
        t0 = time.time()
        num_examples = len(dataset)
        completions: list[list[str]] = [[] for _ in range(num_examples)]
        attacks, losses, times, prompts, token_list = [], [], [], [], []

        for prompt_dict, target_string in dataset:
            pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens = (
                prepare_tokens(
                    tokenizer,
                    prompt_dict["content"],
                    target_string,
                    attack=self.config.optim_str_init,
                    placement=self.config.placement,
                )
            )
            prompts.append(prompt_dict)

            initial_ppl = self.get_perplexity(
                model,
                pre_tokens,
                prompt_tokens,
                [attack_tokens],
                post_tokens,
                target_tokens,
            )[0]
            per_sample_attacks = [""]
            per_sample_losses = [torch.log(torch.tensor(initial_ppl)).item()]
            beams = self.sample(
                model,
                k=self.config.k1,
                pre_tokens=pre_tokens,
                prompt_tokens=prompt_tokens,
                attack_token_list=[attack_tokens],
                post_tokens=post_tokens
            )[0]  # shape is (k1,)
            beams: list[torch.LongTensor] = [torch.tensor([b]) for b in beams]
            for i in (pbar := trange(1, self.config.num_steps)):
                # Get next K1 x K2 candidates
                candidates = []
                next_tokens = self.sample(
                    model,
                    self.config.k2,
                    pre_tokens,
                    prompt_tokens,
                    beams,
                    post_tokens,
                )  # (k1, k2)

                for beam, next_token in zip(beams, next_tokens):
                    candidates.extend([torch.cat((beam, t.unsqueeze(0))) for t in next_token])

                # Score them
                scores = self.get_perplexity(
                    model,
                    pre_tokens,
                    prompt_tokens,
                    candidates,
                    post_tokens,
                    target_tokens,
                )
                # Take the K1 best by lowest score
                sorting = sorted(range(len(scores)), key=lambda i: scores[i])
                beams = [candidates[i] for i in sorting[: self.config.k1]]

                best_suffix = tokenizer.decode(candidates[sorting[0]])
                best_score = scores[sorting[0]]
                per_sample_attacks.append(best_suffix)
                per_sample_losses.append(torch.log(torch.tensor(best_score)).item())
                pbar.set_postfix({"loss": best_score, "attack": best_suffix})
                times.append(time.time() - t0)
            attacks.append(per_sample_attacks)
            losses.append(per_sample_losses)

            token_list.extend(
                [
                    torch.cat(
                        prepare_tokens(
                            tokenizer,
                            prompt_dict["content"],
                            target_string,
                            attack=attack,
                            placement=self.config.placement,
                        )[:4]
                    )
                    for attack in per_sample_attacks
                ]
            )

        outputs = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            initial_batch_size=128,
            max_new_tokens=self.config.max_new_tokens,
        )
        for i, output in enumerate(outputs):
            completions[i // len(attacks[0])].append(output)

        return AttackResult(
            attacks=attacks,
            losses=losses,
            prompts=prompts,
            completions=completions,
            times=times,
        )

    def get_perplexity(
        self,
        model,
        pre_tokens,
        prompt_tokens,
        attack_tokens_list,
        post_tokens,
        target_tokens,
    ) -> float:
        tensor = torch.stack([torch.cat(
            [
                pre_tokens,
                prompt_tokens,
                torch.tensor(attack_tokens),
                post_tokens,
                target_tokens,
            ]
        ) for attack_tokens in attack_tokens_list])

        def get_log_probs(target_tokens, x):
            # Get the relevant logits and softmax
            logits = model(input_ids=x.to(model.device)).logits
            output_logits = logits[:, -target_tokens.shape[0] :]
            log_probs = torch.nn.functional.log_softmax(output_logits, dim=-1).cpu()

            # Repeat target_tokens to match batch size
            target_tokens = target_tokens.unsqueeze(0).repeat(log_probs.size(0), 1).unsqueeze(-1)

            # Calculate perplexity
            gathered_log_probs = log_probs.gather(2, target_tokens).squeeze(-1)
            return gathered_log_probs.mean(dim=1)

        get_log_probs = partial(get_log_probs, target_tokens)
        # Push everything but the last token through
        mean_log_probs = with_max_batchsize(get_log_probs, tensor[:, :-1], initial_batch_size=128)

        perplexity = torch.exp(-mean_log_probs).tolist()
        return perplexity

    def sample(
        self,
        model,
        k,
        pre_tokens,
        prompt_tokens,
        attack_token_list,
        post_tokens,
    ) -> torch.LongTensor:
        tensor = torch.stack([
            torch.cat(
                [pre_tokens, prompt_tokens, torch.tensor(attack_tokens), post_tokens]
            )
            for attack_tokens in attack_token_list
        ]).to(model.device)
        logits = model(tensor).logits[:, -1, :]
        probs = torch.softmax(logits / self.config.temperature, dim=-1)
        tokens = torch.multinomial(probs, k, replacement=False)
        return tokens.cpu()  # 1-D
