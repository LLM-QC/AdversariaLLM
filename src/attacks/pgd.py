"""Implementation of a embedding-space continuous attack."""

import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from accelerate.utils import find_executable_batch_size
from tqdm import trange

from .attack import Attack, AttackResult


@dataclass
class PGDConfig:
    name: str = "pgd"
    type: str = "continuous"
    placement: str = "command"
    generate_completions: Literal["all", "best", "last"] = "last"
    num_steps: int = 30
    seed: int = 0
    batch_size: int = 2
    optim_str_init: str = ""
    epsilon: float = 100000.0
    alpha: float = 0.005
    generation_steps: int = 256


class PGDAttack(Attack):
    def __init__(self, config: PGDConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size
        if self.config.placement == "suffix":
            assert self.config.optim_str_init
        elif self.config.placement == "command":
            assert not self.config.optim_str_init

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        """The attack is in embedding"""
        num_examples = len(dataset)

        x: list = []
        attack_masks: list = []
        target_masks: list = []
        prompts = []
        for msg, target in dataset:
            token, attack_mask, target_mask = self._prepare_tokens(
                tokenizer, msg, target,
                attack_string=self.config.optim_str_init,
                placement=self.config.placement
            )
            prompts.append(msg)
            x.append(token)
            attack_masks.append(attack_mask)
            target_masks.append(target_mask)

        x = pad_sequence(
            x, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(model.device)
        target_masks = pad_sequence(target_masks, batch_first=True).to(model.device)
        attack_masks = pad_sequence(attack_masks, batch_first=True).to(model.device)
        attention_mask = (x != tokenizer.pad_token_id).long()

        y = x.clone()
        y[:, :-1] = x[:, 1:]

        def attack(batch_size):
            losses = [[] for _ in range(num_examples)]
            completions = [[] for _ in range(num_examples)]
            # Perform the actual attack
            for i in range(0, num_examples, batch_size):
                original_embeddings = model.get_input_embeddings()(
                    x[i : i + batch_size]
                )
                perturbed_embeddings = original_embeddings.clone().detach()
                for _ in trange(self.config.num_steps, file=sys.stderr):
                    perturbed_embeddings.requires_grad = True
                    model.zero_grad()
                    logits = model(
                        inputs_embeds=perturbed_embeddings,
                        attention_mask=attention_mask[i : i + batch_size],
                    ).logits
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y[i : i + batch_size].view(-1),
                        reduction="none",
                    )
                    loss = loss * target_masks[i : i + batch_size].view(-1)
                    loss = loss.view(perturbed_embeddings.size(0), -1).mean(dim=1)
                    loss.mean().backward()
                    for j, l in enumerate(loss.detach().tolist()):
                        losses[i + j].append(l)

                    with torch.no_grad():
                        perturbed_embeddings = (
                            perturbed_embeddings
                            - self.config.alpha
                            * perturbed_embeddings.grad.sign()
                            * attack_masks[i : i + batch_size, ..., None]
                        )
                        delta = self.project_l2(
                            perturbed_embeddings - original_embeddings
                        )
                        perturbed_embeddings = (original_embeddings + delta).detach()

                    if self.config.generate_completions == "all":
                        completion = self.get_completions(
                            model,
                            tokenizer,
                            perturbed_embeddings,
                            attack_masks[i : i + batch_size],
                            self.config.generation_steps,
                        )
                        for j, c in enumerate(completion):
                            completions[i + j].append(c)
                    elif self.config.generate_completions == "best":
                        for j in range(batch_size):
                            if losses[i + j][-1] == min(losses[i + j]):
                                completion = self.get_completions(
                                    model,
                                    tokenizer,
                                    perturbed_embeddings[j : j + 1],
                                    attack_masks[i + j : i + j + 1],
                                    self.config.generation_steps,
                                )
                                completions[i + j] = [completion[0]]
                if self.config.generate_completions == "last":
                    completion = self.get_completions(
                        model,
                        tokenizer,
                        perturbed_embeddings,
                        target_masks[i : i + batch_size],
                        self.config.generation_steps,
                    )
                    for j, c in enumerate(completion):
                        completions[i + j].append(c)

            return losses, completions

        # Run the attack
        losses, completions = find_executable_batch_size(attack, self.batch_size)()
        # assemble the results
        return AttackResult(
            attacks=[None] * num_examples,
            completions=completions,
            losses=losses,
            prompts=prompts,
        )

    @staticmethod
    @torch.no_grad
    def get_completions(
        model, tokenizer, perturbed_embeddings, loss_masks, gen_steps: int = 256
    ):
        # We first pad the embeddings to the maximum context length of the model.
        # Positions after the current one will be ignored via the attention mask.
        B, T = perturbed_embeddings.size()[:2]
        padding = (0, 0, 0, T + gen_steps + 2)
        padded_embeddings = F.pad(perturbed_embeddings, padding)
        # Then we generate the completions
        # find first non-zero index of loss mask
        next_token_idx = torch.argmax(loss_masks, dim=1)

        tokens = []
        done = torch.zeros(B, dtype=torch.bool, device=model.device)
        for step in range(gen_steps):
            max_idx = next_token_idx.max()
            logits = model(inputs_embeds=padded_embeddings[:, :max_idx + 1]).logits
            next_tokens = logits.argmax(dim=-1)[torch.arange(B), next_token_idx]
            next_embeds = model.get_input_embeddings()(next_tokens)
            padded_embeddings[torch.arange(B), next_token_idx + 1] = next_embeds

            tokens.append(next_tokens)
            next_token_idx += 1
            done |= (next_tokens == tokenizer.eos_token_id)
            if done.all():
                break

        completion = tokenizer.batch_decode(
            torch.stack(tokens, dim=0).T, skip_special_tokens=False
        )
        completion = [c.split(tokenizer.eos_token)[0] for c in completion]
        return completion

    def project_l2(self, delta):
        norm = delta.norm(p=2, dim=-1, keepdim=True)
        mask = norm > self.config.epsilon
        return torch.where(mask, delta * self.config.epsilon / norm, delta)

    def _prepare_tokens(self, tokenizer, prompt, target, attack_string=None, placement="suffix"):
        if placement == "prompt":
            attack_string = prompt['content']
            prompt = ""
        test_chat = [{'role': "user", 'content': 'XXXX'}]
        template = tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
        # Sometimes, the chat template adds the BOS token to the beginning of the template.
        # The tokenizer adds it again later, so we need to remove it to avoid duplication.
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")

        pre, post = template.split('XXXX')[:2]
        post_tokens = tokenizer(post, add_special_tokens=False, return_tensors="pt").input_ids
        # Fix for Llama tokenizer adding an extra underline token at the beginning:
        # https://github.com/huggingface/transformers/issues/26273
        if post_tokens[0][0] == 29871:
            post_tokens = post_tokens[0, 1:].unsqueeze(0)

        prompt = pre + prompt['content']
        attack = attack_string + post
        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
        attack_tokens = tokenizer(attack, return_tensors="pt", add_special_tokens=False).input_ids
        target_tokens = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids

        # Sanity check
        assert torch.allclose(attack_tokens, torch.cat((tokenizer(attack_string, return_tensors="pt", add_special_tokens=False).input_ids, post_tokens), dim=1))

        original_tokens = torch.cat([prompt_tokens, attack_tokens, target_tokens], dim=1)
        attack_mask = torch.zeros_like(original_tokens)
        # -post_tokens because we dont optimize the end-of-turn token
        attack_mask[:, prompt_tokens.size(1) : prompt_tokens.size(1) + attack_tokens.size(1) - post_tokens.size(1)] = 1
        target_mask = torch.zeros_like(original_tokens)
        target_mask[:, - target_tokens.size(1) - 1:-1] = 1  #  -1 to match with the label shift
        return original_tokens[0], attack_mask[0], target_mask[0]