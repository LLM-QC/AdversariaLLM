"""Implementation of a embedding-space continuous attack."""

import sys
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from accelerate.utils import find_executable_batch_size
from tqdm import trange

from src.lm_utils import prepare_tokens, get_batched_completions
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
    max_new_tokens: int = 256


class PGDAttack(Attack):
    def __init__(self, config: PGDConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size
        if self.config.placement == "suffix":
            assert self.config.optim_str_init
        elif self.config.placement == "command":
            assert not self.config.optim_str_init

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        x: list = []
        attack_masks: list = []
        target_masks: list = []
        prompts = []
        for msg, target in dataset:
            pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens = (
                prepare_tokens(
                    tokenizer,
                    msg['content'],
                    target,
                    attack=self.config.optim_str_init,
                    placement=self.config.placement,
                )
            )
            tokens = torch.cat(
                [pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens]
            )
            attack_mask = torch.cat(
                [
                    torch.zeros_like(pre_tokens),
                    torch.zeros_like(prompt_tokens),
                    torch.ones_like(attack_tokens),
                    torch.zeros_like(post_tokens),
                    torch.zeros_like(target_tokens),
                ]
            )
            target_mask = torch.cat(
                [
                    torch.zeros_like(pre_tokens),
                    torch.zeros_like(prompt_tokens),
                    torch.zeros_like(attack_tokens),
                    torch.zeros_like(post_tokens),
                    torch.ones_like(target_tokens),
                ]
            ).roll(-1, 0)
            prompts.append(msg)
            x.append(tokens)
            attack_masks.append(attack_mask)
            target_masks.append(target_mask)

        x = pad_sequence(
            x, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        target_masks = pad_sequence(target_masks, batch_first=True)
        attack_masks = pad_sequence(attack_masks, batch_first=True)
        attention_mask = (x != tokenizer.pad_token_id).long()

        y = x.clone()
        y[:, :-1] = x[:, 1:]
        # Run the attack
        losses, completions = find_executable_batch_size(self.attack_batched, self.batch_size)(x, y, attention_mask, attack_masks, target_masks, model, tokenizer)
        # assemble the results
        return AttackResult(
            attacks=[None] * x.size(0),
            completions=completions,
            losses=losses,
            prompts=prompts,
        )

    def attack_batched(self, batch_size, x, y, attention_mask, attack_masks, target_masks, model, tokenizer):
        num_examples = x.size(0)
        losses = [[] for _ in range(num_examples)]
        completions = [[] for _ in range(num_examples)]
        perturbed_embeddings_list = [[] for _ in range(num_examples)]
        # Perform the actual attack
        for i in range(0, num_examples, batch_size):
            x_batch = x[i : i + batch_size].to(model.device)
            y_batch = y[i : i + batch_size].to(model.device)
            attention_mask_batch = attention_mask[i : i + batch_size].to(model.device)
            attack_masks_batch = attack_masks[i : i + batch_size].to(model.device)
            target_masks_batch = target_masks[i : i + batch_size].to(model.device)

            original_embeddings = model.get_input_embeddings()(x_batch)
            perturbed_embeddings = original_embeddings.clone().detach()
            for _ in (pbar := trange(self.config.num_steps)):
                perturbed_embeddings.requires_grad = True
                model.zero_grad()
                logits = model(
                    inputs_embeds=perturbed_embeddings,
                    attention_mask=attention_mask_batch,
                ).logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_batch.view(-1),
                    reduction="none",
                )
                loss = loss.view(perturbed_embeddings.size(0), -1)
                loss = loss * target_masks_batch
                loss = loss.mean(dim=1)
                loss.mean().backward()
                for j, l in enumerate(loss.detach().tolist()):
                    losses[i + j].append(l)

                with torch.no_grad():
                    perturbed_embeddings = (
                        perturbed_embeddings
                        - self.config.alpha
                        * perturbed_embeddings.grad.sign()
                        * attack_masks_batch[..., None]
                    )
                    delta = self.project_l2(
                        perturbed_embeddings - original_embeddings
                    )
                    perturbed_embeddings = (original_embeddings + delta).detach()
                pbar.set_postfix({"loss": loss.mean().item()})
                if self.config.generate_completions == "all":
                    for j, (pe, tm) in enumerate(zip(perturbed_embeddings, target_masks_batch)):
                        perturbed_embeddings_list[i + j].append(pe[~(tm.roll(1, 0).cumsum(0).bool())].detach())
            if self.config.generate_completions == "last":
                embedding_list = [
                    pe[~(tm.roll(1, 0).cumsum(0).bool())].detach()
                    for pe, tm in zip(perturbed_embeddings, target_masks_batch)
                ]
                perturbed_embeddings_list[i:i+batch_size] = [[el] for el in embedding_list]

        flattened_embeddings = [e for el in perturbed_embeddings_list for e in el]
        outputs = find_executable_batch_size(self.get_completions, 64)(flattened_embeddings, model, tokenizer, self.config.max_new_tokens)

        for i, output in enumerate(outputs):
            completions[i // len(perturbed_embeddings_list[0])].append(output)
        return losses, completions

    @staticmethod
    def get_completions(batch_size, embedding_list, model, tokenizer, max_new_tokens=256):
        outputs = []
        for i in trange(0, len(embedding_list), batch_size):
            output = get_batched_completions(
                model,
                tokenizer,
                embedding_list=embedding_list[i : i + batch_size],
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
            outputs.extend(output)
        return outputs

    def project_l2(self, delta):
        norm = delta.norm(p=2, dim=-1, keepdim=True)
        mask = norm > self.config.epsilon
        return torch.where(mask, delta * self.config.epsilon / norm, delta)
