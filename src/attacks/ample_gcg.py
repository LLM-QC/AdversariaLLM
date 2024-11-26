"""Single-file implementation of the AmpleGCG attack."""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from accelerate.utils import find_executable_batch_size
from tqdm import trange
from transformers import GenerationConfig

from src.io_utils import free_vram, load_model_and_tokenizer
from src.lm_utils import get_batched_completions, get_batched_losses, prepare_tokens

from .attack import Attack, AttackResult


@dataclass
class AmpleGCGAttackResult:
    """A class to store the attack result for a single instance in a dataset."""

    attacks: list[str]
    losses: list[float]
    prompt: list
    completions: list[int | None] = None


@dataclass
class AmpleGCGConfig:
    name: str = "AmpleGCG"
    type: str = "discrete"
    generate_completions: Literal["all", "best", "last"] = "last"


class AmpleGCGAttack(Attack):
    def __init__(self, config: AmpleGCGConfig):
        super().__init__(config)

    @torch.no_grad
    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        results = AttackResult([], [], [], [])
        for msg, target in dataset:
            # Temporarily move target model to cpu
            device = model.device
            model.cpu()
            free_vram()
            attacks = self.get_attack_prompts(f"### Query:{msg['content']} ### Prompt:")
            model.to(device)
            losses = find_executable_batch_size(
                self.get_losses, self.config.target_lm.batch_size
            )(msg, attacks, target, model, tokenizer)
            completions = find_executable_batch_size(
                self.get_completions, self.config.target_lm.batch_size
            )(msg, attacks, model, tokenizer)
            results.attacks.append(attacks)
            results.losses.append(losses)
            results.prompts.append(msg)
            results.completions.append(completions)
        return results

    def get_attack_prompts(self, msg):
        prompter_model = PrompterModel(self.config.prompter_lm)
        prompter_model.eval()
        attacks = prompter_model.run([msg])
        free_vram()
        return attacks

    @staticmethod
    def _format_prompt(prompt, attack, tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{prompt} {attack}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def get_completions(self, batch_size, prompt, attacks, model, tokenizer):
        outputs = []
        for i in trange(0, len(attacks), batch_size, desc="Target Model"):
            token_list = [
                prepare_tokens(
                    tokenizer,
                    prompt=prompt["content"],
                    target="ZZZZZ",  #  need dummy target (probably)
                    attack=attack,
                )
                for attack in attacks[i : i + batch_size]
            ]
            token_list = [
                torch.cat([a, b, c, d], dim=0) for a, b, c, d, _ in token_list
            ]
            output = get_batched_completions(
                model,
                tokenizer,
                token_list=token_list,
                max_new_tokens=self.config.target_lm.max_new_tokens,
                use_cache=True
            )
            outputs.extend(output)
        return outputs

    def get_losses(self, batch_size, prompt, attacks, target, model, tokenizer):
        outputs = []
        for i in trange(0, len(attacks), batch_size, desc="Target Model"):
            token_list = [
                prepare_tokens(
                    tokenizer,
                    prompt=prompt["content"],
                    target="ZZZZZ",  # need dummy target (probably)
                    attack=attack,
                )
                for attack in attacks[i : i + batch_size]
            ]
            target_lengths = [e.size(0) for a, b, c, d, e in token_list]
            token_list = [torch.cat(t, dim=0) for t in token_list]
            targets = [t.roll(-1, 0) for t in token_list]

            losses = get_batched_losses(
                model,
                targets=targets,
                token_list=token_list,
            )
            losses = [l[-tl:].mean().item() for l, tl in zip(losses, target_lengths)]
            outputs.extend(losses)
        return outputs


class PrompterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.model, self.tokenizer = load_model_and_tokenizer(config.id, config)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
        }
        self.gen_config = GenerationConfig(
            **config.generation_config, **self.gen_kwargs
        )

    def _prompterlm_run_batch(self, batch):
        input_ids = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = input_ids.to(self.model.device)
        # Does a beam search to generate multiple completions per prompt
        output = self.model.generate(**input_ids, generation_config=self.gen_config)
        output = output[:, input_ids["input_ids"].shape[-1] :]
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_text

    # q_s questions, p_s prompts
    def run(self, q_s):
        outputs = []
        batch_size = self.batch_size
        for i in trange(0, len(q_s), batch_size, desc="Prompter Model"):
            outputs.extend(self._prompterlm_run_batch(q_s[i : i + batch_size]))
        return outputs
