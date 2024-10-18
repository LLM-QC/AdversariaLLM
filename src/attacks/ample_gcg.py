"""Single-file implementation of the AmpleGCG attack."""

from dataclasses import dataclass
from typing import Literal

import torch
from accelerate.utils import find_executable_batch_size
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from src.lm_utils import get_batched_completions
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

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:

        results = AttackResult([], [], [], [])
        for msg, target in dataset:
            attacks = self.get_attack_prompts(f"### Query:{msg['content']} ### Prompt:")
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
        prompter_model = PrompterModel(self.config.prompter_lm).to("cuda")
        prompter_model.eval()
        return prompter_model.run([msg])

    @staticmethod
    def _format_prompt(prompt, attack, tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{prompt} {attack}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # q_s questions, p_s prompts
    def get_completions(
        self, batch_size, prompt, attacks, model, tokenizer
    ):
        outputs = []
        for i in trange(0, len(attacks), batch_size, desc="Target Model"):
            batch = [
                self._format_prompt(prompt["content"], attack, tokenizer)
                for attack in attacks[i : i + batch_size]
            ]
            # token_list = [tokenizer.encode(b, return_tensors="pt")[0] for b in batch]
            # output = get_batched_completions(
            #     model,
            #     tokenizer,
            #     token_list=token_list,
            #     max_new_tokens=self.config.target_lm.generation_steps,
            # )
            input_ids = tokenizer(batch, return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.no_grad():
                output = model.generate(
                    **input_ids,
                    max_new_tokens=self.config.target_lm.generation_steps,
                )
            output = tokenizer.batch_decode(output[:, input_ids.input_ids.shape[1] :], skip_special_tokens=True)
            outputs.extend(output)
        return outputs

    def get_losses(self, batch_size, prompt, attacks, target, model, tokenizer):
        outputs = []
        for i in trange(0, len(attacks), batch_size, desc="Target Model"):
            batch = [
                self._format_prompt(prompt["content"], attack, tokenizer)
                for attack in attacks[i : i + batch_size]
            ]
            input_ids = tokenizer(
                batch, return_tensors="pt", padding=True
            ).input_ids.to(model.device)
            target_ids = tokenizer(
                [target] * len(batch), return_tensors="pt"
            ).input_ids.to(model.device)
            logits = model(input_ids=torch.cat([input_ids, target_ids], dim=1)).logits
            logits = logits[:, input_ids.shape[-1] :]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids[:, -logits.size(1) :].view(-1),
                reduction="none",
            )
            loss = loss.view(len(batch), -1).mean(dim=1)
            outputs.extend(loss.cpu().tolist())
        return outputs


def cal_loss_avg(loss):
    non_zero_mask = loss != 0

    average_ignoring_zeros = torch.zeros(loss.size(0))
    for i in range(loss.size(0)):
        non_zero_values = loss[i, non_zero_mask[i]]
        if len(non_zero_values) > 0:
            average_ignoring_zeros[i] = non_zero_values.mean()
        else:
            average_ignoring_zeros[i] = float("nan")
    return average_ignoring_zeros


def check_torch_dtype(config):
    kwargs = {}
    if config.torch_dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif config.torch_dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif config.torch_dtype == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["trust_remote_code"] = True
    return kwargs


class PrompterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size

        kwargs = check_torch_dtype(config)
        self.model = AutoModelForCausalLM.from_pretrained(config.id, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(config.id)
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
        input_ids = self.tokenizer(batch, return_tensors="pt", padding=True).to(
            self.model.device
        )
        # Does a beam search to generate multiple completions per prompt
        output = self.model.generate(**input_ids, generation_config=self.gen_config)
        output = output[:, input_ids["input_ids"].shape[-1] :]
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_text

    # q_s questions, p_s prompts
    @torch.no_grad()
    def run(self, q_s):
        outputs = []
        batch_size = self.batch_size
        for i in trange(0, len(q_s), batch_size, desc="Prompter Model"):
            outputs.extend(self._prompterlm_run_batch(q_s[i : i + batch_size]))
        return outputs
