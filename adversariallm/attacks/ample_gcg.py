"""Single-file implementation of the AmpleGCG attack.

@article{liao2024amplegcg,
  title={Amplegcg: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed llms},
  author={Liao, Zeyi and Sun, Huan},
  journal={arXiv preprint arXiv:2404.07921},
  year={2024}
}
"""
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import GenerationConfig as HuggingFaceGenerationConfig

from ..io_utils import free_vram, load_model_and_tokenizer
from ..lm_utils import generate_ragged_batched, get_losses_batched, prepare_conversation

from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult


@dataclass
class PrompterLMConfig:
    id: str
    tokenizer_id: str
    chat_template: Optional[str]
    short_name: str
    developer_name: str
    batch_size: int
    dtype: str
    attn_implementation: Optional[str]
    trust_remote_code: bool
    compile: bool
    generation_config: dict


@dataclass
class AmpleGCGConfig:
    name: str = "ample_gcg"
    type: str = "discrete"
    version: str = ""
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    num_steps: int = 200
    prompter_lm: PrompterLMConfig = field(default_factory=PrompterLMConfig)


class AmpleGCGAttack(Attack):
    def __init__(self, config: AmpleGCGConfig):
        super().__init__(config)

    @torch.no_grad
    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        runs = []
        for conversation in tqdm(dataset, file=sys.stdout):
            assert len(conversation) == 2, "Current AmpleGCG only supports single-turn conversations"
            msg = conversation[0]["content"]
            target = conversation[1]["content"]
            # Temporarily move target model to cpu
            device = model.device
            model.cpu()
            free_vram()
            t0 = time.time()
            batch_attacks = self.get_attack_prompts(f"### Query:{msg} ### Prompt:")
            model.to(device)
            attack_conversations = [
                [{"role": "user", "content": f"{msg}{attack}"}, {"role": "assistant", "content": target}]
                for attack in batch_attacks
            ]
            logging.info("Prepared attack conversations")
            batch_losses = self.get_losses(attack_conversations, model, tokenizer)
            logging.info("Calculated losses")
            token_list = [
                torch.cat(prepare_conversation(tokenizer, attack_conversation)[0][:-1])
                for attack_conversation in attack_conversations
            ]
            logging.info("Prepared token lists")
            batch_completions = generate_ragged_batched(
                model,
                tokenizer,
                token_list=token_list,
                max_new_tokens=self.config.generation_config.max_new_tokens,
                temperature=self.config.generation_config.temperature,
                top_p=self.config.generation_config.top_p,
                top_k=self.config.generation_config.top_k,
                num_return_sequences=self.config.generation_config.num_return_sequences,
            )
            logging.info("Generated completions")
            for i in range(len(attack_conversations)):
                attack_conversations[i][-1]["content"] = ""
            t1 = time.time()
            step_results = [
                AttackStepResult(
                    step=i,
                    model_completions=batch_completions[i],
                    time_taken=(t1 - t0) / len(batch_attacks),
                    loss=batch_losses[i],
                    model_input=attack_conversations[i],
                    model_input_tokens=token_list[i].tolist()
                )
                for i in range(len(batch_attacks))
            ]
            single_attack_run_result = SingleAttackRunResult(
                original_prompt=conversation,
                steps=step_results,
                total_time=t1 - t0
            )
            runs.append(single_attack_run_result)
            logging.info("Created single attack run result")
        return AttackResult(runs)

    def get_attack_prompts(self, msg):
        prompter_model = PrompterModel(self.config.prompter_lm, self.config.num_steps)
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

    def get_losses(self, attack_conversations, model, tokenizer):
        token_list = [
            prepare_conversation(tokenizer, attack_conversation)[0]
            for attack_conversation in attack_conversations
        ]
        target_lengths = [t[-1].size(0) for t in token_list]
        token_list = [torch.cat(t, dim=0) for t in token_list]
        targets = [t.roll(-1, 0) for t in token_list]

        with torch.no_grad():
            losses = get_losses_batched(
                model,
                targets=targets,
                token_list=token_list,
            )
        losses = [l[-tl:].mean().item() for l, tl in zip(losses, target_lengths)]
        return losses


class PrompterModel(nn.Module):
    def __init__(self, config, num_steps):
        super().__init__()
        self.batch_size = config.batch_size
        self.model, self.tokenizer = load_model_and_tokenizer(config)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "num_return_sequences": num_steps,
            "num_beams": num_steps,
            # Set num_beam_groups to 1 to use regular beam search instead of group beam search
            "num_beam_groups": 1,
        }
        self.gen_config = HuggingFaceGenerationConfig(
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
