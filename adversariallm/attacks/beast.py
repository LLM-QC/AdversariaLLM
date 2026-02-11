"""Implementation of the BEAST attack.

@article{sadasivan2024fast,
  title={Fast adversarial attacks on language models in one gpu minute},
  author={Sadasivan, Vinu Sankar and Saha, Shoumik and Sriramanan, Gaurang and Kattakinda, Priyatham and Chegini, Atoosa and Feizi, Soheil},
  journal={arXiv preprint arXiv:2402.15570},
  year={2024}
}

Implementation adapted from https://github.com/dreadnode/research/blob/main/notebooks/Mistral%20-%20BEAST%20Beam%20Attack.ipynb
"""

import copy
import sys
import time
from dataclasses import dataclass, field
from functools import partial

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attack import (Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult)
from ..lm_utils import (generate_ragged_batched, get_disallowed_ids, get_flops,
                          prepare_conversation, with_max_batchsize)
from ..types import Conversation


@dataclass
class BEASTConfig:
    name: str = "beast"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    num_steps: int = 30  # also the suffix length
    seed: int = 0
    optim_str_init: str = ""
    k1: int = 10
    k2: int = 10
    search_temperature: float = 1.0
    allow_non_ascii: bool = False
    allow_special: bool = False
    use_prefix_cache: bool = True
    mask_undecided_tokens: bool = False


class BEASTAttack(Attack):
    def __init__(self, config: BEASTConfig):
        super().__init__(config)
        assert self.config.search_temperature > 0.0, "Temperature must be greater than 0 for BEAST"
        self.prefix_cache = None

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
        t_start = time.time()
        attacks, losses, times, prompts, token_list, flops_list = [], [], [], [], [], []

        # Get disallowed token IDs once
        self.disallowed_ids = get_disallowed_ids(
            tokenizer,
            self.config.allow_non_ascii,
            self.config.allow_special
        ).to(model.device)
        self.disallowed_ids = self.disallowed_ids[self.disallowed_ids < model.get_input_embeddings().weight.size(0)]

        for conversation in dataset:
            t0 = time.time()
            assert len(conversation) == 2, "Currently BEAST only supports single-turn conversations"
            prompts.append(conversation)
            attack: Conversation = [
                {"role": "user", "content": conversation[0]["content"] + self.config.optim_str_init},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
            pre_tokens, _, prompt_tokens, attack_tokens, post_tokens, target_tokens = prepare_conversation(
                tokenizer,
                conversation,
                attack,
            )[0]

            # Compute KV cache for prefix tokens
            if self.config.use_prefix_cache and "gemma" not in model.name_or_path:
                self.populate_prefix_cache(model, pre_tokens, prompt_tokens)
                flops_prefix = get_flops(model, pre_tokens.numel() + prompt_tokens.numel(), 0, "forward")
            else:
                flops_prefix = 0

            initial_ppl, flops_ppl = self.get_perplexity(
                model,
                tokenizer,
                pre_tokens,
                prompt_tokens,
                [attack_tokens],
                post_tokens,
                target_tokens,
            )
            initial_ppl = initial_ppl[0]
            flops = flops_prefix + flops_ppl
            per_sample_attacks = [""]
            per_sample_times = [time.time() - t0]
            per_sample_losses = [torch.log(torch.tensor(initial_ppl)).item()]
            per_sample_flops = [flops]
            beams: list[torch.LongTensor] = [torch.LongTensor([]) for _ in range(self.config.k1)]
            for i in (pbar := trange(1, self.config.num_steps, file=sys.stdout)):
                flops = 0
                t1 = time.time()
                # Get next K1 x K2 candidates
                next_tokens, flops_sample = self.sample(
                    model,
                    self.config.k2,
                    pre_tokens,
                    prompt_tokens,
                    beams,
                    post_tokens,
                )  # (k1, k2)

                # Create all candidates
                candidates = []
                for beam, next_token in zip(beams, next_tokens):
                    candidates.extend([torch.cat((beam, t.unsqueeze(0))) for t in next_token])

                # Score candidates
                scores, flops_ppl = self.get_perplexity(
                    model,
                    tokenizer,
                    pre_tokens,
                    prompt_tokens,
                    candidates,
                    post_tokens,
                    target_tokens,
                )

                # Take the K1 best by lowest score
                sorting_indices = torch.tensor(scores).argsort().tolist()
                beams = [candidates[i] for i in sorting_indices[:self.config.k1]]
                flops = flops_sample + flops_ppl

                # Record best result
                best_idx = sorting_indices[0]
                best_suffix = tokenizer.decode(candidates[best_idx])
                best_score = scores[best_idx]
                best_loss = torch.log(torch.tensor(best_score)).item()

                per_sample_attacks.append(best_suffix)
                per_sample_losses.append(best_loss)
                per_sample_flops.append(flops)
                pbar.set_postfix({"loss": best_loss, "attack": best_suffix})
                per_sample_times.append(time.time() - t1)

            losses.append(per_sample_losses)
            times.append(per_sample_times)
            flops_list.append(per_sample_flops)
            # Prepare tokens for generation
            attack_conversations = []
            token_list_batch = []
            for attack in per_sample_attacks:
                attack_conv = [
                    {"role": "user", "content": f"{conversation[0]['content']}{attack}"},
                    {"role": "assistant", "content": ""}
                ]
                attack_conversations.append(attack_conv)

                tokens = prepare_conversation(tokenizer, attack_conv)[0][:5]
                token_list_batch.append(torch.cat(tokens))
            attacks.append(attack_conversations)
            token_list.extend(token_list_batch)

        # Generate completions in batches
        outputs = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            initial_batch_size=512,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )  # (B*num_steps, num_return_sequences, T)

        # Aggregate results
        t_final = time.time()
        runs = []
        for i in range(len(dataset)):
            step_results = []
            for j in range(self.config.num_steps):
                step_result = AttackStepResult(
                    step=j,
                    model_completions=outputs[i*self.config.num_steps + j],
                    time_taken=times[i][j],
                    loss=losses[i][j],
                    flops=flops_list[i][j],
                    model_input=attacks[i][j],
                    model_input_tokens=token_list[i*self.config.num_steps + j].tolist()
                )
                step_results.append(step_result)
            run = SingleAttackRunResult(
                original_prompt=prompts[i],
                steps=step_results,
                total_time=(t_final - t_start) / len(prompts)
            )
            runs.append(run)

        return AttackResult(runs=runs)

    def populate_prefix_cache(self, model, pre_tokens, prompt_tokens):
        """Compute KV cache for prefix tokens to avoid recomputing them."""
        prefix_input = torch.cat([pre_tokens, prompt_tokens]).unsqueeze(0).to(model.device)
        outputs = model(prefix_input, use_cache=True)
        self.prefix_cache = outputs.past_key_values

    def get_perplexity(
        self,
        model,
        tokenizer,
        pre_tokens,
        prompt_tokens,
        attack_tokens_list,
        post_tokens,
        target_tokens,
    ) -> tuple[list[float], int]:
        # Create tensor based on whether prefix cache is available
        T_cur = attack_tokens_list[0].size(0)
        T = self.config.num_steps
        # Pad attack tokens to ensure all have the same length
        padding_length = T - T_cur
        padded_attack_tokens_list = []
        for attack_tokens in attack_tokens_list:
            # Calculate padding needed
            if padding_length > 0:
                # Create padding tensor with pad token ID (usually 0)
                padding = torch.full(
                    (padding_length,),
                    tokenizer.pad_token_id,
                    dtype=attack_tokens.dtype,
                    device=attack_tokens.device
                )
                # Concatenate attack tokens with padding
                padded_attack = torch.cat([attack_tokens, padding])
                padded_attack_tokens_list.append(padded_attack)
            else:
                # No padding needed
                padded_attack_tokens_list.append(attack_tokens)
        # Replace original list with padded version
        attack_tokens_list = padded_attack_tokens_list
        if self.config.mask_undecided_tokens:
            attention_mask = torch.zeros(T, dtype=torch.long, device=attack_tokens_list[0].device)
            attention_mask[:T_cur] = 1
        else:
            attention_mask = None

        if self.prefix_cache is not None:
            # With prefix cache, we don't need to include prefix tokens
            tokens_to_concat = [
                torch.cat([attack_tokens, post_tokens, target_tokens])
                for attack_tokens in attack_tokens_list
            ]
        else:
            # Without prefix cache, include all tokens
            tokens_to_concat = [
                torch.cat([pre_tokens, prompt_tokens, attack_tokens, post_tokens, target_tokens])
                for attack_tokens in attack_tokens_list
            ]
        if attention_mask is not None:
            attention_mask = torch.cat([torch.ones(pre_tokens.size(0) + prompt_tokens.size(0)), attention_mask, torch.ones(post_tokens.size(0) + target_tokens.size(0))])
            attention_mask = attention_mask[:-1].to(model.device)

        tensor = torch.stack(tokens_to_concat)

        def get_log_probs(target_tokens, attention_mask, x):
            B = x.size(0)
            # Expand prefix cache to match batch size if available
            if self.prefix_cache is not None:
                cache = copy.deepcopy(self.prefix_cache)
                for i, layer in enumerate(cache.layers):
                    layer.keys = layer.keys.expand(B, -1, -1, -1)
                    layer.values = layer.values.expand(B, -1, -1, -1)
            else:
                cache = None
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0).repeat(B, 1).to(model.device)

            # Get logits and compute log probabilities
            logits = model(input_ids=x.to(model.device), past_key_values=cache, attention_mask=attention_mask).logits
            output_logits = logits[:, -target_tokens.shape[0]:]
            log_probs = torch.nn.functional.log_softmax(output_logits, dim=-1).cpu()

            # Repeat target_tokens to match batch size
            target_tokens_expanded = target_tokens.unsqueeze(0).repeat(log_probs.size(0), 1).unsqueeze(-1)

            # Calculate perplexity
            gathered_log_probs = log_probs.gather(2, target_tokens_expanded).squeeze(-1)
            return gathered_log_probs.mean(dim=1)

        # Partial function to avoid repeating target_tokens
        get_log_probs_fn = partial(get_log_probs, target_tokens, attention_mask)

        # Process in batches
        mean_log_probs = with_max_batchsize(get_log_probs_fn, tensor[:, :-1], initial_batch_size=512)
        flops = get_flops(model, tensor[:, :-1].numel(), 0, "forward")

        perplexity = torch.exp(-mean_log_probs).tolist()
        return perplexity, flops

    def sample(
        self,
        model,
        k: int,
        pre_tokens,
        prompt_tokens,
        attack_token_list,
        post_tokens,
    ) -> tuple[torch.LongTensor, int]:
        if self.prefix_cache is not None:
            # Use the prefix cache to avoid recomputing the prefix
            tensor = torch.stack([
                torch.cat([attack_tokens, post_tokens])
                for attack_tokens in attack_token_list
            ]).to(model.device)

            # Expand cache to match batch size
            cache = copy.deepcopy(self.prefix_cache)
            for i, layer in enumerate(cache.layers):
                layer.keys = layer.keys.expand(tensor.size(0), -1, -1, -1)
                layer.values = layer.values.expand(tensor.size(0), -1, -1, -1)
        else:
            # Fall back to the original implementation if prefix cache is not available
            tensor = torch.stack([
                torch.cat([pre_tokens, prompt_tokens, attack_tokens, post_tokens])
                for attack_tokens in attack_token_list
            ]).to(model.device)
            cache = None

        # Get logits for next token prediction
        logits = model(input_ids=tensor, past_key_values=cache).logits[:, -1, :]
        flops = get_flops(model, tensor.numel(), 0, "forward")

        # Filter out disallowed tokens
        if self.disallowed_ids is not None:
            logits[:, self.disallowed_ids] = float('-inf')

        probs = torch.softmax(logits / self.config.search_temperature, dim=-1)
        tokens = torch.multinomial(probs, k, replacement=False)
        return tokens.cpu(), flops   # Return as CPU tensor
