"""Implementation of a one-hot-input space continuous attack. Needs tuning."""
import functools
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from src.attacks import (Attack, AttackResult, AttackStepResult,
                         GenerationConfig, SingleAttackRunResult)
from src.lm_utils import (generate_ragged_batched, get_disallowed_ids,
                          prepare_conversation, with_max_batchsize)


@dataclass
class RandomSearchConfig:
    name: str = "random_search"
    type: str = "discrete"
    version: str = ""
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    num_steps: int = 100
    seed: int = 0
    batch_size: int = 16
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"


class RandomSearchAttack(Attack):
    def __init__(self, config: RandomSearchConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size
        if self.config.placement == "suffix":
            assert self.config.optim_str_init
        elif self.config.placement == "prompt":
            assert not self.config.optim_str_init

    @torch.no_grad()
    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        self.disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)

        x: list[torch.Tensor] = []
        attack_masks: list[torch.Tensor] = []
        target_masks: list[torch.Tensor] = []
        original_conversations = []
        for conversation in dataset:
            assert len(conversation) == 2, "Random search attack only supports two-turn conversations"
            original_conversations.append(conversation)

            if self.config.placement == "suffix":
                conversation_opt = [
                    {"role": "user", "content": conversation[0]["content"] + self.config.optim_str_init},
                    {"role": "assistant", "content": conversation[1]["content"]}
                ]
                prep_conv_arg = conversation
            elif self.config.placement == "prompt":
                initial_user_content = self.config.optim_str_init if self.config.optim_str_init else ""
                prep_conv_arg = [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": conversation[1]["content"]}
                ]
                conversation_opt = [
                    {"role": "user", "content": initial_user_content},
                    {"role": "assistant", "content": conversation[1]["content"]}
                ]
            else:
                raise ValueError(f"Unknown placement type: {self.config.placement}")

            pre_tokens, attack_prefix_tokens, prompt_tokens, attack_suffix_tokens, post_tokens, target_tokens = (
                prepare_conversation(tokenizer, prep_conv_arg, conversation_opt)[0]
            )
            tokens = torch.cat(
                [pre_tokens, attack_prefix_tokens, prompt_tokens, attack_suffix_tokens, post_tokens, target_tokens]
            )
            attack_mask = torch.cat(
                [
                    torch.zeros_like(pre_tokens),
                    torch.ones_like(attack_prefix_tokens),
                    torch.zeros_like(prompt_tokens),
                    torch.ones_like(attack_suffix_tokens),
                    torch.zeros_like(post_tokens),
                    torch.zeros_like(target_tokens),
                ]
            )
            target_mask = torch.cat(
                [
                    torch.zeros_like(pre_tokens),
                    torch.zeros_like(attack_prefix_tokens),
                    torch.zeros_like(prompt_tokens),
                    torch.zeros_like(attack_suffix_tokens),
                    torch.zeros_like(post_tokens),
                    torch.ones_like(target_tokens),
                ]
            ).roll(-1, 0)
            x.append(tokens)
            attack_masks.append(attack_mask)
            target_masks.append(target_mask)

        x = pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_masks = pad_sequence(target_masks, batch_first=True)
        attack_masks = pad_sequence(attack_masks, batch_first=True)
        attention_mask = (x != tokenizer.pad_token_id).long()

        y = x.clone()
        y[:, :-1] = x[:, 1:]

        attack_fn = functools.partial(self.attack_batch, model, tokenizer)
        runs = with_max_batchsize(attack_fn, original_conversations, x, y, attention_mask, attack_masks, target_masks)

        return AttackResult(runs=runs)

    def _get_random_allowed_tokens(self, model, tokenizer, k, happy_set, weights):
        allowed_mask = torch.ones(len(tokenizer), dtype=torch.bool, device=model.device)
        allowed_mask[self.disallowed_ids] = False

        # Get indices of allowed tokens
        allowed_indices = torch.nonzero(allowed_mask, as_tuple=True)[0]

        # Create a list of token weights, giving higher weight to tokens in happy_set
        weights = torch.softmax(-weights/0.1, dim=-1)[allowed_mask]

        if happy_set:
            for token_id in happy_set:
                token_idx = (allowed_indices == token_id).nonzero(as_tuple=True)[0]
                weights[token_idx] *= 50.0  # Increase weight for tokens in happy_set

        # Sample k random indices from the allowed tokens with weights
        random_indices = torch.multinomial(weights, k, replacement=True)
        # Return the actual token IDs
        return allowed_indices[random_indices]

    def attack_batch(self, model, tokenizer, original_conversations_batch, x, y, attention_mask, attack_masks, target_masks) -> list[SingleAttackRunResult]:
        t0 = time.time()
        num_examples = x.size(0)

        batch_attack_indices = [torch.nonzero(mask, as_tuple=True)[0] for mask in attack_masks]

        candidates = x.to(model.device)
        y = y.to(model.device)
        attention_mask = attention_mask.to(model.device)
        attack_masks = attack_masks.to(model.device)
        target_masks = target_masks.to(model.device)

        all_step_conversations: list[list[list[dict[str, str]]]] = [[] for _ in range(num_examples)]
        all_step_token_ids: list[list[torch.Tensor]] = [[] for _ in range(num_examples)]
        losses: list[list[float]] = [[] for _ in range(num_examples)]
        times: list[list[float]] = [[] for _ in range(num_examples)]

        best_sequence = candidates.clone()
        best_losses = torch.full((candidates.size(0),), float('inf'), device=candidates.device)
        happy_set: list[set[int]] = [set() for _ in range(num_examples)]
        mean_losses_history = []
        mean_diffs_history = []

        def mutate_candidates(current_candidates, current_attack_masks, current_happy_sets, weights_for_sampling):
            mutated_candidates = current_candidates.clone()
            effective_k = 1

            attack_positions = torch.nonzero(current_attack_masks, as_tuple=True)[0]
            n = len(attack_positions)

            attack_indices_to_modify_relative = torch.multinomial(torch.ones(n, device=model.device), effective_k, replacement=False)
            positions_to_modify_absolute = attack_positions[attack_indices_to_modify_relative]

            random_allowed_tokens = self._get_random_allowed_tokens(model, tokenizer, effective_k, current_happy_sets, weights_for_sampling)

            # Apply mutation to the cloned tensor
            mutated_candidates[positions_to_modify_absolute] = random_allowed_tokens[:effective_k]

            return mutated_candidates

        avg_loss_per_token = torch.zeros(len(tokenizer), device=model.device)
        pbar = trange(self.config.num_steps)
        for step_num in pbar:
            t1 = time.time()

            # --- Store conversations and tokens for this step ---
            current_step_conversations = []
            current_step_token_ids = []
            for b in range(candidates.size(0)):
                current_attack_indices = batch_attack_indices[b]
                attack_tokens = candidates[b, current_attack_indices]
                # Decode carefully, handle potential special tokens if needed
                attack_str = tokenizer.decode(attack_tokens, skip_special_tokens=False)

                original_conv = original_conversations_batch[b]
                if self.config.placement == "suffix":
                    user_content = original_conv[0]["content"] + attack_str
                elif self.config.placement == "prompt":
                    user_content = attack_str
                reconstructed_conversation = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": ""}
                ]
                step_tokens = candidates[b].clone()
                current_step_conversations.append(reconstructed_conversation)
                current_step_token_ids.append(step_tokens)
                all_step_conversations[b].append(reconstructed_conversation)
                all_step_token_ids[b].append(step_tokens)
            # --- End storing ---

            # --- Model Forward and Loss Calculation ---
            logits = model(
                input_ids=candidates,
                attention_mask=attention_mask,
            ).logits
            loss_per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction="none",
            )
            loss_per_token = loss_per_token * target_masks.view(-1)
            loss_per_example = loss_per_token.view(candidates.size(0), -1)
            target_lengths = target_masks.sum(dim=1)
            mean_loss_per_example = torch.where(target_lengths > 0, loss_per_example.sum(dim=1) / target_lengths, torch.zeros_like(target_lengths, dtype=torch.float))

            # --- Update Best Sequence and Stats ---
            current_mean_loss = mean_loss_per_example.mean().item()
            mean_losses_history.append(current_mean_loss)

            for i in range(num_examples):
                if mean_loss_per_example[i] < best_losses[i]:
                    best_losses[i] = mean_loss_per_example[i]
                    best_sequence[i] = candidates[i].clone()
                    happy_set[i].update(set(candidates[i, batch_attack_indices[i]].tolist()))
                losses[i].append(mean_loss_per_example[i].item())

            for i, current_loss in enumerate(mean_loss_per_example.detach()):
                current_attack_indices_i = batch_attack_indices[i]
                for token_pos in current_attack_indices_i:
                    token_id = candidates[i, token_pos].item()
                    avg_loss_per_token[token_id] = current_loss - best_losses[i]

            step_time = time.time() - t1
            for i in range(num_examples):
                times[i].append(step_time)

            # --- Generate Next Candidates ---
            # Mutate the overall best sequence found so far (across all examples)
            next_candidates = best_sequence.clone()
            for i in range(num_examples):
                next_candidates[i] = mutate_candidates(best_sequence[i], attack_masks[i], happy_set[i], avg_loss_per_token)

            # Record differences between the previous best sequence and the new candidates
            diffs_list = []
            for b in range(best_sequence.size(0)):
                indices = batch_attack_indices[b]
                diffs_list.append((best_sequence[b, indices] != next_candidates[b, indices]).any().float())
            diffs = torch.stack(diffs_list) if diffs_list else torch.tensor(0.0, device=best_sequence.device)
            mean_diffs_history.append(diffs.mean().item())

            # Update candidates for the next iteration
            candidates = next_candidates

            # --- Logging and Plotting ---
            pbar.set_postfix({"loss": f"{current_mean_loss:.4f}", "best_loss": f"{best_losses.mean().item():.4f}"})
            if (step_num+1) % 100 == 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 8))

                # Plot losses over time
                plt.subplot(3, 1, 1)
                plt.scatter(range(len(mean_losses_history)), mean_losses_history, s=1, alpha=0.2, label='One-Hot Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title('Loss vs. Training Step')
                plt.legend()
                plt.grid(True)

                plt.subplot(3, 1, 3)
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(mean_diffs_history, 'r-', label='Edit Distance [in one-hot space]')
                ax2.set_ylabel('Fraction Changed', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                ax1.set_xlabel('Step')
                ax1.set_title('One-Hot Token Changes per Step')
                ax1.grid(True)

                plt.legend()

                plt.tight_layout()
                plt.savefig('loss_analysis.pdf')
                plt.close()

        # --- Generation Step ---
        # Flatten the per-example, per-step token lists for batch generation
        flat_token_list = [token_tensor for example_tokens in all_step_token_ids for token_tensor in example_tokens]
        outputs = []
        outputs = generate_ragged_batched(
            model, tokenizer, token_list=flat_token_list,
            initial_batch_size=self.batch_size,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p, top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences
        )

        # --- Collate Results ---
        runs = []
        num_steps = self.config.num_steps

        for i in range(num_examples):
            steps_for_prompt = []
            for step in range(num_steps):
                step_outputs = outputs[i*num_steps + step]
                step_result = AttackStepResult(
                    step=step,
                    model_completions=step_outputs,
                    time_taken=times[i][step],
                    loss=losses[i][step],
                    model_input=all_step_conversations[i][step],
                    model_input_tokens=all_step_token_ids[i][step].tolist(),
                )
                steps_for_prompt.append(step_result)

            run = SingleAttackRunResult(
                original_prompt=original_conversations_batch[i],
                steps=steps_for_prompt,
                total_time=(time.time() - t0) / num_examples
            )
            runs.append(run)
        return runs
