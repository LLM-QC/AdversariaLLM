"""Implementation of a one-hot-input space continuous attack with discretization.

Also implements a discretization attack based on Geisler et al. (2024).
"""
import copy
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
import matplotlib.pyplot as plt # Keep import for plotting code

from src.attacks import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from src.lm_utils import (generate_ragged_batched, get_disallowed_ids, prepare_conversation,
                          with_max_batchsize, TokenMergeError)



@dataclass
class LRSchedulerConfig:
    type: Literal["constant", "cosine"] = "constant"
    factor: float = 1.0
    eta_min: float = 0.325
    T_0: int = 60
    total_iters: int = 100


@dataclass
class PGDDiscreteConfig:
    name: str = "pgd_one_hot"
    type: str = "continuous"
    placement: str = "suffix" # Note: Not explicitly used in provided code structure
    version: str = ""
    num_steps: int = 100
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    optimizer: Literal["Adam", "SAM"] = "Adam"
    projection: Literal["l2", "simplex", "l1", None] = "simplex"
    alpha: float = 0.001
    restart_every: int = 100
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class PGDAttackStepResult(AttackStepResult):
    continuous_loss: float


class PGDDiscreteAttack(Attack):
    def __init__(self, config: PGDDiscreteConfig):
        super().__init__(config)

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        x, attack_masks, target_masks, conversations = self._prepare_dataset(dataset, tokenizer)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        attention_mask = (x != tokenizer.pad_token_id).long()
        y = x.clone()
        y[:, :-1] = x[:, 1:]

        attack_fn = functools.partial(self.attack_batch, model, tokenizer)
        runs = with_max_batchsize(
            attack_fn,
            x,
            y,
            conversations,
            attention_mask,
            attack_masks,
            target_masks,
        )
        return AttackResult(runs=runs)

    def _prepare_dataset(self, dataset, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        all_tokens = []
        all_attack_masks = []
        all_target_masks = []
        all_conversations = []

        for conversation in dataset:
            all_conversations.append(conversation)
            try:
                tokens, attack_mask, target_mask = self._prepare_single_conversation(
                    conversation, tokenizer, self.config.optim_str_init
                )
            except TokenMergeError:
                logging.warning("TokenMergeError encountered, retrying with added space.")
                tokens, attack_mask, target_mask = self._prepare_single_conversation(
                    conversation, tokenizer, " " + self.config.optim_str_init
                )

            all_tokens.append(tokens)
            all_attack_masks.append(attack_mask)
            all_target_masks.append(target_mask)

        all_tokens = pad_sequence(all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        all_target_masks = pad_sequence(all_target_masks, batch_first=True)
        all_attack_masks = pad_sequence(all_attack_masks, batch_first=True)

        return all_tokens, all_attack_masks, all_target_masks, all_conversations

    def _prepare_single_conversation(self, conversation, tokenizer, optim_str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        if self.config.placement == "suffix":
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
        elif self.config.placement == "prefix":
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"]},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
        elif self.config.placement == "prompt":
            attack_conversation = copy.deepcopy(conversation)
            conversation = copy.deepcopy(conversation)
            conversation[0]["content"] = ""
        else:
            raise ValueError(f"Invalid placement: {self.config.placement}")
        parts = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
        pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks = parts

        tokens = torch.cat(parts)

        attack_mask = torch.zeros_like(tokens, dtype=torch.bool)
        offset = pre_toks.size(0)
        attack_mask[offset:offset + attack_prefix_toks.size(0)] = True
        offset += attack_prefix_toks.size(0) + prompt_toks.size(0)
        attack_mask[offset:offset + attack_suffix_toks.size(0)] = True

        target_mask = torch.zeros_like(tokens, dtype=torch.bool)
        target_start_idx = len(tokens) - target_toks.size(0)
        target_mask[target_start_idx:] = True
        target_mask = target_mask.roll(-1, 0)
        target_mask[-1] = False

        return tokens, attack_mask.long(), target_mask.long()

    def _initialize_optimizer(self, params):
        if self.config.optimizer == "Adam":
            return torch.optim.Adam(params, lr=self.config.alpha)
        elif self.config.optimizer == "SAM":
            base_optimizer = torch.optim.Adam
            return SAM(params, base_optimizer, rho=0.05, lr=self.config.alpha)
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

    def _initialize_scheduler(self, optimizer):
        cfg = self.config.lr_scheduler
        if cfg.type == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=cfg.factor, total_iters=cfg.total_iters
            )
        elif cfg.type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0, eta_min=cfg.eta_min
            )
        else:
            raise ValueError(f"Invalid learning rate scheduler: {cfg.type}")

    def _initialize_perturbed_one_hots(self, x_batch, attack_masks_batch, disallowed_mask_indices, model):
        perturbed_one_hots = F.one_hot(x_batch, num_classes=model.config.vocab_size).to(model.dtype)
        # Create and apply disallowed mask
        attack_mask_expanded = attack_masks_batch.unsqueeze(-1)  # (B, T, 1)
        disallowed_mask = torch.zeros(model.config.vocab_size, device=model.device, dtype=torch.bool)
        disallowed_mask[disallowed_mask_indices] = True  # (V,)
        disallowed_mask_expanded = disallowed_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, V)
        combined_disallowed_mask = attack_mask_expanded & disallowed_mask_expanded
        perturbed_one_hots.masked_fill_(combined_disallowed_mask, 0.0)
        perturbed_one_hots = perturbed_one_hots.detach().requires_grad_(True)
        return perturbed_one_hots

    def _calculate_continuous_loss(self, model, perturbed_one_hots, emb_matrix, attention_mask, y_batch, target_masks_batch):
        outputs = model(
            inputs_embeds=perturbed_one_hots @ emb_matrix,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_batch.view(-1),
            reduction="none",
        )
        loss = loss.view(y_batch.shape[0], -1) * target_masks_batch  # (B, L)
        loss_per_sample = loss.sum(dim=1) / (target_masks_batch.sum(dim=1).float() + 1e-6)  # (B,)
        return loss_per_sample, logits  # Return logits for discrete loss calc

    def _modify_gradient(self, grad, attack_mask, disallowed_ids):
        if grad is not None:
            grad.data[..., disallowed_ids] = 0  # Zero out gradients for disallowed token embeddings
            grad.data[~attack_mask] = 0  # Zero out gradients for non-attack tokens

    def _perform_optimizer_step(self, optimizer, perturbed_one_hots, model, emb_matrix, attention_mask, y_batch, target_masks_batch, disallowed_ids, attack_masks_batch):
        if self.config.optimizer == "SAM":
            # First SAM step (ascent)
            optimizer.first_step(zero_grad=True)
            # Apply projection after first step if needed (e.g., simplex)
            self._apply_projection(perturbed_one_hots)
            # Second SAM step (descent) - requires re-calculating loss and gradients
            # Closure for SAM's second step
            def closure():
                 optimizer.zero_grad()
                 loss_per_sample, _ = self._calculate_continuous_loss(
                     model, perturbed_one_hots, emb_matrix, attention_mask, y_batch, target_masks_batch
                 )
                 mean_loss = loss_per_sample.mean()
                 mean_loss.backward()
                 # Modify gradients again before second step update
                 self._modify_gradient(perturbed_one_hots.grad, attack_masks_batch, disallowed_ids)

            closure() # Calculate gradients at the ascended point
            optimizer.second_step(zero_grad=True) # Perform the actual update
        elif self.config.optimizer == "Adam":
            optimizer.step()
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

    def _apply_projection(self, perturbed_one_hots):
        if self.config.projection is None:
            return # No projection needed

        # Reshape for projection functions that expect (N, D)
        original_shape = perturbed_one_hots.shape
        B, T, V = original_shape
        one_hots_flat = perturbed_one_hots.view(B * T, V)

        if self.config.projection == "simplex":
            projected_flat = self.simplex_projection(one_hots_flat)
        elif self.config.projection == "l2":
            projected_flat = self.lp_projection(one_hots_flat, p=2)
        elif self.config.projection == "l1":
             projected_flat = self.lp_projection(one_hots_flat, p=1)
        else:
             # Should not happen if config validation is done, but good practice
             raise ValueError(f"Invalid projection type: {self.config.projection}")

        perturbed_one_hots.data = projected_flat.view(original_shape).data

    def _calculate_discrete_loss(self, model, discrete_one_hots, emb_matrix, attention_mask, y_batch, target_masks_batch):
        with torch.no_grad():
            logits_one_hot = model(
                inputs_embeds=discrete_one_hots @ emb_matrix,
                attention_mask=attention_mask,
            ).logits
            loss_one_hot = F.cross_entropy(
                logits_one_hot.view(-1, logits_one_hot.size(-1)),
                y_batch.view(-1),
                reduction="none",
            )
            loss_one_hot = loss_one_hot.view(y_batch.shape[0], -1) * target_masks_batch # (B, L)
            loss_one_hot_per_sample = loss_one_hot.sum(dim=1) / (target_masks_batch.sum(dim=1).float() + 1e-6) # (B,)
        return loss_one_hot_per_sample

    def _handle_restart(self, step, best_perturbed_one_hots, perturbed_one_hots):
        if step > 0 and step % self.config.restart_every == 0:
             logging.info(f"Restarting optimization at step {step}")
             with torch.no_grad():
                 # Reset to best found discrete state so far
                 perturbed_one_hots.data = best_perturbed_one_hots.data
                 # Re-apply projection to ensure consistency if needed
                 self._apply_projection(perturbed_one_hots)
             return True
        return False

    def attack_batch(
        self,
        model,
        tokenizer,
        x_batch,
        y_batch,
        original_conversations_batch,
        attention_mask_batch,
        attack_masks_batch,
        target_masks_batch
    ) -> list[SingleAttackRunResult]:

        t_start_batch = time.time()
        device = model.device
        B = x_batch.size(0)
        disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)
        emb_matrix = model.get_input_embeddings().weight # V, D

        # --- Initialization ---
        batch_losses = [[] for _ in range(B)] # Continuous loss history
        batch_losses_one_hot = [[] for _ in range(B)] # Discrete loss history
        batch_perturbed_embeddings_list = [[] for _ in range(B)] # Embeddings for generation
        batch_times = [[] for _ in range(B)] # Step timing

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        attack_masks_batch = attack_masks_batch.to(device).bool()
        target_masks_batch = target_masks_batch.to(device).bool()

        perturbed_one_hots = self._initialize_perturbed_one_hots(
            x_batch, attack_masks_batch, disallowed_ids, model
        )
        optimizer = self._initialize_optimizer([perturbed_one_hots])
        scheduler = self._initialize_scheduler(optimizer)

        # --- Tracking & Plotting Initialization ---
        mean_losses_hist = []
        mean_losses_one_hot_hist = []
        mean_diffs_hist = []
        mean_diffs_one_hot_hist = []
        learning_rates_hist = []
        last_one_hots = F.one_hot(perturbed_one_hots.argmax(dim=-1), num_classes=model.config.vocab_size).to(model.dtype).detach()
        best_perturbed_one_hots = last_one_hots.clone().detach()
        best_loss = torch.full((B,), float('inf'), device=device, dtype=model.dtype)

        # --- Attack Loop ---
        pbar = trange(self.config.num_steps, postfix={"loss": "N/A"})
        for step in pbar:
            t_step_start = time.time()
            optimizer.zero_grad()

            # Handle restarts
            restarted = self._handle_restart(step, best_perturbed_one_hots, perturbed_one_hots)
            if restarted:
                 # Optionally reset optimizer state if needed for restart
                 # optimizer.state = {} # Example: Reset Adam state
                 pass

            perturbed_one_hots_prev_for_diff = perturbed_one_hots.clone().detach()

            # Calculate continuous loss and gradients
            loss_per_sample, _ = self._calculate_continuous_loss(
                model, perturbed_one_hots, emb_matrix, attention_mask_batch, y_batch, target_masks_batch
            )
            mean_loss = loss_per_sample.mean()
            mean_loss.backward()

            # Modify gradients (before optimizer step)
            self._modify_gradient(perturbed_one_hots.grad, attack_masks_batch, disallowed_ids)

            # Optimizer step
            self._perform_optimizer_step(
                optimizer, perturbed_one_hots, model, emb_matrix, attention_mask_batch, y_batch, target_masks_batch, disallowed_ids, attack_masks_batch
            )

            # Apply projection
            self._apply_projection(perturbed_one_hots)

            # Step the scheduler
            scheduler.step()

            # --- Discretization, Discrete Loss, and Best State Tracking ---
            with torch.no_grad():
                 argmax_indices = perturbed_one_hots.argmax(dim=-1)
                 discrete_perturbed_one_hots = F.one_hot(argmax_indices, num_classes=model.config.vocab_size).to(model.dtype)

                 loss_one_hot_per_sample = self._calculate_discrete_loss(
                     model, discrete_perturbed_one_hots, emb_matrix, attention_mask_batch, y_batch, target_masks_batch
                 )
                 mean_loss_one_hot = loss_one_hot_per_sample.mean()

                 # Update best state per sample
                 is_better = loss_one_hot_per_sample < best_loss
                 best_loss = torch.where(is_better, loss_one_hot_per_sample, best_loss)
                 best_perturbed_one_hots = torch.where(is_better.unsqueeze(-1).unsqueeze(-1), discrete_perturbed_one_hots, best_perturbed_one_hots)

                 # --- Store Metrics ---
                 current_time = time.time() - t_step_start
                 current_lr = scheduler.get_last_lr()[0]
                 for i in range(B):
                     batch_losses[i].append(loss_per_sample[i].item())
                     batch_losses_one_hot[i].append(loss_one_hot_per_sample[i].item())
                     batch_times[i].append(current_time)
                     # Store embeddings derived from *discrete* state for generation
                     discrete_embeddings = discrete_perturbed_one_hots[i] @ emb_matrix
                     pert_emb_cpu = self.select_tokens(discrete_embeddings, target_masks_batch[i])
                     batch_perturbed_embeddings_list[i].append(pert_emb_cpu)

                 # History for plotting
                 mean_losses_hist.append(mean_loss.item())
                 mean_losses_one_hot_hist.append(mean_loss_one_hot.item())
                 learning_rates_hist.append(current_lr)

                 # Calculate diffs for plotting
                 diffs_one_hot = (last_one_hots != discrete_perturbed_one_hots).any(dim=-1).float()[attack_masks_batch]
                 mean_diffs_one_hot_hist.append(diffs_one_hot.mean().item() if diffs_one_hot.numel() > 0 else 0.0)
                 diffs = (perturbed_one_hots_prev_for_diff - perturbed_one_hots).norm(dim=-1)[attack_masks_batch].mean()
                 mean_diffs_hist.append(diffs.item() if diffs_one_hot.numel() > 0 else 0.0)

                 last_one_hots = discrete_perturbed_one_hots.clone().detach() # Update for next step diff calc

            pbar.set_postfix({
                "loss": f"{mean_loss.item():.4f}",
                "loss_1hot": f"{mean_loss_one_hot.item():.4f}",
                "best_1hot": f"{best_loss.mean().item():.4f}", # Mean best loss across batch
                "lr": f"{current_lr:.4f}"
            })

            # --- Plotting (Optional, kept from original) ---
            if (step + 1) % 100 == 0:
                self._plot_metrics(mean_losses_hist, mean_losses_one_hot_hist, mean_diffs_hist, mean_diffs_one_hot_hist, learning_rates_hist)

        # --- Generation ---
        # Use the stored discrete embeddings from each step
        flattened_embeddings = [e for el in batch_perturbed_embeddings_list for e in el]
        outputs = generate_ragged_batched(
            model,
            tokenizer,
            embedding_list=flattened_embeddings,
            initial_batch_size=512,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )
        logging.info(f"Generated {len(outputs)}x{self.config.generation_config.num_return_sequences} completions")

        # --- Result Formatting ---
        t_end_batch = time.time()
        runs = []
        for i in range(B):
            steps = []
            for step_idx in range(self.config.num_steps):
                # Calculate the index in the flattened outputs list
                output_idx = i * self.config.num_steps + step_idx
                steps.append(PGDAttackStepResult(
                    step=step_idx,
                    model_completions=outputs[output_idx],
                    time_taken=batch_times[i][step_idx],
                    loss=batch_losses_one_hot[i][step_idx], # Discrete loss
                    continuous_loss=batch_losses[i][step_idx], # Continuous loss
                ))
            runs.append(SingleAttackRunResult(
                original_prompt=original_conversations_batch[i],
                steps=steps,
                total_time=(t_end_batch - t_start_batch) # Total time for this batch
            ))
        return runs

    def _plot_metrics(self, mean_losses, mean_losses_one_hot, mean_diffs, mean_diffs_one_hot, learning_rates):
        # Keep the plotting logic encapsulated
        plt.figure(figsize=(12, 10))
        plt.subplot(4, 1, 1)
        plt.plot(mean_losses, label='Continuous Loss')
        plt.plot(mean_losses_one_hot, label='One-Hot Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss vs. Training Step')
        plt.legend(loc='lower left'); plt.grid(True)
        plt.subplot(4, 1, 2)
        plt.scatter(mean_losses_one_hot, mean_losses, c=range(len(mean_losses)), cmap='viridis', alpha=0.5); plt.colorbar(label='Step')
        min_val = min(min(mean_losses_one_hot, default=0), min(mean_losses, default=0))
        max_val = max(max(mean_losses_one_hot, default=1), max(mean_losses, default=1))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('One-Hot Loss'); plt.ylabel('Continuous Loss'); plt.title('Loss Correlation'); plt.grid(True)
        plt.subplot(4, 1, 3)
        ax1 = plt.gca(); ax2 = ax1.twinx()
        ln1 = ax1.plot(mean_diffs, 'b-', label='Update Norm [relaxed]')
        ax1.set_yscale('log'); ax1.set_ylabel('Update Norm (log)', color='b'); ax1.tick_params(axis='y', labelcolor='b')
        ln2 = ax2.plot(mean_diffs_one_hot, 'r-', label='Edit Distance [discrete]')
        ax2.set_ylabel('Edit Distance', color='r'); ax2.tick_params(axis='y', labelcolor='r')
        lns = ln1 + ln2; labs = [l.get_label() for l in lns]; ax1.legend(lns, labs, loc='upper left')
        ax1.set_xlabel('Step'); ax1.set_title('Token Changes per Step'); ax1.grid(True)
        plt.subplot(4, 1, 4)
        plt.plot(learning_rates, 'g-')
        plt.xlabel('Step'); plt.ylabel('Learning Rate'); plt.title('Learning Rate Schedule'); plt.grid(True)
        plt.tight_layout(); plt.savefig('loss_analysis.pdf'); plt.close()

    # --- Projection Methods (Static or instance methods as appropriate) ---
    @staticmethod
    def simplex_projection(values):
        # Implementation from original code
        def sort_projection(values):
            b, d = values.shape
            cat_indices = torch.arange(d, device=values.device)
            batch_indices = torch.arange(b, device=values.device)
            values = torch.clamp_min(values, 0.)
            values_sorted = -(-values).sort(-1).values
            values_cumulative = torch.cumsum(values_sorted, axis=-1) - 1
            condition = values_sorted - values_cumulative / (cat_indices + 1) > 0
            rho = torch.count_nonzero(condition, axis=-1)
            # Prevent division by zero for rho=0 case
            rho_safe = torch.clamp_min(rho, 1)
            theta = values_cumulative[batch_indices, rho_safe - 1] / rho_safe
            # Only apply theta where rho > 0
            values = torch.clamp_min(values - theta[:, None] * (rho > 0)[:, None], 0.)
            return values

        values = values.clone()
        # Avoid potential NaN from sum(clamp(0,1)) if values contains NaN initially
        values_clamped = torch.clamp(values.nan_to_num(0.0), 0, 1)
        exceeds_budget = values_clamped.sum(-1) > 1

        if exceeds_budget.any():
            values[exceeds_budget] = sort_projection(values[exceeds_budget])
            # Clamp non-exceeding values *after* potential NaN handling
            values[~exceeds_budget] = torch.clamp(values[~exceeds_budget].nan_to_num(0.0), min=0, max=1)
        else:
            values = torch.clamp(values.nan_to_num(0.0), min=0, max=1)

        # Handle degenerate case (all zeros)
        sum_values = values.sum(-1, keepdims=True)
        is_degenerate = torch.isclose(sum_values, torch.tensor(0., device=values.device, dtype=values.dtype))
        # Add small random noise only to degenerate rows
        rand_offset = torch.rand_like(values) * is_degenerate
        values += rand_offset
        # Renormalize - clamp denominator to avoid division by zero
        values = values / torch.clamp_min(values.sum(-1, keepdims=True), 1e-8)

        return values

    @staticmethod
    def lp_projection(values: torch.Tensor, p: float = 2) -> torch.Tensor:
        # Implementation from original code
        values = values.clone() # Avoid modifying input tensor
        values.clamp_min_(0)
        norm = values.norm(dim=-1, keepdim=True, p=p)
        # Avoid division by zero
        values.div_(torch.clamp_min(norm, 1e-8))
        return values

    @staticmethod
    def select_tokens(embeddings, mask):
        # Implementation from original code
        # Selects embeddings corresponding to the input part (before target)
        input_mask = ~(mask.roll(1, 0).cumsum(0).bool())
        return embeddings[input_mask].detach().cpu()


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
