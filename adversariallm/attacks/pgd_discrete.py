"""
Implementation of a one-hot-input space continuous attack with discretization.

@article{geisler2024attacking,
  title={Attacking Large Language Models with Projected Gradient Descent},
  author={Geisler, Simon and Wollschl{\"a}ger, Tom and Abdalla, MHI and Gasteiger, Johannes and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2402.09154},
  year={2024}
}

Also implements a discretization attack based on Geisler et al. (2024).
WARNING: This implementation currently lacks some of the features of the reference implementation
and leads to worse results. This will be fixed in a future version.
Use https://github.com/sigeisler/reinforce-attacks-llms to reproduce official results.
"""
import copy
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt  # Keep import for plotting code
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from .attack import (Attack, AttackResult, AttackStepResult, GenerationConfig,
                     SingleAttackRunResult)
from ..lm_utils import (TokenMergeError, generate_ragged_batched,
                        get_disallowed_ids, prepare_conversation,
                        with_max_batchsize)


@dataclass
class LRSchedulerConfig:
    type: Literal["constant", "cosine", "sequential"] = "sequential"
    factor: float = 1.0
    eta_min: float = 0.325
    T_0: int = 60
    total_iters: int = 100


@dataclass
class PGDDiscreteConfig:
    name: str = "pgd_discrete"
    type: str = "hybrid"
    version: str = "0.0.1"
    placement: str = "suffix"
    num_steps: int = 5000
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    optimizer: Literal["Adam", "SAM"] = "Adam"
    projection: Literal["l2", "simplex", "l1", None] = "simplex"
    alpha: float = 0.1
    restart_every: int = 100
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    # New parameters to match reference implementation
    initialization: str = "random"  # "random" or "one_hot"
    entropy_factor: float = 0.4
    anneal: bool = True
    anneal_config: dict = field(default_factory=lambda: dict(
        start=0,
        duration=250,
        attrs=['entropy_factor'],
        mode='uniform',
        init_entropy_factor=0.,
        end_entropy_factor=0.4
    ))
    grad_clip_value: float = 20.0
    grad_clip_strategy: str = "token_norm"  # "norm", "token_norm", "value"
    tsallis_q2_proj_config: dict = field(default_factory=lambda: dict(
        iter=1,
        exclude_already_zero=True
    ))
    entropy_factor_scale_by_relaxation_gap: float = 0.1
    entropy_factor_alternate_scheduler: bool = True
    langevin_dynamics_std: float = None


@dataclass
class PGDAttackStepResult(AttackStepResult):
    continuous_loss: float


class PGDDiscreteAttack(Attack):
    def __init__(self, config: PGDDiscreteConfig):
        super().__init__(config)
        logging.warning("This implementation is WIP. To reproduce Geisler et al. (2024), use their official code.")

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
        elif self.config.placement == "prefix_suffix":
            attack_conversation = [
                {"role": "user", "content": optim_str + conversation[0]["content"] + optim_str},
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
        elif cfg.type == "sequential":
            # Match reference implementation's sophisticated scheduling
            scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.T_0, eta_min=cfg.eta_min
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[scheduler1, scheduler2], milestones=[100]
            )
        else:
            raise ValueError(f"Invalid learning rate scheduler: {cfg.type}")

    def _initialize_perturbed_one_hots(self, x_batch, attack_masks_batch, disallowed_mask_indices, model):
        # Initialize based on configuration
        if self.config.initialization == "one_hot":
            perturbed_one_hots = F.one_hot(x_batch, num_classes=model.config.vocab_size).to(model.dtype)
        else:  # random initialization
            shape = (x_batch.size(0), x_batch.size(1), model.config.vocab_size)
            perturbed_one_hots = torch.rand(shape, dtype=model.dtype, device=model.device)

        # Create and apply disallowed mask
        attack_mask_expanded = attack_masks_batch.unsqueeze(-1)  # (B, T, 1)
        disallowed_mask = torch.zeros(model.config.vocab_size, device=model.device, dtype=torch.bool)
        disallowed_mask[disallowed_mask_indices] = True  # (V,)
        disallowed_mask_expanded = disallowed_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, V)
        combined_disallowed_mask = attack_mask_expanded & disallowed_mask_expanded
        perturbed_one_hots.masked_fill_(combined_disallowed_mask, 0.0)

        # Proper normalization after masking (critical fix)
        self.eps = torch.finfo(perturbed_one_hots.dtype).eps
        perturbed_one_hots = perturbed_one_hots / torch.clamp_min(
            perturbed_one_hots.sum(-1, keepdims=True), self.eps
        )

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
            # Apply gradient clipping
            self._maybe_clip_gradient(grad, self.config.grad_clip_strategy)

    def _maybe_clip_gradient(self, grad: torch.Tensor, grad_clip_strategy: str) -> torch.Tensor:
        """Gradient clipping from reference implementation"""
        if grad_clip_strategy == 'norm':
            norm = torch.linalg.norm(grad)
            if norm > self.config.grad_clip_value:
                grad *= self.config.grad_clip_value / (norm + self.eps)
        elif grad_clip_strategy == 'token_norm':
            norm = torch.linalg.norm(grad, axis=-1, keepdim=True)
            grad_ = torch.where(
                norm > self.config.grad_clip_value,
                self.config.grad_clip_value * grad / (norm + self.eps),
                grad
            )
            grad.copy_(grad_)
        elif grad_clip_strategy == 'value':
            grad.clamp_(-self.config.grad_clip_value, self.config.grad_clip_value)
        return grad

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
                 total_loss = loss_per_sample.sum()
                 total_loss.backward()
                 # Modify gradients again before second step update
                 self._modify_gradient(perturbed_one_hots.grad, attack_masks_batch, disallowed_ids)

            closure() # Calculate gradients at the ascended point
            optimizer.second_step(zero_grad=True) # Perform the actual update
        elif self.config.optimizer == "Adam":
            optimizer.step()
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

    def _apply_projection(self, perturbed_one_hots, entropy_factor_overwrite=None,
                         disallowed_ids=None, verbose=False):
        if self.config.projection is None:
            return # No projection needed

        # Get current entropy factor
        entropy_factor = getattr(self, 'current_entropy_factor', self.config.entropy_factor)
        if entropy_factor_overwrite is not None:
            entropy_factor = entropy_factor_overwrite

        # Reshape for projection functions that expect (N, D)
        original_shape = perturbed_one_hots.shape
        B, T, V = original_shape
        one_hots_flat = perturbed_one_hots.view(B * T, V)

        # Phase 1: Simplex projection
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
        # Phase 2: Langevin dynamics (if enabled)
        if self.config.langevin_dynamics_std is not None:
            self._randomize_embedding_factors(perturbed_one_hots)
            # Re-apply simplex projection after adding noise
            one_hots_flat = perturbed_one_hots.view(B * T, V)
            if self.config.projection == "simplex":
                projected_flat = self.simplex_projection(one_hots_flat)
                perturbed_one_hots.data = projected_flat.view(original_shape).data

        # Phase 3: Tsallis entropy projection (if entropy factor > 0)
        if entropy_factor > 0 and self.config.projection == "simplex":
            one_hots_flat = perturbed_one_hots.view(B * T, V)
            projected_flat = self.tsallis_q2_projection(one_hots_flat, entropy_factor, disallowed_ids)
            perturbed_one_hots.data = projected_flat.view(original_shape).data

    def _randomize_embedding_factors(self, perturbed_one_hots):
        """Add Langevin dynamics noise"""
        noise = torch.randn_like(perturbed_one_hots)
        noise *= self.config.langevin_dynamics_std
        perturbed_one_hots.add_(noise)

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

    def anneal_step(self, step: int):
        """Annealing functionality from reference implementation"""
        def weighting_uniform(step: int, init_value: float, end_value: float, duration: int) -> float:
            return init_value + (end_value - init_value) * (min(step, duration) / duration)

        def weighting_uniform_log(step: int, init_value: float, end_value: float, duration: int) -> float:
            return init_value * (end_value / init_value)**(min(step, duration) / duration)

        if self.config.anneal_config['mode'] == 'uniform':
            weighting = weighting_uniform
        elif self.config.anneal_config['mode'] == 'uniform_log':
            weighting = weighting_uniform_log
        else:
            raise ValueError(f"Anneal mode {self.config.anneal_config['mode']} not supported")

        for attr in self.config.anneal_config['attrs']:
            init_value = self.config.anneal_config.get(f'init_{attr}', 0)
            end_value = self.config.anneal_config.get(f'end_{attr}', 1)
            step_ = step - self.config.anneal_config.get(f'start_{attr}', self.config.anneal_config['start'])
            step_ = 0 if step_ < 0 else step_
            duration = self.config.anneal_config.get(f'duration_{attr}', self.config.anneal_config['duration'])

            # Dynamically update the current entropy factor
            current_value = weighting(step_, init_value, end_value, duration)
            if attr == 'entropy_factor':
                self.current_entropy_factor = current_value

    def maybe_anneal(self, step: int, relaxation_gap: torch.Tensor | None = None):
        """Apply annealing if enabled"""
        if self.config.anneal and step >= 0:
            self.anneal_step(step)

    def dynamic_entropy_factor(self, relaxation_gap: torch.Tensor | None,
                             scheduler = None) -> float | torch.Tensor:
        """Calculate dynamic entropy factor based on relaxation gap and scheduler"""
        entropy_factor_overwrite = None

        if (self.config.entropy_factor_scale_by_relaxation_gap and relaxation_gap is not None):
            relaxation_gap_scale = relaxation_gap.clamp(0, 1)
            squeeze = 1 / (1 - self.config.entropy_factor_scale_by_relaxation_gap)
            relaxation_gap_scale = torch.where(
                relaxation_gap_scale < self.config.entropy_factor_scale_by_relaxation_gap,
                torch.sigmoid(squeeze * relaxation_gap_scale),  # Simplified x_bounded_sigmoid
                torch.ones_like(relaxation_gap_scale)
            )
            # Handle both scalar and tensor cases
            if relaxation_gap_scale.dim() == 0:
                # Scalar case - just use the value directly
                entropy_factor_overwrite = (relaxation_gap_scale *
                                           getattr(self, 'current_entropy_factor', self.config.entropy_factor))
            else:
                # Tensor case - add dimension for broadcasting
                relaxation_gap_scale = relaxation_gap_scale[:, None]
                entropy_factor_overwrite = (relaxation_gap_scale *
                                           getattr(self, 'current_entropy_factor', self.config.entropy_factor))

        if self.config.entropy_factor_alternate_scheduler and scheduler is not None:
            if hasattr(scheduler, '_schedulers'):
                base_lr = scheduler._schedulers[0].base_lrs[0]
                # To allow reverting the annealing (eta_min > learning rate)
                if hasattr(scheduler._schedulers[-1], 'eta_min'):
                    base_lr = max(base_lr, scheduler._schedulers[-1].eta_min)
            else:
                base_lr = scheduler.base_lrs[0]
                if hasattr(scheduler, 'eta_min'):
                    base_lr = max(base_lr, scheduler.eta_min)

            last_lr = scheduler.get_last_lr()[0]
            if entropy_factor_overwrite is None:
                entropy_factor_overwrite = getattr(self, 'current_entropy_factor', self.config.entropy_factor)
            entropy_factor_overwrite *= last_lr / base_lr

        return entropy_factor_overwrite

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
        if hasattr(model.get_input_embeddings(), "embed_scale"):  # For gemma
            emb_matrix = emb_matrix * model.get_input_embeddings().embed_scale.to(emb_matrix)

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

        # Store discrete tokens for each step for model_input reconstruction
        batch_discrete_tokens = [[] for _ in range(B)]

        # Initialize current entropy factor for annealing
        self.current_entropy_factor = self.config.entropy_factor

        # Track relaxation gap and discrete losses for dynamic entropy factor
        discrete_loss = None
        relaxed_loss = None
        relaxation_gap = None

        # --- Attack Loop ---
        pbar = trange(self.config.num_steps, postfix={"loss": "N/A"})
        for step in pbar:
            t_step_start = time.time()

            # Apply annealing (before optimization step)
            self.maybe_anneal(step, relaxation_gap)

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
            total_loss = loss_per_sample.sum()
            relaxed_loss = {"mean": loss_per_sample.mean().item(), "per_sample": loss_per_sample}
            total_loss.backward()

            # Modify gradients (before optimizer step)
            self._modify_gradient(perturbed_one_hots.grad, attack_masks_batch, disallowed_ids)

            # Optimizer step
            self._perform_optimizer_step(
                optimizer, perturbed_one_hots, model, emb_matrix, attention_mask_batch, y_batch, target_masks_batch, disallowed_ids, attack_masks_batch
            )

            # Calculate relaxation gap for dynamic entropy factor
            if discrete_loss is not None:
                relaxation_gap = (discrete_loss["mean"] - relaxed_loss["mean"]) / discrete_loss["mean"]
            else:
                relaxation_gap = None

            # Calculate dynamic entropy factor
            entropy_factor_overwrite = self.dynamic_entropy_factor(
                torch.tensor(relaxation_gap) if relaxation_gap is not None else None,
                scheduler
            )

            # Apply projection with proper ordering and dynamic entropy factor
            self._apply_projection(
                perturbed_one_hots,
                entropy_factor_overwrite=entropy_factor_overwrite,
                disallowed_ids=disallowed_ids,
                verbose=False
            )

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
                discrete_loss = {"mean": mean_loss_one_hot.item(), "per_sample": loss_one_hot_per_sample}

                # Update best state per sample
                is_better = loss_one_hot_per_sample < best_loss
                best_loss = torch.where(is_better, loss_one_hot_per_sample, best_loss)
                best_perturbed_one_hots = torch.where(is_better.unsqueeze(-1).unsqueeze(-1), discrete_perturbed_one_hots, best_perturbed_one_hots)
                # --- Store Metrics ---
                current_time = time.time() - t_step_start
                current_lr = scheduler.get_last_lr()[0]
                for i in range(B):
                    batch_losses[i].append(loss_per_sample[i].item())
                    print(loss_one_hot_per_sample[i].item())
                    batch_losses_one_hot[i].append(loss_one_hot_per_sample[i].item())
                    batch_times[i].append(current_time)
                    # Store embeddings derived from *discrete* state for generation
                    discrete_embeddings = discrete_perturbed_one_hots[i] @ emb_matrix
                    pert_emb_cpu = self.select_tokens(discrete_embeddings, target_masks_batch[i])
                    batch_perturbed_embeddings_list[i].append(pert_emb_cpu)
                    # Store discrete tokens for model_input reconstruction
                    discrete_tokens = argmax_indices[i][attack_masks_batch[i]]
                    batch_discrete_tokens[i].append(discrete_tokens.cpu())
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
                # Reconstruct the attack conversation for this step
                attack_conversation = self._reconstruct_attack_conversation(
                    original_conversations_batch[i],
                    batch_discrete_tokens[i][step_idx],
                    tokenizer
                )
                steps.append(PGDAttackStepResult(
                    step=step_idx,
                    model_completions=outputs[output_idx],
                    time_taken=batch_times[i][step_idx],
                    loss=batch_losses_one_hot[i][step_idx], # Discrete loss
                    continuous_loss=batch_losses[i][step_idx], # Continuous loss
                    model_input=attack_conversation,
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
        eps = torch.finfo(values.dtype).eps
        values = values / torch.clamp_min(values.sum(-1, keepdims=True), eps)

        return values

    @staticmethod
    def lp_projection(values: torch.Tensor, p: float = 2) -> torch.Tensor:
        # Implementation from original code
        values = values.clone() # Avoid modifying input tensor
        values.clamp_min_(0)
        norm = values.norm(dim=-1, keepdim=True, p=p)
        # Avoid division by zero
        eps = torch.finfo(values.dtype).eps
        values.div_(torch.clamp_min(norm, eps))
        return values

    def tsallis_q2_projection(self, values: torch.Tensor, entropy_factor: float | torch.Tensor,
                             disallowed_tokens: torch.Tensor = None) -> torch.Tensor:
        """Tsallis entropy projection (q=2)"""
        assert isinstance(self.config.tsallis_q2_proj_config['iter'], int)

        # Preserve original dtype
        original_dtype = values.dtype

        # Handle disallowed tokens
        if disallowed_tokens is not None:
            normal = torch.ones((values.shape[-1],), device=values.device, dtype=original_dtype)
            normal[disallowed_tokens] = 0
        else:
            normal = torch.ones((values.shape[-1],), device=values.device, dtype=original_dtype)

        for _ in range(self.config.tsallis_q2_proj_config['iter']):
            if self.config.tsallis_q2_proj_config['exclude_already_zero']:
                is_close_to_zero = torch.isclose(values, torch.tensor(0., device=values.device, dtype=original_dtype))
                normal = torch.broadcast_to(normal[None], is_close_to_zero.shape).clone()
                normal[is_close_to_zero] = 0
                normal = normal / normal.norm(dim=-1, keepdim=True)
            else:
                normal = normal / normal.norm()

            non_zero_components = normal > 0
            d = non_zero_components.sum(-1)
            target_entropy = (1 - entropy_factor) * (d - 1) / d
            center = 1 / d[..., None] * non_zero_components

            dist_to_hyperplane = (values * normal).sum(-1)
            projection_radius = torch.sqrt(torch.clamp(1 - target_entropy - dist_to_hyperplane**2, 0))[..., None]

            direction = values - center
            direction_norm = torch.linalg.norm(direction, axis=-1, keepdims=True)
            direction_norm = torch.clamp_min(direction_norm, self.eps)
            exceeds_budget = (direction_norm < projection_radius)[..., 0]

            if not exceeds_budget.any():
                break

            values_ = projection_radius / direction_norm * direction + center
            # Fallback to simplex projection for problematic cases
            values_[exceeds_budget] = self.simplex_projection(values_[exceeds_budget])
            values = torch.where(exceeds_budget[..., None], values_, values)

        # Ensure output maintains original dtype
        return values.to(original_dtype)

    @staticmethod
    def select_tokens(embeddings, mask):
        # Implementation from original code
        # Selects embeddings corresponding to the input part (before target)
        input_mask = ~(mask.roll(1, 0).cumsum(0).bool())
        return embeddings[input_mask].detach().cpu()

    def _reconstruct_attack_conversation(self, original_conversation, discrete_tokens, tokenizer):
        """Reconstruct the attack conversation from discrete tokens."""
        # Decode the optimized tokens to string
        optim_str = tokenizer.decode(discrete_tokens, skip_special_tokens=True)

        # Reconstruct the conversation based on placement
        if self.config.placement == "suffix":
            attack_conversation = [
                {"role": "user", "content": original_conversation[0]["content"] + optim_str},
                {"role": "assistant", "content": original_conversation[1]["content"]}
            ]
        elif self.config.placement == "prefix":
            attack_conversation = [
                {"role": "user", "content": optim_str + original_conversation[0]["content"]},
                {"role": "assistant", "content": original_conversation[1]["content"]}
            ]
        elif self.config.placement == "prompt":
            attack_conversation = copy.deepcopy(original_conversation)
            attack_conversation[0]["content"] = optim_str + attack_conversation[0]["content"]
        else:
            raise ValueError(f"Invalid placement: {self.config.placement}")

        return attack_conversation


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
