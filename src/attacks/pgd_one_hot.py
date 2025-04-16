"""Implementation of a one-hot-input space continuous attack. Needs tuning.

Also implements a discretization attack based on Geisler et al. (2024).
"""
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from src.attacks import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult
from src.lm_utils import generate_ragged_batched, get_disallowed_ids, prepare_conversation, with_max_batchsize, TokenMergeError


@dataclass
class LRSchedulerConfig:
    type: Literal["constant", "cosine"] = "constant"
    factor: float = 1.0
    eta_min: float = 0.325
    T_0: int = 60
    total_iters: int = 100


@dataclass
class PGDOneHotConfig:
    name: str = "pgd_one_hot"
    type: str = "continuous"
    placement: str = "suffix"
    version: str = ""
    num_steps: int = 100
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    batch_size: int = 2
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    epsilon: float = 100000.0
    alpha: float = 0.001
    projection: Literal["l2", "simplex", None] = "simplex"
    normalize_gradient: bool = False
    optimizer: Literal["Adam", "SAM"] = "Adam"
    restart_every: int = 100
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)


@dataclass
class PGDAttackStepResult(AttackStepResult):
    continuous_loss: float


class PGDOneHotAttack(Attack):
    def __init__(self, config: PGDOneHotConfig):
        super().__init__(config)
        self.batch_size = self.config.batch_size

    def run(self, model: torch.nn.Module, tokenizer, dataset) -> AttackResult:
        x: list = []
        attack_masks: list = []
        target_masks: list = []
        conversations = []

        for conversation in dataset:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + self.config.optim_str_init},
                {"role": "assistant", "content": conversation[1]["content"]}
            ]
            try:
                pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
            except TokenMergeError:
                attack_conversation = [
                    {"role": "user", "content": conversation[0]["content"] + " " + self.config.optim_str_init},
                    {"role": "assistant", "content": conversation[1]["content"]}
                ]
                pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks = prepare_conversation(tokenizer, conversation, attack_conversation)[0]

            tokens = torch.cat(
                [pre_toks, attack_prefix_toks, prompt_toks, attack_suffix_toks, post_toks, target_toks]
            )
            attack_mask = torch.cat(
                [
                    torch.zeros_like(pre_toks),
                    torch.ones_like(attack_prefix_toks),
                    torch.zeros_like(prompt_toks),
                    torch.ones_like(attack_suffix_toks),
                    torch.zeros_like(post_toks),
                    torch.zeros_like(target_toks),
                ]
            )
            target_mask = torch.cat(
                [
                    torch.zeros_like(pre_toks),
                    torch.zeros_like(attack_prefix_toks),
                    torch.zeros_like(prompt_toks),
                    torch.zeros_like(attack_suffix_toks),
                    torch.zeros_like(post_toks),
                    torch.ones_like(target_toks),
                ]
            ).roll(-1, 0)
            conversations.append(attack_conversation)
            x.append(tokens)
            attack_masks.append(attack_mask)
            target_masks.append(target_mask)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        x = pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_masks = pad_sequence(target_masks, batch_first=True)
        attack_masks = pad_sequence(attack_masks, batch_first=True)
        attention_mask = (x != tokenizer.pad_token_id).long()

        y = x.clone()
        y[:, :-1] = x[:, 1:]

        # Run the attack with dynamic batch size adjustment
        attack_fn = functools.partial(self.attack_batch, model, tokenizer)
        runs = with_max_batchsize(
            attack_fn,
            x,
            y,
            conversations,
            attention_mask,
            attack_masks,
            target_masks
        )
        return AttackResult(runs=runs)

    def _make_random_one_hots(self, x, attack_masks, disallowed_mask, vocab_size, dtype):
        random_one_hots = F.one_hot(x, num_classes=vocab_size).to(dtype)
        random_one_hots += (
            2*torch.rand_like(random_one_hots) / random_one_hots.size(-1) - random_one_hots
        ) * attack_masks[..., None]
        random_one_hots.masked_fill(disallowed_mask, 0.0)
        random_one_hots = F.one_hot(random_one_hots.argmax(dim=-1), num_classes=vocab_size).to(dtype)
        return random_one_hots

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
        disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)
        B = x_batch.size(0)
        losses = [[] for _ in range(B)]
        losses_one_hot = [[] for _ in range(B)]
        perturbed_embeddings_list = [[] for _ in range(B)]
        times = [[] for _ in range(B)]

        emb = model.get_input_embeddings().weight
        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)
        attention_mask_batch = attention_mask_batch.to(model.device)
        attack_masks_batch = attack_masks_batch.to(model.device)
        target_masks_batch = target_masks_batch.to(model.device)

        perturbed_one_hots = (
            F.one_hot(x_batch, num_classes=model.config.vocab_size)
            .to(model.dtype)
            .detach()
        )
        # Zero out disallowed token IDs for attack tokens
        attack_mask_expanded = attack_masks_batch.bool().unsqueeze(-1)  # Shape: (B, T, 1)
        disallowed_mask = torch.zeros(model.config.vocab_size, device=model.device, dtype=torch.bool)
        disallowed_mask[disallowed_ids] = True  # Shape: (D)
        disallowed_mask_expanded = disallowed_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D)
        disallowed_mask = attack_mask_expanded & disallowed_mask_expanded
        perturbed_one_hots = perturbed_one_hots.masked_fill(disallowed_mask, 0.0)

        # Make perturbed_one_hots a parameter for optimization
        perturbed_one_hots = perturbed_one_hots.detach().requires_grad_(True)
        for i in range(B):
            logging.info("Discrete suffix: " + tokenizer.decode(perturbed_one_hots[i].argmax(dim=-1)[attack_masks_batch[i].bool()].tolist()))
        # Initialize Adam optimizer with zero momentum
        if self.config.optimizer == "Adam":
            optimizer = torch.optim.Adam([perturbed_one_hots], lr=self.config.alpha)
        elif self.config.optimizer == "SAM":
            base_optimizer = torch.optim.Adam
            optimizer = SAM([perturbed_one_hots], base_optimizer, rho=1, lr=self.config.alpha)
        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

        # Initialize cosine learning rate schedule
        # Using constant learning rate instead of cosine annealing
        if self.config.lr_scheduler.type == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=self.config.lr_scheduler.factor,
                total_iters=self.config.lr_scheduler.total_iters
            )
        elif self.config.lr_scheduler.type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.config.lr_scheduler.T_0,
                eta_min=self.config.lr_scheduler.eta_min
            )
        else:
            raise ValueError(f"Invalid learning rate scheduler: {self.config.lr_scheduler.type}")

        t0 = time.time()
        pbar = trange(self.config.num_steps, postfix={"loss": "N/A"})

        # Lists to store losses for plotting
        mean_losses = []
        mean_losses_one_hot = []
        mean_diffs = []
        mean_diffs_one_hot = []
        learning_rates = []  # Store learning rates for plotting
        last_one_hots = F.one_hot(perturbed_one_hots.argmax(dim=-1), num_classes=model.config.vocab_size).to(model.dtype).detach()
        best_perturbed_one_hots = last_one_hots.clone().detach()
        best_loss = float('inf')
        for _step in pbar:
            optimizer.zero_grad()
            # Reinitialize optimizer and randomize perturbed_one_hots every 100 steps
            if _step > 0 and _step % self.config.restart_every == 0:
                with torch.no_grad():
                    # Create a new random distribution for attack tokens & project onto simplex
                    random_one_hots = best_perturbed_one_hots
                    perturbed_one_hots.data = random_one_hots.data
                    perturbed_one_hots.data = self.simplex_projection(perturbed_one_hots.clone().view(-1, perturbed_one_hots.size(-1))).view_as(perturbed_one_hots).data

            perturbed_one_hots_prev = perturbed_one_hots.clone().detach()
            # Forward pass
            outputs = model(
                inputs_embeds=perturbed_one_hots @ emb,
                attention_mask=attention_mask_batch,
            )
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1),
                reduction="none",
            )
            loss = loss.view(B, -1) * target_masks_batch  # (B, L)
            loss = loss.sum(dim=1) / (target_masks_batch.sum(dim=1) + 1e-6)  # (B,)
            mean_loss = loss.mean()
            mean_loss.backward()
            # Zero out disallowed token IDs for attack tokens
            perturbed_one_hots.grad[..., disallowed_ids] = 0
            perturbed_one_hots.grad[~attack_masks_batch.bool()] = 0
            if self.config.optimizer == "SAM":
                optimizer.first_step(zero_grad=True)
                projected = self.simplex_projection(perturbed_one_hots.clone().view(-1, perturbed_one_hots.size(-1)))
                perturbed_one_hots.data = projected.view_as(perturbed_one_hots).data
                embeddings = perturbed_one_hots @ emb
                logits = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask_batch,
                ).logits
                (F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y_batch.view(-1),
                    reduction="none",
                ) * target_masks_batch.view(-1)).view(embeddings.size(0), -1).mean().backward()
                perturbed_one_hots.grad[..., disallowed_ids] = 0
                perturbed_one_hots.grad[~attack_masks_batch.bool()] = 0
                optimizer.second_step(zero_grad=True)
            elif self.config.optimizer == "Adam":
                optimizer.step()
            else:
                raise ValueError(f"Invalid optimizer: {self.config.optimizer}")

            # Record losses
            for j, l in enumerate(loss.detach().tolist()):
                losses[j].append(l)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)

            # Manually enforce constraints/projections
            with torch.no_grad():
                if self.config.projection == "simplex":
                    perturbed_one_hots.data = self.simplex_projection(perturbed_one_hots.clone().view(-1, perturbed_one_hots.size(-1))).view_as(perturbed_one_hots).data
                elif self.config.projection == "l2":
                    perturbed_one_hots.data = self.lp_projection(perturbed_one_hots.clone().view(-1, perturbed_one_hots.size(-1))).view_as(perturbed_one_hots).data
                elif self.config.projection == "l1":
                    perturbed_one_hots.data = self.lp_projection(perturbed_one_hots.clone().view(-1, perturbed_one_hots.size(-1)), p=1).view_as(perturbed_one_hots).data

                argmax_indices = perturbed_one_hots.argmax(dim=-1)
                discrete_perturbed_one_hots = F.one_hot(argmax_indices, num_classes=model.config.vocab_size).to(model.dtype)

                for i in range(B):
                    logging.info("Discrete suffix: " + tokenizer.decode(discrete_perturbed_one_hots[i].argmax(dim=-1)[attack_masks_batch[i].bool()].tolist()))

                logits_one_hot = model(
                    inputs_embeds=discrete_perturbed_one_hots @ emb,
                    attention_mask=attention_mask_batch,
                ).logits
                loss_one_hot = F.cross_entropy(
                    logits_one_hot.view(-1, logits_one_hot.size(-1)),
                    y_batch.view(-1),
                    reduction="none",
                )
                loss_one_hot = loss_one_hot.view(B, -1) * target_masks_batch  # (B, L)
                loss_one_hot = loss_one_hot.sum(dim=1) / (target_masks_batch.sum(dim=1) + 1e-6)  # (B,)
                # Record losses
                for j, l in enumerate(loss_one_hot.detach().tolist()):
                    losses_one_hot[j].append(l)

                mean_loss_one_hot = loss_one_hot.mean()
                # tsallis_projected = self.tsallis_q2_projection(perturbed_one_hots.clone().view(-1, perturbed_one_hots.size(-1)), 0.1, disallowed_ids)
                # perturbed_one_hots.data = tsallis_projected.view_as(perturbed_one_hots).data

                # Store losses for plotting
                mean_losses.append(mean_loss.item())
                mean_losses_one_hot.append(mean_loss_one_hot.item())

                diffs_one_hot = (last_one_hots != discrete_perturbed_one_hots).any(dim=-1).float()[attack_masks_batch.bool()]
                mean_diffs_one_hot.append(diffs_one_hot.mean().item())

                diffs = (perturbed_one_hots_prev - perturbed_one_hots).norm(dim=-1).mean()
                mean_diffs.append(diffs.item())

                last_one_hots = discrete_perturbed_one_hots
                if mean_loss_one_hot < best_loss:
                    best_loss = mean_loss_one_hot
                    best_perturbed_one_hots = discrete_perturbed_one_hots.clone().detach()

            for i, (pe, tm) in enumerate(zip(discrete_perturbed_one_hots @ emb, target_masks_batch)):
                perturbed_embeddings_list[i].append(self.select_tokens(pe, tm).detach())

            t = time.time() - t0
            for i in range(x_batch.size(0)):
                times[i].append(t)
            pbar.set_postfix({"loss": f"{mean_loss.item():.4f}", "loss_one_hot": f"{mean_loss_one_hot.item():.4f}", "best_loss": f"{best_loss.item():.4f}", "lr": f"{current_lr:.2f}"})
            # Plot losses
            if (_step+1) % 100 == 0:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 10))  # Increased figure size to accommodate 4 plots

                # Plot losses over time
                plt.subplot(4, 1, 1)  # Changed to 4 rows
                plt.plot(mean_losses, label='Continuous Loss')
                plt.plot(mean_losses_one_hot, label='One-Hot Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title('Loss vs. Training Step')
                plt.legend(loc='lower left')
                plt.grid(True)

                # Plot correlation
                plt.subplot(4, 1, 2)  # Changed to 4 rows
                plt.scatter(mean_losses_one_hot, mean_losses, c=range(len(mean_losses)), cmap='viridis', alpha=0.5)
                plt.colorbar(label='Step')

                # Add x=y dashed line
                min_val = min(min(mean_losses_one_hot), min(mean_losses))
                max_val = max(max(mean_losses_one_hot), max(mean_losses))
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('One-Hot Loss')
                plt.ylabel('Continuous Loss')
                plt.title('Loss Correlation')
                plt.grid(True)

                # Plot number of changed elements with two y-axes
                plt.subplot(4, 1, 3)  # Changed to 4 rows
                ax1 = plt.gca()
                ax2 = ax1.twinx()

                # Plot update norm with log scale
                ln1 = ax1.plot(mean_diffs, 'b-', label='Update Norm [in relaxed space]')
                ax1.set_yscale('log')
                ax1.set_ylabel('Update Norm (log scale)', color='b')
                ax1.tick_params(axis='y', labelcolor='b')

                # Plot edit distance with linear scale
                ln2 = ax2.plot(mean_diffs_one_hot, 'r-', label='Edit Distance [in one-hot space]')
                ax2.set_ylabel('Edit Distance (linear scale)', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                # Add combined legend
                lns = ln1 + ln2
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs, loc='upper left')

                ax1.set_xlabel('Step')
                ax1.set_title('One-Hot Token Changes per Step')
                ax1.grid(True)

                # Add learning rate plot
                plt.subplot(4, 1, 4)
                plt.plot(learning_rates, 'g-')
                plt.xlabel('Step')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.grid(True)

                plt.tight_layout()
                plt.savefig('loss_analysis.pdf')
                plt.close()

        flattened_embeddings = [e for el in perturbed_embeddings_list for e in el]
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
        logging.info(f"Generated {len(outputs)}x{len(outputs[0])} completions")

        runs = []
        for i in range(B):
            steps = []
            for step in range(self.config.num_steps):
                steps.append(PGDAttackStepResult(
                    step=step,
                    model_completions=outputs[i*self.config.num_steps + step],
                    time_taken=times[i][step],
                    loss=losses_one_hot[i][step],
                    continuous_loss=losses[i][step],
                ))

            runs.append(SingleAttackRunResult(
                original_prompt=original_conversations_batch[i],
                steps=steps,
                total_time=times[i][-1]
            ))
        return runs

    @staticmethod
    def simplex_projection(values):
        """L2 optimal projection onto the simplex.
        From https://github.com/sigeisler/reinforce-attacks-llms/blob/main/baselines/reinforce/pgd_attack.py

        Args:
            values: A tensor of shape (batch_size, num_tokens) containing the values to project onto the simplex.
        Returns:
            A tensor of shape (batch_size, num_tokens) containing the projected values.
        """
        def sort_projection(values):
            b, d = values.shape
            cat_indices = torch.arange(d, device=values.device)
            batch_indices = torch.arange(b, device=values.device)

            values = torch.clamp_min(values, 0.)

            values_sorted = -(-values).sort(-1).values
            values_cumulative = torch.cumsum(values_sorted, axis=-1) - 1
            condition = values_sorted - values_cumulative / (cat_indices + 1) > 0
            rho = torch.count_nonzero(condition, axis=-1)
            theta = values_cumulative[batch_indices, rho - 1] / rho
            values = torch.clamp_min(values - theta[:, None], 0.)
            return values
        values = values.clone()
        exceeds_budget = torch.clamp(values, 0, 1).sum(-1) > 1
        if exceeds_budget.any():
            values[exceeds_budget] = sort_projection(values[exceeds_budget])
            values[~exceeds_budget] = torch.clamp(values[~exceeds_budget], min=0, max=1)
        else:
            values = torch.clamp(values, min=0, max=1)

        # Handle degenerate case where weights for token are all 0
        all_values_zero_offset = (
            torch.isclose(values.sum(-1, keepdims=True), torch.tensor(0., device=values.device, dtype=values.dtype)) *
            torch.rand_like(values))
        values += all_values_zero_offset
        values = values / torch.clamp_min(values.sum(-1, keepdims=True), 1e-8)

        return values

    def lp_projection(self, values: torch.Tensor, p: float = 2) -> torch.Tensor:
        """L_p projection onto the simplex.
        Args:
            values: A tensor of shape (batch_size, num_tokens) containing the values to project onto the simplex.
        Returns:
            A tensor of shape (batch_size, num_tokens) containing the projected values.
        """
        values.clamp_min_(0)
        values.div_(values.norm(dim=-1, keepdim=True, p=p))
        return values

    def tsallis_q2_projection(self, values: torch.Tensor, entropy_factor: float | torch.Tensor, disallowed_tokens: torch.Tensor) -> torch.Tensor:
        """Entropy factor within (0, 1] that scales between max and min."""
        # Ensure values is in float32 to avoid precision issues with bfloat16
        original_dtype = values.dtype

        normal = torch.ones((values.shape[-1], ), device=values.device, dtype=values.dtype)
        normal[disallowed_tokens] = 0

        for _ in range(2):
            if True:
                is_close_to_zero = torch.isclose(values, torch.tensor(0., device=values.device, dtype=values.dtype))
                normal = torch.broadcast_to(normal, is_close_to_zero.shape).clone()
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
            direction_norm = torch.clamp_min(direction_norm, 1e-9)
            exceeds_budget = (direction_norm < projection_radius)[..., 0]

            if not exceeds_budget.any():
                break

            values_ = projection_radius / direction_norm * direction + center
            values_[exceeds_budget] = self.simplex_projection(values_[exceeds_budget])
            values = torch.where(exceeds_budget[..., None], values_, values)

        values = values.to(original_dtype)

        return values

    @staticmethod
    def select_tokens(embeddings, mask):
        return embeddings[~(mask.roll(1, 0).cumsum(0).bool())].detach().cpu()


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        SAM optimizer. Implementation from https://github.com/davda54/sam
        Originally proposed in https://arxiv.org/pdf/2010.01412v3
        Published under MIT license.

        Args:
            params: list of parameters to optimize
            base_optimizer: optimizer to use for optimization
            rho: rho parameter for SAM
            adaptive: whether to use adaptive SAM
        """
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
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
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
