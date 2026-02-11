"""
Implementation of an embedding-space continuous attack inspired by Soft Prompt Threats.

@article{schwinn2024soft,
  title={Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space},
  author={Schwinn, Leo and Dobre, David and Xhonneux, Sophie and Gidel, Gauthier and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2402.09063},
  year={2024}
}

This attack optimizes the embeddings directly. In addition, we also support a "one-hot" attack
where we optimize the continuously relaxed one-hot encoded attack tokens.
This is quite different from the embedding-space attack, and the attack is less strong.
"""

import functools
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from .attack import (Attack, AttackResult, AttackStepResult, GenerationConfig,
                     SingleAttackRunResult)
from ..lm_utils import (TokenMergeError, generate_ragged_batched, get_disallowed_ids,
                          prepare_conversation, with_max_batchsize)
from ..types import Conversation


@dataclass
class OptimizerConfig:
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class PGDConfig:
    name: str = "pgd"
    type: str = "continuous"
    num_steps: int = 100
    version: str = ""
    seed: int = 0
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    epsilon: float = 100000.0
    alpha: float = 0.001
    max_new_tokens: int = 256
    embedding_scale: Optional[float] = None
    normalize_alpha: bool = False
    normalize_gradient: bool = False
    original_model: Optional[str] = None
    loss: str = "ce"
    tie_logits: float = 0.0
    tie_features: float = 0.0
    optimizer: str = "Adam"
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    projection: str = "l2"
    attack_space: str = "embedding"
    random_restart_interval: int = 0
    random_restart_epsilon: float = 0.1
    log_embeddings: bool = False


class PGDAttack(Attack):
    def __init__(self, config: PGDConfig):
        super().__init__(config)
        self.zero_init_attack = False  # Consider making this a config option if needed

    def run(self, model: PreTrainedModel, tokenizer, dataset) -> AttackResult:
        self._initialize_embedding_scale(model)
        original_model = self._maybe_load_original_model()

        x, attack_masks, target_masks, conversations = self._prepare_dataset(dataset, tokenizer)
        logging.info(f"Prepared {len(conversations)} conversations for attack")

        assert isinstance(tokenizer.pad_token_id, int), "pad_token_id must be an integer"
        attention_mask = (x != tokenizer.pad_token_id).long()
        y = x.clone()
        y[:, :-1] = x[:, 1:]

        attack_fn = functools.partial(self.attack_batch, model, tokenizer, original_model)
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

    def _initialize_embedding_scale(self, model: PreTrainedModel):
        # we compute and store the embedding scale for the projection and the lr
        # important: we do not store them back in the config because the config will later
        # be saved to disk. Future runs should be able to use the config to avoid duplications.
        if self.config.embedding_scale is None:
            embeddings = model.get_input_embeddings().weight
            assert isinstance(embeddings, torch.Tensor), "embeddings are expected to be a tensor"
            if hasattr(model.get_input_embeddings(), "embed_scale"):  # For gemma
                embed_scale = model.get_input_embeddings().embed_scale
                assert isinstance(embed_scale, torch.Tensor), "embed_scale are expected to be a tensor"
                embeddings = embeddings * embed_scale.to(embeddings)
            if self.config.projection == "l2":
                self.embedding_scale = embeddings.norm(dim=-1).mean().item()
            elif self.config.projection == "l1":
                self.embedding_scale = embeddings.norm(dim=-1, p=1).mean().item()
            else:
                logging.warning(f"Unknown projection {self.config.projection}, embedding_scale not set.")
                self.embedding_scale = 1.0
        else:
            self.embedding_scale = 1.0
        self.lr = self.embedding_scale * self.config.alpha
        if self.config.normalize_gradient:
            embeddings = model.get_input_embeddings().weight
            assert isinstance(embeddings, torch.Tensor), "embeddings are expected to be a tensor"
            self.lr /= embeddings.size(-1) ** 0.5
        logging.info(f"Embedding scale set to {self.embedding_scale} based on projection={self.config.projection}")

    def _initialize_optimizer(self, params):
        logging.info(f"Initializing optimizer with lr={self.lr}")
        if self.config.optimizer == "FGSM":
            return FGSMOptimizer(params, lr=self.lr, **self.config.optimizer_config)
        else:
            return torch.optim.Adam(params, lr=self.lr, **self.config.optimizer_config)

    def _maybe_load_original_model(self) -> Optional[PreTrainedModel]:
        if self.config.original_model:
            logging.info(f"Loading {self.config.original_model} for logit/feature tying")
            return AutoModelForCausalLM.from_pretrained(
                self.config.original_model,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            ).eval()
        return None

    def _prepare_dataset(self, dataset, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Conversation]]:
        all_tokens = []
        all_attack_masks = []
        all_target_masks = []
        all_conversations = []

        for conversation in dataset:
            try:
                tokens, attack_mask, target_mask, attack_conversation = self._prepare_single_conversation(
                    conversation, tokenizer, self.config.optim_str_init
                )
            except TokenMergeError:
                logging.warning("TokenMergeError encountered, retrying with added space.")
                tokens, attack_mask, target_mask, attack_conversation = self._prepare_single_conversation(
                    conversation, tokenizer, " " + self.config.optim_str_init
                )

            all_tokens.append(tokens)
            all_attack_masks.append(attack_mask)
            all_target_masks.append(target_mask)
            all_conversations.append(attack_conversation)
        all_tokens = pad_sequence(all_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
        all_target_masks = pad_sequence(all_target_masks, batch_first=True)
        all_attack_masks = pad_sequence(all_attack_masks, batch_first=True)
        return all_tokens, all_attack_masks, all_target_masks, all_conversations

    def _prepare_single_conversation(self, conversation, tokenizer, optim_str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Conversation]:
        attack_conversation = [
            {"role": "user", "content": conversation[0]["content"] + optim_str},
            {"role": "assistant", "content": conversation[1]["content"]}
        ]
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
        target_mask[-1] = False # Last token has no target

        return tokens, attack_mask.long(), target_mask.long(), attack_conversation

    def attack_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        original_model: Optional[PreTrainedModel],
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        original_conversations_batch: List[Conversation],
        attention_mask_batch: torch.Tensor,
        attack_masks_batch: torch.Tensor,
        target_masks_batch: torch.Tensor
    ) -> list[SingleAttackRunResult]:
        t_start = time.time()
        device = model.device
        B, L = x_batch.shape
        disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        attack_masks_batch = attack_masks_batch.to(device).bool()
        target_masks_batch = target_masks_batch.to(device).bool()

        original_embeddings = model.get_input_embeddings()(x_batch)
        if self.config.attack_space == "one-hot":
            perturbed_embeddings_or_one_hot = (
                F.one_hot(x_batch, num_classes=model.config.vocab_size)
                .to(model.dtype)
                .to(device)
                .detach()
            )
        elif self.config.attack_space == "embedding":
            perturbed_embeddings_or_one_hot = original_embeddings.detach().clone()
        else:
            raise ValueError(f"Unknown attack space {self.config.attack_space}")

        if self.zero_init_attack:
            perturbed_embeddings_or_one_hot[attack_masks_batch] = 0

        benign_ref_data = None
        if self.config.tie_logits > 0 or self.config.tie_features > 0:
             benign_ref_data = self._setup_benign_reference(model, tokenizer, B, device)

        batch_losses = [[] for _ in range(B)]
        batch_perturbed_embeddings_list = [[] for _ in range(B)]
        batch_times = [[] for _ in range(B)]

        t_start = time.time()
        pbar = trange(self.config.num_steps, desc=f"Running PGD Attack Loop on {B} conversations", file=sys.stdout)
        perturbed_embeddings_or_one_hot.requires_grad = True
        optimizer = self._initialize_optimizer([perturbed_embeddings_or_one_hot])

        for step in pbar:
            t0 = time.time()
            perturbed_embeddings = self._maybe_convert_to_embeddings(perturbed_embeddings_or_one_hot, model)
            outputs = model(
                inputs_embeds=perturbed_embeddings,
                attention_mask=attention_mask_batch,
                output_hidden_states=True
            )

            loss = self._calculate_loss(outputs.logits, y_batch, target_masks_batch, tokenizer)

            kl_div_loss = 0.0
            if original_model is not None and (self.config.tie_logits > 0 or self.config.tie_features > 0):
                kl_div_loss = self._calculate_tying_loss(
                    model, original_model, perturbed_embeddings, attention_mask_batch,
                    attack_masks_batch, outputs, benign_ref_data, device
                )

            total_loss = loss + kl_div_loss
            total_loss.mean().backward()

            grad = perturbed_embeddings_or_one_hot.grad

            with torch.no_grad():
                grad = self._modify_gradient(grad, attack_masks_batch, disallowed_ids)
                perturbed_embeddings_or_one_hot = self._perform_optimizer_step(
                    optimizer, perturbed_embeddings_or_one_hot, original_embeddings, grad, attack_masks_batch, step
                )

            model.zero_grad()
            pbar.set_postfix({"loss": loss.mean().item(), "kl_div": kl_div_loss.item() if isinstance(kl_div_loss, torch.Tensor) else kl_div_loss})
            if original_model is not None:
                 original_model.zero_grad()

            current_time = time.time() - t0
            step_losses = loss.detach().tolist()
            for i in range(B):
                batch_times[i].append(current_time)
                batch_losses[i].append(step_losses[i])
                # Storing only attack embeddings might be more memory efficient if needed later
                # For now, storing relevant segment as per original logic
                pert_emb_cpu = self._select_embeddings_for_generation(perturbed_embeddings_or_one_hot[i], target_masks_batch[i])
                batch_perturbed_embeddings_list[i].append(pert_emb_cpu)

        # Generation after all steps
        final_perturbed_embeddings_flat = []
        # We need the embeddings corresponding to the *input* for generation, not just attack tokens
        # Assuming the generation should start after the prompt+attack string
        for i in range(B):
             # Find the last index of the non-target part (pre+attack+prompt+post)
             end_of_input_idx = torch.where(~target_masks_batch[i].roll(1,0))[0][-1].item()
             for step in range(self.config.num_steps):
                input_embeds_or_one_hot = batch_perturbed_embeddings_list[i][step][:end_of_input_idx + 1]
                input_embeds = self._maybe_convert_to_embeddings(input_embeds_or_one_hot.to(model.device), model).cpu()
                final_perturbed_embeddings_flat.append(input_embeds.cpu())

        # Generate based on the *final* perturbation state for each item in the batch
        logging.info(f"Attacks done, generating completions...")
        outputs = generate_ragged_batched(
            model,
            tokenizer,
            embedding_list=final_perturbed_embeddings_flat,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )
        logging.info(f"Generated {len(outputs)}x{self.config.generation_config.num_return_sequences} completions")

        # Structure results
        t_end = time.time()
        runs = []
        for i in range(B):
             # Create step results, but only the last one has meaningful completions here
             steps = []
             for step in range(self.config.num_steps):
                if self.config.log_embeddings:
                     model_input_embeddings = batch_perturbed_embeddings_list[i][step].cpu()
                else:
                    model_input_embeddings = None
                steps.append(AttackStepResult(
                     step=step,
                     model_completions=outputs[i * self.config.num_steps + step],
                     time_taken=batch_times[i][step],
                     loss=batch_losses[i][step],
                     model_input_embeddings=model_input_embeddings,
                     model_input=original_conversations_batch[i],
                 ))
             input_conversation = original_conversations_batch[i]
             runs.append(SingleAttackRunResult(
                 original_prompt=input_conversation,
                 steps=steps,
                 total_time=(t_end - t_start) / B
             ))
        return runs

    def _maybe_convert_to_embeddings(self, embeddings_or_one_hot, model):
        if self.config.attack_space == "one-hot":
            embeddings = embeddings_or_one_hot @ model.get_input_embeddings().weight
            if hasattr(model.get_input_embeddings(), "embed_scale"):  # For gemma
                embeddings = embeddings * model.get_input_embeddings().embed_scale.to(embeddings)
            return embeddings
        else:
            return embeddings_or_one_hot

    def _calculate_loss(self, logits, targets, mask, tokenizer: Optional[PreTrainedTokenizerBase] = None):
        if self.config.loss == "ce":
            return self._calculate_ce_loss(logits, targets, mask)
        elif self.config.loss == "entropy_allowed":
            assert tokenizer is not None, "tokenizer is required for entropy_allowed loss"
            return self._calculate_entropy_allowed_loss(logits, mask, tokenizer=tokenizer)
        elif self.config.loss == "entropy_first_token":
            return self._calculate_entropy_first_token_loss(logits, mask)
        else:
            raise ValueError(f"Unknown loss {self.config.loss}")

    def _calculate_ce_loss(self, logits, targets, mask):
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        )
        loss = loss.view(targets.shape[0], -1) * mask  # (B, L)
        loss = loss.sum(dim=1) / (mask.sum(dim=1).float() + 1e-6)  # (B,)
        return loss

    def _calculate_entropy_allowed_loss(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase
    ) -> torch.Tensor:
        """
        KL divergence loss to encourage uniform distribution over allowed tokens
        on the first token position.

        Args
        ----
        logits : (B, L, V)  un-normalised logits
        mask   : (B, L)     1 = real token, 0 = padding
        adaptive_threshold : float (unused, kept for compatibility)

        Returns
        -------
        loss : (B,)   one scalar per sequence in the batch
        """
        disallowed_ids = get_disallowed_ids(tokenizer, allow_non_ascii=False, allow_special=False)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        B, L, D = log_probs.shape
        N_valid = D - len(disallowed_ids)
        tgt_dist = torch.full((1, 1, D), device=log_probs.device, fill_value=1 / N_valid)
        tgt_dist[0, 0, disallowed_ids] = 0
        model_probs = log_probs.exp()
        log_tgt = torch.log(tgt_dist + 1e-30)

        # Find the first nonzero index of mask for each sequence in the batch
        cum_mask = mask.float().cumsum(dim=1)
        temp_mask = (cum_mask == 1)
        first_token_indices = temp_mask.nonzero(as_tuple=True)[1]

        kl_loss = F.kl_div(
            log_tgt.expand(B, L, -1),
            model_probs,
            reduction="none"
        )  # (B, L, D)

        # Sum over vocabulary dimension and select first token for each sequence
        kl_loss = kl_loss.sum(dim=-1)  # (B, L)
        loss = kl_loss[torch.arange(B, device=logits.device), first_token_indices]  # (B,)

        return loss

    def _calculate_entropy_first_token_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        * maximise entropy of the **first** token (→ encourage diverse samples)

        Args
        ----
        logits : (B, L, V)  un-normalised logits
        mask   : (B, L)     1 = real token, 0 = padding
        adaptive_threshold : float
            confidence level that triggers the “low-entropy after the first
            token” term (defaults to 0.7)

        Returns
        -------
        loss : (B,)   one scalar per sequence in the batch
        """
        # ---- probabilities & entropy -----------------------------------------
        log_probs = F.log_softmax(logits, dim=-1)  # (B, L, D)
        probs = log_probs.exp()  # (B, L, D)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)

        # ---- first-token (exploration) term -----------------------------------
        cum_mask = mask.float().cumsum(dim=1)
        first_token_loss = - (entropy * (cum_mask == 1)).sum(dim=1)
        loss = first_token_loss
        return loss

    def _setup_benign_reference(self, model, tokenizer, batch_size, device):
        """This function pre-calculates embeddings and logits for a fixed benign prompt
        to be used in the tying loss calculation."""
        with torch.no_grad():
            prompt_benign = "Can you recommend a chocolate cake recipe for two?"
            conversation_benign = [{"role": "user", "content": prompt_benign}, {"role": "assistant", "content": ""}]
            parts = prepare_conversation(tokenizer, conversation_benign)[0]
            pre_toks, atk_pre, prompt_toks, atk_suf, post_toks, _ = parts  # Ignoring target tokens for input prep

            benign_input_ids = [torch.cat([pre_toks, atk_pre, prompt_toks, atk_suf, post_toks]).to(device)]
            target_tokens_benign = generate_ragged_batched(model, tokenizer, benign_input_ids, max_new_tokens=64, return_tokens=True)[0][0]

            all_parts_ids = (pre_toks, atk_pre, prompt_toks, atk_suf, post_toks, target_tokens_benign)
            all_embeds = [model.get_input_embeddings()(ids.to(device)).unsqueeze(0) for ids in all_parts_ids]  # list[(1, SeqLen, D)]

            # Select necessary parts and repeat for batch size
            pre_embeds = all_embeds[0].repeat(batch_size, 1, 1)
            prompt_embeds = all_embeds[2].repeat(batch_size, 1, 1)
            post_embeds = all_embeds[4].repeat(batch_size, 1, 1)
            target_embeds = all_embeds[5].repeat(batch_size, 1, 1)

            gen_size = post_toks.size(0) + target_tokens_benign.size(0)
            ref_inputs_embeds = torch.cat([pre_embeds, prompt_embeds, post_embeds, target_embeds], dim=1)

            # Calculate reference logits only once
            ref_logits = model(inputs_embeds=ref_inputs_embeds).logits[:, -gen_size:].detach()

            return {
                "pre_embeds": pre_embeds,
                "prompt_embeds": prompt_embeds,
                "post_embeds": post_embeds,
                "target_embeds": target_embeds,
                "ref_logits": ref_logits,
                "gen_size": gen_size
            }

    def _calculate_tying_loss(self, model, original_model, perturbed_embeddings, attention_mask,
                             attack_masks_batch, model_outputs, benign_ref_data, device):
        # Calculates KL divergence for logits and cosine similarity for features.
        kl_div_loss = torch.tensor(0.0, device=device)

        # 1. KL divergence on the main adversarial sequence logits
        with torch.no_grad():  # Original model pass should not accumulate gradients here
             original_outputs = original_model(
                 inputs_embeds=perturbed_embeddings.detach(),
                 attention_mask=attention_mask,
                 output_hidden_states=True
             )

        if self.config.tie_logits > 0:
            kl_div_loss += F.kl_div(
                F.log_softmax(model_outputs.logits, dim=-1),
                F.softmax(original_outputs.logits, dim=-1),
                reduction="batchmean",
                log_target=False # Use softmax on target
            ) * self.config.tie_logits

        # 2. Cosine similarity loss on hidden features
        if self.config.tie_features > 0:
            for perturbed_layer, original_layer in zip(model_outputs.hidden_states, original_outputs.hidden_states):
                 # Apply cosine similarity only on attack token positions? Or all tokens? Assuming all for now.
                layer_similarity_loss = (1 - F.cosine_similarity(perturbed_layer, original_layer.detach(), dim=-1).mean())
                kl_div_loss += layer_similarity_loss * self.config.tie_features

        # 3. KL divergence on the benign prompt output when attack applied
        if self.config.tie_logits > 0 and benign_ref_data is not None:
             # This part assumes all attack strings have the same length, which might not be true if optim_str_init varies
             # A safer approach might involve padding or handling ragged tensors if lengths differ significantly.
             # Assuming fixed length based on optim_str_init for now.
             attack_embeds_batch = perturbed_embeddings[attack_masks_batch].view(perturbed_embeddings.size(0), -1, perturbed_embeddings.size(-1))

             # Construct input for benign prompt check
             benign_check_inputs = torch.cat([
                 benign_ref_data["pre_embeds"],
                 benign_ref_data["prompt_embeds"],
                 attack_embeds_batch,             # Inserted attack
                 benign_ref_data["post_embeds"],
                 benign_ref_data["target_embeds"]
             ], dim=1)

             # Adjust attention mask if needed, assuming simple concatenation works for now
             benign_check_logits = model(inputs_embeds=benign_check_inputs).logits
             benign_check_logits_target = benign_check_logits[:, -benign_ref_data["gen_size"]:]

             kl_div_loss += F.kl_div(
                 F.log_softmax(benign_check_logits_target, dim=-1),
                 F.softmax(benign_ref_data["ref_logits"].detach(), dim=-1),
                 reduction="batchmean",
                 log_target=False
             ) * self.config.tie_logits  # Use the same tying factor?

        return kl_div_loss

    def _modify_gradient(self, grad, attack_mask, disallowed_ids):
        if self.config.attack_space == "one-hot":
            grad[..., disallowed_ids] = 0
        grad[~attack_mask] = 0
        return grad

    @torch.no_grad()
    def _perform_optimizer_step(self, optimizer, perturbed_embeds, original_embeds, grad, attack_mask, current_step):
        optimizer.step()
        # Project delta back into epsilon ball
        if self.config.random_restart_interval > 0 and (current_step + 1) % self.config.random_restart_interval == 0:
            perturbed_embeds.data = original_embeds.clone() + torch.randn_like(original_embeds) * self.config.random_restart_epsilon
            return perturbed_embeds

        if self.config.attack_space == "embedding":
            delta = perturbed_embeds - original_embeds
            if self.config.projection == "l2":
                perturbed_embeds.data = original_embeds + self.project_l2(delta)
            elif self.config.projection == "l1":
                perturbed_embeds.data = original_embeds + self.project_l1(delta)
            else:
                raise ValueError(f"Unknown projection {self.config.projection}")
            return perturbed_embeds
        elif self.config.attack_space == "one-hot":
            if self.config.projection == "simplex":
                perturbed_embeds.data = self.project_simplex_opt(perturbed_embeds)
            elif self.config.projection == "lp_simplex":
                perturbed_embeds.data = self.project_lp_simplex(perturbed_embeds)
            else:
                raise ValueError(f"Unknown projection {self.config.projection}")
            return perturbed_embeds
        else:
            raise ValueError(f"Unknown attack space {self.config.attack_space}")

    def project_l2(self, delta: torch.Tensor) -> torch.Tensor:
        norm = delta.norm(p=2, dim=-1, keepdim=True)
        eps_normalized = self.config.epsilon * self.embedding_scale
        mask = norm > eps_normalized
        scaling_factor = torch.where(mask, eps_normalized / (norm + 1e-9), torch.ones_like(norm))
        return delta * scaling_factor

    def project_l1(self, delta):
        b, t, d = delta.shape
        eps = self.config.epsilon * self.embedding_scale
        original_shape = delta.shape
        dtype = delta.dtype
        delta_flat = delta.view(b * t, -1)

        # Mask for entries already within the L1 ball
        norm_l1 = torch.norm(delta_flat, p=1, dim=1)
        mask = (norm_l1 <= eps).float().unsqueeze(1)

        # Calculations for projection (Duchi et al., 2008)
        mu, _ = torch.sort(torch.abs(delta_flat), dim=1, descending=True)
        cumsum = torch.cumsum(mu, dim=1)
        arange = torch.arange(1, d + 1, device=delta.device, dtype=dtype)

        # Find rho: max{j | mu_j - (1/j) * (sum_{i=1}^j mu_i - eps) > 0}
        # Simplified check: mu * arange > cumsum - eps
        rho_check = (mu * arange > (cumsum - eps))
        # Ensure rho is at least 1 if any check passes, handle cases where all are false (rho=0 implicitly)
        rho = torch.sum(rho_check, dim=1) # Number of elements > theta
        # Need indices for gather, rho values are counts (1-based index)
        rho_indices = (rho - 1).clamp(min=0) # Convert to 0-based index safely

        # Calculate theta = (sum_{i=1}^{rho} mu_i - eps) / rho
        # Use gather to select correct cumsum value based on calculated rho
        sum_mu_rho = torch.gather(cumsum, 1, rho_indices.unsqueeze(1)).squeeze(1)
        # Avoid division by zero if rho is 0
        theta = torch.where(rho > 0, (sum_mu_rho - eps) / rho.to(dtype), torch.zeros_like(rho, dtype=dtype))

        # Projection: sign(x) * max(abs(x) - theta, 0)
        proj = (torch.abs(delta_flat) - theta.unsqueeze(1)).clamp(min=0)
        delta_proj = proj * torch.sign(delta_flat)

        # Combine projected and non-projected parts
        delta_final_flat = mask * delta_flat + (1 - mask) * delta_proj
        return delta_final_flat.view(original_shape).to(dtype)

    @staticmethod
    def project_simplex_opt(values):
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

    def project_lp_simplex(self, values: torch.Tensor, p: float = 2) -> torch.Tensor:
        """L_p projection onto the simplex.
        Args:
            values: A tensor of shape (batch_size, num_tokens) containing the values to project onto the simplex.
        Returns:
            A tensor of shape (batch_size, num_tokens) containing the projected values.
        """
        values.clamp_min_(0)
        values.div_(values.norm(dim=-1, keepdim=True, p=p))
        return values

    @staticmethod
    def _select_embeddings_for_generation(embeddings: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        # Selects embeddings up to the start of the target sequence
        # target_mask indicates target tokens. We want tokens *before* the first target token.
        # Rolling makes the mask True for the token *before* the target starts.
        # cumsum identifies all tokens up to and including that point.
        # Negating selects the desired input prefix.
        input_mask = ~(target_mask.roll(1, 0).cumsum(0).bool())
        return embeddings[input_mask].detach().cpu()


class FGSMOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float, **kwargs):
        # We pass an empty defaults dict since we handle config ourselves
        super().__init__(params, defaults={})
        self.lr = lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # FGSM-style update: perturb in direction of gradient sign
                p.data.sub_(self.lr * p.grad.sign())

        return loss
