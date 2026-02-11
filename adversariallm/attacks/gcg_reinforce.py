"""WIP GCG REINFORCE attack implementation.

This implements the REINFORCE version of the GCG attack, which uses a judge model
to provide rewards for generated completions and optimizes using REINFORCE gradients.

@article{geisler2025reinforce,
  title={Reinforce adversarial attacks on large language models: An adaptive, distributional, and semantic objective},
  author={Geisler, Simon and Wollschl{\"a}ger, Tom and Abdalla, MHI and Cohen-Addad, Vincent and Gasteiger, Johannes and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2502.17254},
  year={2025}
}
"""

import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from functools import partial

from judgezoo import Judge
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..dataset import PromptDataset
from ..lm_utils import (
    TokenMergeError,
    filter_suffix,
    generate_ragged_batched,
    get_disallowed_ids,
    get_flops,
    get_losses_batched,
    prepare_conversation,
    with_max_batchsize,
)
from ..types import Conversation
from .attack import (
    Attack,
    AttackResult,
    AttackStepResult,
    GenerationConfig,
    SingleAttackRunResult,
)


@dataclass
class GCGReinforceConfig:
    name: str = "gcg_reinforce"
    type: str = "discrete"
    version: str = "0.0.1"
    placement: str = "suffix"
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    num_steps: int = 500
    seed: int = 0
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    mellowmax_alpha: float = 1.0
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    allow_special: bool = False
    filter_ids: bool = True
    verbosity: str = "WARNING"
    token_selection: str = "default"
    # REINFORCE parameters
    judge_model_id: str = "strong_reject"
    optim_max_new_tokens: int = 256
    optim_temperature: float = 0.7
    optim_top_p: float = 1.0
    optim_top_k: int = 0
    reward_baseline: float = 0.1  # all generations below this are penalized no matter what
    token_position_weight_ratio: float = 5.0
    token_position_weight_type: str = "linear"
    loss_include_categories: tuple[str] = ("samples", "greedy", "buffer", "affirmative")
    selection_include_categories: tuple[str] = ("greedy", "buffer", "affirmative")
    optim_num_return_sequences: int = 2
    elite_buffer_size: int = 1
    optim_num_return_sequences_for_grad: int = 0
    harmful_threshold: float = 0.5  # for exploit_greedy
    entropy_losses_weight: float = 0.0


class GCGReinforceAttack(Attack[GCGReinforceConfig]):
    def __init__(self, config: GCGReinforceConfig):
        super().__init__(config)
        self.tokenizer = None  # Will be set in run()
        self.model = None
        self.elite_buffer = []  # List of (generation, reward) tuples
        self.previous_greedy_reward = 0.0  # Track previous greedy reward for exploit_greedy
        self.previous_optim_ids = None  # Track previous optim_ids for exploit_greedy
        self.previous_rewards = None  # Track previous rewards for exploit_greedy
        self.previous_generations = None  # Track previous generations for exploit_greedy
        self.previous_categories = None  # Track previous categories for exploit_greedy
        self.reward_cache = {}  # Cache of rewards for each completion

        self.logger = logging.getLogger("gcg_reinforce")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: PromptDataset,
    ) -> AttackResult:
        self.judge = Judge.from_name(self.config.judge_model_id)
        self.model = model
        self.tokenizer = tokenizer  # Store tokenizer as instance variable
        self.not_allowed_ids = get_disallowed_ids(
            tokenizer, self.config.allow_non_ascii, self.config.allow_special
        ).to(model.device)
        # need to have this filter here for models like gemma-3 which add extra tokens that do not have embeddings
        # we cannot filter the ids inside the get_disallowed_ids function because we need
        # the embedding layer weights to see the correct sizes
        self.not_allowed_ids = self.not_allowed_ids[
            self.not_allowed_ids < model.get_input_embeddings().weight.size(0)
        ]
        runs = []
        for conversation in dataset:
            runs.append(
                self._attack_single_conversation(model, tokenizer, conversation)
            )
        return AttackResult(runs=runs)

    def _attack_single_conversation(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, conversation: Conversation
    ) -> SingleAttackRunResult:
        t0 = time.time()
        pre_prompt_ids, attack_suffix_ids, post_ids, target_ids = self._tokenize_conversation(conversation)

        # Embed everything that doesn't get optimized
        embedding_layer = model.get_input_embeddings()
        pre_prompt_embeds, post_embeds, target_embeds = [
            embedding_layer(ids) for ids in (pre_prompt_ids, post_ids, target_ids)
        ]

        self.target_ids = target_ids
        self.pre_prompt_embeds = pre_prompt_embeds
        self.post_embeds = post_embeds
        self.target_embeds = target_embeds

        # Initialize the attack buffer
        optim_ids = attack_suffix_ids

        rewards = []
        times = []
        flops = []
        optim_strings = []
        losses = []
        for i in (pbar := trange(self.config.num_steps, file=sys.stdout)):
            current_reward, loss, time_for_step, optim_ids, flops_for_step = (
                self._single_step(conversation, optim_ids)
            )
            rewards.append(current_reward.mean().item())
            losses.append(loss)
            times.append(time_for_step)
            optim_strings.append(self.tokenizer.batch_decode(optim_ids)[0])

            if i == 0:
                flops.append(flops_for_step)
            else:
                flops.append(flops_for_step)
            pbar.set_postfix(
                {
                    "Mean Reward": current_reward.mean().item(),
                    "Max Reward": current_reward.max().item(),
                    "Loss": loss,
                    "Best Attack": optim_strings[-1][:80].replace('\n', '\\n'),
                }
            )
        token_list = []
        attack_conversations = []
        for attack in optim_strings:
            attack_conversation = [
                {"role": "user", "content": conversation[0]["content"] + attack},
                {"role": "assistant", "content": ""},
            ]
            tokens = prepare_conversation(tokenizer, conversation, attack_conversation)[0]
            token_list.append(torch.cat(tokens[:5]))
            attack_conversations.append(attack_conversation)

        batch_completions = generate_ragged_batched(
            model,
            tokenizer,
            token_list=token_list,
            initial_batch_size=len(token_list),
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
        )  # (N_steps, N_return_sequences, T)

        steps = []
        t1 = time.time()
        for i in range(len(optim_strings)):
            step = AttackStepResult(
                step=i,
                model_completions=batch_completions[i],
                time_taken=times[i],
                loss=losses[i],
                flops=flops[i],
                model_input=attack_conversations[i],
                model_input_tokens=token_list[i].tolist(),
            )
            steps.append(step)

        run = SingleAttackRunResult(
            original_prompt=conversation,
            steps=steps,
            total_time=t1 - t0,
        )
        return run

    def _single_step(self, conversation, optim_ids):
        """Each GCG step consists of three basic steps:

        1) Compute gradients w.r.t a differentiable loss function
        2) Sample & Filter candidate token sequences (possibly with help of the gradients)
        3) Compute the loss for each candidate token sequence and select the best one

        In our case, the loss function is the REINFORCE loss.

        Parameters
        ----------
        conversation : Conversation
            The original conversation to attack
        optim_ids : Tensor, shape = (1, T)
            The current attack token ids
        """
        t0a = time.time()

        t_rewards_start = time.time()
        rewards, generations, categories, flops_loss = self.compute_candidate_rewards(conversation, optim_ids)
        t_rewards = time.time() - t_rewards_start
        self.logger.info(f"  Compute rewards: {t_rewards:.2f}s")

        # Once the greedy generation has become harmful, we only accept new optim_ids if
        # the greedy generation remains harmful.
        if "greedy" in self.config.loss_include_categories and self.previous_greedy_reward > self.config.harmful_threshold:
            if self._get_greedy_reward(rewards) <= self.config.harmful_threshold:
                self.logger.info("Exploit greedy: Rejected new optim_ids")
                self.logger.info(f"  {self.tokenizer.decode(optim_ids.squeeze(0))} with {self._get_greedy_reward(rewards):.3f}")
                self.logger.info(f"  {self.tokenizer.decode(self.previous_optim_ids.squeeze(0))}  had {self.previous_greedy_reward:.3f}")
                optim_ids = self.previous_optim_ids
                rewards = self.previous_rewards
                generations = self.previous_generations
                categories = self.previous_categories

        self.previous_optim_ids = optim_ids
        self.previous_rewards = rewards
        self.previous_generations = generations
        self.previous_categories = categories
        self.previous_greedy_reward = self._get_greedy_reward(rewards)

        t_grad_start = time.time()
        advantages = self.compute_advantages(rewards)  # (num_gens,)

        optim_ids_one_hot = F.one_hot(optim_ids, num_classes=self.model.get_input_embeddings().weight.size(0)).to(self.model.dtype).requires_grad_(True)
        loss_for_grad, flops_loss = self.compute_reinforce_loss(generations, advantages, optim_ids_one_hot, rewards, categories)

        grad = torch.autograd.grad([loss_for_grad], [optim_ids_one_hot])[0].squeeze(0)

        t_grad = time.time()
        print(f"  Compute gradients: {t_grad - t_grad_start:.2f}s")

        if grad.isinf().all() or grad.isnan().all():
            candidate_ids, candidate_ids_pos = _random_overall(
                ids=optim_ids.squeeze(0),
                vocab_size=grad.size(-1),
                search_width=self.config.search_width,
                not_allowed_ids=self.not_allowed_ids,
            )
        else:
            candidate_ids, candidate_ids_pos = _sample_ids_from_grad(
                ids=optim_ids.squeeze(0),
                grad=grad,
                search_width=self.config.search_width,
                topk=self.config.topk,
                n_replace=self.config.n_replace,
                not_allowed_ids=self.not_allowed_ids,
            )
        t_sample_ids = time.time()
        print(f"  Sample ids: {t_sample_ids - t_grad:.2f}s")

        with torch.no_grad():
            # Sample candidate token sequences
            if self.config.filter_ids:
                # We're trying to be as strict as possible here, so we filter
                # the entire prompt, not just the attack sequence in an isolated
                # way. This is because the prompt and attack can affect each
                # other's tokenization in some cases.
                idx = filter_suffix(
                    self.tokenizer,
                    conversation,
                    [[None, candidate_ids.cpu()]],
                )
                candidate_ids = candidate_ids[idx] # (B, T)
                candidate_ids_pos = candidate_ids_pos[idx] # (B, T)
            t_filter_ids = time.time()
            print(f"  Filter ids: {t_filter_ids - t_sample_ids:.2f}s")

            keep_idx = [i for i, c in enumerate(categories) if c in self.config.selection_include_categories]
            gens_for_selection = [generations[i] for i in keep_idx]
            advantages_for_selection = advantages[keep_idx]
            categories_for_selection = [categories[i] for i in keep_idx]
            print("For selection:\n--------------------------------")
            self._print_info(
                gens_for_selection,
                advantages_for_selection,
                rewards[keep_idx],
                categories_for_selection
            )

            compute_loss_fn = partial(
                self.compute_candidates_loss,
                gens_for_selection,
                advantages_for_selection,
                categories_for_selection
            )

            loss_for_candidates, flops_loss = with_max_batchsize(compute_loss_fn, candidate_ids)
            t_compute_loss = time.time()
            print(f"  Compute loss: {t_compute_loss - t_filter_ids:.2f}s")

            optim_ids = candidate_ids[loss_for_candidates.argmin()].unsqueeze(0)
            current_loss = loss_for_candidates[loss_for_candidates.argmin()].item()
            flops_for_step = flops_loss.sum().long().item()
            # Update the buffer based on the loss

        return rewards, current_loss, time.time() - t0a, optim_ids, flops_for_step

    @torch.no_grad()
    def compute_candidate_rewards(
        self,
        conversation: Conversation,
        attack_suffix_ids: Tensor,
    ) -> tuple[Tensor, list[Tensor], list[str], int]:
        """Computes the GCG reward on all candidate token id sequences.

        Args:
            attack_suffix_ids : Tensor, shape = (1, T)
                the attack token ids to evaluate
            conversation : Conversation
                the original dataset conversation to evaluate

        Returns:
            reward : Tensor, shape = (B,)
                the GCG reward on all candidate sequences
            completions : list[Tensor], each tensor is shape (T)
                the completions for all candidate sequences
            categories : list[str]
                Categories corresponding to each candidate sequence
            flops : int
                Estimate of FLOPS used for all computations in this function
        """
        flops = 0

        # Convert candidate_ids to one_hot and get embeddings
        embeds_to_generate_with = torch.cat(
            [
                self.pre_prompt_embeds,
                self.model.get_input_embeddings()(attack_suffix_ids),
                self.post_embeds,
            ],
            dim=1,
        )[0] # (T, V)

        completions = generate_ragged_batched(
            model=self.model,
            tokenizer=self.tokenizer,
            embedding_list=[embeds_to_generate_with],
            max_new_tokens=self.config.optim_max_new_tokens,
            temperature=self.config.optim_temperature,
            top_p=self.config.optim_top_p,
            top_k=self.config.optim_top_k,
            num_return_sequences=self.config.optim_num_return_sequences,
            return_tokens=True
        )[0]

        # Track categories
        categories = ["samples"] * len(completions)
        n_input_tokens = embeds_to_generate_with.size(0) * self.config.optim_num_return_sequences
        # Take into account prefix caching.
        n_redundant_tokens = self.pre_prompt_embeds.size(1) * (self.config.optim_num_return_sequences - 1)
        n_input_tokens -= n_redundant_tokens
        n_output_tokens = sum(len(c) for c in completions)
        flops += get_flops(self.model, n_input_tokens, n_output_tokens, "forward")

        if "greedy" in self.config.loss_include_categories:
            # Generate greedy response (temperature=0)
            greedy_completion = generate_ragged_batched(
                model=self.model,
                tokenizer=self.tokenizer,
                embedding_list=[embeds_to_generate_with],
                max_new_tokens=self.config.optim_max_new_tokens,
                temperature=0.0,  # Greedy sampling
                top_p=1.0,
                top_k=0,
                num_return_sequences=1,
                return_tokens=True
            )[0][0]  # Get the first (and only) completion
            completions.append(greedy_completion)
            categories.append("greedy")
            n_input_tokens = embeds_to_generate_with.size(0) - self.pre_prompt_embeds.size(1)
            n_output_tokens = len(greedy_completion)
            flops += get_flops(self.model, n_input_tokens, n_output_tokens, "forward")

        if "buffer" in self.config.loss_include_categories and self.config.elite_buffer_size > 0:
            elite_completions = self._get_elite_completions()
            completions.extend(elite_completions)
            categories.extend(["buffer"] * len(elite_completions))

        if "affirmative" in self.config.loss_include_categories:
            completions.append(self.target_ids[0])  # add y_affirmative
            categories.append("affirmative")

        conversation_with_completions: Conversation = [
            [
                {"role": "user", "content": conversation[0]["content"] },
                {"role": "assistant", "content": c},
            ] for c in self.tokenizer.batch_decode(completions)
        ]

        to_judge = [c for c in conversation_with_completions if c[1]["content"] not in self.reward_cache]
        # Compute judgements for the entire conversation
        if to_judge:
            judgements = self.judge(to_judge)["p_harmful"]
            for c, j in zip(to_judge, judgements):
                self.reward_cache[c[1]["content"]] = j

        rewards = torch.tensor([self.reward_cache[c[1]["content"]] for c in conversation_with_completions]).to(self.model.device)

        if self.config.judge_model_id == "strong_reject":
            inputs = [
                self.judge.tokenizer(self.judge.SR_CLS_PROMPT.format(behavior=c[0]["content"], response=c[1]["content"])).input_ids
                for c in conversation_with_completions
            ]
            n_input_tokens = sum(len(pr) for pr in inputs)
            n_output_tokens = 1 * len(inputs)
            flops += get_flops(self.judge.classifier, n_input_tokens, n_output_tokens, "forward")
        elif self.config.judge_model_id == "harmbench":
            cls_prompt = self.judge.tokenizer.apply_chat_template(
                self.judge.CLS_CONVERSATION, tokenize=False
            )
            inputs = [
                self.judge.tokenizer(cls_prompt.format(behavior=c[0]["content"], generation=c[1]["content"])).input_ids
                for c in conversation_with_completions
            ]
            n_input_tokens = sum(len(pr) for pr in inputs)
            n_output_tokens = 1 * len(inputs)
            flops += get_flops(self.judge.classifier, n_input_tokens, n_output_tokens, "forward")
        else:
            logging.warning(f"We do not compute flops for the judge `{self.config.judge_model_id}` right now")

        if "affirmative" in self.config.loss_include_categories:
            rewards[-1] = max(rewards[-1], 0.5)

        if "buffer" in self.config.loss_include_categories and self.config.elite_buffer_size > 0:
            self._update_elite_buffer(completions, rewards)

        # Select top-k sequences by reward for gradient computation
        if self.config.optim_num_return_sequences_for_grad != 0 and len(rewards) > self.config.optim_num_return_sequences_for_grad:
            top_k_indices = torch.topk(rewards, self.config.optim_num_return_sequences_for_grad, sorted=False).indices
            completions = [completions[i] for i in top_k_indices]
            rewards = rewards[top_k_indices]
            categories = [categories[i] for i in top_k_indices]

        gc.collect()
        torch.cuda.empty_cache()

        return rewards, completions, categories, flops

    def compute_advantages(self, rewards):
        """Computes the reinforce advantages for this set of generations with their rewards.
        We use the leave-one-out estimator from Koop et al. 2019 to compute the advantages.

        Args:
            rewards: (B,)
            generations: (B, T)
        """
        total_sum = rewards.sum() + self.config.reward_baseline
        n = rewards.size(0)
        if self.config.reward_baseline > 0:
            n += 1
        advantages = (rewards * n - total_sum) / (n - 1)
        return advantages + 1e-8

    @torch.no_grad()
    def compute_first_token_entropy(self, optim_ids: Tensor) -> float:
        """Computes the entropy of the first predicted token distribution.

        Args:
            optim_ids: Tensor, shape = (1, T) - Current attack token ids

        Returns:
            entropy: float - Entropy of the first token distribution
        """
        # Get embeddings for the current attack sequence
        optim_embeds = self.model.get_input_embeddings()(optim_ids)  # (1, T, D)

        # Prepare full input embeddings
        full_embeds = torch.cat([
            self.pre_prompt_embeds,  # (1, L_pre, D)
            optim_embeds,            # (1, T, D)
            self.post_embeds         # (1, L_post, D)
        ], dim=1)  # (1, L_total, D)

        # Forward pass to get logits
        outputs = self.model(inputs_embeds=full_embeds)

        # Get logits for the first generated token (position after input sequence)
        first_token_logits = outputs.logits[0, -1, :]  # (V,)

        # Compute entropy: H(p) = -sum(p * log(p))
        probs = F.softmax(first_token_logits.float(), dim=-1)  # (V,)
        log_probs = F.log_softmax(first_token_logits.float(), dim=-1)  # (V,)
        entropy = -(probs * log_probs).sum().item()

        return entropy

    def _get_greedy_reward(self, rewards: Tensor) -> float:
        """Extract the greedy completion reward from the rewards tensor.

        The greedy completion is added after the regular generated completions
        but before elite buffer and affirmative completions.

        Args:
            rewards: Tensor of shape (N,) containing all completion rewards

        Returns:
            greedy_reward: float - The reward for the greedy completion
        """
        assert "greedy" in self.config.loss_include_categories, "Greedy reward is only available if greedy is in loss_include_categories"

        # The greedy completion is at index optim_num_return_sequences
        greedy_idx = self.config.optim_num_return_sequences
        return rewards[greedy_idx].item()

    @torch.no_grad()
    def _evaluate_greedy_reward(self, conversation: Conversation, attack_suffix_ids: Tensor) -> float:
        """Evaluate the greedy completion reward for a given attack string.

        Args:
            conversation: The original conversation
            attack_suffix_ids: Tensor of shape (1, T) - Attack token ids to evaluate

        Returns:
            greedy_reward: float - The reward for the greedy completion
        """
        # Get embeddings for the attack sequence
        embeds_to_generate_with = torch.cat(
            [
                self.pre_prompt_embeds,
                self.model.get_input_embeddings()(attack_suffix_ids),
                self.post_embeds,
            ],
            dim=1,
        )[0]  # (T, V)

        # Generate greedy response
        greedy_completion = generate_ragged_batched(
            model=self.model,
            tokenizer=self.tokenizer,
            embedding_list=[embeds_to_generate_with],
            max_new_tokens=self.config.optim_max_new_tokens,
            temperature=0.0,  # Greedy sampling
            top_p=1.0,
            top_k=0,
            num_return_sequences=1,
            return_tokens=True
        )[0][0]  # Get the first (and only) completion

        # Create conversation for judging
        conversation_with_completion = [
            {"role": "user", "content": conversation[0]["content"]},
            {"role": "assistant", "content": self.tokenizer.decode(greedy_completion)},
        ]

        # Get judgment
        judgement = self.judge([conversation_with_completion])["p_harmful"][0]
        return judgement

    def compute_entropy_loss(self, candidate_ids: Tensor) -> Tensor:
        """Computes the entropy maximization loss using kl_allowed_fwd for candidate token id sequences.

        This implements the entropy maximization loss that encourages the model to produce diverse
        tokens while avoiding disallowed tokens, similar to the kl_allowed_fwd implementation in gcg.py.

        Args:
            candidate_ids: (N, T) tensor

        Returns:
            loss: (N,) tensor
        """
        # Get embeddings directly from candidate_ids
        optim_embeds = self.model.get_input_embeddings()(candidate_ids)  # (N, T, D)

        N = optim_embeds.size(0)  # (N, T, V)
        # Prepare input embeddings
        pre_prompt: Tensor = self.pre_prompt_embeds  # (L_pre, D)
        post: Tensor = self.post_embeds  # (L_post, D)
        pre_prompt = pre_prompt.expand(N, -1, -1)  # (N, L_pre, D)
        post = post.expand(N, -1, -1)  # (N, L_post, D)

        # We only need the first token's logits for entropy computation, no generation required
        full_embeds = torch.cat([pre_prompt, optim_embeds, post], dim=1)  # (N, L_pre + L_opt + L_post, D)
        outputs = self.model(inputs_embeds=full_embeds)
        first_token_logits = outputs.logits[:, -1, :].clone()  # (N, T, V) -> (N, V)

        # Get the log probs for the last position
        # This corresponds to the first generated token
        log_probs = F.log_softmax(first_token_logits.float(), dim=-1)  # (N, V)

        # Create target distribution: uniform over allowed tokens, zero for disallowed
        V = first_token_logits.size(-1)
        N_valid = V - len(self.not_allowed_ids)
        tgt_dist = torch.full((V,), device=log_probs.device, fill_value=1 / N_valid)
        tgt_dist[self.not_allowed_ids] = 0

        model_probs = log_probs.exp()  # (N, V)
        log_tgt = torch.log(tgt_dist + 1e-30)  # (V,) - tiny ε avoids log(0) → -inf

        loss = F.kl_div(
            log_tgt.unsqueeze(0).expand(N, -1),  # (N, V)
            model_probs,  # (N, V)
            reduction="none"
        ).sum(dim=-1)  # (N, V) -> (N,)

        return loss

    def compute_reinforce_loss(self, generations: list[Tensor], advantages: Tensor, candidate_ids: Tensor, rewards: Tensor = None, categories: list[str] = None) -> Tensor:
        """Computes the REINFORCE loss for candidate token id sequences.

        Args:
            generations: list of B (T,) tensors
            advantages: (B,) tensor in [0, 1]
            candidate_ids: (N, T) tensor or (N, T, V) tensor

        Returns:
            loss: (N,) tensor
        """
        N = candidate_ids.size(0)
        B = len(generations)
        V = self.model.get_input_embeddings().weight.size(0)

        # Convert candidate_ids to one_hot and get embeddings
        if candidate_ids.dim() == 2:
            # Can't compute gradients in this scenario
            candidate_ids_one_hot = F.one_hot(candidate_ids, num_classes=V).to(self.model.dtype)
        else:
            candidate_ids_one_hot = candidate_ids
        embedding_layer = self.model.get_input_embeddings()
        optim_embeds = candidate_ids_one_hot @ embedding_layer.weight  # (N, T, V)
        if hasattr(embedding_layer, "embed_scale"):  # For gemma
            optim_embeds = optim_embeds * embedding_layer.embed_scale.to(optim_embeds)

        # Create input embeddings for each candidate and generation combination
        pre = self.pre_prompt_embeds.expand(N, -1, -1)  # (N, L_pre , V)
        post = self.post_embeds.expand(N, -1, -1)  # (N, L_post, V)
        non_gen_embeds = torch.cat([pre, optim_embeds, post], dim=1)  # (N, L_pre + L_opt + L_post, V)

        embedding_list = []
        targets = []

        for gen in generations:
            gen = gen.to(self.model.device)
            gen_embeds: Tensor = self.model.get_input_embeddings()(gen)  # (L_gen , V)
            gen_embeds = gen_embeds.unsqueeze(0).expand(N, -1, -1) # (N, L_gen , V)
            full_embeds = torch.cat([non_gen_embeds, gen_embeds], dim=1) # (N, L_pre + L_opt + L_post + L_gen, V)

            seq_len = full_embeds.size(1)
            tgt = torch.zeros((N, seq_len), dtype=torch.long, device=self.model.device)
            tgt[:, -len(gen) - 1:-1] = gen

            embedding_list.extend([e for e in full_embeds])
            targets.extend([t for t in tgt])
        # Get losses using get_losses_batched
        losses_list = get_losses_batched(
            model=self.model,
            targets=targets,  # len(generations) * N * (T,)
            embedding_list=embedding_list,  # len(generations) * N * (T, V)
            padding_side="right",
            initial_batch_size=512,
        )
        if candidate_ids_one_hot.requires_grad:
            flops_loss = get_flops(self.model, non_gen_embeds.size(0)*non_gen_embeds.size(1), sum(len(gen) for gen in generations), "forward_and_backward")
        else:
            flops_loss = get_flops(self.model, non_gen_embeds.size(0)*non_gen_embeds.size(1), sum(len(gen) for gen in generations), "forward")

        if self.config.token_position_weight_type == "linear":
            token_weights = torch.linspace(self.config.token_position_weight_ratio, 1, self.config.optim_max_new_tokens, device=self.model.device)
        elif self.config.token_position_weight_type == "exponential":
            token_weights = torch.arange(self.config.optim_max_new_tokens, device=self.model.device)
            token_weights = self.config.token_position_weight_ratio ** token_weights
        else:
            raise ValueError(f"Invalid token position weight type: {self.config.token_position_weight_type}")
        # Reshape losses back to (B, N) and normalize by sequence length
        # TODO: try out
        # clipping max per-token loss
        # rescaling by ce magnitude
        reinforce_losses = torch.zeros((B, N), device=self.model.device)

        for i, gen in enumerate(generations):
            gen_len = len(gen)
            if gen_len == 0:
                continue
            start_idx = i * N
            end_idx = start_idx + N
            weights = token_weights[:gen_len].clone()
            weights = weights / weights.sum()

            # batch_losses = torch.stack([(losses_list[idx][-gen_len:]/(losses_list[idx][-gen_len:].detach().abs() + 1e-3) * weights).sum() for idx in range(start_idx, end_idx)])
            batch_losses = torch.stack([(losses_list[idx][-gen_len:] * weights).sum() for idx in range(start_idx, end_idx)])
            reinforce_losses[i, :] = batch_losses
        # Apply advantages and take mean over generations
        if N == 1 and rewards is not None:
            self._print_info(generations, advantages, rewards, categories, reinforce_losses)
        reinforce_losses = (reinforce_losses * advantages.unsqueeze(1)).mean(dim=0)  # (N,)
        return reinforce_losses, torch.tensor(flops_loss).expand_as(reinforce_losses) / N

    def _print_info(self, generations, advantages, rewards, categories, reinforce_losses=None):
        # Define color codes for each category
        colors = {
            "samples": "\033[94m",      # Blue
            "greedy": "\033[92m",       # Green
            "buffer": "\033[93m",       # Yellow
            "affirmative": "\033[95m"   # Magenta
        }
        reset_color = "\033[0m"

        # Print formatted header based on whether we have losses
        if reinforce_losses is not None:
            print(f"{'Generation':<100} {'Category':<12} {'Advantage':<12} {'Raw Reward':<12} {'Loss':<12}")
            print("-" * 148)
        else:
            print(f"{'Generation':<100} {'Category':<12} {'Advantage':<12} {'Raw Reward':<12}")
            print("-" * 136)

        for i, (gen, adv, reward, category) in enumerate(zip(generations, advantages, rewards, categories)):
            generation_text = self.tokenizer.decode(gen)[:100].replace('\n', '\\n')
            color = colors.get(category, "")

            # Format values
            advantage_val = f"{adv.item():.6f}"
            reward_val = f"{reward.item():.6f}"

            # Print with or without loss column
            if reinforce_losses is not None:
                loss_val = f"{reinforce_losses[i].item():.6f}"
                print(f"{color}{generation_text:<100} {category:<12} {advantage_val:<12} {reward_val:<12} {loss_val:<12}{reset_color}")
            else:
                print(f"{color}{generation_text:<100} {category:<12} {advantage_val:<12} {reward_val:<12}{reset_color}")

    @torch.no_grad()
    def compute_candidates_loss(self, generations: list[Tensor], advantages: Tensor, categories: list[str], candidate_ids: Tensor):
        """Computes the combined REINFORCE and affirmative loss for candidate token id sequences.

        Args:
            generations: list of B (T,) tensors
            advantages: (B,) tensor in [0, 1]
            candidate_ids: (N, T) tensor

        Returns:
            loss: (N,) tensor
        """
        t0_candidates = time.time()
        # Compute REINFORCE loss
        t_reinforce_start = time.time()
        reinforce_losses, flops_loss = self.compute_reinforce_loss(generations, advantages, candidate_ids, categories=categories)
        t_reinforce = time.time() - t_reinforce_start
        print(f"    REINFORCE loss computation: {t_reinforce:.3f}s")
        losses = reinforce_losses
        # Compute entropy maximization loss
        if self.config.entropy_losses_weight != 0:
            t_entropy_start = time.time()
            entropy_losses = self.compute_entropy_loss(candidate_ids)
            t_entropy = time.time() - t_entropy_start
            reinforce_losses_avg = reinforce_losses.mean()
            entropy_losses_avg = entropy_losses.mean()
            print(f"    Entropy loss computation: {t_entropy:.3f}s")
            print(reinforce_losses_avg.item() / entropy_losses_avg.item())
            entropy_losses_norm = entropy_losses * (reinforce_losses_avg / entropy_losses_avg)
            losses = losses + entropy_losses_norm * self.config.entropy_losses_weight

        t_total_candidates = time.time() - t0_candidates
        print(f"    Total candidates loss: {t_total_candidates:.3f}s")
        return losses, flops_loss

    def _get_elite_completions(self) -> list[Tensor]:
        """Returns the top completions from the elite buffer."""
        if not self.elite_buffer:
            return []

        # Sort by reward (descending) and take top completions
        sorted_buffer = sorted(self.elite_buffer, key=lambda x: x[1], reverse=True)
        elite_completions = [completion for completion, reward in sorted_buffer[:self.config.elite_buffer_size] if reward > self.config.harmful_threshold]
        return elite_completions

    def _update_elite_buffer(self, completions: list[Tensor], rewards: Tensor):
        """Updates the elite buffer with new completions and rewards."""
        # Determine how many completions are new (excluding affirmative, greedy, and elite buffer samples)
        n_new = self.config.optim_num_return_sequences + ("greedy" in self.config.loss_include_categories)

        # Add new completions to buffer
        new_completions = completions[:n_new]
        new_rewards = rewards[:n_new]

        for completion, reward in zip(new_completions, new_rewards):
            self.elite_buffer.append((completion.clone(), reward.item()))

        # Keep only the top k completions
        self.elite_buffer.sort(key=lambda x: x[1], reverse=True)
        self.elite_buffer = self.elite_buffer[:self.config.elite_buffer_size]

    def _tokenize_conversation(self, conversation: Conversation) -> list[str]:
        try:
            attack_conversation: Conversation = [
                {
                    "role": "user",
                    "content": conversation[0]["content"] + self.config.optim_str_init,
                },
                {"role": "assistant", "content": conversation[1]["content"]},
            ]
            (
                pre_ids,
                attack_prefix_ids,
                prompt_ids,
                attack_suffix_ids,
                post_ids,
                target_ids,
            ) = prepare_conversation(self.tokenizer, conversation, attack_conversation)[0]
        except TokenMergeError:
            attack_conversation: Conversation = [
                {
                    "role": "user",
                    "content": conversation[0]["content"]
                    + " "
                    + self.config.optim_str_init,
                },
                {"role": "assistant", "content": conversation[1]["content"]},
            ]
            (
                pre_ids,
                attack_prefix_ids,
                prompt_ids,
                attack_suffix_ids,
                post_ids,
                target_ids,
            ) = prepare_conversation(self.tokenizer, conversation, attack_conversation)[0]
        device = self.model.device

        pre_ids = pre_ids.to(device)
        attack_prefix_ids = attack_prefix_ids.to(device)
        assert attack_prefix_ids.size(0) == 0, "Attack prefix ids should be empty in the current implementation"
        prompt_ids = prompt_ids.to(device)
        pre_prompt_ids = torch.cat([pre_ids, attack_prefix_ids, prompt_ids], dim=0).unsqueeze(0)
        attack_suffix_ids = attack_suffix_ids.to(device).unsqueeze(0)
        post_ids = post_ids.to(device).unsqueeze(0)
        target_ids = target_ids.to(device).unsqueeze(0)
        return pre_prompt_ids, attack_suffix_ids, post_ids, target_ids


def _sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = None,
):
    """Returns `search_width` combinations of token ids based on the token gradient.
    Original GCG does this.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence
        not_allowed_ids: Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    # Initial gradient computation
    n_optim_ids = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        if grad.isinf().all() or grad.isnan().all():
            raise ValueError("Gradient is all inf or nan")
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    # fmt: off
    topk_ids = grad.topk(topk, dim=1, largest=False, sorted=False).indices  # (n_optim_ids, topk)
    sampled_ids_pos = torch.randint(0, n_optim_ids, (search_width, n_replace), device=grad.device)  # (search_width, n_replace)
    sampled_topk_idx = torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)  # (search_width, n_replace, 1)

    sampled_ids_val = topk_ids[sampled_ids_pos].gather(2, sampled_topk_idx).squeeze(2)  # (search_width, n_replace)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)  # (search_width, n_optim_ids)
    # fmt: on

    return new_ids, sampled_ids_pos


def _random_overall(
    ids: Tensor,
    vocab_size: int,
    search_width: int,
    not_allowed_ids: Tensor = None,
):
    """Returns `search_width` random token substitutions.

    Args:
        ids : Tensor, shape = (n_optim_ids,)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = ids.shape[0]
    original_ids = ids.repeat(search_width, 1)

    # Create valid token mask
    valid_tokens = torch.ones(vocab_size, dtype=torch.bool, device=ids.device)
    if not_allowed_ids is not None:
        valid_tokens[not_allowed_ids.to(ids.device)] = False

    # Sample positions and token indices
    sampled_ids_pos = torch.randint(0, n_optim_tokens, (search_width, 1), device=ids.device)
    valid_token_indices = torch.nonzero(valid_tokens).squeeze()
    sampled_topk_idx = valid_token_indices[torch.randint(0, valid_token_indices.size(0), (search_width, 1), device=ids.device)]

    # Create new sequences with substitutions
    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_topk_idx)
    return new_ids, sampled_ids_pos
