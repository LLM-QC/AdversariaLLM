"""Single-file implementation of the GCG attack with dynamic adversarial prefix creation.
"""
import gc
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Literal

import torch
import transformers
from accelerate.utils import find_executable_batch_size
from torch import Tensor
from tqdm import trange

from src.lm_utils import generate_ragged_batched, prepare_tokens

from .attack import Attack, AttackResult


@dataclass
class GCGRefusalConfig:
    name: str = "gcg_refusal"
    type: str = "discrete"
    placement: str = "suffix"
    generate_completions: Literal["all", "best", "last"] = "all"
    num_steps: int = 250
    seed: int = 0
    batch_size: int = 512
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    use_constrained_gradient: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    allow_special: bool = False
    filter_ids: bool = True
    verbosity: str = "WARNING"
    token_selection: str = "default"
    max_new_tokens: int = 256
    max_new_target_tokens: int = 64


def get_disallowed_ids(tokenizer, allow_non_ascii, allow_special, device="cpu"):
    disallowed_ids = []

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    # Important to loop over len(tokenizer), not just tokenizer.vocab_size, because
    # special tokens added post-hoc are not counted to vocab_size.
    if not allow_non_ascii:
        for i in range(len(tokenizer)):
            if not is_ascii(tokenizer.decode([i])):
                disallowed_ids.append(i)

    if not allow_special:
        for i in range(len(tokenizer)):
            if not tokenizer.decode([i], skip_special_tokens=True):
                disallowed_ids.append(i)

    if tokenizer.bos_token_id is not None:
        disallowed_ids.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        disallowed_ids.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        disallowed_ids.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        disallowed_ids.append(tokenizer.unk_token_id)
    disallowed_ids = sorted(list(set(disallowed_ids)))
    return torch.tensor(disallowed_ids, device=device)


def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return (
        1.0 / alpha * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
        )
    )


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer, prompt, target):
    """Filters out sequences of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
        tokenizer : ~transformers.PreTrainedTokenizer
        prompt : str
        target : str

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    attacks_decoded = tokenizer.batch_decode(ids)
    filtered_idx = []

    for i, attack in enumerate(attacks_decoded):
        pre_ids, prompt_ids, attack_ids, post_ids, target_ids = prepare_tokens(
            tokenizer,
            prompt,
            target,
            attack=attack,
            placement="suffix",
        )
        if torch.equal(ids[i], attack_ids.to(ids.device)):
            filtered_idx.append(i)

    if not filtered_idx:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
            "An example of the token sequence that failed:"
            f"{ids[-1]}"
            "\n->\n"
            f"{attacks_decoded[-1]}"
            "\n->\n"
            f"{attack_ids}"
        )
    return filtered_idx


class GCGRefusalAttack(Attack):
    def __init__(self, config: GCGRefusalConfig):
        super().__init__(config)
        self.logger = logging.getLogger("nanogcg")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def run(self, model, tokenizer, dataset) -> AttackResult:
        not_allowed_ids = get_disallowed_ids(tokenizer, self.config.allow_non_ascii, self.config.allow_special, device=model.device)
        results = AttackResult([], [], [], [], [])

        fwd_pre_hooks, fwd_hooks = toxify(model, tokenizer, from_cache=True)

        for msg, target in dataset:
            msg: dict[str, str]
            target: str
            t0 = time.time()

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                fluent_target = generate_ragged_batched(
                    model,
                    tokenizer,
                    [torch.cat(prepare_tokens(tokenizer, msg["content"], "", placement="prompt")[:4], dim=0)],
                    max_new_tokens=self.config.max_new_target_tokens,
                )[0]
                print(target, "->", fluent_target)
                target = fluent_target

            pre_ids, prompt_ids, attack_ids, post_ids, target_ids = prepare_tokens(
                tokenizer,
                msg["content"],
                target,
                attack=self.config.optim_str_init,
                placement="suffix",
            )
            pre_ids = pre_ids.unsqueeze(0).to(model.device)
            prompt_ids = prompt_ids.unsqueeze(0).to(model.device)
            pre_prompt_ids = torch.cat([pre_ids, prompt_ids], dim=1)
            attack_ids = attack_ids.unsqueeze(0).to(model.device)
            post_ids = post_ids.unsqueeze(0).to(model.device)
            target_ids = target_ids.unsqueeze(0).to(model.device)

            # Embed everything that doesn't get optimized
            embedding_layer = model.get_input_embeddings()
            pre_prompt_embeds, post_embeds, target_embeds = [
                embedding_layer(ids) for ids in (pre_prompt_ids, post_ids, target_ids)
            ]

            # Compute the KV Cache for tokens that appear before the optimized tokens
            if self.config.use_prefix_cache and model.name_or_path != "google/gemma-2-2b-it":
                with torch.no_grad():
                    output = model(inputs_embeds=pre_prompt_embeds, use_cache=True)
                    self.prefix_cache = output.past_key_values
            else:
                self.prefix_cache = None

            self.target_ids = target_ids
            self.pre_prompt_embeds = pre_prompt_embeds
            self.post_embeds = post_embeds
            self.target_embeds = target_embeds

            # Initialize the attack buffer
            buffer = self.init_buffer(model, attack_ids)
            optim_ids = buffer.get_best_ids()
            token_selection = SubstitutionSelectionStrategy(self.config.token_selection, self.config, self.prefix_cache, self.pre_prompt_embeds, self.post_embeds, self.target_embeds, self.target_ids)

            losses = []
            times = []
            optim_strings = []
            self.stop_flag = False
            current_loss = buffer.get_lowest_loss()
            for _ in (pbar := trange(self.config.num_steps)):
                # Compute the token gradient
                sampled_ids, sampled_ids_pos, grad = token_selection(
                    optim_ids.squeeze(0),
                    model,
                    self.config.search_width,
                    self.config.topk,
                    self.config.n_replace,
                    not_allowed_ids=not_allowed_ids,
                )
                with torch.no_grad():
                    # Sample candidate token sequences
                    if self.config.filter_ids:
                        # We're trying to be as strict as possible here, so we filter
                        # the entire prompt, not just the attack sequence in an isolated
                        # way. This is because the prompt and attack can affect each
                        # other's tokenization in some cases.
                        idx = filter_ids(
                            sampled_ids,
                            tokenizer,
                            msg["content"],
                            target,
                        )

                        sampled_ids = sampled_ids[idx]
                        sampled_ids_pos = sampled_ids_pos[idx]

                    new_search_width = sampled_ids.shape[0]

                    # Compute loss on all candidate sequences
                    batch_size = (
                        new_search_width
                        if self.config.batch_size is None
                        else self.config.batch_size
                    )
                    if self.prefix_cache:
                        input_embeds = torch.cat(
                            [
                                embedding_layer(sampled_ids),
                                post_embeds.repeat(new_search_width, 1, 1),
                                target_embeds.repeat(new_search_width, 1, 1),
                            ],
                            dim=1,
                        )
                    else:
                        input_embeds = torch.cat(
                            [
                                pre_prompt_embeds.repeat(new_search_width, 1, 1),
                                embedding_layer(sampled_ids),
                                post_embeds.repeat(new_search_width, 1, 1),
                                target_embeds.repeat(new_search_width, 1, 1),
                            ],
                            dim=1,
                        )
                    loss = find_executable_batch_size(
                        self.compute_candidates_loss, batch_size
                    )(input_embeds, model)
                    import matplotlib.pyplot as plt
                    import numpy as np

                    # Get the gradient values for the sampled positions and tokens
                    batch_indices = torch.arange(new_search_width, device=grad.device)
                    selected_grads = grad[sampled_ids_pos.squeeze(), sampled_ids[batch_indices, sampled_ids_pos.squeeze()]]

                    # Calculate loss differences relative to current best loss
                    loss_diffs = loss - current_loss

                    # Calculate embedding distances
                    current_token_embeds = embedding_layer(optim_ids.squeeze())[sampled_ids_pos.squeeze()]
                    sampled_token_embeds = embedding_layer(sampled_ids[batch_indices, sampled_ids_pos.squeeze()])
                    embed_dists = torch.norm(sampled_token_embeds - current_token_embeds, dim=1)

                    # Convert to numpy for plotting
                    grads_np = selected_grads.to(torch.float32).cpu().numpy()
                    loss_diffs_np = loss_diffs.to(torch.float32).cpu().numpy()
                    dists_np = embed_dists.to(torch.float32).cpu().numpy()
                    positions_np = sampled_ids_pos.squeeze().cpu().numpy()

                    # Normalize values to [0,1] for HSV color mapping
                    dists_norm = (dists_np - dists_np.min()) / (dists_np.max() - dists_np.min())
                    pos_norm = positions_np / positions_np.max()

                    # Create HSV colors combining distance and position
                    colors = np.zeros((len(dists_norm), 3))
                    colors[:, 0] = pos_norm  # Hue from position
                    colors[:, 1] = 0.8  # Fixed saturation
                    colors[:, 2] = 1 - dists_norm  # Value from embedding distance

                    # Convert HSV to RGB
                    colors = plt.cm.hsv(colors[:, 0])

                    # Create scatter plot
                    plt.figure(figsize=(8, 6))
                    plt.scatter(grads_np, loss_diffs_np, c=colors, alpha=0.5)

                    # Add two colorbars
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    ax = plt.gca()
                    divider = make_axes_locatable(ax)
                    cax1 = divider.append_axes("right", size="5%", pad=0.05)
                    cax2 = divider.append_axes("right", size="5%", pad=0.15)

                    # Create colorbars
                    sm1 = plt.cm.ScalarMappable(cmap='hsv')
                    sm1.set_array([])
                    plt.colorbar(sm1, cax=cax1, label='Sequence Position')

                    sm2 = plt.cm.ScalarMappable(cmap='viridis')
                    sm2.set_array([])
                    plt.colorbar(sm2, cax=cax2, label='Embedding Distance')

                    plt.xlabel('Token Gradient')
                    plt.ylabel('Loss Difference')
                    plt.title('Loss Difference vs Token Gradient\nColored by Position and Distance')

                    # Save plot
                    plt.savefig(f'gcg_loss_grad_step_{_}.png')
                    plt.close()

                    current_loss = loss.min().item()
                    optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                    # Update the buffer based on the loss
                    losses.append(current_loss)
                    times.append(time.time() - t0)
                    if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                        buffer.add(current_loss, optim_ids)

                optim_ids = buffer.get_best_ids()
                optim_str = tokenizer.batch_decode(optim_ids)[0]
                optim_strings.append(optim_str)
                pbar.set_postfix({"Loss": current_loss, "Best Attack": optim_str[:50]})

                if self.stop_flag:
                    self.logger.info("Early stopping due to finding a perfect match.")
                    break

            # Generate completions
            match self.config.generate_completions:
                case "all":
                    attacks = optim_strings
                case "best":
                    attacks = [optim_strings[losses.index(min(losses))]]
                case "last":
                    attacks = [optim_strings[-1]]
                case _:
                    raise ValueError(
                        f"Unknown value for generate_completions: {self.config.generate_completions}"
                    )

            token_list = [
                torch.cat(prepare_tokens(
                    tokenizer,
                    prompt=msg["content"],
                    target="ZZZZZ",  # need dummy target (probably)
                    attack=attack,
                )[:4])
                for attack in attacks
            ]
            completions = generate_ragged_batched(
                model,
                tokenizer,
                token_list=token_list,
                initial_batch_size=self.config.batch_size,
                max_new_tokens=self.config.max_new_tokens
            )
            results.losses.append(losses)
            results.attacks.append(optim_strings)
            results.prompts.append(msg)
            results.completions.append(completions)
            results.times.append(times)
        return results

    def init_buffer(self, model, init_buffer_ids):
        config = self.config

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)
        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat(
                [
                    model.get_input_embeddings()(init_buffer_ids),
                    self.post_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds = torch.cat(
                [
                    self.pre_prompt_embeds.repeat(true_buffer_size, 1, 1),
                    model.get_input_embeddings()(init_buffer_ids),
                    self.post_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )

        init_buffer_losses = find_executable_batch_size(
            self.compute_candidates_loss, true_buffer_size
        )(init_buffer_embeds, model)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
        return buffer

    def compute_candidates_loss(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        model: transformers.PreTrainedModel,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.prefix_cache:
                    if (
                        not prefix_cache_batch
                        or current_batch_size != search_batch_size
                    ):
                        prefix_cache_batch = [
                            [
                                x.expand(current_batch_size, -1, -1, -1)
                                for x in self.prefix_cache[i]
                            ]
                            for i in range(len(self.prefix_cache))
                        ]

                    outputs = model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=prefix_cache_batch,
                        use_cache=True,
                    )
                else:
                    outputs = model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(
                        shift_logits, -1, shift_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    loss = mellowmax(
                        -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    )

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.config.early_stop:
                    if (shift_logits.argmax(-1) == shift_labels).all(-1).any().item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]


class SubstitutionSelectionStrategy:
    def __init__(self, strategy: str, config: GCGRefusalConfig, prefix_cache: list[tuple[Tensor, Tensor]], pre_prompt_embeds: Tensor, post_embeds: Tensor, target_embeds: Tensor, target_ids: Tensor):
        self.config = config
        self.strategy = strategy
        self.prefix_cache = prefix_cache
        self.pre_prompt_embeds = pre_prompt_embeds
        self.post_embeds = post_embeds
        self.target_embeds = target_embeds
        self.target_ids = target_ids

    def __call__(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int,
        n_replace: int,
        not_allowed_ids: Tensor,
        *args,
        **kwargs,
    ):
        if self.strategy == "default":
            return self._sample_ids_from_grad(
                ids,
                model,
                search_width,
                topk,
                n_replace,
                not_allowed_ids,
                *args,
                **kwargs,
            )
        elif self.strategy == "random_overall":
            return self._random_overall(
                ids,
                model,
                search_width,
                topk,
                n_replace,
                not_allowed_ids,
                *args,
                **kwargs,
            )
        elif self.strategy == "random_per_position":
            return self._random_per_position(
                ids,
                model,
                search_width,
                topk,
                n_replace,
                not_allowed_ids,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid replacement selection strategy: {self.strategy}")

    def _sample_ids_from_grad(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor | None = None,
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
        grad = self.compute_token_gradient(ids.unsqueeze(0), model).squeeze(0)  # (n_optim_ids, vocab_size)
        n_optim_tokens = len(ids)
        original_ids = ids.repeat(search_width, 1)
        if not_allowed_ids is not None:
            grad[:, not_allowed_ids.to(grad.device)] = float("inf")
        # (n_optim_ids, topk)
        topk_ids = grad.topk(topk, dim=1, largest=False, sorted=False).indices

        sampled_ids_pos = torch.randint(
            0, n_optim_tokens, (search_width, n_replace), device=grad.device
        )  # (search_width, n_replace)
        sampled_topk_idx = torch.randint(
            0, topk, (search_width, n_replace, 1), device=grad.device
        )

        sampled_ids_val = (
            topk_ids[sampled_ids_pos].gather(2, sampled_topk_idx).squeeze(2)
        )  # (search_width, n_replace)

        new_ids = original_ids.scatter_(
            1, sampled_ids_pos, sampled_ids_val
        )  # (search_width, n_optim_ids)

        return new_ids, sampled_ids_pos, grad

    @torch.no_grad()
    def _random_overall(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor | None = None,
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
            not_allowed_ids: Tensor, shape = (n_ids,)
                the token ids that should not be used in optimization

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
        """
        vocab_size = model.get_input_embeddings().weight.size(0)
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

    @torch.no_grad()
    def _random_per_position(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor | None = None,
    ):
        """Returns `search_width` random token substitutions.

        Args:
            ids : Tensor, shape = (n_optim_ids,)
                the sequence of token ids that are being optimized
            model : transformers.PreTrainedModel
                the model to compute the gradient with respect to
            search_width : int
                the number of candidate sequences to return
            topk : int
                the topk to be used when sampling from the gradient
            n_replace: int
                the number of token positions to update per sequence
            not_allowed_ids: Tensor, shape = (n_ids,)
                the token ids that should not be used in optimization

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
        """
        # Sample search_width//ids.shape[0] substitutions at each position
        samples_per_position = search_width // ids.shape[0]
        positions = torch.arange(ids.shape[0], device=ids.device)
        original_ids = ids.repeat(search_width, 1)

        # Get valid ids for each position (all except not_allowed_ids)
        valid_ids = torch.ones((ids.shape[0], model.get_input_embeddings().weight.size(0)), dtype=torch.bool, device=ids.device)
        if not_allowed_ids is not None:
            valid_ids[:, not_allowed_ids.to(ids.device)] = False

        # Sample indices for each position in parallel
        sampled_ids = torch.empty((ids.shape[0], samples_per_position), dtype=torch.long, device=ids.device)
        rand_perm = torch.argsort(torch.rand_like(valid_ids.float()), dim=1)
        valid_perm = torch.masked_select(rand_perm, valid_ids).reshape(ids.shape[0], -1)
        sampled_ids = valid_perm[:, :samples_per_position]

        # Reshape to (total_samples, 1) format
        sampled_topk_idx = sampled_ids.reshape(-1)
        sampled_ids_pos = positions.repeat_interleave(samples_per_position)
        original_ids = original_ids[:samples_per_position * ids.shape[0]]
        new_ids = original_ids.scatter_(1, sampled_ids_pos.unsqueeze(1), sampled_topk_idx.unsqueeze(1))
        return new_ids, sampled_ids_pos

    def _lowest_gradient_magnitude(
        self,
        ids: Tensor,
        model: transformers.PreTrainedModel,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor | None = None,
    ):
        """Returns `search_width` combinations of token ids with the lowest token gradient.

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
        grad = self.compute_token_gradient(ids.unsqueeze(0), model).squeeze(0)  # (n_optim_ids, vocab_size)
        n_optim_ids = len(ids)
        original_ids = ids.repeat(search_width, 1)

        if not_allowed_ids is not None:
            grad[:, not_allowed_ids.to(grad.device)] = float("inf")

        # We have 32768 * 20 = 655360 substitutions to evaluate
        # Here we crop this down with the smallest gradient heuristic to topk * 20
        topk_ids = (
            grad.abs()
            .view(-1)
            .topk(topk * n_optim_ids, largest=False, sorted=False)
            .indices
        )  # (n_optim_ids, topk)
        topk_ids = torch.randperm(grad.view(-1), device=topk_ids.device)[
            : topk * n_optim_ids
        ]  # (n_optim_ids, topk)

        # We then crop again randomly to search_width candidates
        topk_ids = topk_ids[
            torch.randperm(topk_ids.size(0), device=topk_ids.device)[:search_width]
        ].unsqueeze(
            1
        )  # (search_width, 1)

        sampled_ids_pos = topk_ids // grad.size(1)
        sampled_topk_idx = topk_ids % grad.size(1)

        new_ids = original_ids.scatter_(
            1, sampled_ids_pos, sampled_topk_idx
        )  # (search_width, n_optim_ids)

        return new_ids, sampled_ids_pos

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
        model: transformers.PreTrainedModel,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
        optim_ids : Tensor, shape = (N, n_optim_ids)
            the sequence of token ids that are being optimized
        model : transformers.PreTrainedModel
            the model to compute the gradient with respect to

        Returns:
            grad : Tensor, shape = (N, n_optim_ids, vocab_size)
                the gradient of the GCG loss computed with respect to the one-hot token embeddings
        """
        embedding_layer = model.get_input_embeddings()

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(dtype=model.dtype, device=model.device)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        if self.config.use_constrained_gradient:
            optim_embeds = (
                optim_ids_onehot / optim_ids_onehot.sum(dim=-1, keepdim=True)
            ) @ embedding_layer.weight
        else:
            optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat(
                [optim_embeds, self.post_embeds, self.target_embeds], dim=1
            )
            output = model(
                inputs_embeds=input_embeds, past_key_values=self.prefix_cache, use_cache=True
            )
        else:
            input_embeds = torch.cat(
                [
                    self.pre_prompt_embeds,
                    optim_embeds,
                    self.post_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )
            output = model(inputs_embeds=input_embeds)
        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[
            ..., shift - 1 : -1, :
        ].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(
                shift_logits, -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[loss],
            inputs=[optim_ids_onehot],
            create_graph=False,
            retain_graph=False
        )[0]
        return optim_ids_onehot_grad


from src.dataset import RefusalDirectionDataDataset, RefusalDirectionDataConfig
from tqdm import tqdm
from functools import partial
import contextlib
from typing import List, Tuple, Callable
from einops import rearrange
import math
from torch.nn.utils.rnn import pad_sequence


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_mean_activations_pre_hook(layer, cache: Tensor, n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Tensor = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn


def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Tensor = input[0]
        else:
            activation: Tensor = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Tensor = output[0]
        else:
            activation: Tensor = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_activation_addition_input_pre_hook(vector: Tensor, coeff: Tensor):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Tensor = input[0]
        else:
            activation: Tensor = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def refusal_score(
    logits: Tensor,
    refusal_toks: Tensor,
    epsilon: float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

def get_refusal_scores(model, tokenizer, instructions, refusal_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    refusal_score_fn = partial(refusal_score, refusal_toks=refusal_toks)

    refusal_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        inputs = [torch.cat(prepare_tokens(tokenizer, hd, target="", placement="prompt")[:4], dim=0) for hd in instructions[i:i+batch_size]]

        # Left-pad the tokens to ensure the post-tokens are aligned at the right position
        inputs = pad_sequence([inp.flip(0) for inp in inputs], batch_first=True, padding_value=tokenizer.pad_token_id).flip(1).to(model.device)
        attention_mask = inputs != tokenizer.pad_token_id

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=inputs.to(model.device),
                attention_mask=attention_mask.to(model.device),
            ).logits

        refusal_scores[i:i+batch_size] = refusal_score_fn(logits=logits)

    return refusal_scores

def kl_div_fn(
    logits_a: Tensor,
    logits_b: Tensor,
    mask: Tensor | None = None,
    epsilon: float = 1e-6
) -> Tensor:
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)

def masked_mean(seq, mask=None, dim=1, keepdim=False):
    if mask is None:
        return seq.mean(dim=dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)

    masked_mean = numer / denom.clamp(min=1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean


# returns True if the direction should be filtered out
def filter_fn(refusal_score, steering_score, kl_div_score, layer, n_layer, kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False


def get_blocks(model):
    if hasattr(model, "model"):
        model = model.model
    # we search through all attributes of the model until we find a nn.modulelist
    for attr in dir(model):
        if isinstance(getattr(model, attr), torch.nn.ModuleList):
            return getattr(model, attr)
    raise ValueError("No nn.ModuleList found in model")


def get_attn_modules(model):
    blocks = get_blocks(model)
    attn = []
    for block in blocks:
        if hasattr(block, "attn"):
            attn.append(block.attn)
        elif hasattr(block, "self_attn"):
            attn.append(block.self_attn)
    if not attn:
        raise ValueError("No attention modules found in model")
    return attn

def get_mlp_modules(model):
    blocks = get_blocks(model)
    return [blocks[i].mlp for i in range(len(blocks))]


def get_all_direction_ablation_hooks(
    model,
    direction: Tensor,
):
    blocks = get_blocks(model)
    attn_modules = get_attn_modules(model)
    mlp_modules = get_mlp_modules(model)
    fwd_pre_hooks = [(block, get_direction_ablation_input_pre_hook(direction=direction)) for block in blocks]
    fwd_hooks = [(attn, get_direction_ablation_output_hook(direction=direction)) for attn in attn_modules]
    fwd_hooks += [(mlp, get_direction_ablation_output_hook(direction=direction)) for mlp in mlp_modules]

    return fwd_pre_hooks, fwd_hooks


def select_direction(
    model,
    tokenizer,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Tensor,
    kl_threshold=0.1,  # directions larger KL score are filtered out
    induce_refusal_threshold=0.0,  # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage=0.2,  # discard the directions extracted from the last 20% of the model
    batch_size=32,
):

    n_pos, n_layer, d_model = candidate_directions.shape

    refusal_toks = tokenizer.encode("I", add_special_tokens=False)   # generally `I` is sufficient as refusal token according to refusal direction paper (https://arxiv.org/abs/2406.11717)

    baseline_refusal_scores_harmful = get_refusal_scores(model, tokenizer, harmful_instructions, refusal_toks, fwd_hooks=[], batch_size=batch_size)
    baseline_refusal_scores_harmless = get_refusal_scores(model, tokenizer, harmless_instructions, refusal_toks, fwd_hooks=[], batch_size=batch_size)

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), device=model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), device=model.device, dtype=torch.float64)
    baseline_harmless_logits = torch.zeros((len(harmless_instructions), model.config.vocab_size), device=model.device, dtype=torch.float32)
    # we evaluate the activations around the post-instruction tokens
    for i in tqdm(range(0, len(harmless_instructions), batch_size)):
        inputs = [torch.cat(prepare_tokens(tokenizer, hd, target="", placement="prompt")[:4], dim=0) for hd in harmless_instructions[i:i+batch_size]]
        # Left-pad the tokens to ensure the post-tokens are aligned at the right position
        inputs = pad_sequence([inp.flip(0) for inp in inputs], batch_first=True, padding_value=tokenizer.pad_token_id).flip(1).to(model.device)
        attention_mask = inputs != tokenizer.pad_token_id
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=[]):
            logits = model(input_ids=inputs, attention_mask=attention_mask).logits[:, -1]
            baseline_harmless_logits[i:i+batch_size] = logits

    blocks = get_blocks(model)
    attn_modules = get_attn_modules(model)
    mlp_modules = get_mlp_modules(model)
    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):
            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(block, get_direction_ablation_input_pre_hook(direction=ablation_dir)) for block in blocks]
            fwd_hooks = [(attn, get_direction_ablation_output_hook(direction=ablation_dir)) for attn in attn_modules]
            fwd_hooks += [(mlp, get_direction_ablation_output_hook(direction=ablation_dir)) for mlp in mlp_modules]

            intervention_logits = torch.zeros((len(harmless_instructions), model.config.vocab_size), device=model.device, dtype=torch.float32)
            # we evaluate the activations around the post-instruction tokens
            for i in tqdm(range(0, len(harmless_instructions), batch_size)):
                inputs = [torch.cat(prepare_tokens(tokenizer, hd, target="", placement="prompt")[:4], dim=0) for hd in harmless_instructions[i:i+batch_size]]
                # Left-pad the tokens to ensure the post-tokens are aligned at the right position
                inputs = pad_sequence([inp.flip(0) for inp in inputs], batch_first=True, padding_value=tokenizer.pad_token_id).flip(1).to(model.device)
                attention_mask = inputs != tokenizer.pad_token_id
                with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                    logits = model(input_ids=inputs, attention_mask=attention_mask).logits[:, -1]
                    intervention_logits[i:i+batch_size] = logits
            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):
            ablation_dir = candidate_directions[source_pos, source_layer]
            fwd_pre_hooks = [(block, get_direction_ablation_input_pre_hook(direction=ablation_dir)) for block in blocks]
            fwd_hooks = [(attn, get_direction_ablation_output_hook(direction=ablation_dir)) for attn in attn_modules]
            fwd_hooks += [(mlp, get_direction_ablation_output_hook(direction=ablation_dir)) for mlp in mlp_modules]
            refusal_scores = get_refusal_scores(model, tokenizer, harmful_instructions, refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal addition for source position {source_pos}"):
            refusal_vector = candidate_directions[source_pos, source_layer]
            coeff = torch.tensor(1.0)

            fwd_pre_hooks = [(blocks[source_layer], get_activation_addition_input_pre_hook(vector=refusal_vector, coeff=coeff))]
            fwd_hooks = []

            refusal_scores = get_refusal_scores(model, tokenizer, harmless_instructions, refusal_toks, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            # we sort the directions in descending order (from highest to lowest score)
            # the intervention is better at bypassing refusal if the refusal score is low, so we multiply by -1
            sorting_score = -refusal_score

            # we filter out directions if the KL threshold
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")

    return pos, layer, candidate_directions[pos, layer]

def filter_data(model, tokenizer, harmful, harmless):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        to_keep = [i for i, score in enumerate(scores.tolist()) if comparison(score, threshold)]
        print(f"Filtering {len(dataset)} examples to {len(to_keep)}, average score: {scores.mean().item()}")
        dataset.messages = [dataset.messages[i] for i in range(len(dataset)) if i in to_keep]
    refusal_toks = tokenizer.encode("I", add_special_tokens=False)   # generally `I` is sufficient as refusal token according to refusal direction paper (https://arxiv.org/abs/2406.11717)

    harmful_scores = get_refusal_scores(model, tokenizer, harmful, refusal_toks)
    harmless_scores = get_refusal_scores(model, tokenizer, harmless, refusal_toks)
    filter_examples(harmful, harmful_scores, 0, lambda x, y: x > y)
    filter_examples(harmless, harmless_scores, 0, lambda x, y: x < y)


@torch.no_grad()
def toxify(model, tokenizer, batch_size=16, from_cache=True):
    cache_root = "/ceph/ssd/staff/beyer/llm-quick-check/cache"
    if from_cache and os.path.exists(f"{cache_root}/{model.name_or_path}/ablation_dir.pt"):
        ablation_dir = torch.load(f"{cache_root}/{model.name_or_path}/ablation_dir.pt")
        blocks = get_blocks(model)
        attn_modules = get_attn_modules(model)
        mlp_modules = get_mlp_modules(model)
        fwd_pre_hooks = [(block, get_direction_ablation_input_pre_hook(direction=ablation_dir["direction"])) for block in blocks]
        fwd_hooks = [(attn, get_direction_ablation_output_hook(direction=ablation_dir["direction"])) for attn in attn_modules]
        fwd_hooks += [(mlp, get_direction_ablation_output_hook(direction=ablation_dir["direction"])) for mlp in mlp_modules]
        return fwd_pre_hooks, fwd_hooks
    data_root = "/ceph/ssd/staff/beyer/llm-quick-check/data"
    harmless_cfg = RefusalDirectionDataConfig(
        name="refusal_direction_data",
        path=f"{data_root}/refusal_direction",
        split="train",
        type="harmless",
        n_samples=256,
    )
    harmless_data = RefusalDirectionDataDataset(harmless_cfg)
    harmful_cfg = RefusalDirectionDataConfig(
        name="refusal_direction_data",
        path=f"{data_root}/refusal_direction",
        split="train",
        type="harmful",
        n_samples=256,
    )
    harmful_data = RefusalDirectionDataDataset(harmful_cfg)
    filter_data(model, tokenizer, harmful_data, harmless_data)

    block_modules = model.model.layers
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    # we evaluate the activations around the post-instruction tokens
    post_tokens = prepare_tokens(tokenizer, "", target="", placement="prompt")[3]
    positions = (-torch.arange(len(post_tokens)) - 1).flip(0)

    # Get difference of means
    mean_activations = {
        "harmless": torch.zeros((len(positions), n_layers, d_model), dtype=torch.float32, device=model.device),
        "harmful": torch.zeros((len(positions), n_layers, d_model), dtype=torch.float32, device=model.device),
    }
    for dataset in [harmless_data, harmful_data]:
        n_samples = len(dataset)
        # we store the mean activations in high-precision to avoid numerical issues
        fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations[dataset.config.type], n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

        for i in tqdm(range(0, len(dataset), batch_size)):
            inputs = [torch.cat(prepare_tokens(tokenizer, hd, target="", placement="prompt")[:4], dim=0) for hd in dataset[i:i+batch_size]]
            # Left-pad the tokens to ensure the post-tokens are aligned at the right position
            inputs = pad_sequence([inp.flip(0) for inp in inputs], batch_first=True, padding_value=tokenizer.pad_token_id).flip(1).to(model.device)
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(input_ids=inputs, attention_mask=inputs != tokenizer.pad_token_id)
            torch.cuda.empty_cache()
    mean_diffs = mean_activations["harmful"] - mean_activations["harmless"]
    # 2. Select the most effective refusal direction
    harmful_val_cfg = RefusalDirectionDataConfig(
        name="refusal_direction_data",
        path=f"{data_root}/refusal_direction",
        split="val",
        type="harmful",
        n_samples=128,
    )
    harmful_val = RefusalDirectionDataDataset(harmful_val_cfg)
    harmless_val_cfg = RefusalDirectionDataConfig(
        name="refusal_direction_data",
        path=f"{data_root}/refusal_direction",
        split="val",
        type="harmless",
        n_samples=128,
    )
    harmless_val = RefusalDirectionDataDataset(harmless_val_cfg)
    filter_data(model, tokenizer, harmful_val, harmless_val)
    pos, layer, ablation_dir = select_direction(model, tokenizer, harmful_val, harmless_val, mean_diffs)
    save_path = f"{cache_root}/{model.name_or_path}/ablation_dir.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"position": pos, "layer": layer, "direction": ablation_dir}, save_path)

    blocks = get_blocks(model)
    attn_modules = get_attn_modules(model)
    mlp_modules = get_mlp_modules(model)
    # fmt: off
    fwd_pre_hooks = [(block, get_direction_ablation_input_pre_hook(direction=ablation_dir)) for block in blocks]
    fwd_hooks = [(attn, get_direction_ablation_output_hook(direction=ablation_dir)) for attn in attn_modules]
    fwd_hooks += [(mlp, get_direction_ablation_output_hook(direction=ablation_dir)) for mlp in mlp_modules]
    # fmt: on
    return fwd_pre_hooks, fwd_hooks
