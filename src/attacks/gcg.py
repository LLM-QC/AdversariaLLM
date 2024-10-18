import gc
import logging
from dataclasses import dataclass
from typing import Literal

import torch
import transformers
from accelerate.utils import find_executable_batch_size
from torch import Tensor
from tqdm import tqdm

from src.lm_utils import get_batched_completions

from .attack import Attack, AttackResult


@dataclass
class GCGConfig:
    name: str = "gcg"
    type: str = "discrete"
    placement: Literal["suffix"] = "suffix"
    generate_completions: Literal["all", "best", "last"] = "last"
    num_steps: int = 250
    seed: int = 0
    batch_size: int | None = None
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
    filter_ids: bool = True
    add_space_before_target: bool = False
    verbosity: str = "WARNING"



def get_nonascii_toks(tokenizer, device="cpu"):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(nonascii_toks, device=device)


def mellowmax(t: Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
        )
    )


def filter_ids(ids: Tensor, pos: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
        tokenizer : ~transformers.PreTrainedTokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []
    filtered_pos = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(
            ids_decoded[i], return_tensors="pt", add_special_tokens=False
        ).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])
            filtered_pos.append(pos[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids), torch.stack(filtered_pos)


class GCGAttack(Attack):
    def __init__(self, config: GCGConfig):
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
        not_allowed_ids = (
            None
            if self.config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )

        results = AttackResult([], [], [], [])
        for message, target in dataset:
            message: dict[str, str]
            target: str
            prompt = message["content"]
            message["content"] = message["content"] + self.config.optim_str_init

            template = tokenizer.apply_chat_template(
                [message], tokenize=False, add_generation_prompt=True
            )
            # Remove the BOS token -- this will get added when tokenizing, if necessary
            if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                template = template.replace(tokenizer.bos_token, "")
            before_str, after_str = template.split(self.config.optim_str_init)

            # Tokenize everything that doesn't get optimized
            before_ids = (
                tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"]
                .to(model.device)
                .to(torch.int64)
            )
            after_ids = (
                tokenizer([after_str], add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ]
                .to(model.device)
                .to(torch.int64)
            )
            target_ids = (
                tokenizer([target], add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ]
                .to(model.device)
                .to(torch.int64)
            )

            # Embed everything that doesn't get optimized
            embedding_layer = model.get_input_embeddings()
            before_embeds, after_embeds, target_embeds = [
                embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
            ]

            # Compute the KV Cache for tokens that appear before the optimized tokens
            if self.config.use_prefix_cache:
                with torch.no_grad():
                    output = model(inputs_embeds=before_embeds, use_cache=True)
                    self.prefix_cache = output.past_key_values

            self.target_ids = target_ids
            self.before_embeds = before_embeds
            self.after_embeds = after_embeds
            self.target_embeds = target_embeds

            # Initialize the attack buffer
            buffer = self.init_buffer(model, tokenizer)
            optim_ids = buffer.get_best_ids()
            token_selection = SubstitutionSelectionStrategy("gcg")

            losses = []
            optim_strings = []
            self.stop_flag = False

            for _ in tqdm(range(self.config.num_steps)):
                # Compute the token gradient
                optim_ids_onehot_grad = self.compute_token_gradient(optim_ids, model)

                with torch.no_grad():
                    # Sample candidate token sequences
                    sampled_ids, sampled_ids_pos = token_selection(
                        optim_ids.squeeze(0),
                        optim_ids_onehot_grad.squeeze(0),
                        self.config.search_width,
                        self.config.topk,
                        self.config.n_replace,
                        not_allowed_ids=not_allowed_ids,
                    )
                    if self.config.filter_ids:
                        sampled_ids, sampled_ids_pos = filter_ids(
                            sampled_ids, sampled_ids_pos, tokenizer
                        )

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
                                after_embeds.repeat(new_search_width, 1, 1),
                                target_embeds.repeat(new_search_width, 1, 1),
                            ],
                            dim=1,
                        )
                    else:
                        input_embeds = torch.cat(
                            [
                                before_embeds.repeat(new_search_width, 1, 1),
                                embedding_layer(sampled_ids),
                                after_embeds.repeat(new_search_width, 1, 1),
                                target_embeds.repeat(new_search_width, 1, 1),
                            ],
                            dim=1,
                        )
                    loss = find_executable_batch_size(
                        self.compute_candidates_loss, batch_size
                    )(input_embeds, model)

                    current_loss = loss.min().item()
                    optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                    # Update the buffer based on the loss
                    losses.append(current_loss)
                    if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                        buffer.add(current_loss, optim_ids)

                optim_ids = buffer.get_best_ids()
                optim_str = tokenizer.batch_decode(optim_ids)[0]
                optim_strings.append(optim_str)

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
            all_modified_message = []

            for string in attacks:
                msg = {k: v for k, v in message.items()}
                msg["content"] = msg["content"] + " " + string
                all_modified_message.append([msg])

            # Apply chat templates in batch
            templates = tokenizer.apply_chat_template(
                all_modified_message, tokenize=False, add_generation_prompt=True
            )

            # Remove BOS token in batch
            for j in range(len(templates)):
                if tokenizer.bos_token and templates[j][0] == tokenizer.bos_token:
                    templates[j] = templates[j][1:]

            token_list = [
                tokenizer(t, return_tensors="pt", add_special_tokens=True)
                .input_ids[0]
                .to(model.device)
                for t in templates
            ]
            completions = get_batched_completions(
                model,
                tokenizer,
                token_list=token_list,
                max_new_tokens=getattr(self.config, "generation_steps", 512),
                return_tokens=False,
            )
            results.losses.append(losses)
            results.attacks.append(optim_strings)
            results.prompts.append(prompt)
            results.completions.append(completions)
        return results

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
        model: transformers.PreTrainedModel,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
        optim_ids : Tensor, shape = (1, n_optim_ids)
            the sequence of token ids that are being optimized
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
            # optim_embeds = (optim_ids_onehot / optim_ids_onehot.norm(dim=-1, keepdim=True)) @ embedding_layer.weight
        else:
            optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat(
                [optim_embeds, self.after_embeds, self.target_embeds], dim=1
            )
            output = model(
                inputs_embeds=input_embeds, past_key_values=self.prefix_cache
            )
        else:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
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
            outputs=[loss], inputs=[optim_ids_onehot]
        )[0]
        return optim_ids_onehot_grad

    def init_buffer(self, model, tokenizer):
        config = self.config

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        init_buffer_ids = tokenizer(
            config.optim_str_init, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)
        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat(
                [
                    model.get_input_embeddings()(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
                    self.target_embeds.repeat(true_buffer_size, 1, 1),
                ],
                dim=1,
            )
        else:
            init_buffer_embeds = torch.cat(
                [
                    self.before_embeds.repeat(true_buffer_size, 1, 1),
                    model.get_input_embeddings()(init_buffer_ids),
                    self.after_embeds.repeat(true_buffer_size, 1, 1),
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
                    if torch.any(
                        torch.all(
                            torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1
                        )
                    ).item():
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
            return

        self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]


class SubstitutionSelectionStrategy:
    def __init__(self, strategy: str):
        self.strategy = strategy

    def __call__(
        self,
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        topk: int,
        n_replace: int,
        not_allowed_ids: Tensor,
        *args,
        **kwargs,
    ):
        if self.strategy == "gcg":
            return self._sample_ids_from_grad(
                ids,
                grad,
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
                grad,
                search_width,
                topk,
                n_replace,
                not_allowed_ids,
                *args,
                **kwargs,
            )
        elif self.strategy == "single_position":
            return self._single_position(
                ids,
                grad,
                search_width,
                topk,
                n_replace,
                not_allowed_ids,
                *args,
                **kwargs,
            )
        elif self.strategy == "cosine_constraint":
            return self._cosine_constraint(
                ids,
                grad,
                search_width,
                topk,
                n_replace,
                not_allowed_ids,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid replacement selection strategy: {self.strategy}")

    @staticmethod
    def _sample_ids_from_grad(
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor = False,
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
        n_optim_tokens = len(ids)
        original_ids = ids.repeat(search_width, 1)

        if not_allowed_ids is not None:
            grad[:, not_allowed_ids.to(grad.device)] = float("inf")
        topk_ids = grad.topk(
            topk, dim=1, largest=False, sorted=False
        ).indices  # (n_optim_ids, topk)

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

        return new_ids, sampled_ids_pos

    @staticmethod
    def _random_overall(
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor = False,
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
            not_allowed_ids: Tensor, shape = (n_ids)
                the token ids that should not be used in optimization

        Returns:
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                sampled token ids
        """
        original_ids = ids.repeat(search_width, 1)

        if not_allowed_ids is not None:
            grad[:, not_allowed_ids.to(grad.device)] = float("inf")
        # We have 32768 * 20 = 655360 substitutions to evaluate
        # Here we crop this down with the smallest gradient heuristic to topk * 20
        sampled_ids = torch.randperm(grad.view(-1).numel(), device=grad.device)
        sampled_ids = sampled_ids[~grad.view(-1)[sampled_ids].isinf()][
            :search_width
        ].unsqueeze(
            1
        )  # (search_width, 1)

        sampled_ids_pos = sampled_ids // grad.size(1)
        sampled_topk_idx = sampled_ids % grad.size(1)

        new_ids = original_ids.scatter_(
            1, sampled_ids_pos, sampled_topk_idx
        )  # (search_width, n_optim_ids)
        return new_ids, sampled_ids_pos

    @staticmethod
    def _lowest_gradient_magnitude(
        ids: Tensor,
        grad: Tensor,
        search_width: int,
        topk: int = 256,
        n_replace: int = 1,
        not_allowed_ids: Tensor = False,
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
