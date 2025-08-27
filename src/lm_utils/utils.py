"""
General utility functions for language model operations.

This module provides utility functions for token handling, model introspection,
and other general-purpose operations.
"""

from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_disallowed_ids(tokenizer: PreTrainedTokenizerBase, allow_non_ascii: bool, allow_special: bool) -> torch.Tensor:
    disallowed_ids = set()

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    # Important to loop over len(tokenizer), not just tokenizer.vocab_size, because
    # special tokens added post-hoc are not counted to vocab_size.
    if not allow_non_ascii:
        for i in range(len(tokenizer)):
            if not is_ascii(tokenizer.decode([i])):
                disallowed_ids.add(i)

    if not allow_special:
        for i in range(len(tokenizer)):
            if not tokenizer.decode([i], skip_special_tokens=True):
                disallowed_ids.add(i)

        is_gemma = "gemma" in tokenizer.name_or_path.lower()
        if is_gemma:
            disallowed_ids.add(tokenizer.convert_tokens_to_ids("[@BOS@]"))
            for i in range(10000):  # gemma-3 has ~8k unused tokens
                unused_id = tokenizer.convert_tokens_to_ids(f"<unused{i}>")
                if unused_id == tokenizer.unk_token_id:
                    break
                disallowed_ids.add(unused_id)

    if tokenizer.bos_token_id is not None:
        disallowed_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        disallowed_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        disallowed_ids.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        disallowed_ids.add(tokenizer.unk_token_id)

    disallowed_ids = sorted(list(disallowed_ids))
    return torch.tensor(disallowed_ids)


def get_stop_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    stop_ids = []
    eos_model = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_model is not None:
        if isinstance(eos_model, list):
            stop_ids.extend(eos_model)
        else:
            stop_ids.append(eos_model)

    stop_ids.append(tokenizer.eos_token_id)
    if hasattr(tokenizer, "eot_token_id"):
        stop_ids.append(tokenizer.eot_token_id)

    return torch.tensor(list(set(stop_ids)), dtype=torch.long)


def get_flops(model: PreTrainedModel, n_tokens_in: int, n_tokens_out: int, type: Literal["forward", "backward", "forward_and_backward"]) -> int:
    """Estimate FLOPS for a model using Kaplan et al. (2020).
    Basic formula is 2 * n_params * n_tokens, and twice that for backwards passes

    Parameters
    ----------
    model : PreTrainedModel
        The model to estimate FLOPS for.
    n_tokens_in : int
        The number of tokens to estimate FLOPS for.
    n_tokens_out : int
        The number of tokens to estimate FLOPS for.
    type : Literal["forward", "backward", "forward_and_backward"]
        Operations to include in the FLOPS calculation.

    Returns
    -------
    int
        The estimated FLOPS.
    """
    n_tokens = n_tokens_in + n_tokens_out
    if type == "forward":
        return model.num_parameters(exclude_embeddings=True) * n_tokens * 2
    elif type == "backward":
        return model.num_parameters(exclude_embeddings=True) * n_tokens * 4
    elif type == "forward_and_backward":
        return model.num_parameters(exclude_embeddings=True) * n_tokens * 6
    else:
        raise ValueError(f"Invalid type: {type}")