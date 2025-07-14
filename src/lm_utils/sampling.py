"""
Sampling utilities for language model generation.

This module provides sampling functions including top-p and top-k filtering
for controlling generation behavior.
"""

import torch
import torch.nn.functional as F


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling.

    Parameters
    ----------
    logits: torch.Tensor, shape (B, T, V) or (B, V)
        The logits to filter.
    top_p: float
        The top-p threshold.

    Returns
    -------
    torch.Tensor
    """
    single_token_only = logits.ndim == 2
    if single_token_only:
        logits = logits.unsqueeze(1)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    if single_token_only:
        logits = logits.squeeze(1)
    return logits


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits using top-k sampling.

    Parameters
    ----------
    logits: torch.Tensor, shape (B, T, V) or (B, V)
        The logits to filter.
    top_k: int
        The top-k threshold.

    Returns
    -------
    torch.Tensor
        Filtered logits with values below top-k threshold set to -inf.
    """
    single_token_only = logits.ndim == 2
    if single_token_only:
        logits = logits.unsqueeze(1)

    values, _ = torch.topk(logits, top_k)
    # Get minimum value of top-k tokens
    min_values = values[..., -1, None]
    # Zero out everything below min values
    logits[logits < min_values] = float('-inf')

    if single_token_only:
        logits = logits.squeeze(1)
    return logits