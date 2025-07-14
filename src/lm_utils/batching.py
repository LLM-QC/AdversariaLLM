"""
Batch processing utilities with automatic memory management.

This module provides utilities for processing data in batches with automatic
batch size adjustment when out-of-memory errors occur.
"""

import sys
import torch
from typing import Callable
from tqdm import tqdm

from ..io_utils import free_vram


def with_max_batchsize(function: Callable, *inputs, initial_batch_size: int | None = None, verbose: bool = False):
    """
    Dynamically adjust batch size if an OOM error occurs.
    TODO: Try increasing batch size again if we have enough VRAM.

    Args:
        function: Callable
            A function that takes one or more arguments and returns a tensor or list of tensors.
            All input arguments should have the same length in their first dimension (batch dimension).
        *inputs:
            The inputs to pass to the function. All inputs must be tensors or lists and have the same length in their first dimension.
        initial_batch_size:
            Starting batch size for execution.
        verbose:
            Whether to print progress.
    Returns:
        The output of the function.
    """
    if not inputs:
        raise ValueError("At least one input must be provided")

    # Verify all inputs have the same length
    input_length = len(inputs[0])
    for i, inp in enumerate(inputs[1:], 1):
        if len(inp) != input_length:
            raise ValueError(f"All inputs must have the same length. Input 0 has length {input_length}, but input {i} has length {len(inp)}")

    outputs = []
    i = 0

    def next_power_of_two(n):
        return 1 << (n - 1).bit_length()

    if initial_batch_size is None:
        initial_batch_size = next_power_of_two(input_length)

    batch_size = min(initial_batch_size, next_power_of_two(input_length))
    pbar = tqdm(total=input_length, desc=f"Running function b={batch_size}", file=sys.stdout) if verbose else None

    while i < input_length:
        try:
            free_vram()
            # Create chunks for all inputs
            chunks = [inp[i:i + batch_size] for inp in inputs]
            output = function(*chunks)
            outputs.append(output)
            i += batch_size  # Move to the next batch
            if verbose:
                pbar.update(batch_size)
        except torch.cuda.OutOfMemoryError:
            # If we hit OOM, reduce batch size and retry the same chunk
            batch_size = batch_size // 2
            if verbose:
                pbar.set_description(f"Running function b={batch_size}")
            if batch_size < 1:
                raise RuntimeError(
                    "OOM even with batch_size=1; cannot generate further."
                )
    if verbose:
        pbar.close()

    if all(isinstance(x, torch.Tensor) for x in outputs):
        outputs = torch.cat(outputs, dim=0)
        assert len(outputs) == input_length
    elif all(isinstance(x, tuple) for x in outputs):
        # Transpose and concatenate tuple outputs
        # Handle both tensors and lists within tuples
        outputs_processed = []
        for i in range(len(outputs[0])):
            elements = [x[i] for x in outputs]
            if all(isinstance(e, torch.Tensor) for e in elements):
                outputs_processed.append(torch.cat(elements, dim=0))
            elif all(isinstance(e, list) for e in elements):
                outputs_processed.append([item for sublist in elements for item in sublist])
            else:
                types = ", ".join(f"{type(e).__name__}" for e in elements)
                raise TypeError(f"Wrapped functions may only return Tensors or lists, not {types}")
        outputs = tuple(outputs_processed)
        assert all(len(o) == input_length for o in outputs)
    elif all(isinstance(x, dict) for x in outputs):
        outputs = {k: [item for o in outputs for item in o[k]] for k in outputs[0].keys()}
        assert all(len(v) == input_length for v in outputs.values())
    else:
        outputs = [item for sublist in outputs for item in sublist]
        assert len(outputs) == input_length
    return outputs
