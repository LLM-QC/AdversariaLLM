"""
Data analysis and result collection utilities.

This module provides functions for analyzing attack results, collecting
metrics, and processing experimental data.
"""

import os
from collections import defaultdict
import safetensors.torch

from .json_utils import cached_json_load
from .model_loading import num_model_params


def normalize_value_for_grouping(value):
    """
    Normalize a value for consistent grouping.

    Converts numeric values to a canonical form to ensure that 0 and 0.0,
    or 1 and 1.0, etc. are treated as identical for grouping purposes.
    For dictionaries and lists, recursively normalizes all contained values.
    """
    if isinstance(value, dict):
        return {k: normalize_value_for_grouping(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        normalized_list = [normalize_value_for_grouping(item) for item in value]
        return type(value)(normalized_list)  # preserve the original type (list or tuple)
    elif isinstance(value, (int, float)):
        # Convert to int if it's a whole number, otherwise keep as float
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    return value


def get_nested_value(data: dict, path: list[str], default="unknown"):
    """
    Safely retrieves a value from a nested dictionary using a path list/tuple.

    Args:
        data (dict): The dictionary to search within.
        path (list or tuple): A list/tuple of keys representing the path.
        default: The value to return if the path is invalid or value not found.

    Returns:
        The value found at the path, or the default value.
    """
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default # Path doesn't exist or intermediate element isn't a dict
    return current


def _gather(value, prefix: tuple[str], out):
    """
    Recursively collect every numeric leaf (float / int or list of them)
    and store it under its full path.
    """
    # leaf node: number or list of numbers
    if isinstance(value, (int, float, str)) or isinstance(value, list) and value and isinstance(value[0], (int, float, str)):
        if len(prefix) == 1:
            prefix = prefix[0]
        out[prefix].append(value)
    elif isinstance(value, dict):                     # keep descending
        for k, v in value.items():
            _gather(v, prefix + (k,), out)
    elif isinstance(value, list):                     # list of containers
        if value and isinstance(value[0], (dict, list)):
            for v in value:
                _gather(v, prefix, out)               # sub-lists of dicts


def collect_results(paths, infer_sampling_flops=False) -> dict[tuple[str], dict[str|tuple[str], list[float]]]:
    """
    Loads JSONs corresponding to a list of paths from disk.

    Parameters
    ----------
    paths : dict
        A dictionary where keys are group identifiers (tuples of strings)
        and values are lists of log paths.
    infer_sampling_flops : bool
        If True, the number of FLOPS for sampling is inferred from the model parameters and the max_new_tokens.

    Returns
    -------
    dict
        A dictionary where keys are group identifiers (tuples of strings)
        and values are dictionaries of collected metrics.
    """
    all_results = {}
    for k, v in paths.items():
        aggregated_results = defaultdict(list)
        for path in v:
            try:
                results = cached_json_load(path)
                for run in results["runs"]:
                    collected_metrics = defaultdict(list)
                    for step in run["steps"]:
                        if infer_sampling_flops:
                            max_new_tokens = results["config"]["attack_params"]["generation_config"]["max_new_tokens"]
                            model_params = num_model_params(results["config"]["model_params"]["id"])
                            step["flops_sampling_prefill_cache"] = model_params * len(step["model_input_tokens"]) * 2
                            step["flops_sampling_generation"] = model_params * max_new_tokens * 2
                        for metric in step.keys():
                            # this will fill collected_metrics with values from step[metric]
                            # and handles nested containers
                            _gather(step[metric], (metric,), collected_metrics)
                    for metric, v in collected_metrics.items():
                        aggregated_results[metric].append(v)
            except Exception as e:
                print(f"Error loading {path}")
                raise e
        all_results[k] = aggregated_results
    return all_results




def load_embedding(path):
    """
    Load an embedding from a file.

    Args:
        path (str): Path to the embedding file.

    Returns:
        np.ndarray: The loaded embedding.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")

    # Load the embedding from the file
    return safetensors.torch.load_file(path, device="cpu")["embeddings"]