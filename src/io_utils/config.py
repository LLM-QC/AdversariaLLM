"""
Configuration and filtering utilities.

This module provides configuration management and filtering functions
for experimental runs and database operations.
"""

from dataclasses import dataclass
from omegaconf import OmegaConf

from .database import get_mongodb_connection


@dataclass
class RunConfig:
    model: str
    dataset: str
    attack: str
    model_params: dict
    dataset_params: dict
    attack_params: dict


def filter_config(run_config: RunConfig, dset_len: int, overwrite: bool = False) -> RunConfig | None:
    db = get_mongodb_connection()
    collection = db.runs

    OmegaConf.resolve(run_config.attack_params)
    OmegaConf.resolve(run_config.dataset_params)
    OmegaConf.resolve(run_config.model_params)
    original_idx = run_config.dataset_params.idx

    if original_idx is None:
        idx = list(range(dset_len))
    elif isinstance(original_idx, int):
        idx = [original_idx]
    else:
        idx = original_idx

    filtered_idx = []
    for i in idx:
        run_config.dataset_params.idx = i
        config_data = OmegaConf.to_container(OmegaConf.structured(run_config), resolve=True)
        if not overwrite and collection.find_one({"config": config_data}):
            print(f"Skipping {run_config.model} {run_config.dataset} {run_config.attack} idx={i} because it already exists at {collection.find_one({'config': config_data})['log_file']}")
            continue
        filtered_idx.append(i)

    if not filtered_idx:
        return None
    run_config.dataset_params.idx = filtered_idx
    return run_config