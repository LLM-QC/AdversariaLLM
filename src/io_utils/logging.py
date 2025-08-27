"""
Attack result logging and management utilities.

This module provides functions for logging attack results to JSON files
and managing tensor offloading for large model embeddings.
"""

import copy
import hashlib
import json
import logging
import os
from dataclasses import asdict

import safetensors.torch
from omegaconf import DictConfig, OmegaConf

from ..attacks import AttackResult
from .database import log_config_to_db
from .json_utils import CompactJSONEncoder


def offload_tensors(run_config, result: AttackResult, embed_dir: str):
    """Offload tensors from the AttackResult to separate .safetensors"""
    assert embed_dir is not None, "embed_dir must be set to offload tensors"
    if not any(step.model_input_embeddings is not None for run in result.runs for step in run.steps):
        return

    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir, exist_ok=True)

    for i, run in enumerate(result.runs):
        for j, step in enumerate(run.steps):
            if step.model_input_embeddings is not None:
                run_hash = hashlib.sha256(f"{str(run_config)} {run.original_prompt} {step.model_completions}, {step.step}".encode()).hexdigest()
                filename = f"{run_hash}.safetensors"
                embed_path = os.path.join(embed_dir, filename)
                safetensors.torch.save_file({"embeddings": step.model_input_embeddings}, embed_path)
                logging.info(f"Saved embeddings for run {i}, step {j} to {embed_path}")
                step.model_input_embeddings = embed_path


def log_attack(run_config, result: AttackResult, cfg: DictConfig, date_time_string: str):
    """Logs the attack results to a JSON file and MongoDB."""
    save_dir = cfg.save_dir
    embed_dir = cfg.embed_dir
    for idx, run in enumerate(result.runs):
        subrun_config = copy.deepcopy(run_config)
        subrun_config.dataset_params["idx"] = [run_config.dataset_params["idx"][idx]]
        subrun_result = AttackResult(runs=[run])

        # Create a structured log message as a JSON object
        OmegaConf.resolve(subrun_config.attack_params)
        OmegaConf.resolve(subrun_config.dataset_params)
        OmegaConf.resolve(subrun_config.model_params)
        log_message = {
            "config": OmegaConf.to_container(OmegaConf.structured(subrun_config), resolve=True)
        }
        offload_tensors(subrun_config, subrun_result, embed_dir)

        log_message.update(asdict(subrun_result))
        # Find the first available run_i.json file
        i = 0
        log_dir = os.path.join(save_dir, date_time_string)
        while os.path.exists(os.path.join(log_dir, str(i), f"run.json")):
            i += 1
        log_file = os.path.join(log_dir, str(i), f"run.json")

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(log_message, f, indent=2, cls=CompactJSONEncoder)
        logging.info(f"Attack logged to {log_file}")
        log_config_to_db(subrun_config, subrun_result, log_file)
