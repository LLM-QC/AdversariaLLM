import os
from dataclasses import dataclass
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.attacks import Attack
from src.datasets import Dataset
from src.errors import print_exceptions
from src.io_utils import load_model_and_tokenizer, log_attack

torch.use_deterministic_algorithms(True, warn_only=True)  # determinism
torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class RunConfig:
    model: str
    dataset: str
    attack: str
    model_params: dict
    dataset_params: dict
    attack_params: dict
    config: dict


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    log_file_path = os.path.join(cfg.save_dir, f"{date_time_string}/run.json")
    logging.info("-------------------")
    logging.info(f"Commencing run `{date_time_string}`")
    logging.info(f"Saving to: {log_file_path}")
    logging.info("-------------------")

    if cfg.model_name not in cfg.models.keys():
        raise ValueError(f"Model `{cfg.model_name}` not found in config; should be one of {cfg.models.keys()}")
    if cfg.dataset_name not in cfg.datasets.keys():
        raise ValueError(f"Dataset `{cfg.dataset_name}` not found in config; should be one of {cfg.datasets.keys()}")
    if cfg.attack_name not in cfg.attacks.keys():
        raise ValueError(f"Attack `{cfg.attack_name}` not found in config; should be one of {cfg.attacks.keys()}")

    for model_name, model_params in cfg.models.items():
        if cfg.model_name is not None and model_name != cfg.model_name:
            continue
        logging.info(f"Target: {model_name}\n{OmegaConf.to_yaml(model_params)}")
        model, tokenizer = load_model_and_tokenizer(model_params)
        for dataset_name, dataset_params in cfg.datasets.items():
            if cfg.dataset_name is not None and dataset_name != cfg.dataset_name:
                continue
            logging.info(
                f"Dataset: {dataset_name}\n{OmegaConf.to_yaml(dataset_params)}"
            )
            dataset = Dataset.from_name(dataset_name)(dataset_params)
            for attack_name, attack_params in cfg.attacks.items():
                if cfg.attack_name is not None and attack_name != cfg.attack_name:
                    continue
                logging.info(
                    f"Attack: {attack_name}\n{OmegaConf.to_yaml(attack_params)}"
                )
                attack = Attack.from_name(attack_name)(attack_params)
                results = attack.run(model, tokenizer, dataset)
                # get combined config for this run
                run_config = RunConfig(
                    model_name,
                    dataset_name,
                    attack_name,
                    model_params,
                    dataset_params,
                    attack_params,
                    cfg,
                )
                log_attack(run_config, results, log_file_path)


if __name__ == "__main__":
    main()
