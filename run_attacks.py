import os
import gc
from dataclasses import dataclass
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.attacks import Attack
from src.dataset import Dataset
from src.errors import print_exceptions
from src.io_utils import load_model_and_tokenizer, log_attack

torch.use_deterministic_algorithms(True, warn_only=True)
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


def should_run(cfg: DictConfig, name: str | None) -> list[tuple[str, DictConfig]]:
    if name is not None:
        return [(name, cfg[name])]
    return list(cfg.items())


def validate_cfg(cfg):
    if cfg.model_name not in cfg.models.keys():
        raise ValueError(f"Model `{cfg.model_name}` not found in config; should be one of {cfg.models.keys()}")
    if cfg.dataset_name not in cfg.datasets.keys():
        raise ValueError(f"Dataset `{cfg.dataset_name}` not found in config; should be one of {cfg.datasets.keys()}")
    if cfg.attack_name not in cfg.attacks.keys():
        raise ValueError(f"Attack `{cfg.attack_name}` not found in config; should be one of {cfg.attacks.keys()}")


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    if cfg.log_file_path is None:
        log_file_path = os.path.join(cfg.save_dir, f"{date_time_string}/run.json")
    else:
        log_file_path = cfg.log_file_path
    validate_cfg(cfg)

    if cfg.hf_offline_mode:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        logging.info(f"HF Offline Mode: {os.environ.get('HF_HUB_OFFLINE')}")

    logging.info("-------------------")
    logging.info(f"Commencing run `{date_time_string}`")
    logging.info(f"Saving to: {log_file_path}")
    logging.info("-------------------")

    if wandb_cfg := cfg.get("wandb", None):
        logging.info("Logging system metrics to Weights & Biases")
        import wandb
        wandb.init(
            project=wandb_cfg.project, 
            entity=wandb_cfg.entity, 
            config=dict(cfg),
            settings=wandb.Settings(
                system_sample_interval=1,
            )
        )

    models_to_run = should_run(cfg.models, cfg.model_name)
    datasets_to_run = should_run(cfg.datasets, cfg.dataset_name)
    attacks_to_run = should_run(cfg.attacks, cfg.attack_name)

    for model_name, model_params in models_to_run:
        logging.info(f"Target: {model_name}\n{OmegaConf.to_yaml(model_params)}")
        model, tokenizer = load_model_and_tokenizer(model_params)

        for dataset_name, dataset_params in datasets_to_run:
            logging.info(f"Dataset: {dataset_name}\n{OmegaConf.to_yaml(dataset_params)}")
            dataset = Dataset.from_name(dataset_name)(dataset_params)

            for attack_name, attack_params in attacks_to_run:
                logging.info(f"Attack: {attack_name}\n{OmegaConf.to_yaml(attack_params)}")

                gc.collect()
                torch.cuda.empty_cache()

                attack = Attack.from_name(attack_name)(attack_params)
                results = attack.run(model, tokenizer, dataset, log_full_results=cfg.log_toks_strs)

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
