import os
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.attacks import Attack
from src.dataset import PromptDataset
from src.errors import print_exceptions
from src.io_utils import load_model_and_tokenizer, log_attack, RunConfig, filter_config

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True


def select_configs(cfg: DictConfig, name: str | ListConfig | None) -> list[tuple[str, DictConfig]]:
    if name is not None:
        if isinstance(name, ListConfig):
            return [(n, cfg[n]) for n in name]
        return [(name, cfg[name])]
    return list(cfg.items())


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def run_attacks(cfg: DictConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    logging.info("-------------------")
    logging.info(f"Commencing run at `{date_time_string}`")
    logging.info("-------------------")

    models_to_run = select_configs(cfg.models, cfg.model_name)
    datasets_to_run = select_configs(cfg.datasets, cfg.dataset_name)
    attacks_to_run = select_configs(cfg.attacks, cfg.attack_name)

    all_run_configs = []
    for model_name, model_params in models_to_run:
        for dataset_name, dataset_params in datasets_to_run:
            temp_dataset = PromptDataset.from_name(dataset_name)(dataset_params)
            dset_len = len(temp_dataset)
            dataset_params["idx"] = temp_dataset.config_idx
            for attack_name, attack_params in attacks_to_run:
                run_config = RunConfig(
                    model_name,
                    dataset_name,
                    attack_name,
                    model_params,
                    dataset_params,
                    attack_params,
                )
                run_config = filter_config(run_config, dset_len, overwrite=cfg.overwrite)
                if run_config is not None:
                    all_run_configs.append(run_config)

    last_model = None
    last_dataset = None
    last_attack = None
    for run_config in all_run_configs:
        if last_model != run_config.model:
            logging.info(f"Target: {run_config.model}\n{OmegaConf.to_yaml(run_config.model_params, resolve=True)}")
            last_model = run_config.model
            model, tokenizer = load_model_and_tokenizer(run_config.model_params)
        if last_dataset != run_config.dataset:
            logging.info(f"Dataset: {run_config.dataset}\n{OmegaConf.to_yaml(run_config.dataset_params, resolve=True)}")
            last_dataset = run_config.dataset
            dataset = PromptDataset.from_name(run_config.dataset)(run_config.dataset_params)
        if last_attack != run_config.attack:
            logging.info(f"Attack: {run_config.attack}\n{OmegaConf.to_yaml(run_config.attack_params, resolve=True)}")
            last_attack = run_config.attack

        attack = Attack.from_name(run_config.attack)(run_config.attack_params)
        results = attack.run(model, tokenizer, dataset)

        log_attack(run_config, results, cfg.save_dir, date_time_string)


if __name__ == "__main__":
    run_attacks()
