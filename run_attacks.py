import os
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.attacks import Attack
from src.dataset import PromptDataset
from src.errors import print_exceptions
from src.io_utils import load_model_and_tokenizer, log_attack, RunConfig

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True


def should_run(cfg: DictConfig, name: str | None) -> list[tuple[str, DictConfig]]:
    if name is not None:
        return [(name, cfg[name])]
    return list(cfg.items())


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def run_attacks(cfg: DictConfig) -> None:
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    log_file_path = os.path.join(cfg.save_dir, f"{date_time_string}/run.json")
    logging.info("-------------------")
    logging.info(f"Commencing run `{date_time_string}`")
    logging.info("-------------------")

    models_to_run = should_run(cfg.models, cfg.model_name)
    datasets_to_run = should_run(cfg.datasets, cfg.dataset_name)
    attacks_to_run = should_run(cfg.attacks, cfg.attack_name)

    for model_name, model_params in models_to_run:
        logging.info(f"Target: {model_name}\n{OmegaConf.to_yaml(model_params, resolve=True)}")
        model, tokenizer = load_model_and_tokenizer(model_params)
        for dataset_name, dataset_params in datasets_to_run:
            logging.info(f"Dataset: {dataset_name}\n{OmegaConf.to_yaml(dataset_params, resolve=True)}")
            dataset = PromptDataset.from_name(dataset_name)(dataset_params)
            for attack_name, attack_params in attacks_to_run:
                logging.info(f"Attack: {attack_name}\n{OmegaConf.to_yaml(attack_params, resolve=True)}")
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
                )
                log_attack(run_config, results, log_file_path)


if __name__ == "__main__":
    run_attacks()
