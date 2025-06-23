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
from src.lm_utils import free_vram
from run_judges import run_judges

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True


def select_configs(cfg: DictConfig, name: str | ListConfig | None) -> list[tuple[str, DictConfig]]:
    if name is not None:
        if isinstance(name, ListConfig):
            return [(n, cfg[n]) for n in name]
        return [(name, cfg[name])]
    return list(cfg.items())


def collect_configs(cfg: DictConfig) -> list[RunConfig]:
    models_to_run = select_configs(cfg.models, cfg.model)
    datasets_to_run = select_configs(cfg.datasets, cfg.dataset)
    attacks_to_run = select_configs(cfg.attacks, cfg.attack)

    all_run_configs = []
    for model, model_params in models_to_run:
        for dataset, dataset_params in datasets_to_run:
            temp_dataset = PromptDataset.from_name(dataset)(dataset_params)
            dset_len = len(temp_dataset)
            dataset_params["idx"] = temp_dataset.config_idx
            for attack, attack_params in attacks_to_run:
                run_config = RunConfig(
                    model,
                    dataset,
                    attack,
                    model_params,
                    dataset_params,
                    attack_params,
                )
                run_config = filter_config(run_config, dset_len, overwrite=cfg.overwrite)
                if run_config is not None:
                    all_run_configs.append(run_config)
    return all_run_configs


def run_attacks(all_run_configs: list[RunConfig], cfg: DictConfig, date_time_string: str) -> None:
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

        log_attack(run_config, results, cfg, date_time_string)


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    logging.info("-------------------")
    logging.info(f"Commencing run at `{date_time_string}`")
    logging.info("-------------------")

    # 1. Parse/Collect configs for the judge and the attacks
    OmegaConf.set_struct(cfg, False)
    judges_to_run = cfg.pop('classifiers')  # remove from config to avoid saving to mongodb
    OmegaConf.set_struct(cfg, True)

    all_run_configs = collect_configs(cfg)

    # 2. Run the attacks
    run_attacks(all_run_configs, cfg, date_time_string)

    # 3. Run the judges
    if judges_to_run is None:
        return
    for judge in judges_to_run:
        # Create judge config
        judge_cfg = OmegaConf.create({
            "classifier": judge,
            "suffixes": [date_time_string.split('/')[-1]],  # Use the timestamp from this run
        })
        free_vram()
        # Run the judge
        run_judges(judge_cfg)


if __name__ == "__main__":
    main()
