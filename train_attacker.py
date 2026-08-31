"""Thin CLI wrapper to train an IHO diffusion attacker: ``python train_attacker.py model=...``.

The training loop itself lives in :mod:`adversariallm.training.attacker_loop` and is shared with
``run_attacks.py attack=iho`` (which trains as part of the attack). This wrapper is a convenience so
"just produce an adapter" stays a one-liner; it is equivalent to
``run_attacks.py attack=iho attacks.iho.eval_after_train=false ...`` with the same ``training.*``
overrides. Config: ``conf/attacker_training.yaml`` (whose ``training:`` block is the shared
``conf/iho_training/default.yaml`` group, identical to ``attacks.iho.training``).
"""

import os

os.umask(0o002)  # keep artifacts group-writable on the shared cluster filesystem (AGENTS.md)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # determinism, matches run_attacks.py

import logging

import hydra
from omegaconf import DictConfig

from adversariallm.dataset import PromptDataset
from adversariallm.defenses import build_target_system
from adversariallm.errors import print_exceptions
from adversariallm.io_utils import load_model_and_tokenizer

# Re-exported for backward compatibility (older imports of `from train_attacker import ...`).
from adversariallm.training.attacker_loop import (  # noqa: F401
    TrainingRunResult,
    _model_short,
    resolve_attacker_id,
    train_attacker_run,
)


@hydra.main(config_path="./conf", config_name="attacker_training", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    logging.info("-------------------")
    logging.info("Commencing attacker training run")
    logging.info("-------------------")

    if cfg.get("model") is None:
        raise ValueError("`model` is required, e.g. `python train_attacker.py model=google/gemma-3-1b-it`.")

    model_params = cfg.models[cfg.model]
    model, tokenizer = load_model_and_tokenizer(model_params)

    defense_name = cfg.defense if cfg.defense is not None else "none"
    defense_params = None if defense_name == "none" else cfg.defenses[defense_name]
    target = build_target_system(defense_params, model=model, tokenizer=tokenizer)

    dataset = PromptDataset.from_name(cfg.dataset)(cfg.datasets[cfg.dataset])

    model_id = str(model_params.get("id", cfg.model))
    result = train_attacker_run(
        target=target,
        dataset=dataset,
        training=cfg.training,
        model_id=model_id,
        dataset_name=str(cfg.dataset),
        defense_name=str(defense_name),
        defense_params=defense_params,
        attacker_dir=str(cfg.attacker_dir),
        generation_config=cfg.get("generation_config"),
        model_short=model_params.get("short_name") or _model_short(model_id),
    )

    logging.info(
        "Attacker training complete: id=%s run_dir=%s best_checkpoint=%s best_metric=%s total_flops=%.3e",
        result.attacker_id, result.run_dir, result.best_checkpoint, result.best_metric, result.total_training_flops,
    )


if __name__ == "__main__":
    main()
