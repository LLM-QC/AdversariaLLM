"""The IHO attacker-training loop (plan phase 3, §4.2-§4.6) -- the shared engine.

A training run turns a target model + a behavior split into a LoRA adapter written under
``${attacker_dir}/<attacker_id>/`` with the exact on-disk layout ``build_iho_phase_table`` reads
(plan §4.3). :func:`train_attacker_run` is called by BOTH entry points: ``run_attacks.py
attack=iho`` (which trains as part of the attack, then evaluates) and the thin ``train_attacker.py``
CLI wrapper at the repo root (``python train_attacker.py model=...``). Keeping the loop here, in the
library, avoids the old layering wart of the attack importing from a top-level script.

Both paths sample and score through :class:`~adversariallm.attacks.iho.IHOSampler`, so a training
sample and an evaluation sample are produced by identical code. ``IHOAttack`` imports this module
lazily (``iho`` <-> ``attacker_loop`` are a genuine mutual dependency via the shared sampler).
"""

import os

os.umask(0o002)  # keep artifacts group-writable on the shared cluster filesystem (AGENTS.md)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # determinism, matches run_attacks.py

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from adversariallm.attackers import build_attacker
from adversariallm.attacks.iho import IHOSample, IHOSampler
from adversariallm.dataset import PromptDataset
from adversariallm.dataset.index_mapping import map_internal_to_original
from adversariallm.defenses import TargetSystem
from adversariallm.io_utils import free_vram
from adversariallm.training import (
    DiffusionDPOTrainer,
    DPOConfig,
    FlopsLedger,
    LoggingConfig,
    PairingConfig,
    PreferenceDataset,
    RunLogger,
    build_preference_pairs,
)
from adversariallm.training.dpo import DEFAULT_SCORE_COL

logger = logging.getLogger(__name__)

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True

#: Judge role that drives preference pairing + early stopping (plan §4.5).
TRAINING_ROLE = "training"
#: Judge role logged only, never influencing checkpoint selection (plan §4.5).
VALIDATION_ROLE = "validation"

_EPOCH_RE = re.compile(r"dpo_epoch(\d+)")
_CYCLE_SAMPLE_RE = re.compile(r"cycle_(\d+)\.parquet$")

#: Persisted per-cycle best metric, so a resumed run can still pick the global winning cycle.
CYCLE_METRICS_FILENAME = "cycle_metrics.json"


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------


@dataclass
class TrainingRunResult:
    """What a completed (or resumed-and-completed) training run produced.

    ``best_checkpoint`` is the resolved adapter directory of the winning cycle (not the ``best``
    symlink), so a caller can hand it straight to ``DiffusionAttackerConfig.lora_checkpoint``.
    """

    attacker_id: str
    run_dir: str
    best_checkpoint: str | None
    best_metric: float | None
    winning_cycle: int | None
    n_cycles_completed: int
    total_training_flops: int
    n_behaviors_amortised: int
    train_config: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# attacker_id (plan §4.4)
# --------------------------------------------------------------------------------------


def _canonical(obj: Any) -> Any:
    """Plain, JSON-serialisable container for hashing / config dumps (OmegaConf- and dataclass-aware)."""
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def resolve_attacker_id(
    attacker_id_cfg: Any,
    *,
    model_id: str,
    model_short: str,
    dataset_name: str,
    defense_name: str,
    defense_params: Any,
    training: dict,
) -> str:
    """``auto`` -> ``f"{model_short}_{dataset}_{defense}_{sha256(...)[:12]}"`` (plan §4.4).

    The hash covers everything that determines the adapter and nothing that does not: target
    model id, defense name+params, dataset name, ``train_idx``/``eval_idx``, the attacker / dpo /
    pairing configs, ``n_cycles``, ``n_samples``, ``train_judge`` and ``seed``. Logging config,
    ``save_*`` flags, ``resume``, paths, and validation-only knobs (``val_idx``/``val_judges``)
    are excluded, so toggling them still resolves to the same directory and resumes.
    """
    if attacker_id_cfg not in (None, "auto"):
        return str(attacker_id_cfg)

    payload = {
        "model_id": model_id,
        "defense": {"name": defense_name, "params": _canonical(defense_params)},
        "dataset": dataset_name,
        "train_idx": _canonical(training.get("train_idx")),
        "eval_idx": _canonical(training.get("eval_idx")),
        "attacker": _canonical(training.get("attacker")),
        "dpo": _canonical(training.get("dpo")),
        "pairing": _canonical(training.get("pairing")),
        "n_cycles": training.get("n_cycles"),
        "n_samples": _canonical(training.get("n_samples")),
        "attacker_batch_size": training.get("attacker_batch_size"),
        "train_judge": training.get("train_judge"),
        "seed": training.get("seed"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    hash12 = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize(model_short)}_{_sanitize(dataset_name)}_{_sanitize(defense_name)}_{hash12}"


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")


def _model_short(model_id: str) -> str:
    """Last path segment of the model id, e.g. ``google/gemma-3-1b-it`` -> ``gemma-3-1b-it``."""
    return model_id.split("/")[-1]


# --------------------------------------------------------------------------------------
# Behaviors / judges
# --------------------------------------------------------------------------------------


def _behaviors_for(dataset: PromptDataset, idx: list[int]) -> list[tuple[int, str, str]]:
    """``[(internal_idx, goal, target)]`` for the given indices into the constructed dataset."""
    behaviors: list[tuple[int, str, str]] = []
    for i in idx:
        conversation = dataset[i]
        assert len(conversation) == 2, "IHO training assumes single-turn (user, assistant) behaviors."
        behaviors.append((i, conversation[0]["content"], conversation[1]["content"]))
    return behaviors


def _default_judge_loader(name: str):
    """Load a judgezoo judge by name (lazy import: judgezoo is heavy and optional at import time)."""
    from judgezoo import Judge  # noqa: PLC0415

    return Judge.from_name(name)


def _apply_judge(
    samples: list[IHOSample],
    judge: Any,
    *,
    role: str,
    ledger: FlopsLedger,
    phase: str,
    tokenizer: Any,
) -> None:
    """Score ``samples`` with a single resident judge, writing ``judge_scores[role]`` in place.

    A best-effort ``judge`` FLOPs entry is booked (plan §5.2/§5.4): the parameter count comes from
    the judge's own model when reachable, otherwise it is recorded as ``0`` with
    ``detail={"unknown_params": True}`` rather than guessed. Token counts are a proxy from the
    supplied tokenizer, which is enough for a component-level accounting.
    """
    if not samples:
        return
    chats = [
        [
            {"role": "user", "content": s.goal},
            {"role": "assistant", "content": s.completion or ""},
        ]
        for s in samples
    ]
    p_harmful = judge(chats)["p_harmful"]
    for s, score in zip(samples, p_harmful):
        s.judge_scores[role] = score

    _book_judge_flops(judge, samples=samples, role=role, ledger=ledger, phase=phase, tokenizer=tokenizer)


def _book_judge_flops(
    judge: Any, *, samples: list[IHOSample], role: str, ledger: FlopsLedger, phase: str, tokenizer: Any
) -> None:
    judge_model = getattr(judge, "model", None)
    n_params = 0
    unknown = True
    model_id = type(judge).__name__
    if judge_model is not None and hasattr(judge_model, "num_parameters"):
        try:
            n_params = int(judge_model.num_parameters(exclude_embeddings=True))
            unknown = False
            model_id = getattr(judge_model, "name_or_path", None) or model_id
        except Exception:  # noqa: BLE001 - accounting must never break a training run
            n_params, unknown = 0, True

    total_tokens = 0
    if tokenizer is not None:
        for s in samples:
            total_tokens += _safe_token_count(tokenizer, s.goal) + _safe_token_count(tokenizer, s.completion or "")

    ledger.add(
        component="judge",
        phase=phase,
        model_id=str(model_id),
        n_params=n_params,
        n_tokens_in=total_tokens,
        pass_type="forward",
        n_passes=1,
        lora=False,
        detail={"role": role, "n_samples": len(samples), "unknown_params": unknown},
    )


def _safe_token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:  # noqa: BLE001 - a proxy count must never break accounting
        return 0


# --------------------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------------------


def _last_valid_cycle(run_dir: str) -> int | None:
    """Highest cycle ``i`` for which ``checkpoints/best_cycle_<i>/`` exists and is non-empty.

    A cycle counts as resumable only once it has a saved adapter, so a run interrupted mid-cycle
    is re-run from the start of that cycle rather than silently skipped.
    """
    ckpt_root = Path(run_dir, "checkpoints")
    if not ckpt_root.is_dir():
        return None
    valid: list[int] = []
    for child in ckpt_root.glob("best_cycle_*"):
        if not child.is_dir() or not any(child.iterdir()):
            continue
        m = re.search(r"best_cycle_(\d+)$", child.name)
        if m:
            valid.append(int(m.group(1)))
    return max(valid) if valid else None


def _load_previous_cycle_frames(run_dir: str, upto_cycle_exclusive: int) -> list[pd.DataFrame]:
    """Reload ``samples/cycle_<i>.parquet`` for cycles ``< upto_cycle_exclusive`` (for expanding pairing)."""
    frames: list[pd.DataFrame] = []
    for i in range(upto_cycle_exclusive):
        path = Path(run_dir, "samples", f"cycle_{i}.parquet")
        if path.exists():
            frames.append(pd.read_parquet(path))
    return frames


def _read_cycle_metrics(run_dir: str) -> dict[int, float | None]:
    path = Path(run_dir, CYCLE_METRICS_FILENAME)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}
    return {int(k): (None if v is None else float(v)) for k, v in raw.items()}


def _write_cycle_metrics(run_dir: str, metrics: dict[int, float | None]) -> None:
    Path(run_dir, CYCLE_METRICS_FILENAME).write_text(
        json.dumps({str(k): v for k, v in sorted(metrics.items())}, indent=2, sort_keys=True)
    )


# --------------------------------------------------------------------------------------
# The cycle loop (shared by the train_attacker.py CLI and IHOAttack's adaptive training path)
# --------------------------------------------------------------------------------------


def train_attacker_run(
    *,
    target: TargetSystem,
    dataset: PromptDataset,
    training: DictConfig | dict,
    model_id: str,
    dataset_name: str,
    defense_name: str,
    defense_params: Any,
    attacker_dir: str,
    generation_config: Any,
    model_short: str | None = None,
    attacker: Any = None,
    judge_loader: Callable[[str], Any] | None = None,
) -> TrainingRunResult:
    """Run the multi-cycle DPO training loop and return where the winning adapter landed.

    ``attacker`` and ``judge_loader`` are injection points (mirroring
    :class:`~adversariallm.attacks.iho.IHOSampler`): production passes neither and one diffusion
    attacker / real judgezoo judges are built; tests pass a fake trainable attacker and a fake
    judge loader so the whole loop runs on CPU without weights, GPU, or network.
    """
    training = dict(_canonical(training) or {})
    judge_loader = judge_loader or _default_judge_loader
    model_short = model_short or _model_short(model_id)

    attacker_id = resolve_attacker_id(
        training.get("attacker_id", "auto"),
        model_id=model_id,
        model_short=model_short,
        dataset_name=dataset_name,
        defense_name=defense_name,
        defense_params=defense_params,
        training=training,
    )
    run_dir = os.path.join(attacker_dir, attacker_id)
    os.makedirs(run_dir, exist_ok=True)

    dpo_cfg = _to_dataclass(DPOConfig, training.get("dpo"))
    pairing_cfg = _to_dataclass(PairingConfig, training.get("pairing"))
    logging_cfg = _to_dataclass(LoggingConfig, training.get("logging"))
    n_cycles = int(training.get("n_cycles", 1))
    seed = int(training.get("seed", 0))
    attacker_batch_size = int(training.get("attacker_batch_size", 64))
    # TOTAL sample budgets, pooled across each split's behaviors (AAPL's num_sampled_attacks
    # semantics -- IHOSampler.generate_pooled), not per-behavior counts.
    n_samples = _canonical(training.get("n_samples")) or {}
    n_train, n_eval, n_val = int(n_samples.get("train", 0)), int(n_samples.get("eval", 0)), int(n_samples.get("val", 0))
    train_judge_name = training.get("train_judge")
    val_judge_names = list(_canonical(training.get("val_judges")) or [])
    save_cycle_samples = bool(training.get("save_cycle_samples", True))
    save_dpo_samples = bool(training.get("save_dpo_samples", True))
    resume = bool(training.get("resume", True))

    # Splits: indices into the constructed dataset object (post-shuffle/post-idx; footgun §6.2).
    all_idx = list(range(len(dataset)))
    train_idx = _resolve_split(training.get("train_idx"), all_idx)
    eval_idx = _resolve_split(training.get("eval_idx"), train_idx)  # eval defaults to the train split
    val_idx_cfg = _canonical(training.get("val_idx"))
    val_idx = list(val_idx_cfg) if val_idx_cfg else None
    _log_resolved_original_rows(dataset, train_idx, eval_idx, val_idx)

    train_behaviors = _behaviors_for(dataset, train_idx)
    eval_behaviors = _behaviors_for(dataset, eval_idx)
    val_behaviors = _behaviors_for(dataset, val_idx) if val_idx else []

    # Resume: find the last cycle with a saved adapter and continue from the next one.
    start_cycle = 0
    resume_ckpt: str | None = None
    if resume:
        last_valid = _last_valid_cycle(run_dir)
        if last_valid is not None:
            start_cycle = last_valid + 1
            resume_ckpt = os.path.join(run_dir, "checkpoints", f"best_cycle_{last_valid}")
            logger.info("Resuming %s from cycle %d (adapter %s).", attacker_id, start_cycle, resume_ckpt)

    attacker = _build_or_inject_attacker(attacker, training.get("attacker"), resume_ckpt=resume_ckpt)

    ledger = FlopsLedger()
    train_config = _build_train_config_dict(
        attacker_id=attacker_id,
        model_id=model_id,
        model_short=model_short,
        dataset_name=dataset_name,
        defense_name=defense_name,
        defense_params=defense_params,
        training=training,
        generation_config=_canonical(generation_config),
        train_idx=train_idx,
        eval_idx=eval_idx,
        val_idx=val_idx,
    )

    sampler = IHOSampler(training.get("attacker"), ledger=ledger, seed=seed, attacker=attacker)
    sampler.gen_cfg = _generation_config(generation_config)
    sampler.target = target

    cycle_metrics = _read_cycle_metrics(run_dir) if resume else {}
    previous_frames = _load_previous_cycle_frames(run_dir, start_cycle) if pairing_cfg.expanding else []

    with RunLogger(logging_cfg, run_dir, train_config) as run_logger:
        # AAPL rebuilds the DPO trainer (a cold AdamW) and reloads the attacker from the previous
        # cycle's *best* checkpoint before every cycle. We mirror that here: `last_best_ckpt` is the
        # adapter to restore from (seeded by a resume checkpoint, if any) and `global_step` is
        # carried across the per-cycle trainers so offline logging stays monotonic. Sharing one
        # trainer/optimizer across cycles (the previous behaviour) let AdamW state and end-of-cycle
        # weights compound past the validated best, diverging from AAPL and the paper.
        last_best_ckpt = resume_ckpt
        global_step = 0

        for cycle in range(start_cycle, n_cycles):
            logger.info("=== IHO training cycle %d/%d (%s) ===", cycle, n_cycles - 1, attacker_id)

            # Reset the attacker to the previous cycle's best adapter, discarding the extra
            # early-stop-overshoot epochs. Cycle `start_cycle` already holds the right weights: a
            # fresh run injected the base LoRA, a resumed run loaded `resume_ckpt` at build time.
            if cycle > start_cycle and last_best_ckpt is not None:
                attacker.load_lora(last_best_ckpt)

            trainer = DiffusionDPOTrainer(
                attacker, dpo_cfg, ledger=ledger, logger=run_logger, score_col=DEFAULT_SCORE_COL
            )
            trainer.global_step = global_step

            train_df = _sample_split(
                sampler, train_behaviors, n_train, phase=f"cycle{cycle}:pre",
                batch_size=attacker_batch_size, target=target, ledger=ledger,
                judge_loader=judge_loader, train_judge_name=train_judge_name, val_judge_names=val_judge_names,
            )
            if save_cycle_samples:
                _write_parquet(train_df, run_dir, "samples", f"cycle_{cycle}.parquet")

            if val_behaviors:
                val_df = _sample_split(
                    sampler, val_behaviors, n_val, phase=f"cycle{cycle}:val",
                    batch_size=attacker_batch_size, target=target, ledger=ledger,
                    judge_loader=judge_loader, train_judge_name=train_judge_name, val_judge_names=val_judge_names,
                )
                if save_cycle_samples:
                    _write_parquet(val_df, run_dir, "samples_validation", f"cycle_{cycle}.parquet")

            pairs = build_preference_pairs(
                current=train_df, previous=previous_frames, cfg=pairing_cfg, score_col=DEFAULT_SCORE_COL
            )
            pref_dataset = PreferenceDataset(pairs)
            if save_cycle_samples:
                pref_dataset.to_parquet(os.path.join(run_dir, "prefs", f"cycle_{cycle}.parquet"))

            if len(pref_dataset) == 0:
                logger.warning("Cycle %d produced zero preference pairs; skipping DPO for this cycle.", cycle)
                cycle_metrics[cycle] = None
                _write_cycle_metrics(run_dir, cycle_metrics)
                if pairing_cfg.expanding:
                    previous_frames.append(train_df)
                continue

            eval_fn = _make_eval_fn(
                sampler=sampler, eval_behaviors=eval_behaviors, n_eval=n_eval, cycle=cycle,
                batch_size=attacker_batch_size, target=target, ledger=ledger, run_dir=run_dir,
                judge_loader=judge_loader, train_judge_name=train_judge_name, val_judge_names=val_judge_names,
                save_dpo_samples=save_dpo_samples,
            )
            cycle_result = trainer.train_cycle(pref_dataset, cycle_id=cycle, run_dir=run_dir, eval_fn=eval_fn)
            global_step = trainer.global_step
            if cycle_result.best_checkpoint is not None:
                last_best_ckpt = cycle_result.best_checkpoint

            cycle_metrics[cycle] = cycle_result.best_metric
            _write_cycle_metrics(run_dir, cycle_metrics)
            run_logger.log(
                {"cycle/id": cycle, "cycle/best_metric": cycle_result.best_metric if cycle_result.best_metric is not None else float("nan"),
                 "cycle/epochs_run": cycle_result.epochs_run, "cycle/stopped_early": int(cycle_result.stopped_early)},
            )
            if pairing_cfg.expanding:
                previous_frames.append(train_df)

        winning_cycle, best_metric = _winning_cycle(cycle_metrics)
        best_checkpoint = _finalize_best(run_dir, winning_cycle, attacker)

        ledger.write_json(os.path.join(run_dir, "flops.json"))
        OmegaConf.save(OmegaConf.create(train_config), os.path.join(run_dir, "train_config.yaml"))
        run_logger.summary(
            {"attacker_id": attacker_id, "winning_cycle": winning_cycle if winning_cycle is not None else -1,
             "best_metric": best_metric if best_metric is not None else float("nan"),
             "total_training_flops": ledger.total()}
        )

    return TrainingRunResult(
        attacker_id=attacker_id,
        run_dir=run_dir,
        best_checkpoint=best_checkpoint,
        best_metric=best_metric,
        winning_cycle=winning_cycle,
        n_cycles_completed=n_cycles,
        total_training_flops=ledger.total(),
        n_behaviors_amortised=len(train_idx),
        train_config=train_config,
    )


# --------------------------------------------------------------------------------------
# Loop helpers
# --------------------------------------------------------------------------------------


def _sample_split(
    sampler: IHOSampler,
    behaviors: list[tuple[int, str, str]],
    n_total: int,
    *,
    phase: str,
    batch_size: int,
    target: TargetSystem,
    ledger: FlopsLedger,
    judge_loader: Callable[[str], Any],
    train_judge_name: Any,
    val_judge_names: list,
) -> pd.DataFrame:
    """Generate + score one split with the current adapter; returns the plan §4.6 DataFrame.

    ``n_total`` is the split's TOTAL sample budget (AAPL's ``num_sampled_attacks`` semantics),
    pooled across ``behaviors`` rather than applied to each one -- see
    :meth:`IHOSampler.generate_pooled`.

    Judges are resident one at a time (plan §4.5): the training judge is loaded, scores into
    ``judge_score_training``, then freed before any validation judge is loaded.
    """
    samples = sampler.generate_pooled(behaviors, n_total, phase=phase, batch_size=batch_size)
    sampler.score(samples, target, judges=None, phase=phase, compute_target_loss=True)
    _score_with_all_judges(
        samples, target=target, ledger=ledger, phase=phase,
        judge_loader=judge_loader, train_judge_name=train_judge_name, val_judge_names=val_judge_names,
    )
    return sampler.to_dataframe(samples)


def _score_with_all_judges(
    samples: list[IHOSample],
    *,
    target: TargetSystem,
    ledger: FlopsLedger,
    phase: str,
    judge_loader: Callable[[str], Any],
    train_judge_name: Any,
    val_judge_names: list,
) -> None:
    if train_judge_name:
        judge = judge_loader(str(train_judge_name))
        _apply_judge(samples, judge, role=TRAINING_ROLE, ledger=ledger, phase=phase, tokenizer=target.tokenizer)
        del judge
        free_vram()
    # Validation judges are logged only. The single judge_score_validation column holds the last
    # one; with several val_judges the extra scores are not persisted per-column (documented).
    for name in val_judge_names:
        judge = judge_loader(str(name))
        _apply_judge(samples, judge, role=VALIDATION_ROLE, ledger=ledger, phase=phase, tokenizer=target.tokenizer)
        del judge
        free_vram()


def _make_eval_fn(
    *,
    sampler: IHOSampler,
    eval_behaviors: list[tuple[int, str, str]],
    n_eval: int,
    cycle: int,
    batch_size: int,
    target: TargetSystem,
    ledger: FlopsLedger,
    run_dir: str,
    judge_loader: Callable[[str], Any],
    train_judge_name: Any,
    val_judge_names: list,
    save_dpo_samples: bool,
) -> Callable[[str], pd.DataFrame]:
    """Build the ``eval_fn(phase_id) -> DataFrame`` the trainer calls mid-training (plan §3.7).

    The trainer sets the attacker to eval mode around this call; we sample the eval split with the
    current (in-training) adapter, score it, write ``dpo_samples/cycle_<i>_epoch_<e>.parquet``, and
    return the scored frame so the trainer can compute its selection metric.
    """

    def eval_fn(phase: str) -> pd.DataFrame:
        m = _EPOCH_RE.search(phase)
        epoch = int(m.group(1)) if m else 0
        df = _sample_split(
            sampler, eval_behaviors, n_eval, phase=phase, batch_size=batch_size, target=target, ledger=ledger,
            judge_loader=judge_loader, train_judge_name=train_judge_name, val_judge_names=val_judge_names,
        )
        if save_dpo_samples:
            _write_parquet(df, run_dir, "dpo_samples", f"cycle_{cycle}_epoch_{epoch}.parquet")
        # The sampler always emits a `judge_score_validation` column; when no validation judge ran
        # it is all-null, and the trainer would try to quantile it. Drop it for the trainer only
        # (the saved parquet keeps the full §4.6 schema).
        if "judge_score_validation" in df.columns and df["judge_score_validation"].isna().all():
            return df.drop(columns=["judge_score_validation"])
        return df

    return eval_fn


def _resolve_split(idx_cfg: Any, default: list[int]) -> list[int]:
    resolved = _canonical(idx_cfg)
    if resolved is None:
        return list(default)
    return [int(i) for i in resolved]


def _log_resolved_original_rows(
    dataset: PromptDataset, train_idx: list[int], eval_idx: list[int], val_idx: list[int] | None
) -> None:
    """Log the original dataset rows behind each split (footgun §6.2), best-effort."""
    cfg = getattr(dataset, "config", None)
    seed = int(getattr(cfg, "seed", 0) or 0)
    shuffle = bool(getattr(cfg, "shuffle", True))
    config_idx = getattr(cfg, "idx", None)
    # `dataset.idx` (a tensor of the selected original rows) is the authoritative internal->original
    # map; its length is the full universe only when config.idx is None. Prefer it, fall back to the
    # helper for a defensively-computed value.
    dataset_idx = getattr(dataset, "idx", None)

    def _rows(internal: list[int]) -> list[int]:
        try:
            if dataset_idx is not None:
                return [int(dataset_idx[i]) for i in internal]
            result = map_internal_to_original(internal, len(dataset), seed, shuffle, config_idx)
            return result if isinstance(result, list) else [result]
        except Exception as exc:  # noqa: BLE001 - auditing must not break training
            logger.warning("Could not resolve original dataset rows (%s).", exc)
            return []

    logger.info("Training split -> original dataset rows: train=%s", _rows(train_idx))
    logger.info("Eval split -> original dataset rows:     eval=%s", _rows(eval_idx))
    if val_idx:
        logger.info("Validation split -> original dataset rows: val=%s", _rows(val_idx))


def _winning_cycle(cycle_metrics: dict[int, float | None]) -> tuple[int | None, float | None]:
    scored = {c: m for c, m in cycle_metrics.items() if m is not None}
    if not scored:
        return None, None
    winning = max(scored, key=lambda c: scored[c])
    return winning, scored[winning]


def _finalize_best(run_dir: str, winning_cycle: int | None, attacker: Any) -> str | None:
    """Point ``<run_dir>/best`` at the winning cycle's adapter and return its resolved path.

    If no cycle ever checkpointed (e.g. ``checkpoint_every=0``), the current adapter is saved to
    ``checkpoints/final`` so there is still a usable adapter, and ``best`` points there.
    """
    if winning_cycle is not None:
        target_dir = os.path.join(run_dir, "checkpoints", f"best_cycle_{winning_cycle}")
    else:
        target_dir = os.path.join(run_dir, "checkpoints", "final")
        if hasattr(attacker, "save_lora") and getattr(attacker, "has_lora", False):
            os.makedirs(target_dir, exist_ok=True)
            try:
                attacker.save_lora(target_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not save fallback final adapter (%s).", exc)
                return None
        else:
            return None

    link = Path(run_dir, "best")
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(target_dir, run_dir))
    except OSError as exc:
        logger.warning("Could not create best symlink (%s); use the checkpoint path directly.", exc)
    return target_dir


def _write_parquet(df: pd.DataFrame, run_dir: str, subdir: str, filename: str) -> None:
    out_dir = Path(run_dir, subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / filename, index=False)


# --------------------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------------------


def _to_dataclass(cls: type, node: Any):
    """Instantiate a config dataclass from an OmegaConf node / dict, keeping only known fields."""
    container = _canonical(node) or {}
    known = {f for f in getattr(cls, "__dataclass_fields__", {})}
    kwargs = {k: v for k, v in container.items() if k in known}
    return cls(**kwargs)


def _generation_config(generation_config: Any):
    """Coerce the target-generation config into the ``GenerationConfig`` the sampler expects."""
    from adversariallm.attacks.attack import GenerationConfig  # noqa: PLC0415

    if isinstance(generation_config, GenerationConfig):
        return generation_config
    container = _canonical(generation_config) or {}
    known = set(GenerationConfig.__dataclass_fields__)
    return GenerationConfig(**{k: v for k, v in container.items() if k in known})


def _build_or_inject_attacker(attacker: Any, attacker_cfg_node: Any, *, resume_ckpt: str | None):
    """Return an attacker to train: the injected one if given, else built from config.

    On resume the built attacker loads the previous cycle's adapter (``lora_checkpoint``) instead
    of a fresh ``lora`` block (the two are mutually exclusive). An injected attacker is used as-is.
    """
    if attacker is not None:
        if resume_ckpt is not None:
            logger.info("Resuming with an injected attacker; the checkpoint at %s is not reloaded.", resume_ckpt)
        return attacker

    cfg = OmegaConf.create(_canonical(attacker_cfg_node) or {})
    if resume_ckpt is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.lora = None
        cfg.lora_checkpoint = resume_ckpt
    return build_attacker(cfg)


def _build_train_config_dict(
    *,
    attacker_id: str,
    model_id: str,
    model_short: str,
    dataset_name: str,
    defense_name: str,
    defense_params: Any,
    training: dict,
    generation_config: Any,
    train_idx: list[int],
    eval_idx: list[int],
    val_idx: list[int] | None,
) -> dict:
    return {
        "attacker_id": attacker_id,
        "model": {"id": model_id, "short_name": model_short},
        "dataset": dataset_name,
        "defense": {"name": defense_name, "params": _canonical(defense_params)},
        "generation_config": _canonical(generation_config),
        "training": _canonical(training),
        "resolved_splits": {"train_idx": train_idx, "eval_idx": eval_idx, "val_idx": val_idx},
    }


# --------------------------------------------------------------------------------------
# The CLI entrypoint that wraps this loop lives in the thin ``train_attacker.py`` at the repo root;
# ``attack=iho`` (run_attacks.py) calls :func:`train_attacker_run` directly. Both share this module.
