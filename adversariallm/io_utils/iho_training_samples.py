"""Link and read IHO training-phase attack samples from a ``run.json``.

An adaptive IHO run (``attack=iho`` with a ``training`` block) trains the diffusion attacker by
repeatedly inpainting adversarial prompts, running them against the target, and judging the
responses. Those are genuine attack attempts, but there are far too many (~15k/cycle) to inline
into ``run.json`` the way a normal attack's ``steps`` are, so training persists them as parquet
files under the attacker ``run_dir``:

* ``samples/cycle_<c>.parquet``                 -- pre-DPO samples (all attempts, every cycle)
* ``dpo_samples/cycle_<c>_epoch_<e>.parquet``   -- mid-training eval samples (per checkpoint)
* ``samples_validation/cycle_<c>.parquet``      -- optional validation-behavior samples

Each row already carries the adversarial prompt, the target ``attacked_output`` (response), the
per-judge scores, the ``phase_id``/``cycle_id``, and per-component FLOPs. This module writes a
compact *manifest* of those parquets so ``run.json`` links them (via ``resume_metadata``), and
reads them back as per-behavior score sequences -- so the training attempts are usable by the same
metric layer (``compute_asr_from_sequences`` / ``evoc_from_score_sequences``) as any inline attack,
without bloating ``run.json``.

Design notes:
* The transfer/eval samples stay inline in ``run.json`` ``steps`` (repo-normal); only the *training*
  attempts live in the linked parquets. So a transfer-only IHO run has no manifest and looks like
  every other attack.
* The manifest declares a **judge -> score-column** mapping, so adding a judge later means adding a
  column + a manifest entry -- never restructuring the parquet.
* Paths are stored relative to ``run_dir`` (plus the absolute ``run_dir``), so the manifest survives
  the run.json being copied; ``base_dir`` overrides resolution if the parquets themselves moved.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any

import pandas as pd

#: Parquet column holding the behaviour id (raw JailbreakBench row) each sample attacks.
DEFAULT_BEHAVIOR_COL = "jb_index"
#: Parquet column holding the target's response to the adversarial prompt.
DEFAULT_RESPONSE_COL = "attacked_output"
#: Parquet column holding the (inpainted) adversarial prompt text.
DEFAULT_PROMPT_COL = "attacking_prompt_text"
#: The two score columns training writes today: the train judge and (optionally) a second judge.
_TRAIN_SCORE_COL = "judge_score_training"
_VAL_SCORE_COL = "judge_score_validation"

_PRE_DPO_RE = re.compile(r"cycle_(\d+)\.parquet$")
_DPO_EVAL_RE = re.compile(r"cycle_(\d+)_epoch_(\d+)\.parquet$")

#: Phase names used in the manifest. ``pre_dpo`` = samples/, ``dpo_eval`` = dpo_samples/,
#: ``validation`` = samples_validation/.
PHASE_PRE_DPO = "pre_dpo"
PHASE_DPO_EVAL = "dpo_eval"
PHASE_VALIDATION = "validation"


def build_training_samples_manifest(
    run_dir: str,
    *,
    train_judge: str | None = None,
    val_judges: list[str] | None = None,
    behavior_col: str = DEFAULT_BEHAVIOR_COL,
    response_col: str = DEFAULT_RESPONSE_COL,
    prompt_col: str = DEFAULT_PROMPT_COL,
) -> dict[str, Any] | None:
    """Enumerate the training-phase parquets under ``run_dir`` into a linkable manifest.

    Returns ``None`` when no training parquets exist (e.g. a transfer-only run), so the caller can
    simply skip attaching a manifest. Filenames drive the phase/cycle/epoch metadata; contents are
    not read here (cheap, and keeps this usable before scoring is finalised).
    """
    phases: list[dict[str, Any]] = []

    for path in sorted(glob.glob(os.path.join(run_dir, "samples", "cycle_*.parquet"))):
        m = _PRE_DPO_RE.search(os.path.basename(path))
        if m:
            phases.append({"phase": PHASE_PRE_DPO, "cycle": int(m.group(1)),
                           "path": os.path.relpath(path, run_dir)})

    for path in sorted(glob.glob(os.path.join(run_dir, "dpo_samples", "cycle_*_epoch_*.parquet"))):
        m = _DPO_EVAL_RE.search(os.path.basename(path))
        if m:
            phases.append({"phase": PHASE_DPO_EVAL, "cycle": int(m.group(1)), "epoch": int(m.group(2)),
                           "path": os.path.relpath(path, run_dir)})

    for path in sorted(glob.glob(os.path.join(run_dir, "samples_validation", "cycle_*.parquet"))):
        m = _PRE_DPO_RE.search(os.path.basename(path))
        if m:
            phases.append({"phase": PHASE_VALIDATION, "cycle": int(m.group(1)),
                           "path": os.path.relpath(path, run_dir)})

    if not phases:
        return None

    # judge -> column. Mirrors run.json's multi-judge ``scores`` dict so more judges are additive.
    score_columns: dict[str, str] = {}
    if train_judge:
        score_columns[str(train_judge)] = _TRAIN_SCORE_COL
    if val_judges:
        # Only one validation score column exists today; map the first val judge to it.
        score_columns[str(val_judges[0])] = _VAL_SCORE_COL
    if not score_columns:
        # No judge names supplied: expose the raw columns so the reader can still find scores.
        score_columns = {_TRAIN_SCORE_COL: _TRAIN_SCORE_COL, _VAL_SCORE_COL: _VAL_SCORE_COL}

    return {
        "run_dir": os.path.abspath(run_dir),
        "behavior_col": behavior_col,
        "response_col": response_col,
        "prompt_col": prompt_col,
        "score_columns": score_columns,
        "phases": phases,
    }


def get_training_manifest(run_json: dict[str, Any]) -> dict[str, Any] | None:
    """Dig the training-samples manifest out of a loaded ``run.json`` (or a bare metadata dict).

    Accepts either a full ``run.json`` payload (``{"runs": [{"resume_metadata": {...}}, ...]}``),
    a single ``SingleAttackRunResult`` dict, a ``resume_metadata`` dict, or a manifest itself.
    Returns ``None`` if no manifest is present.
    """
    if not isinstance(run_json, dict):
        return None
    # Already a manifest.
    if "phases" in run_json and "score_columns" in run_json:
        return run_json
    # A resume_metadata / provenance dict.
    iho = run_json.get("iho_training")
    if isinstance(iho, dict) and isinstance(iho.get("training_samples"), dict):
        return iho["training_samples"]
    # A single run dict.
    rm = run_json.get("resume_metadata")
    if isinstance(rm, dict):
        found = get_training_manifest(rm)
        if found is not None:
            return found
    # A full run.json payload.
    for run in run_json.get("runs", []) or []:
        if isinstance(run, dict):
            found = get_training_manifest(run)
            if found is not None:
                return found
    return None


def _resolve_score_column(manifest: dict[str, Any], judge: str | None) -> str:
    score_columns: dict[str, str] = manifest.get("score_columns", {}) or {}
    if judge is not None:
        if judge in score_columns:
            return score_columns[judge]
        # Allow passing a raw column name directly.
        if judge in score_columns.values():
            return judge
        raise KeyError(
            f"Judge {judge!r} not in manifest score_columns {sorted(score_columns)}; "
            f"pass one of those judge names or a column name."
        )
    # Default: prefer the train-judge column, else the first declared column.
    if _TRAIN_SCORE_COL in score_columns.values():
        return _TRAIN_SCORE_COL
    if score_columns:
        return next(iter(score_columns.values()))
    return _TRAIN_SCORE_COL


def load_training_score_sequences(
    manifest_or_run_json: dict[str, Any],
    *,
    judge: str | None = None,
    phases: list[str] | None = None,
    base_dir: str | None = None,
) -> dict[int, list[float]]:
    """Read the linked training parquets into ``{behavior_id: [scores...]}``.

    ``judge`` selects the score column via the manifest's judge->column map (or pass a column name
    directly; ``None`` uses the train-judge column). ``phases`` filters by phase name
    (``pre_dpo`` / ``dpo_eval`` / ``validation``); ``None`` means all. ``base_dir`` overrides where
    the parquets are read from (defaults to the manifest's ``run_dir``). Missing files/columns are
    skipped rather than raising, so a partially-written run still yields what exists.
    """
    manifest = get_training_manifest(manifest_or_run_json)
    if manifest is None:
        return {}

    score_col = _resolve_score_column(manifest, judge)
    behavior_col = manifest.get("behavior_col", DEFAULT_BEHAVIOR_COL)
    root = base_dir or manifest.get("run_dir") or ""
    wanted = set(phases) if phases is not None else None

    out: dict[int, list[float]] = {}
    for entry in manifest.get("phases", []) or []:
        if wanted is not None and entry.get("phase") not in wanted:
            continue
        path = entry.get("path")
        if not path:
            continue
        abs_path = path if os.path.isabs(path) else os.path.join(root, path)
        if not os.path.exists(abs_path):
            continue
        try:
            df = pd.read_parquet(abs_path, columns=[behavior_col, score_col])
        except Exception:
            # Column missing in this phase's parquet, or unreadable: skip it.
            continue
        for behavior_id, group in df.groupby(behavior_col):
            out.setdefault(int(behavior_id), []).extend(float(v) for v in group[score_col].to_numpy())
    return out
