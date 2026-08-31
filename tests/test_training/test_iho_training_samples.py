"""CPU tests for linking/reading IHO training-phase attack samples (io_utils/iho_training_samples).

No GPU or real models: we fabricate the parquet layout an adaptive IHO run writes under its
``run_dir`` and check that the manifest links every phase and that the reader returns the right
per-behavior score sequences under phase/judge filtering.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adversariallm.io_utils.iho_training_samples import (
    PHASE_DPO_EVAL,
    PHASE_PRE_DPO,
    PHASE_VALIDATION,
    build_training_samples_manifest,
    get_training_manifest,
    load_training_score_sequences,
)


def _write(path, rows):
    """rows: list of (jb_index, judge_score_training, judge_score_validation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "jb_index": [r[0] for r in rows],
            "attacking_prompt_text": ["prompt" for _ in rows],
            "attacked_output": ["response" for _ in rows],
            "judge_score_training": [r[1] for r in rows],
            "judge_score_validation": [r[2] for r in rows],
        }
    )
    df.to_parquet(path, index=False)


@pytest.fixture
def run_dir(tmp_path):
    # cycle 0: pre-DPO + one dpo-eval epoch; cycle 1: pre-DPO + a validation set.
    _write(tmp_path / "samples" / "cycle_0.parquet",
           [(0, 0.10, 0.01), (0, 0.20, 0.02), (7, 0.90, 0.80)])
    _write(tmp_path / "dpo_samples" / "cycle_0_epoch_41.parquet",
           [(0, 0.55, 0.05), (7, 0.60, 0.50)])
    _write(tmp_path / "samples" / "cycle_1.parquet",
           [(0, 0.30, 0.03)])
    _write(tmp_path / "samples_validation" / "cycle_1.parquet",
           [(3, 0.42, 0.40)])
    return tmp_path


def test_manifest_enumerates_every_phase_with_metadata(run_dir):
    m = build_training_samples_manifest(str(run_dir), train_judge="strong_reject", val_judges=["harmbench"])
    assert m is not None
    assert m["run_dir"] == str(run_dir.resolve()) or m["run_dir"] == str(run_dir)
    # judge -> column map (mirrors run.json multi-judge scores; additive for future judges).
    assert m["score_columns"] == {"strong_reject": "judge_score_training", "harmbench": "judge_score_validation"}

    phases = m["phases"]
    kinds = sorted((p["phase"], p.get("cycle"), p.get("epoch")) for p in phases)
    assert kinds == sorted([
        (PHASE_PRE_DPO, 0, None),
        (PHASE_PRE_DPO, 1, None),
        (PHASE_DPO_EVAL, 0, 41),
        (PHASE_VALIDATION, 1, None),
    ])
    # Paths are stored relative to run_dir.
    assert all(not p["path"].startswith("/") for p in phases)


def test_manifest_is_none_without_parquets(tmp_path):
    assert build_training_samples_manifest(str(tmp_path)) is None


def test_reader_all_phases_train_judge(run_dir):
    m = build_training_samples_manifest(str(run_dir), train_judge="strong_reject", val_judges=["harmbench"])
    seqs = load_training_score_sequences(m, judge="strong_reject")
    # behavior 0: pre-DPO c0 [.10,.20] + dpo c0 [.55] + pre-DPO c1 [.30]  (validation is behavior 3)
    assert sorted(seqs[0]) == [0.10, 0.20, 0.30, 0.55]
    assert sorted(seqs[7]) == [0.60, 0.90]
    assert seqs[3] == [0.42]


def test_reader_phase_filter(run_dir):
    m = build_training_samples_manifest(str(run_dir), train_judge="strong_reject", val_judges=["harmbench"])
    only_dpo = load_training_score_sequences(m, judge="strong_reject", phases=[PHASE_DPO_EVAL])
    assert sorted(only_dpo[0]) == [0.55]
    assert sorted(only_dpo[7]) == [0.60]
    assert 3 not in only_dpo  # validation-only behavior absent from dpo-eval phase


def test_reader_judge_selects_the_right_column(run_dir):
    m = build_training_samples_manifest(str(run_dir), train_judge="strong_reject", val_judges=["harmbench"])
    val = load_training_score_sequences(m, judge="harmbench", phases=[PHASE_PRE_DPO])
    # pre-DPO behavior 7 has judge_score_validation 0.80
    assert val[7] == [0.80]


def test_reader_defaults_to_train_column_and_reads_via_run_json(run_dir):
    m = build_training_samples_manifest(str(run_dir), train_judge="strong_reject", val_judges=["harmbench"])
    # Embed the manifest exactly where run.json puts it, then read straight from the payload.
    run_json = {"runs": [{"resume_metadata": {"iho_training": {"training_samples": m}}}]}
    assert get_training_manifest(run_json) == m
    seqs = load_training_score_sequences(run_json)  # judge=None -> train column
    assert sorted(seqs[0]) == [0.10, 0.20, 0.30, 0.55]


def test_reader_unknown_judge_raises(run_dir):
    m = build_training_samples_manifest(str(run_dir), train_judge="strong_reject", val_judges=["harmbench"])
    with pytest.raises(KeyError):
        load_training_score_sequences(m, judge="not_a_judge")


def test_get_manifest_none_for_transfer_run():
    # A transfer run has resume_metadata without an iho_training manifest.
    assert get_training_manifest({"runs": [{"resume_metadata": None, "steps": []}]}) is None
