"""Tests for adversariallm.training.preference.

The expected pairings below are worked out by hand from `aapl/datasets/PreferenceDataset.py`
(`ThresholdedTopKBottomKPairing.generate` + `_build_from_run_path`), not from our port:

    bottom_k  = sorted(rejected, by score asc)[:k]
    top_k     = sorted([s for s in chosen if score >= min_score], by score asc)[-k:]
    pair_k    = min(len(top_k), len(bottom_k))
    pairs     = zip(reversed(top_k[-pair_k:]), bottom_k[:pair_k])

`_aapl_reference_pairs` is a literal transcription of that source, used as a differential
oracle on randomised inputs.
"""

import logging
import random

import pandas as pd
import pytest

from adversariallm.training.preference import (
    BEHAVIOR_COL,
    CYCLE_COL,
    DEFAULT_SCORE_COL,
    FULL_TEXT_COL,
    PAIR_COLUMNS,
    PROMPT_COL,
    PairingConfig,
    PairingStrategy,
    PreferenceDataset,
    PreferencePair,
    ScoredSample,
    ThresholdedTopKBottomKPairing,
    build_preference_pairs,
    dataframe_to_samples,
)


def make_df(scores, *, cycle=0, behavior=0, prompt="goal-A", tag=None) -> pd.DataFrame:
    """Sample frame with one row per score; `full_text` is a unique marker per row."""
    tag = tag if tag is not None else f"c{cycle}b{behavior}"
    return pd.DataFrame(
        {
            BEHAVIOR_COL: [behavior] * len(scores),
            PROMPT_COL: [prompt] * len(scores),
            FULL_TEXT_COL: [f"{tag}_t{i}" for i in range(len(scores))],
            DEFAULT_SCORE_COL: list(scores),
            CYCLE_COL: [cycle] * len(scores),
        }
    )


def empty_df() -> pd.DataFrame:
    return make_df([])


# --------------------------------------------------------------------------------------
# the pairing rule, hand-worked against the AAPL source
# --------------------------------------------------------------------------------------


def test_exact_pairing_order_hand_worked():
    """8 samples, percent_chosen=0.5 -> k=4, min_score=0.2.

    scores by row:      t0=0.9 t1=0.1 t2=0.5 t3=0.3 t4=0.7 t5=0.05 t6=0.4 t7=0.6

    rejected asc:       t5(.05) t1(.10) t3(.30) t6(.40) t2(.50) t7(.60) t4(.70) t0(.90)
    bottom_k (k=4):     t5 t1 t3 t6
    chosen >= 0.2:      t0 t2 t3 t4 t6 t7
    chosen asc:         t3(.30) t6(.40) t2(.50) t7(.60) t4(.70) t0(.90)
    top_k (last 4):     t2 t7 t4 t0
    pair_k = 4          reversed(top_k) = t0 t4 t7 t2   vs   bottom_k = t5 t1 t3 t6
    """
    scores = [0.9, 0.1, 0.5, 0.3, 0.7, 0.05, 0.4, 0.6]
    pairs = build_preference_pairs(
        make_df(scores, tag="x"), [], PairingConfig(min_score=0.2, percent_chosen=0.5, expanding=False)
    )

    assert [(p.chosen, p.rejected) for p in pairs] == [
        ("x_t0", "x_t5"),
        ("x_t4", "x_t1"),
        ("x_t7", "x_t3"),
        ("x_t2", "x_t6"),
    ]
    assert [p.chosen_score for p in pairs] == [0.9, 0.7, 0.6, 0.5]
    assert [p.rejected_score for p in pairs] == [0.05, 0.1, 0.3, 0.4]
    # A sample may be chosen in one pair and rejected in another -- t3 is, exactly as in AAPL.
    assert "x_t3" in {p.rejected for p in pairs} and 0.3 >= 0.2


def test_pair_carries_behavior_prompt_and_cycles_from_the_chosen_sample():
    current = make_df([0.9, 0.1], behavior=7, prompt="how to X", cycle=3, tag="c")
    pairs = build_preference_pairs(current, [], PairingConfig(min_score=0.2, percent_chosen=0.5))
    (pair,) = pairs
    assert isinstance(pair, PreferencePair)
    assert (pair.behavior_idx, pair.prompt) == (7, "how to X")
    assert (pair.chosen, pair.rejected) == ("c_t0", "c_t1")
    assert (pair.chosen_cycle, pair.rejected_cycle) == (3, 3)


def test_best_chosen_is_matched_with_worst_rejected():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pairs = build_preference_pairs(make_df(scores, tag="m"), [], PairingConfig(min_score=0.0, percent_chosen=0.5))
    assert [p.chosen_score for p in pairs] == [0.8, 0.7, 0.6, 0.5]
    assert [p.rejected_score for p in pairs] == [0.1, 0.2, 0.3, 0.4]


def test_ties_follow_row_order_because_the_sort_is_stable():
    """All-equal scores: sorted() is stable, so the result is row order, not arbitrary."""
    pairs = build_preference_pairs(
        make_df([0.5, 0.5, 0.5, 0.5], tag="e"), [], PairingConfig(min_score=0.2, percent_chosen=0.5)
    )
    # bottom_2 = [t0, t1]; top_2 = [t2, t3]; reversed -> [t3, t2]
    assert [(p.chosen, p.rejected) for p in pairs] == [("e_t3", "e_t0"), ("e_t2", "e_t1")]


# --------------------------------------------------------------------------------------
# per-group k arithmetic
# --------------------------------------------------------------------------------------


def test_k_is_computed_per_group_from_the_chosen_pool_size():
    """8-sample group -> k=2, 4-sample group -> k=1, at percent_chosen=0.25."""
    big = make_df([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], behavior=0, prompt="A", tag="A")
    small = make_df([0.9, 0.8, 0.7, 0.6], behavior=1, prompt="B", tag="B")
    current = pd.concat([big, small], ignore_index=True)

    pairs = build_preference_pairs(current, [], PairingConfig(min_score=0.0, percent_chosen=0.25))
    per_behavior = {0: [], 1: []}
    for pair in pairs:
        per_behavior[pair.behavior_idx].append(pair)
    assert len(per_behavior[0]) == 2
    assert len(per_behavior[1]) == 1


def test_grouping_is_by_behavior_and_prompt_together():
    """Same behavior index, two different prompt texts -> two independent groups."""
    a = make_df([0.9, 0.1], behavior=0, prompt="A", tag="A")
    b = make_df([0.8, 0.2], behavior=0, prompt="B", tag="B")
    pairs = build_preference_pairs(
        pd.concat([a, b], ignore_index=True), [], PairingConfig(min_score=0.0, percent_chosen=0.5)
    )
    assert [(p.prompt, p.chosen, p.rejected) for p in pairs] == [("A", "A_t0", "A_t1"), ("B", "B_t0", "B_t1")]


def test_k_zero_yields_no_pairs():
    """int(4 * 0.125) == 0, so a small group contributes nothing."""
    pairs = build_preference_pairs(make_df([0.9, 0.8, 0.7, 0.6]), [], PairingConfig(percent_chosen=0.125))
    assert pairs == []


def test_k_is_truncated_not_rounded():
    """int(5 * 0.5) == 2, not 3."""
    pairs = build_preference_pairs(
        make_df([0.9, 0.8, 0.7, 0.6, 0.5]), [], PairingConfig(min_score=0.0, percent_chosen=0.5)
    )
    assert len(pairs) == 2


# --------------------------------------------------------------------------------------
# expanding
# --------------------------------------------------------------------------------------


def _expanding_frames():
    previous = make_df([0.99], cycle=0, tag="prev")
    current = make_df([0.5, 0.3, 0.1], cycle=1, tag="cur")
    return previous, current


def test_expanding_true_pools_previous_cycles_into_the_chosen_pool():
    """chosen = prev(0.99) + cur(0.5,0.3,0.1) -> k=int(4*0.5)=2; rejected = current only."""
    previous, current = _expanding_frames()
    pairs = build_preference_pairs(
        current, [previous], PairingConfig(min_score=0.2, percent_chosen=0.5, expanding=True)
    )
    assert [(p.chosen, p.rejected) for p in pairs] == [("prev_t0", "cur_t2"), ("cur_t0", "cur_t1")]
    assert [(p.chosen_cycle, p.rejected_cycle) for p in pairs] == [(0, 1), (1, 1)]


def test_expanding_false_uses_the_current_cycle_only():
    """chosen = cur(0.5,0.3,0.1) -> k=int(3*0.5)=1."""
    previous, current = _expanding_frames()
    pairs = build_preference_pairs(
        current, [previous], PairingConfig(min_score=0.2, percent_chosen=0.5, expanding=False)
    )
    assert [(p.chosen, p.rejected) for p in pairs] == [("cur_t0", "cur_t2")]
    assert [(p.chosen_cycle, p.rejected_cycle) for p in pairs] == [(1, 1)]


def test_previous_frames_are_ignored_when_not_expanding():
    _, current = _expanding_frames()
    cfg = PairingConfig(min_score=0.2, percent_chosen=0.5, expanding=False)
    many_previous = [make_df([0.99] * 20, cycle=0, tag="p")]
    assert build_preference_pairs(current, many_previous, cfg) == build_preference_pairs(current, [], cfg)


def test_rejected_pool_never_includes_previous_cycles():
    """A previous cycle full of terrible samples must not become the rejected side."""
    previous = make_df([0.0] * 8, cycle=0, tag="prev")
    current = make_df([0.9, 0.8], cycle=1, tag="cur")
    pairs = build_preference_pairs(
        current, [previous], PairingConfig(min_score=0.2, percent_chosen=0.5, expanding=True)
    )
    assert all(p.rejected.startswith("cur") for p in pairs)
    assert all(p.rejected_cycle == 1 for p in pairs)


# --------------------------------------------------------------------------------------
# min_score filtering and degenerate cases
# --------------------------------------------------------------------------------------


def test_min_score_filters_the_chosen_pool_only():
    """k still comes from the unfiltered chosen pool; the filter only shrinks top_k."""
    # 8 samples -> k=4, but only 2 clear the threshold.
    scores = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    pairs = build_preference_pairs(make_df(scores, tag="f"), [], PairingConfig(min_score=0.5, percent_chosen=0.5))
    assert [(p.chosen, p.rejected) for p in pairs] == [("f_t0", "f_t2"), ("f_t1", "f_t3")]


def test_min_score_is_inclusive():
    pairs = build_preference_pairs(make_df([0.2, 0.0], tag="i"), [], PairingConfig(min_score=0.2, percent_chosen=0.5))
    assert [p.chosen_score for p in pairs] == [0.2]


def test_no_chosen_above_threshold_yields_no_pairs(caplog):
    with caplog.at_level(logging.WARNING, logger="adversariallm.training.preference"):
        pairs = build_preference_pairs(
            make_df([0.1, 0.05, 0.0, 0.01]), [], PairingConfig(min_score=0.9, percent_chosen=0.5)
        )
    assert pairs == []
    assert any("zero preference pairs" in r.message for r in caplog.records)


def test_empty_current_frame(caplog):
    with caplog.at_level(logging.WARNING, logger="adversariallm.training.preference"):
        pairs = build_preference_pairs(empty_df(), [], PairingConfig())
    assert pairs == []
    assert any("zero preference pairs" in r.message for r in caplog.records)


def test_empty_pools_in_the_strategy():
    strategy = ThresholdedTopKBottomKPairing(min_score=0.2)
    sample = ScoredSample(behavior_idx=0, prompt="p", full_text="t", score=0.9, cycle=0)
    assert strategy.generate(chosen_pool=[], rejected_pool=[sample], k=2) == []
    assert strategy.generate(chosen_pool=[sample], rejected_pool=[], k=2) == []
    assert strategy.generate(chosen_pool=[sample], rejected_pool=[sample], k=0) == []
    assert strategy.generate(chosen_pool=[sample], rejected_pool=[sample], k=-1) == []


def test_pair_k_clamps_to_the_smaller_pool():
    """Chosen pool of 5 (k=2 from it) but only 1 rejected sample -> a single pair."""
    strategy = ThresholdedTopKBottomKPairing(min_score=0.0)
    chosen = [ScoredSample(0, "p", f"c{i}", 0.1 * i, 0) for i in range(5)]
    rejected = [ScoredSample(0, "p", "r0", 0.01, 0)]
    pairs = strategy.generate(chosen_pool=chosen, rejected_pool=rejected, k=2)
    assert [(p.chosen, p.rejected) for p in pairs] == [("c4", "r0")]


def test_single_sample_group_produces_nothing():
    """int(1 * 0.5) == 0."""
    assert build_preference_pairs(make_df([0.9]), [], PairingConfig(min_score=0.0, percent_chosen=0.5)) == []


def test_percent_chosen_one_pairs_the_whole_group():
    pairs = build_preference_pairs(make_df([0.9, 0.5], tag="w"), [], PairingConfig(min_score=0.0, percent_chosen=1.0))
    assert [(p.chosen, p.rejected) for p in pairs] == [("w_t0", "w_t1"), ("w_t1", "w_t0")]


def test_behavior_in_rejected_but_not_in_chosen_is_skipped():
    """Unreachable from build_preference_pairs (rejected is a subset of chosen), so the
    strategy contract is checked directly."""
    strategy = ThresholdedTopKBottomKPairing(min_score=0.2)
    rejected = [ScoredSample(3, "unseen", "r", 0.1, 0)]
    assert strategy.generate(chosen_pool=[], rejected_pool=rejected, k=1) == []


# --------------------------------------------------------------------------------------
# DataFrame plumbing
# --------------------------------------------------------------------------------------


def test_score_col_is_a_parameter():
    df = make_df([0.9, 0.1], tag="s").rename(columns={DEFAULT_SCORE_COL: "judge_score_validation"})
    pairs = build_preference_pairs(
        df, [], PairingConfig(min_score=0.2, percent_chosen=0.5), score_col="judge_score_validation"
    )
    assert [(p.chosen, p.rejected) for p in pairs] == [("s_t0", "s_t1")]


def test_missing_columns_raise():
    df = make_df([0.9]).drop(columns=[CYCLE_COL])
    with pytest.raises(KeyError, match=CYCLE_COL):
        dataframe_to_samples(df)
    with pytest.raises(KeyError, match="judge_score_missing"):
        dataframe_to_samples(make_df([0.9]), score_col="judge_score_missing")


def test_dataframe_to_samples_casts_types():
    df = make_df([1, 0])  # ints in a float column
    samples = dataframe_to_samples(df)
    assert all(isinstance(s.score, float) and isinstance(s.behavior_idx, int) for s in samples)
    assert samples[0] == ScoredSample(behavior_idx=0, prompt="goal-A", full_text="c0b0_t0", score=1.0, cycle=0)


def test_config_accepts_ints_where_floats_are_annotated():
    """`min_score: 0` in a yaml config must not be a type error (PEP 484 numeric tower)."""
    cfg = PairingConfig(min_score=0, percent_chosen=1)
    assert build_preference_pairs(make_df([0.9, 0.1], tag="n"), [], cfg) != []


def test_config_still_rejects_genuinely_wrong_types():
    with pytest.raises(Exception, match="min_score"):
        PairingConfig(min_score="high")


def test_custom_strategy_is_used_polymorphically():
    """AAPL rejected every strategy but its own with an isinstance check; ours must not."""

    class TakeNothing(PairingStrategy):
        def __init__(self):
            self.calls = 0

        def generate(self, *, chosen_pool, rejected_pool, k):
            self.calls += 1
            return []

    strategy = TakeNothing()
    pairs = build_preference_pairs(make_df([0.9, 0.1]), [], PairingConfig(), strategy=strategy)
    assert pairs == [] and strategy.calls == 1


# --------------------------------------------------------------------------------------
# PreferenceDataset
# --------------------------------------------------------------------------------------


def test_dataset_yields_chosen_rejected_dicts():
    dataset = PreferenceDataset.from_dataframes(
        make_df([0.9, 0.1], tag="d"), cfg=PairingConfig(min_score=0.2, percent_chosen=0.5)
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert item["chosen"] == "d_t0" and item["rejected"] == "d_t1"
    assert set(item) == set(PAIR_COLUMNS)


def test_dataset_collates_into_lists_of_strings():
    from torch.utils.data import DataLoader

    dataset = PreferenceDataset.from_dataframes(
        make_df([0.9, 0.8, 0.2, 0.1], tag="b"), cfg=PairingConfig(min_score=0.2, percent_chosen=0.5)
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch["chosen"] == ["b_t0", "b_t1"]
    assert batch["rejected"] == ["b_t3", "b_t2"]


def test_empty_dataset_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="adversariallm.training.preference"):
        dataset = PreferenceDataset([])
    assert len(dataset) == 0
    assert any("zero preference pairs" in r.message for r in caplog.records)


def test_parquet_round_trip(tmp_path):
    dataset = PreferenceDataset.from_dataframes(
        make_df([0.9, 0.8, 0.2, 0.1], behavior=5, prompt="goal", tag="r"),
        cfg=PairingConfig(min_score=0.2, percent_chosen=0.5),
    )
    path = tmp_path / "prefs" / "cycle_0.parquet"
    dataset.to_parquet(path)

    assert list(pd.read_parquet(path).columns) == list(PAIR_COLUMNS)
    restored = PreferenceDataset.from_parquet(path)
    assert restored.pairs == dataset.pairs


def test_empty_parquet_round_trip(tmp_path):
    path = tmp_path / "empty.parquet"
    PreferenceDataset([]).to_parquet(path)
    assert list(pd.read_parquet(path).columns) == list(PAIR_COLUMNS)
    assert PreferenceDataset.from_parquet(path).pairs == []


def test_from_parquet_rejects_a_wrong_schema(tmp_path):
    path = tmp_path / "wrong.parquet"
    pd.DataFrame({"jb_index": [1], "chosen": ["a"]}).to_parquet(path, index=False)
    with pytest.raises(KeyError, match="behavior_idx"):
        PreferenceDataset.from_parquet(path)


def test_to_dataframe_column_order():
    dataset = PreferenceDataset.from_dataframes(
        make_df([0.9, 0.1]), cfg=PairingConfig(min_score=0.2, percent_chosen=0.5)
    )
    assert list(dataset.to_dataframe().columns) == list(PAIR_COLUMNS)


# --------------------------------------------------------------------------------------
# differential test against a transcription of the AAPL source
# --------------------------------------------------------------------------------------


def _aapl_reference_pairs(rows, previous_rows, *, min_score, percent_chosen, expanding):
    """Literal transcription of aapl/datasets/PreferenceDataset.py, operating on dicts.

    Rows are dicts with AAPL's own key names. Returns AAPL-shaped pair dicts.
    """
    from collections import defaultdict

    current_samples = list(rows)
    if expanding:
        previous_samples = [s for df in previous_rows for s in df]
        chosen_pool = previous_samples + current_samples
    else:
        chosen_pool = current_samples
    rejected_pool = current_samples

    grouped_chosen = defaultdict(list)
    grouped_rejected = defaultdict(list)
    for s in chosen_pool:
        grouped_chosen[(s["jb_index"], s["original_prompt_text"])].append(s)
    for s in rejected_pool:
        grouped_rejected[(s["jb_index"], s["original_prompt_text"])].append(s)

    preference_pairs = []
    for key in grouped_rejected.keys():
        chosen_samples = grouped_chosen.get(key, [])
        rejected_samples = grouped_rejected[key]
        k = int(len(chosen_samples) * percent_chosen)
        if not chosen_samples or not rejected_samples:
            continue

        if k <= 0 or not rejected_samples:
            continue
        rejected_sorted = sorted(rejected_samples, key=lambda s: s["judge_score_training"])
        bottom_k = rejected_sorted[:k]
        chosen_candidates = [s for s in chosen_samples if s["judge_score_training"] >= min_score]
        if not chosen_candidates:
            continue
        chosen_sorted = sorted(chosen_candidates, key=lambda s: s["judge_score_training"])
        top_k = chosen_sorted[-k:]
        pair_k = min(len(top_k), len(bottom_k))
        if pair_k == 0:
            continue
        for chosen, rejected in zip(reversed(top_k[-pair_k:]), bottom_k[:pair_k]):
            preference_pairs.append(
                {
                    "jb_index": chosen["jb_index"],
                    "prompt": chosen["original_prompt_text"],
                    "chosen": chosen["inpainted_prompt_text_full"],
                    "rejected": rejected["inpainted_prompt_text_full"],
                    "chosen_judge_score": chosen["judge_score_training"],
                    "rejected_judge_score": rejected["judge_score_training"],
                    "chosen_cycle_id": chosen["cycle_id"],
                    "rejected_cycle_id": rejected["cycle_id"],
                }
            )
    return preference_pairs


def _random_frame(rng, cycle, n_behaviors, n_per_behavior):
    rows = []
    for behavior in range(n_behaviors):
        for i in range(n_per_behavior):
            rows.append(
                {
                    "jb_index": behavior,
                    "original_prompt_text": f"goal-{behavior}",
                    "inpainted_prompt_text_full": f"c{cycle}b{behavior}s{i}",
                    "judge_score_training": round(rng.choice([0.0, 0.1, 0.2, 0.5, 0.5, 0.9, 1.0]), 3),
                    "cycle_id": cycle,
                }
            )
    return rows


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("expanding", [False, True])
def test_matches_the_aapl_reference_implementation(seed, expanding):
    rng = random.Random(seed)
    min_score = rng.choice([0.0, 0.2, 0.5, 0.95])
    percent_chosen = rng.choice([0.125, 0.25, 0.5, 1.0])
    n_behaviors = rng.randint(1, 3)
    n_per_behavior = rng.randint(1, 9)
    n_previous = rng.randint(0, 2)

    previous_rows = [_random_frame(rng, c, n_behaviors, n_per_behavior) for c in range(n_previous)]
    current_rows = _random_frame(rng, n_previous, n_behaviors, n_per_behavior)

    expected = _aapl_reference_pairs(
        current_rows, previous_rows, min_score=min_score, percent_chosen=percent_chosen, expanding=expanding
    )
    actual = build_preference_pairs(
        pd.DataFrame(current_rows),
        [pd.DataFrame(rows) for rows in previous_rows],
        PairingConfig(min_score=min_score, percent_chosen=percent_chosen, expanding=expanding),
    )

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.behavior_idx == want["jb_index"]
        assert got.prompt == want["prompt"]
        assert got.chosen == want["chosen"]
        assert got.rejected == want["rejected"]
        assert got.chosen_score == pytest.approx(want["chosen_judge_score"])
        assert got.rejected_score == pytest.approx(want["rejected_judge_score"])
        assert got.chosen_cycle == want["chosen_cycle_id"]
        assert got.rejected_cycle == want["rejected_cycle_id"]
