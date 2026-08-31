"""Tests for adversariallm.training.flops.

The load-bearing check is `test_pass_flops_agrees_with_lm_utils_get_flops`: the ledger must
reproduce the number the rest of the repo already reports, or FLOPs stop being comparable
across attacks.
"""

import json
from dataclasses import dataclass

import pytest

from adversariallm.lm_utils.utils import get_flops
from adversariallm.training.flops import (
    COMPONENTS,
    GROUP_KEY_SEP,
    FlopsEntry,
    FlopsLedger,
    format_group_key,
    model_id_of,
    pass_flops,
)

PASS_TYPES = ["forward", "backward", "forward_and_backward"]


class FakeModel:
    """Duck-typed stand-in for a PreTrainedModel; `get_flops` only calls `num_parameters`."""

    def __init__(self, n_params: int):
        self._n_params = n_params

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        assert exclude_embeddings, "get_flops must exclude embeddings (plan §5.4)"
        return self._n_params


@dataclass
class FakeDenoiseResult:
    n_forward_passes: int
    n_tokens_per_pass: int


class FakeAttacker:
    def __init__(self, n_params: int, has_lora: bool = False, model_id: str = "fake/attacker"):
        self.n_params_no_embed = n_params
        self.has_lora = has_lora
        self.model_id = model_id


# --------------------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pass_type,expected_multiplier",
    [("forward", 2), ("backward", 4), ("forward_and_backward", 6)],
)
def test_pass_flops_dense_multipliers(pass_type, expected_multiplier):
    assert pass_flops(1_000, 7, pass_type) == expected_multiplier * 1_000 * 7


@pytest.mark.parametrize(
    "pass_type,expected_multiplier",
    [("forward", 2), ("backward", 2), ("forward_and_backward", 4)],
)
def test_pass_flops_lora_multipliers(pass_type, expected_multiplier):
    assert pass_flops(1_000, 7, pass_type, lora=True) == expected_multiplier * 1_000 * 7


@pytest.mark.parametrize("pass_type", PASS_TYPES)
def test_pass_flops_agrees_with_lm_utils_get_flops(pass_type):
    """lora=False must be bit-identical to the repo-wide estimator."""
    n_params, n_in, n_out = 6_738_415_616, 123, 456
    model = FakeModel(n_params)
    assert pass_flops(n_params, n_in + n_out, pass_type) == get_flops(model, n_in, n_out, pass_type)


def test_lora_only_changes_backward():
    n_params, n_tokens = 1_000, 10
    assert pass_flops(n_params, n_tokens, "forward", lora=True) == pass_flops(n_params, n_tokens, "forward")
    assert pass_flops(n_params, n_tokens, "backward", lora=True) * 2 == pass_flops(n_params, n_tokens, "backward")
    fwd_bwd = pass_flops(n_params, n_tokens, "forward_and_backward", lora=True)
    assert fwd_bwd == pass_flops(n_params, n_tokens, "forward") + pass_flops(n_params, n_tokens, "backward", lora=True)


def test_pass_flops_rejects_bad_input():
    with pytest.raises(ValueError, match="pass_type"):
        pass_flops(10, 10, "sideways")
    with pytest.raises(ValueError, match="n_params"):
        pass_flops(-1, 10, "forward")
    with pytest.raises(ValueError, match="n_tokens"):
        pass_flops(10, -1, "forward")


def test_pass_flops_is_exact_integer_arithmetic():
    value = pass_flops(6_738_415_616, 4_096, "forward_and_backward")
    assert isinstance(value, int)
    assert value == 6 * 6_738_415_616 * 4_096


# --------------------------------------------------------------------------------------
# ledger: add / validation
# --------------------------------------------------------------------------------------


def test_add_returns_entry_and_computes_flops():
    ledger = FlopsLedger()
    entry = ledger.add(
        component="attacker_policy",
        phase="cycle0:dpo_epoch3",
        model_id="attacker",
        n_params=1_000,
        n_tokens_in=8,
        n_tokens_out=2,
        pass_type="forward_and_backward",
        n_passes=3,
        lora=True,
        detail={"batch_size": 4},
    )
    assert isinstance(entry, FlopsEntry)
    assert entry.n_tokens == 10
    assert entry.flops == 3 * 4 * 1_000 * 10
    assert entry.detail == {"batch_size": 4}
    assert len(ledger) == 1
    assert ledger.total() == entry.flops


def test_add_defaults():
    ledger = FlopsLedger()
    entry = ledger.add(component="judge", phase="eval", model_id="j", n_params=5, n_tokens_in=4, pass_type="forward")
    assert (entry.n_tokens_out, entry.n_passes, entry.lora, entry.detail) == (0, 1, False, {})
    assert entry.flops == 2 * 5 * 4


def test_add_copies_detail_so_the_caller_cannot_mutate_the_ledger():
    ledger = FlopsLedger()
    detail = {"batch_size": 4}
    entry = ledger.add(
        component="judge", phase="eval", model_id="j", n_params=5, n_tokens_in=4, pass_type="forward", detail=detail
    )
    detail["batch_size"] = 999
    assert entry.detail == {"batch_size": 4}


def test_add_rejects_unknown_component():
    ledger = FlopsLedger()
    with pytest.raises(ValueError, match="Unknown component"):
        ledger.add(component="nope", phase="eval", model_id="m", n_params=1, n_tokens_in=1, pass_type="forward")


@pytest.mark.parametrize("component", COMPONENTS)
def test_add_accepts_every_declared_component(component):
    ledger = FlopsLedger()
    ledger.add(component=component, phase="eval", model_id="m", n_params=1, n_tokens_in=1, pass_type="forward")
    assert len(ledger) == 1


def test_add_rejects_negative_counts():
    ledger = FlopsLedger()
    base = dict(component="judge", phase="eval", model_id="m", n_params=1, pass_type="forward")
    with pytest.raises(ValueError, match="n_passes"):
        ledger.add(**base, n_tokens_in=1, n_passes=-1)
    with pytest.raises(ValueError, match="non-negative"):
        ledger.add(**base, n_tokens_in=-1)
    with pytest.raises(ValueError, match="non-negative"):
        ledger.add(**base, n_tokens_in=1, n_tokens_out=-3)


def test_add_rejects_phase_containing_the_group_separator():
    ledger = FlopsLedger()
    with pytest.raises(ValueError, match="must not contain"):
        ledger.add(
            component="judge",
            phase=f"cycle0{GROUP_KEY_SEP}pre",
            model_id="m",
            n_params=1,
            n_tokens_in=1,
            pass_type="forward",
        )


def test_zero_passes_is_allowed_and_costs_nothing():
    ledger = FlopsLedger()
    entry = ledger.add(
        component="judge", phase="eval", model_id="m", n_params=10, n_tokens_in=10, pass_type="forward", n_passes=0
    )
    assert entry.flops == 0


# --------------------------------------------------------------------------------------
# ledger: add_denoise
# --------------------------------------------------------------------------------------


def test_add_denoise_hand_computed():
    """4 sequences x 8 denoise steps, each a full forward over a 64-token canvas."""
    ledger = FlopsLedger()
    attacker = FakeAttacker(n_params=1_000_000)
    result = FakeDenoiseResult(n_forward_passes=8, n_tokens_per_pass=64)

    entry = ledger.add_denoise(phase="cycle0:pre", attacker=attacker, batch_size=4, denoise_result=result)

    assert entry.component == "attacker_denoise"
    assert entry.pass_type == "forward"
    assert entry.n_passes == 4 * 8
    assert entry.n_tokens_in == 64 and entry.n_tokens_out == 0
    assert entry.model_id == "fake/attacker"
    assert entry.flops == 4 * 8 * 2 * 1_000_000 * 64 == 4_096_000_000
    assert entry.detail == {"batch_size": 4, "n_denoise_steps": 8, "canvas_len": 64}


def test_add_denoise_is_not_one_forward_per_generated_token():
    """Regression guard: the diffusion cost is steps x full canvas, not tokens x prefix."""
    ledger = FlopsLedger()
    attacker = FakeAttacker(n_params=1_000)
    result = FakeDenoiseResult(n_forward_passes=4, n_tokens_per_pass=100)
    entry = ledger.add_denoise(phase="p", attacker=attacker, batch_size=1, denoise_result=result)
    assert entry.flops == 4 * 2 * 1_000 * 100
    # An AR model producing 100 tokens would be 2*N*100 in total, 4x cheaper here.
    assert entry.flops == 4 * pass_flops(1_000, 100, "forward")


def test_add_denoise_records_lora_provenance_without_changing_cost():
    ledger = FlopsLedger()
    result = FakeDenoiseResult(n_forward_passes=2, n_tokens_per_pass=16)
    kwargs = dict(phase="p", batch_size=1, denoise_result=result)
    plain = ledger.add_denoise(attacker=FakeAttacker(100, has_lora=False), **kwargs)
    lora = ledger.add_denoise(attacker=FakeAttacker(100, has_lora=True), **kwargs)
    assert (plain.lora, lora.lora) == (False, True)
    assert plain.flops == lora.flops  # a forward pass costs the same either way


def test_add_denoise_extra_detail_merges():
    ledger = FlopsLedger()
    entry = ledger.add_denoise(
        phase="p",
        attacker=FakeAttacker(100),
        batch_size=2,
        denoise_result=FakeDenoiseResult(2, 8),
        detail={"remasking": "low_confidence"},
    )
    assert entry.detail["remasking"] == "low_confidence"
    assert entry.detail["canvas_len"] == 8


def test_add_denoise_rejects_negative_sizes():
    ledger = FlopsLedger()
    with pytest.raises(ValueError, match="batch_size"):
        ledger.add_denoise(phase="p", attacker=FakeAttacker(1), batch_size=-1, denoise_result=FakeDenoiseResult(1, 1))
    with pytest.raises(ValueError, match="n_forward_passes"):
        ledger.add_denoise(phase="p", attacker=FakeAttacker(1), batch_size=1, denoise_result=FakeDenoiseResult(-1, 1))


def test_add_denoise_only_needs_the_duck_typed_attributes():
    class Minimal:
        n_params_no_embed = 7

    ledger = FlopsLedger()
    entry = ledger.add_denoise(phase="p", attacker=Minimal(), batch_size=1, denoise_result=FakeDenoiseResult(1, 1))
    assert entry.n_params == 7
    assert entry.model_id == "Minimal"  # falls back to the class name
    assert entry.lora is False


# --------------------------------------------------------------------------------------
# ledger: add_ar_generation
# --------------------------------------------------------------------------------------


def test_add_ar_generation_hand_computed():
    """Two sequences: 2*N*(3+7) + 2*N*(5+11) with N=100 gives 2000 + 3200 = 5200."""
    ledger = FlopsLedger()
    entry = ledger.add_ar_generation(
        phase="eval", model_id="target", n_params=100, prompt_tokens=[3, 5], new_tokens=[7, 11]
    )
    assert entry.component == "target_generate"
    assert (entry.n_tokens_in, entry.n_tokens_out, entry.n_passes) == (8, 18, 1)
    assert entry.flops == 5_200
    per_sequence = sum(2 * 100 * (p + g) for p, g in zip([3, 5], [7, 11]))
    assert entry.flops == per_sequence
    assert entry.detail["n_sequences"] == 2


def test_add_ar_generation_component_override():
    ledger = FlopsLedger()
    entry = ledger.add_ar_generation(
        phase="eval", model_id="j", n_params=10, prompt_tokens=[4], new_tokens=[0], component="judge"
    )
    assert entry.component == "judge"
    assert entry.flops == 2 * 10 * 4


def test_add_ar_generation_empty_batch():
    ledger = FlopsLedger()
    entry = ledger.add_ar_generation(phase="eval", model_id="t", n_params=10, prompt_tokens=[], new_tokens=[])
    assert entry.flops == 0
    assert entry.detail["n_sequences"] == 0


def test_add_ar_generation_validation():
    ledger = FlopsLedger()
    with pytest.raises(ValueError, match="equal length"):
        ledger.add_ar_generation(phase="p", model_id="t", n_params=1, prompt_tokens=[1, 2], new_tokens=[1])
    with pytest.raises(ValueError, match="non-negative"):
        ledger.add_ar_generation(phase="p", model_id="t", n_params=1, prompt_tokens=[-1], new_tokens=[1])
    with pytest.raises(ValueError, match="non-negative"):
        ledger.add_ar_generation(phase="p", model_id="t", n_params=1, prompt_tokens=[1], new_tokens=[-1])


# --------------------------------------------------------------------------------------
# ledger: aggregation
# --------------------------------------------------------------------------------------


def _populated_ledger() -> FlopsLedger:
    ledger = FlopsLedger()
    ledger.add(
        component="attacker_denoise", phase="cycle0:pre", model_id="a", n_params=10, n_tokens_in=5, pass_type="forward"
    )  # 100
    ledger.add(
        component="target_generate", phase="cycle0:pre", model_id="t", n_params=20, n_tokens_in=5, pass_type="forward"
    )  # 200
    ledger.add(component="judge", phase="eval", model_id="j", n_params=30, n_tokens_in=5, pass_type="forward")  # 300
    return ledger


def test_total_and_by():
    ledger = _populated_ledger()
    assert ledger.total() == 600
    assert ledger.by("phase") == {("cycle0:pre",): 300, ("eval",): 300}
    assert ledger.by("component") == {("attacker_denoise",): 100, ("target_generate",): 200, ("judge",): 300}
    assert ledger.by("phase", "component") == {
        ("cycle0:pre", "attacker_denoise"): 100,
        ("cycle0:pre", "target_generate"): 200,
        ("eval", "judge"): 300,
    }
    assert sum(ledger.by("phase", "component").values()) == ledger.total()


def test_by_keys_are_always_tuples():
    ledger = _populated_ledger()
    assert all(isinstance(key, tuple) for key in ledger.by("phase"))


def test_by_rejects_bad_keys():
    ledger = _populated_ledger()
    with pytest.raises(ValueError, match="at least one"):
        ledger.by()
    with pytest.raises(ValueError, match="Unknown grouping key"):
        ledger.by("not_a_field")
    with pytest.raises(ValueError, match="Unknown grouping key"):
        ledger.by("detail")  # unhashable, so it cannot be a group key


def test_empty_ledger():
    ledger = FlopsLedger()
    assert ledger.total() == 0
    assert ledger.by("phase") == {}
    assert ledger.to_records() == []
    assert len(ledger) == 0


def test_merge_extends_and_leaves_other_alone():
    left, right = _populated_ledger(), _populated_ledger()
    left.merge(right)
    assert len(left) == 6
    assert len(right) == 3
    assert left.total() == 1200


def test_snapshot_is_a_deep_copy():
    ledger = _populated_ledger()
    snap = ledger.snapshot()
    assert snap.total() == ledger.total()

    ledger.add(component="judge", phase="eval", model_id="j", n_params=1, n_tokens_in=1, pass_type="forward")
    assert len(snap) == 3 and len(ledger) == 4

    ledger.entries[0].detail["mutated"] = 1
    assert "mutated" not in snap.entries[0].detail


def test_entries_property_does_not_expose_the_internal_list():
    ledger = _populated_ledger()
    ledger.entries.append(ledger.entries[0])
    assert len(ledger) == 3


def test_snapshot_diff_gives_per_step_attribution():
    """The intended usage: total() before/after a step attributes FLOPs to that step."""
    ledger = _populated_ledger()
    before = ledger.snapshot().total()
    ledger.add_ar_generation(phase="eval", model_id="t", n_params=100, prompt_tokens=[3], new_tokens=[7])
    assert ledger.total() - before == 2 * 100 * 10


# --------------------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------------------


def test_to_records_allows_recomputation_under_the_dense_convention():
    ledger = FlopsLedger()
    ledger.add(
        component="attacker_policy",
        phase="cycle0:dpo_epoch0",
        model_id="a",
        n_params=1_000,
        n_tokens_in=10,
        pass_type="forward_and_backward",
        n_passes=2,
        lora=True,
    )
    (record,) = ledger.to_records()
    assert record["flops"] == 2 * 4 * 1_000 * 10
    n_tokens = record["n_tokens_in"] + record["n_tokens_out"]
    dense = record["n_passes"] * pass_flops(record["n_params"], n_tokens, record["pass_type"])
    assert dense == 2 * 6 * 1_000 * 10  # the entry carries everything needed to redo the model


def test_write_json_round_trip(tmp_path):
    ledger = _populated_ledger()
    path = tmp_path / "nested" / "flops.json"
    ledger.write_json(path)

    payload = json.loads(path.read_text())
    assert set(payload) == {"total", "by_phase", "by_component", "by_phase_component", "entries"}
    assert payload["total"] == 600
    assert payload["by_phase"] == {"cycle0:pre": 300, "eval": 300}
    assert payload["by_component"] == {"attacker_denoise": 100, "target_generate": 200, "judge": 300}
    assert payload["by_phase_component"]["cycle0:pre|attacker_denoise"] == 100
    assert len(payload["entries"]) == 3
    assert payload["entries"][0]["component"] == "attacker_denoise"
    assert sum(payload["by_component"].values()) == payload["total"]


def test_write_json_keys_are_sorted():
    ledger = _populated_ledger()
    text = json.dumps(
        {
            "total": ledger.total(),
            "by_phase": {format_group_key(k): v for k, v in ledger.by("phase").items()},
        },
        sort_keys=True,
    )
    assert text.index('"by_phase"') < text.index('"total"')


def test_write_json_is_stable_across_writes(tmp_path):
    ledger = _populated_ledger()
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    ledger.write_json(first)
    ledger.write_json(second)
    assert first.read_text() == second.read_text()


def test_format_group_key():
    assert format_group_key(("cycle0:pre", "judge")) == f"cycle0:pre{GROUP_KEY_SEP}judge"
    assert format_group_key(("eval",)) == "eval"


def test_model_id_of_fallback_chain():
    class WithCfg:
        cfg = type("Cfg", (), {"id": "org/model"})()

    class WithModel:
        model = type("M", (), {"name_or_path": "org/other"})()

    class Bare:
        pass

    assert model_id_of(FakeAttacker(1, model_id="explicit")) == "explicit"
    assert model_id_of(WithCfg()) == "org/model"
    assert model_id_of(WithModel()) == "org/other"
    assert model_id_of(Bare()) == "Bare"


def test_detail_accepts_ints_floats_strings_and_bools():
    ledger = FlopsLedger()
    entry = ledger.add(
        component="judge",
        phase="eval",
        model_id="j",
        n_params=1,
        n_tokens_in=1,
        pass_type="forward",
        detail={"n": 1, "ratio": 0.5, "name": "x", "flag": True},
    )
    assert entry.detail["ratio"] == 0.5


def test_entry_rejects_a_genuinely_wrong_type():
    with pytest.raises(Exception, match="pass_type"):
        FlopsEntry(
            component="judge",
            phase="eval",
            model_id="j",
            n_params=1,
            n_tokens_in=1,
            n_tokens_out=0,
            pass_type="sideways",
            n_passes=1,
            lora=False,
            flops=0,
        )


def test_entry_is_frozen():
    ledger = _populated_ledger()
    with pytest.raises(Exception):
        ledger.entries[0].flops = 0
