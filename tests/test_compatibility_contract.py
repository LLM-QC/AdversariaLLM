from __future__ import annotations

from adversariallm.attacks.attack import AttackStepResult
from adversariallm.io_utils.config import RunConfig


def test_attack_step_result_backwards_compatible_old_payload():
    old_step = {"step": 0, "model_completions": ["ok"]}
    step = AttackStepResult(**old_step)
    assert step.model_completions == ["ok"]
    assert step.model_completions_raw is None
    assert step.defense_metadata is None


def test_run_config_without_schema_version_fields():
    rc = RunConfig(
        model="m",
        dataset="d",
        attack="a",
        defense=None,
        model_params={},
        dataset_params={},
        attack_params={},
    )
    assert "config_schema_version" not in rc.__dict__
    assert "defense_schema_version" not in rc.__dict__
