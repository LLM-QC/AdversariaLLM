"""Tests for adversariallm.training.run_logging.

wandb is deliberately *not* installed in this environment, so the no-wandb path is exercised
for real. The wandb-present path is exercised with a fake module injected into `sys.modules`,
which is what the lazy `import wandb` inside `_init_wandb` will pick up.
"""

import json
import sys
import types

import pytest

from adversariallm.training import run_logging
from adversariallm.training.run_logging import (
    CONFIG_FILENAME,
    METRICS_FILENAME,
    LoggingConfig,
    RunLogger,
    wandb_enabled,
)


def read_metrics(run_dir) -> list[dict]:
    path = run_dir / METRICS_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FakeSummary(dict):
    pass


class FakeRun:
    def __init__(self, **init_kwargs):
        self.init_kwargs = init_kwargs
        self.logged: list[tuple[dict, int | None]] = []
        self.summary = FakeSummary()
        self.finished = False

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    """Inject a stub `wandb` module and hand the test the runs it created."""
    module = types.ModuleType("wandb")
    module.runs = []

    def init(**kwargs):
        run = FakeRun(**kwargs)
        module.runs.append(run)
        return run

    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


@pytest.fixture(autouse=True)
def reset_import_warning(monkeypatch):
    monkeypatch.setattr(run_logging, "_import_warning_emitted", False)


# --------------------------------------------------------------------------------------
# metrics.jsonl is written unconditionally
# --------------------------------------------------------------------------------------


def test_jsonl_written_without_wandb(tmp_path, monkeypatch):
    """The whole point: no wandb installed, full metrics on disk anyway."""
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", None)  # forces ImportError on `import wandb`

    logger = RunLogger(LoggingConfig(mode="online"), str(tmp_path), {"lr": 1e-5})
    logger.log({"loss": 0.5})
    logger.log({"loss": 0.25})
    logger.finish()

    assert logger.mode == "disabled"  # degraded, but nothing was lost
    records = read_metrics(tmp_path)
    assert [r["loss"] for r in records] == [0.5, 0.25]
    assert [r["_step"] for r in records] == [0, 1]
    assert all(isinstance(r["_wall_time"], float) for r in records)


def test_wandb_is_importable_in_this_env():
    """wandb is now a declared dependency; the missing-wandb path is exercised by
    monkeypatching the import out (see test_missing_wandb_warns_once), not by its real
    absence."""
    import wandb  # noqa: F401

    assert wandb is not None


def test_missing_wandb_warns_once(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", None)
    with caplog.at_level("WARNING", logger=run_logging.__name__):
        RunLogger(LoggingConfig(mode="online"), str(tmp_path / "a"), {})
        RunLogger(LoggingConfig(mode="online"), str(tmp_path / "b"), {})
    assert sum("wandb is not installed" in r.message for r in caplog.records) == 1


def test_config_is_persisted_for_wandb_less_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("ADVERSARIALLM_WANDB", "off")
    RunLogger(LoggingConfig(mode="online"), str(tmp_path), {"beta": 0.25, "nested": {"a": 1}})
    assert json.loads((tmp_path / CONFIG_FILENAME).read_text()) == {"beta": 0.25, "nested": {"a": 1}}


def test_run_dir_is_created(tmp_path):
    run_dir = tmp_path / "does" / "not" / "exist"
    logger = RunLogger(LoggingConfig(), run_dir, {})
    logger.log({"a": 1})
    assert (run_dir / METRICS_FILENAME).exists()


def test_log_appends_across_loggers(tmp_path):
    """A resumed run must not truncate the metrics of the earlier one."""
    RunLogger(LoggingConfig(), str(tmp_path), {}).log({"a": 1})
    RunLogger(LoggingConfig(), str(tmp_path), {}).log({"a": 2})
    assert [r["a"] for r in read_metrics(tmp_path)] == [1, 2]


def test_explicit_step_and_monotonic_autoincrement(tmp_path):
    logger = RunLogger(LoggingConfig(), str(tmp_path), {})
    logger.log({"a": 1})  # step 0
    logger.log({"a": 2}, step=10)  # explicit
    logger.log({"a": 3})  # 11, not 2
    logger.log({"a": 4}, step=5)  # explicit, allowed
    logger.log({"a": 5})  # must not go backwards
    assert [r["_step"] for r in read_metrics(tmp_path)] == [0, 10, 11, 5, 12]


def test_summary_line(tmp_path):
    logger = RunLogger(LoggingConfig(), str(tmp_path), {})
    logger.log({"a": 1})
    logger.summary({"best_asr": 0.75})
    records = read_metrics(tmp_path)
    assert records[-1]["_summary"] is True
    assert records[-1]["best_asr"] == 0.75


def test_non_json_values_do_not_crash_a_run(tmp_path):
    import torch

    logger = RunLogger(LoggingConfig(), str(tmp_path), {})
    logger.log({"loss": torch.tensor(0.5), "vec": torch.tensor([1.0, 2.0]), "obj": object()})
    record = read_metrics(tmp_path)[0]
    assert record["loss"] == 0.5
    assert record["vec"] == [1.0, 2.0]
    assert isinstance(record["obj"], str)


def test_invalid_mode_rejected(tmp_path):
    cfg = LoggingConfig()
    cfg.mode = "verbose"  # bypasses the beartype check on __init__, mimicking a bad yaml value
    with pytest.raises(ValueError, match="Invalid logging mode"):
        RunLogger(cfg, str(tmp_path), {})


# --------------------------------------------------------------------------------------
# kill switch
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "off", "no", "false", "disabled", "OFF", " off "])
def test_kill_switch_values(monkeypatch, value):
    monkeypatch.setenv("ADVERSARIALLM_WANDB", value)
    assert wandb_enabled() is False


@pytest.mark.parametrize("value", ["on", "1", "true", ""])
def test_kill_switch_non_off_values(monkeypatch, value):
    monkeypatch.setenv("ADVERSARIALLM_WANDB", value)
    assert wandb_enabled() is True


def test_kill_switch_default_is_on(monkeypatch):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    assert wandb_enabled() is True


def test_kill_switch_prevents_wandb_init(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.setenv("ADVERSARIALLM_WANDB", "off")
    logger = RunLogger(LoggingConfig(mode="online"), str(tmp_path), {})
    logger.log({"a": 1})
    logger.finish()
    assert fake_wandb.runs == []
    assert logger.mode == "disabled"
    assert read_metrics(tmp_path)[0]["a"] == 1


def test_disabled_mode_never_touches_wandb(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    RunLogger(LoggingConfig(mode="disabled"), str(tmp_path), {}).log({"a": 1})
    assert fake_wandb.runs == []


# --------------------------------------------------------------------------------------
# wandb present
# --------------------------------------------------------------------------------------


def test_wandb_receives_config_and_never_a_hardcoded_entity(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    cfg = LoggingConfig(mode="offline", entity="my-team", project="my-proj", name="run-1", group="g", tags=["iho"])
    logger = RunLogger(cfg, str(tmp_path), {"beta": 0.25})
    logger.log({"loss": 1.0})
    logger.summary({"best": 2.0})
    logger.finish()

    (run,) = fake_wandb.runs
    assert run.init_kwargs["entity"] == "my-team"
    assert run.init_kwargs["project"] == "my-proj"
    assert run.init_kwargs["mode"] == "offline"
    assert run.init_kwargs["tags"] == ["iho"]
    assert run.init_kwargs["config"] == {"beta": 0.25}
    assert run.logged == [({"loss": 1.0}, 0)]
    assert run.summary == {"best": 2.0}
    assert run.finished
    # ... and the jsonl still has everything.
    assert [r.get("loss") for r in read_metrics(tmp_path)] == [1.0, None]


def test_wandb_init_failure_degrades(tmp_path, monkeypatch, fake_wandb, caplog):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)

    def boom(**kwargs):
        raise RuntimeError("no credentials")

    fake_wandb.init = boom
    with caplog.at_level("WARNING", logger=run_logging.__name__):
        logger = RunLogger(LoggingConfig(mode="online"), str(tmp_path), {})
    logger.log({"a": 1})
    assert logger.mode == "disabled"
    assert any("wandb.init failed" in r.message for r in caplog.records)
    assert read_metrics(tmp_path)[0]["a"] == 1


def test_wandb_log_failure_does_not_kill_the_run(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    logger = RunLogger(LoggingConfig(mode="offline"), str(tmp_path), {})

    def boom(metrics, step=None):
        raise RuntimeError("network down")

    fake_wandb.runs[0].log = boom
    logger.log({"a": 1})  # must not raise
    assert read_metrics(tmp_path)[0]["a"] == 1


# --------------------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------------------


def test_context_manager_finishes(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    with RunLogger(LoggingConfig(mode="offline"), str(tmp_path), {}) as logger:
        logger.log({"a": 1})
    assert fake_wandb.runs[0].finished


def test_context_manager_finishes_on_exception_and_propagates(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    logger = RunLogger(LoggingConfig(mode="offline"), str(tmp_path), {})
    with pytest.raises(RuntimeError, match="training blew up"):
        with logger:
            logger.log({"a": 1})
            raise RuntimeError("training blew up")
    assert fake_wandb.runs[0].finished
    assert read_metrics(tmp_path)[0]["a"] == 1


def test_finish_is_idempotent(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    logger = RunLogger(LoggingConfig(mode="offline"), str(tmp_path), {})
    logger.finish()
    logger.finish()
    with logger:
        pass
    assert sum(run.finished for run in fake_wandb.runs) == 1


def test_finish_failure_is_swallowed(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("ADVERSARIALLM_WANDB", raising=False)
    logger = RunLogger(LoggingConfig(mode="offline"), str(tmp_path), {})

    def boom():
        raise RuntimeError("wandb hung")

    fake_wandb.runs[0].finish = boom
    logger.finish()  # must not raise
