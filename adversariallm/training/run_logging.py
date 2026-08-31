"""Run logging for attacker training: ``metrics.jsonl`` always, wandb optionally.

The jsonl file is the reproducibility path. It is written unconditionally, one JSON object per
:meth:`RunLogger.log` call, so a run can be analysed in full by someone with no wandb account --
and by us later, once the wandb project has been moved, expired or deleted. wandb is a strictly
optional mirror: the import is lazy, a missing package degrades to ``mode="disabled"`` with a
warning, and ``ADVERSARIALLM_WANDB=off`` turns it off globally (mirroring ``ADVERSARIALLM_DB``,
see ``io_utils/database.py::db_enabled``).

Entity and project always come from :class:`LoggingConfig`; nothing here may be hardcoded (the
ported AAPL trainer hardcoded ``entity="limbach"``, which is exactly the bug this fixes).
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beartype import beartype
from beartype.typing import Literal

logger = logging.getLogger(__name__)

#: Values of ``ADVERSARIALLM_WANDB`` that turn the wandb integration off. Kept identical to
#: ``io_utils.database._DB_OFF_VALUES`` so the two kill switches behave the same way.
_WANDB_OFF_VALUES = {"0", "off", "no", "false", "disabled"}

#: Always-written metrics file inside the run directory (JSON Lines, appended to).
METRICS_FILENAME = "metrics.jsonl"

#: The ``config`` dict handed to :class:`RunLogger` is also written here, so a disabled-wandb run
#: still records what it was configured with.
CONFIG_FILENAME = "logger_config.json"

MODES = ("online", "offline", "disabled")

_import_warning_emitted = False


def wandb_enabled() -> bool:
    """Whether the wandb integration is allowed to run at all.

    Set ``ADVERSARIALLM_WANDB=off`` on machines without network access or wandb credentials.
    Metrics are still written to ``metrics.jsonl`` in the run directory.
    """
    return os.environ.get("ADVERSARIALLM_WANDB", "on").strip().lower() not in _WANDB_OFF_VALUES


@beartype
@dataclass
class LoggingConfig:
    mode: Literal["online", "offline", "disabled"] = "disabled"
    entity: str | None = None
    project: str = "llm-quick-check-iho"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)


class RunLogger:
    """Writes ``metrics.jsonl`` into ``run_dir``; talks to wandb only when ``mode != "disabled"``.

    ``self.mode`` is the *effective* mode after the kill switch and the wandb import have been
    resolved, and may therefore differ from ``cfg.mode``.
    """

    def __init__(self, cfg: LoggingConfig, run_dir: str | Path, config: dict) -> None:
        if cfg.mode not in MODES:
            raise ValueError(f"Invalid logging mode {cfg.mode!r}. Valid: {list(MODES)}")

        self.cfg = cfg
        self.run_dir = str(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, METRICS_FILENAME)
        self.config_path = os.path.join(self.run_dir, CONFIG_FILENAME)

        self._step = 0
        self._finished = False
        self._run: Any = None

        self._write_config(config)

        self.mode = cfg.mode
        if self.mode != "disabled" and not wandb_enabled():
            logger.info("wandb disabled via ADVERSARIALLM_WANDB; metrics still go to %s", self.metrics_path)
            self.mode = "disabled"
        if self.mode != "disabled":
            self._run = self._init_wandb(config)
            if self._run is None:
                self.mode = "disabled"

    def _write_config(self, config: dict) -> None:
        """Persist the run config next to the metrics so a wandb-less run is self-describing."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2, sort_keys=True, default=_json_default)
        except Exception as exc:  # noqa: BLE001 - logging must never kill a training run
            logger.warning("Could not write %s (%s); continuing.", self.config_path, exc)

    def _init_wandb(self, config: dict) -> Any:
        """Start a wandb run, or return ``None`` if that is not possible."""
        global _import_warning_emitted
        try:
            import wandb  # noqa: PLC0415 - lazy on purpose: wandb is an optional dependency
        except ImportError:
            if not _import_warning_emitted:
                logger.warning(
                    "wandb is not installed; falling back to mode='disabled'. Metrics still go to %s.",
                    METRICS_FILENAME,
                )
                _import_warning_emitted = True
            return None

        try:
            return wandb.init(
                mode=self.mode,
                entity=self.cfg.entity,
                project=self.cfg.project,
                name=self.cfg.name,
                group=self.cfg.group,
                tags=list(self.cfg.tags),
                dir=self.run_dir,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - a logging backend must never kill a training run
            logger.warning("wandb.init failed (%s); falling back to mode='disabled'.", exc)
            return None

    def log(self, metrics: dict, *, step: int | None = None) -> None:
        """Append one record to ``metrics.jsonl`` and mirror it to wandb when enabled.

        ``_step`` and ``_wall_time`` are always included so the jsonl is analysable on its own.
        The auto-increment never goes backwards, because wandb rejects decreasing step numbers.
        """
        current_step = self._step if step is None else step
        self._step = max(self._step, current_step + 1)
        self._append({"_step": current_step, "_wall_time": time.time(), **metrics})
        if self._run is not None:
            try:
                self._run.log(dict(metrics), step=current_step)
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb log failed (%s); continuing with metrics.jsonl only.", exc)

    def summary(self, metrics: dict) -> None:
        """Set wandb summary fields and append a ``{"_summary": true, ...}`` line to the jsonl."""
        self._append({"_summary": True, "_step": self._step, "_wall_time": time.time(), **metrics})
        if self._run is not None:
            try:
                self._run.summary.update(dict(metrics))
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb summary update failed (%s); continuing.", exc)

    def finish(self) -> None:
        """Close the wandb run. Idempotent; safe to call from ``__exit__`` and explicitly."""
        if self._finished:
            return
        self._finished = True
        run, self._run = self._run, None
        if run is not None:
            try:
                run.finish()
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb finish failed (%s); continuing.", exc)

    def _append(self, record: dict) -> None:
        # Opened per call rather than held open: a killed job then still leaves a complete file.
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish()
        return False  # never swallow the exception


def _json_default(value: Any) -> Any:
    """Make torch/numpy scalars jsonl-safe, and stringify anything else rather than crashing."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # noqa: BLE001
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:  # noqa: BLE001
            pass
    return str(value)
