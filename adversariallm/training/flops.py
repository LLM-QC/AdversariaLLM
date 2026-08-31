"""FLOPs accounting: a ledger of typed entries instead of a single scalar.

The repo already estimates FLOPs with :func:`adversariallm.lm_utils.utils.get_flops`, which
returns one number per call under the Kaplan et al. (2020) cost model. That is enough for
attacks whose compute has a single shape (autoregressive generation), but not for attacks that
mix shapes: diffusion sampling is ``num_denoise_steps`` full-canvas forwards per sample, DPO
adds policy forward+backward and reference forward passes, and target generation / judging are
ordinary autoregressive passes on models with different parameter counts. A scalar cannot be
decomposed after the fact.

:class:`FlopsLedger` therefore records one :class:`FlopsEntry` per accounted operation
(component x phase x pass type x token counts) and aggregates only on demand, so the hot path
is a list append. Every entry keeps the inputs to its own cost formula (``n_params``,
``n_tokens_in``, ``n_tokens_out``, ``n_passes``, ``pass_type``, ``lora``), which means
:meth:`FlopsLedger.to_records` is enough to recompute the whole run under a different cost
convention.

This module is deliberately model- and attack-agnostic: it imports nothing from
``adversariallm``. Objects passed to :meth:`FlopsLedger.add_denoise` are duck-typed.
"""

import copy
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

from beartype import BeartypeConf, beartype
from beartype.typing import Literal

logger = logging.getLogger(__name__)

#: beartype with the PEP 484 numeric tower enabled, so an ``int`` is accepted wherever a
#: ``float`` is annotated. Without it, a perfectly reasonable ``detail={"ratio": 1}`` or a yaml
#: ``0`` where ``0.0`` was meant raises a type violation, which is noise rather than a bug.
_typed = beartype(conf=BeartypeConf(is_pep484_tower=True))

PassType = Literal["forward", "backward", "forward_and_backward"]

#: Components a :class:`FlopsEntry` may be attributed to. Extend deliberately: these strings end
#: up in ``flops.json`` and in analysis code that groups by them.
COMPONENTS = (
    "attacker_denoise",  # diffusion sampling: num_denoise_steps forwards over the canvas
    "attacker_policy",  # DPO policy forward/backward
    "attacker_reference",  # DPO reference forward (adapter disabled)
    "target_generate",  # target model generation
    "target_loss",  # target model teacher-forced loss pass
    "judge",  # judge model
    "embedding",  # optional sentence-transformer
)

#: Kaplan et al. (2020): ``2 * N * T`` per forward pass, twice that for the backward (once for
#: input gradients, once for weight gradients), hence ``6 * N * T`` for forward+backward.
#: Identical to ``lm_utils.get_flops`` by construction -- see ``tests/test_training/test_flops.py``.
_MULTIPLIERS_DENSE: dict[str, int] = {"forward": 2, "backward": 4, "forward_and_backward": 6}

#: With LoRA the base weights are frozen, so the backward still computes input gradients but no
#: weight gradients for them: ``2 * N * T`` instead of ``4 * N * T``. The adapter's own weight
#: gradients are neglected (``r << d_model``, a sub-percent correction). This is a modelling
#: choice, not a fact; it is recorded per entry as ``lora=True`` so ``to_records()`` can be
#: replayed under the dense convention instead.
_MULTIPLIERS_LORA: dict[str, int] = {"forward": 2, "backward": 2, "forward_and_backward": 4}

#: Separator used to flatten tuple group keys into JSON object keys in ``flops.json``, e.g. the
#: ``by_phase_component`` key ``"cycle0:pre|attacker_denoise"``. Phase names must not contain it;
#: :meth:`FlopsLedger.add` rejects those that do, so the flattened keys stay unambiguous.
GROUP_KEY_SEP = "|"


class DenoiseResultLike(Protocol):
    """Structural type of the diffusion sampling result :meth:`FlopsLedger.add_denoise` consumes.

    Duck-typed on purpose: this module must not depend on ``adversariallm.attackers``.
    """

    n_forward_passes: int
    n_tokens_per_pass: int


class AttackerLike(Protocol):
    """Structural type of the attacker :meth:`FlopsLedger.add_denoise` consumes."""

    n_params_no_embed: int


def pass_flops(n_params: int, n_tokens: int, pass_type: PassType, lora: bool = False) -> int:
    """Kaplan et al. (2020) cost model, consistent with ``lm_utils.get_flops``.

    ``forward = 2 * N * T``, ``backward = 4 * N * T``, ``forward_and_backward = 6 * N * T``.

    With ``lora=True`` the frozen base weights get no weight-gradient, so ``backward = 2 * N * T``
    and ``forward_and_backward = 4 * N * T``. ``forward`` is unaffected.

    Parameters
    ----------
    n_params
        Non-embedding parameter count, i.e. ``model.num_parameters(exclude_embeddings=True)``.
    n_tokens
        Tokens processed by a single pass (input + output).
    pass_type
        Which operations to include.
    lora
        Whether the base weights are frozen behind a LoRA adapter.
    """
    if n_params < 0:
        raise ValueError(f"n_params must be non-negative, got {n_params}")
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be non-negative, got {n_tokens}")
    multipliers = _MULTIPLIERS_LORA if lora else _MULTIPLIERS_DENSE
    if pass_type not in multipliers:
        raise ValueError(f"Invalid pass_type: {pass_type!r}. Valid: {sorted(multipliers)}")
    return multipliers[pass_type] * n_params * n_tokens


@_typed
@dataclass(frozen=True, kw_only=True)
class FlopsEntry:
    """One accounted batch of identical passes.

    ``n_tokens_in``/``n_tokens_out`` are per pass, so that
    ``flops == n_passes * pass_flops(n_params, n_tokens_in + n_tokens_out, pass_type, lora)``
    always holds. A batch of differently-sized sequences is therefore recorded as a single pass
    over the summed token count (the cost model is linear in tokens), which is what
    :meth:`FlopsLedger.add_ar_generation` does.
    """

    component: str
    phase: str  # "cycle0:pre" | "cycle0:dpo_epoch3" | "eval"
    model_id: str
    n_params: int
    n_tokens_in: int
    n_tokens_out: int
    pass_type: PassType
    n_passes: int
    lora: bool
    flops: int
    detail: dict[str, int | float | str | bool] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        """Tokens per pass."""
        return self.n_tokens_in + self.n_tokens_out


#: Fields that can be used as :meth:`FlopsLedger.by` grouping keys. ``detail`` is excluded
#: because a dict is not hashable and could not act as a group key.
GROUPABLE_KEYS = tuple(f.name for f in fields(FlopsEntry) if f.name != "detail")


class FlopsLedger:
    """Append-only log of :class:`FlopsEntry`, aggregated on demand.

    :meth:`add` is O(1) and does no aggregation, so it is safe to call inside a training loop.
    """

    def __init__(self, entries: list[FlopsEntry] | None = None) -> None:
        self._entries: list[FlopsEntry] = list(entries) if entries else []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_entries={len(self._entries)}, total={self.total():.3e})"

    @property
    def entries(self) -> list[FlopsEntry]:
        """A shallow copy of the recorded entries; entries themselves are frozen."""
        return list(self._entries)

    def add(
        self,
        *,
        component: str,
        phase: str,
        model_id: str,
        n_params: int,
        n_tokens_in: int,
        n_tokens_out: int = 0,
        pass_type: PassType,
        n_passes: int = 1,
        lora: bool = False,
        detail: dict[str, int | float | str | bool] | None = None,
    ) -> FlopsEntry:
        """Record ``n_passes`` passes of ``pass_type`` over ``n_tokens_in + n_tokens_out`` tokens each."""
        if component not in COMPONENTS:
            raise ValueError(f"Unknown component {component!r}. Valid components: {list(COMPONENTS)}")
        if GROUP_KEY_SEP in phase:
            raise ValueError(
                f"phase must not contain {GROUP_KEY_SEP!r} (it separates group keys in flops.json): {phase!r}"
            )
        if n_passes < 0:
            raise ValueError(f"n_passes must be non-negative, got {n_passes}")
        if n_tokens_in < 0 or n_tokens_out < 0:
            raise ValueError(f"Token counts must be non-negative, got in={n_tokens_in} out={n_tokens_out}")

        entry = FlopsEntry(
            component=component,
            phase=phase,
            model_id=model_id,
            n_params=n_params,
            n_tokens_in=n_tokens_in,
            n_tokens_out=n_tokens_out,
            pass_type=pass_type,
            n_passes=n_passes,
            lora=lora,
            flops=n_passes * pass_flops(n_params, n_tokens_in + n_tokens_out, pass_type, lora=lora),
            detail=dict(detail) if detail else {},
        )
        self._entries.append(entry)
        return entry

    def add_denoise(
        self,
        *,
        phase: str,
        attacker: AttackerLike,
        batch_size: int,
        denoise_result: DenoiseResultLike,
        detail: dict[str, int | float | str | bool] | None = None,
    ) -> FlopsEntry:
        """Record one batch of diffusion sampling.

        A diffusion sample costs ``n_forward_passes`` *full-canvas* forwards -- not one forward
        per generated token -- so the batch costs
        ``batch_size * n_forward_passes * pass_flops(N, n_tokens_per_pass, "forward")``.

        ``attacker`` and ``denoise_result`` are duck-typed (see :class:`AttackerLike` /
        :class:`DenoiseResultLike`) so this module stays independent of ``adversariallm.attackers``.
        """
        if batch_size < 0:
            raise ValueError(f"batch_size must be non-negative, got {batch_size}")
        n_forward_passes = int(denoise_result.n_forward_passes)
        n_tokens_per_pass = int(denoise_result.n_tokens_per_pass)
        if n_forward_passes < 0:
            raise ValueError(f"n_forward_passes must be non-negative, got {n_forward_passes}")

        merged_detail: dict[str, int | float | str | bool] = {
            "batch_size": batch_size,
            "n_denoise_steps": n_forward_passes,
            "canvas_len": n_tokens_per_pass,
        }
        if detail:
            merged_detail.update(detail)

        return self.add(
            component="attacker_denoise",
            phase=phase,
            model_id=model_id_of(attacker),
            n_params=int(attacker.n_params_no_embed),
            n_tokens_in=n_tokens_per_pass,
            n_tokens_out=0,
            pass_type="forward",
            n_passes=batch_size * n_forward_passes,
            # A forward pass costs the same either way; recorded for provenance only.
            lora=bool(getattr(attacker, "has_lora", False)),
            detail=merged_detail,
        )

    def add_ar_generation(
        self,
        *,
        phase: str,
        model_id: str,
        n_params: int,
        prompt_tokens: list[int],
        new_tokens: list[int],
        component: str = "target_generate",
        lora: bool = False,
        detail: dict[str, int | float | str | bool] | None = None,
    ) -> FlopsEntry:
        """Record autoregressive generation for a batch of sequences.

        Per sequence the cost is ``2 * N * (P_i + G_i)``; because that is linear in tokens the
        batch is stored as one pass over ``(sum P_i) + (sum G_i)`` tokens, which gives exactly
        ``sum_i 2 * N * (P_i + G_i)``. The per-sequence counts are not kept -- record them in
        ``detail`` if a particular caller needs them.
        """
        if len(prompt_tokens) != len(new_tokens):
            raise ValueError(
                f"prompt_tokens and new_tokens must have equal length, got {len(prompt_tokens)} and {len(new_tokens)}"
            )
        if any(t < 0 for t in prompt_tokens) or any(t < 0 for t in new_tokens):
            raise ValueError(f"Token counts must be non-negative, got prompt={prompt_tokens} new={new_tokens}")

        merged_detail: dict[str, int | float | str | bool] = {"n_sequences": len(prompt_tokens)}
        if detail:
            merged_detail.update(detail)

        return self.add(
            component=component,
            phase=phase,
            model_id=model_id,
            n_params=n_params,
            n_tokens_in=int(sum(prompt_tokens)),
            n_tokens_out=int(sum(new_tokens)),
            pass_type="forward",
            n_passes=1,
            lora=lora,
            detail=merged_detail,
        )

    def total(self) -> int:
        return sum(entry.flops for entry in self._entries)

    def by(self, *keys: str) -> dict[tuple, int]:
        """Aggregate FLOPs grouped by :class:`FlopsEntry` attribute names.

        Keys are always tuples, including for a single grouping key, so callers do not have to
        special-case arity: ``by("phase") -> {("eval",): 123}``.
        """
        if not keys:
            raise ValueError("by() needs at least one grouping key")
        unknown = [k for k in keys if k not in GROUPABLE_KEYS]
        if unknown:
            raise ValueError(f"Unknown grouping key(s) {unknown}. Valid keys: {list(GROUPABLE_KEYS)}")

        grouped: dict[tuple, int] = {}
        for entry in self._entries:
            group = tuple(getattr(entry, key) for key in keys)
            grouped[group] = grouped.get(group, 0) + entry.flops
        return grouped

    def to_records(self) -> list[dict]:
        """All entries as plain dicts, sufficient to recompute the run under another convention."""
        return [asdict(entry) for entry in self._entries]

    def merge(self, other: "FlopsLedger") -> None:
        """Append all entries of ``other`` to this ledger. ``other`` is left unchanged."""
        self._entries.extend(other._entries)

    def snapshot(self) -> "FlopsLedger":
        """Deep copy, e.g. to diff before/after a step for per-step FLOPs attribution."""
        return FlopsLedger(copy.deepcopy(self._entries))

    def write_json(self, path: str | Path) -> None:
        """Write total, aggregations and all entries to ``path``.

        Tuple group keys are flattened with :data:`GROUP_KEY_SEP`, so a ``by_phase_component``
        key looks like ``"cycle0:pre|attacker_denoise"``. Keys are sorted so the file is stable
        across runs and diffable.
        """
        path = Path(path)
        parent = path.parent
        if str(parent):
            os.makedirs(parent, exist_ok=True)
        payload = {
            "total": self.total(),
            "by_phase": _flatten_groups(self.by("phase")),
            "by_component": _flatten_groups(self.by("component")),
            "by_phase_component": _flatten_groups(self.by("phase", "component")),
            "entries": self.to_records(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def format_group_key(key: tuple) -> str:
    """Flatten a :meth:`FlopsLedger.by` tuple key into the string used in ``flops.json``."""
    return GROUP_KEY_SEP.join(str(part) for part in key)


def _flatten_groups(grouped: dict[tuple, int]) -> dict[str, int]:
    return {format_group_key(key): value for key, value in grouped.items()}


def model_id_of(obj: Any) -> str:
    """Best-effort identifier for a model-owning object, for provenance in the ledger.

    Tries, in order: ``model_id``, ``cfg.id``, ``config.id``, ``model.name_or_path``, class name.
    Never raises: a missing identifier must not break FLOPs accounting.
    """
    for getter in (
        lambda a: getattr(a, "model_id", None),
        lambda a: getattr(getattr(a, "cfg", None), "id", None),
        lambda a: getattr(getattr(a, "config", None), "id", None),
        lambda a: getattr(getattr(a, "model", None), "name_or_path", None),
    ):
        try:
            value = getter(obj)
        except Exception:  # noqa: BLE001 - provenance must never break accounting
            value = None
        if isinstance(value, str) and value:
            return value
    return type(obj).__name__
