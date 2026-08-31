"""Diffusion DPO trainer for the IHO attacker.

Ported from ``aapl/trainer/DPOTrainer.py``. The DPO objective, the masking protocol, the warmup
schedule and the early-stopping rule are preserved exactly; the plumbing around them is not:

* the trainer talks to an :class:`Attacker` (plan §3.1) through a structural type, so it never
  imports ``adversariallm.attackers`` and can be unit-tested against a toy ``nn.Module``;
* mid-training resampling goes through ``eval_fn(phase_id) -> DataFrame``, so the trainer knows
  nothing about targets, judges or parquet files -- the caller writes those;
* all metrics go through :class:`~adversariallm.training.run_logging.RunLogger`; ``wandb`` is
  never imported here and no entity/project is hardcoded (AAPL pinned ``entity="limbach"``);
* every forward/backward is booked on a :class:`~adversariallm.training.flops.FlopsLedger`
  (plan §5.2), which AAPL did not account for at all;
* the optimizer only sees ``requires_grad`` parameters -- AAPL handed ``model.parameters()`` to
  ``AdamW``, i.e. the whole frozen base model, which made the optimizer state ~8B entries wide.

The DPO loss itself is unchanged:
``loss = -logsigmoid(beta * (chosen_log_pi - rejected_log_pi - chosen_log_ref + rejected_log_ref)).mean()``
with chosen and rejected masked **independently** -- each gets its own ``mask_tokens`` call and
therefore its own random masking pattern. That asymmetry is AAPL's behaviour and is deliberate:
the DPO comparison is over the model's ability to reconstruct each sequence under an independent
noise draw, not over a shared one.
"""

import logging
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import pandas as pd
import torch
import torch.nn.functional as F
from beartype import BeartypeConf, beartype
from beartype.typing import Literal
from torch.utils.data import DataLoader

from .flops import FlopsLedger, model_id_of
from .preference import PreferenceDataset
from .run_logging import RunLogger

logger = logging.getLogger(__name__)

#: beartype with the PEP 484 numeric tower enabled, so an ``int`` is accepted wherever a ``float``
#: is annotated -- a yaml ``beta: 1`` must not be a type violation.
_typed = beartype(conf=BeartypeConf(is_pep484_tower=True))

MaskingMode = Literal["all", "prompt", "attack"]
SelectionMetric = Literal["mean", "max", "weighted"]

#: Per-behavior grouping column of the scored-sample frames returned by ``eval_fn`` (plan §4.6).
BEHAVIOR_COL = "jb_index"

#: Default score column read from those frames (plan §4.6); load-bearing for
#: ``evaluate/expected_asr_multiphase.py::build_iho_phase_table``.
DEFAULT_SCORE_COL = "judge_score_training"

#: Quantiles reported per behavior at every evaluation, and the metric names they map to.
_PERCENTILES: dict[float, str] = {0.50: "p50", 0.90: "p90", 0.95: "p95", 0.99: "p99", 1.00: "max"}


@runtime_checkable
class AttackerLike(Protocol):
    """Structural view of the plan §3.1 ``Attacker`` interface that DPO training needs.

    Declared locally rather than imported so this module has no dependency on
    ``adversariallm.attackers`` -- which also makes the trainer testable with a fake attacker.
    """

    device: torch.device
    model: torch.nn.Module
    has_lora: bool

    @property
    def n_params_no_embed(self) -> int: ...

    def train(self) -> "AttackerLike": ...

    def eval(self) -> "AttackerLike": ...

    def disable_adapter(self) -> AbstractContextManager: ...

    def save_lora(self, path: str) -> None: ...

    def encode(self, texts: list[str]) -> torch.Tensor: ...

    def mask_tokens(self, token_ids: torch.Tensor, masking_mode: str, mask_all: bool = False): ...

    def compute_log_likelihood(
        self,
        masked_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        target_ids: torch.Tensor,
        use_base_model: bool = False,
    ) -> torch.Tensor: ...


@_typed
@dataclass
class DPOConfig:
    """Plan §3.7. Mirrored 1:1 by the ``training.dpo`` yaml block."""

    learning_rate: float = 1e-5
    beta: float = 0.25
    epochs: int = 200
    batch_size: int = 16
    masking_mode: MaskingMode = "prompt"
    mask_all: bool = False
    max_grad_norm: float | None = 1.0
    checkpoint_every: int = 1  # epochs between eval/checkpoint; 0 disables eval
    patience: int = 1
    warmup_epochs: int = 0
    load_optimizer_state: bool = False
    selection_metric: SelectionMetric = "mean"


@_typed
@dataclass
class CycleResult:
    """Outcome of one :meth:`DiffusionDPOTrainer.train_cycle`."""

    losses: list[float] = field(default_factory=list)  # mean loss per epoch, in epoch order
    best_metric: float | None = None  # None when no evaluation ran
    best_checkpoint: str | None = None  # adapter dir corresponding to ``best_metric``
    epochs_run: int = 0
    stopped_early: bool = False


def dpo_phase_id(cycle_id: int, epoch: int) -> str:
    """Phase string for FLOPs entries and ``eval_fn``, e.g. ``"cycle0:dpo_epoch3"``.

    ``epoch`` is 1-based, matching AAPL's ``dpo_samples/cycle_<i>_epoch_<e>.parquet`` naming
    (plan §4.3), so the caller can derive the sample filename from the phase id alone.
    """
    return f"cycle{cycle_id}:dpo_epoch{epoch}"


def judge_percentiles_avg_by_jb_index(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    """Per-behavior quantiles of ``score_col``, averaged across behaviors.

    If ``jb_index=0`` has ``p90=0.5`` and ``jb_index=1`` has ``p90=0.9``, the returned ``p90`` is
    ``0.7``. Averaging per behavior rather than pooling all rows keeps behaviors with more samples
    from dominating the summary.
    """
    _require_columns(df, [BEHAVIOR_COL, score_col])
    per_behavior = df.groupby(BEHAVIOR_COL)[score_col].quantile(list(_PERCENTILES)).unstack(level=-1)
    averaged = per_behavior.mean(axis=0)
    return {_PERCENTILES[q]: float(averaged[q]) for q in _PERCENTILES}


def select_metric(df: pd.DataFrame, score_col: str, metric: SelectionMetric) -> float:
    """Single scalar summary of ``score_col``, used for checkpoint selection and early stopping.

    * ``"mean"`` -- plain mean over all rows;
    * ``"max"`` -- mean of the per-behavior maxima (best attack per goal, averaged);
    * ``"weighted"`` -- the average of the two, i.e. it rewards a high ceiling without ignoring
      the bulk of the samples.

    Note that ``"weighted"`` is AAPL's implementation, not its docstring (which claimed per-behavior
    p90); the code is what produced every AAPL result, so the code is what is ported.
    """
    _require_columns(df, [BEHAVIOR_COL, score_col])
    if metric == "mean":
        return float(df[score_col].mean())
    if metric == "max":
        return float(df.groupby(BEHAVIOR_COL)[score_col].max().mean())
    if metric == "weighted":
        mean_score = float(df[score_col].mean())
        max_score = float(df.groupby(BEHAVIOR_COL)[score_col].max().mean())
        return (mean_score + max_score) / 2
    raise ValueError(f"Unknown selection metric {metric!r}. Choose from 'mean', 'max', 'weighted'.")


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Scored-sample DataFrame is missing required column(s) {missing}; has {list(df.columns)}")


def _align_target_to_masked(masked_ids: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Return a per-position target the same width as ``masked_ids``.

    ``masking_mode="attack"`` prepends ``prompt_len`` mask tokens to the canvas
    (``LLaDADiffusionAttacker.mask_tokens``), so ``masked_ids`` is wider than the encoded
    ``token_ids`` the trainer holds, while ``compute_log_likelihood`` gathers the target at every
    canvas position and therefore requires a target of the *lengthened* width. The prepended block
    is masked-in and never eligible outside it, so ``masked_ids[:, :prepend]`` holds exactly the
    canvas content there (the mask-token block) -- that is what the attacker scores against, so it
    is the correct target for those positions. Positions after the block keep the original ids.

    ``"all"`` and ``"prompt"`` do not widen the canvas (``prepend == 0``), so ``token_ids`` is
    returned unchanged and the behaviour is identical to AAPL. Lifting the prepended block out of
    ``masked_ids`` (rather than re-deriving it) keeps the trainer agnostic to the attacker's
    mask-token id.
    """
    prepend = masked_ids.shape[-1] - token_ids.shape[-1]
    if prepend == 0:
        return token_ids
    if prepend < 0:
        raise ValueError(
            f"mask_tokens shortened the canvas ({token_ids.shape[-1]} -> {masked_ids.shape[-1]}); "
            "compute_log_likelihood cannot align a target to it."
        )
    return torch.cat([masked_ids[..., :prepend], token_ids], dim=-1)


class DiffusionDPOTrainer:
    """DPO over a masked-diffusion attacker with LoRA adapters (plan §3.7).

    One trainer instance spans all cycles of a training run: the optimizer, the global step
    counter and the ledger/logger are shared, while :meth:`train_cycle` runs the epochs of a
    single cycle against one :class:`~adversariallm.training.preference.PreferenceDataset`.
    """

    def __init__(
        self,
        attacker: AttackerLike,
        cfg: DPOConfig,
        *,
        ledger: FlopsLedger,
        logger: RunLogger,
        score_col: str = DEFAULT_SCORE_COL,
    ) -> None:
        if not getattr(attacker, "has_lora", False):
            raise RuntimeError("DPO training requires an attacker with LoRA adapters (has_lora=True).")

        self.attacker = attacker
        self.cfg = cfg
        self.ledger = ledger
        self.logger = logger
        self.score_col = score_col
        self.global_step = 0

        # AAPL passed `model.parameters()`, i.e. also the frozen base weights, which allocates
        # AdamW moments for billions of parameters that never receive a gradient (plan §6.4).
        self._trainable_params = [p for p in attacker.model.parameters() if p.requires_grad]
        if not self._trainable_params:
            raise RuntimeError("No parameters with requires_grad=True; the LoRA adapter is not trainable.")
        self.optimizer = torch.optim.AdamW(self._trainable_params, lr=cfg.learning_rate)

        self._model_id = model_id_of(attacker)
        self._n_params = int(attacker.n_params_no_embed)

        if cfg.masking_mode == "attack":
            # AAPL defaulted to "prompt" and never exercised "attack" in DPO; here it is supported
            # by re-aligning the target to the prompt_len-wider canvas (see `_align_target_to_masked`).
            # The `logger` argument shadows the module logger in this scope, so name it explicitly.
            logging.getLogger(__name__).info(
                "DPO masking_mode='attack': canvas is widened by prompt_len mask tokens; targets are "
                "aligned to the lengthened masked_ids and FLOPs are booked over the fed (padded) width."
            )

    # ------------------------------------------------------------------
    # Optimizer state persistence
    # ------------------------------------------------------------------

    def save_optimizer_state(self, path: str | Path) -> None:
        """Save the optimizer state into ``<path>/optimizer.pt`` (plan §4.3)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

    def load_optimizer_state(self, path: str | Path) -> None:
        """Load the optimizer state from ``<path>/optimizer.pt``."""
        state_path = os.path.join(path, "optimizer.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"No optimizer state found at {state_path}")
        self.optimizer.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))

    # ------------------------------------------------------------------
    # Warmup schedule
    # ------------------------------------------------------------------

    def _make_warmup_scheduler(self, warmup_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
        """LR 0 for epoch 0, linear ramp to the full LR over ``warmup_epochs``, then flat.

        Epoch 0 deliberately runs at LR 0: with ``checkpoint_every=1`` it produces the *untrained*
        baseline evaluation, so the first checkpoint is a fair reference point rather than a
        model that already took a step. Stepped once per epoch, after the epoch.
        """

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs <= 0:
                return 1.0
            if epoch == 0:
                return 0.0
            if epoch <= warmup_epochs:
                return epoch / warmup_epochs
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_dpo_loss(self, chosen: list[str], rejected: list[str], *, phase: str) -> torch.Tensor:
        """DPO loss for one batch of (chosen, rejected) full-canvas texts.

        Chosen and rejected are encoded and masked independently, then scored by the policy
        (adapter active, gradients on) and by the reference (adapter disabled, ``no_grad``).
        Four FLOPs entries are booked per call -- policy and reference, chosen and rejected --
        because the two sides pad to different lengths and a single entry cannot express that.
        """
        if not self.attacker.model.training:
            raise RuntimeError("Attacker model must be in training mode; call attacker.train() first.")
        if len(chosen) != len(rejected):
            raise ValueError(f"chosen and rejected must have equal length, got {len(chosen)} and {len(rejected)}")
        if not chosen:
            raise ValueError("compute_dpo_loss called with an empty batch")

        chosen_ids = self.attacker.encode(chosen)
        rejected_ids = self.attacker.encode(rejected)

        chosen_masking = self.attacker.mask_tokens(chosen_ids, self.cfg.masking_mode, self.cfg.mask_all)
        rejected_masking = self.attacker.mask_tokens(rejected_ids, self.cfg.masking_mode, self.cfg.mask_all)

        # `masking_mode="attack"` widens the canvas (prepends `prompt_len` mask tokens), so the
        # target must be re-aligned to the lengthened `masked_ids`; "all"/"prompt" leave the width
        # unchanged and this is a no-op. See `_align_target_to_masked`.
        chosen_target = _align_target_to_masked(chosen_masking.masked_ids, chosen_ids)
        rejected_target = _align_target_to_masked(rejected_masking.masked_ids, rejected_ids)

        chosen_log_pi = self.attacker.compute_log_likelihood(
            chosen_masking.masked_ids, chosen_masking.mask_positions, chosen_target, use_base_model=False
        )
        rejected_log_pi = self.attacker.compute_log_likelihood(
            rejected_masking.masked_ids, rejected_masking.mask_positions, rejected_target, use_base_model=False
        )

        # The reference must see the base model: `use_base_model=True` alone is only an argument;
        # `disable_adapter()` is what actually detaches the LoRA weights (plan §3.7).
        with torch.no_grad(), self.attacker.disable_adapter():
            chosen_log_ref = self.attacker.compute_log_likelihood(
                chosen_masking.masked_ids, chosen_masking.mask_positions, chosen_target, use_base_model=True
            )
            rejected_log_ref = self.attacker.compute_log_likelihood(
                rejected_masking.masked_ids, rejected_masking.mask_positions, rejected_target, use_base_model=True
            )

        self._record_batch_flops(
            phase=phase,
            batch_size=len(chosen),
            chosen_seq_len=int(chosen_masking.masked_ids.shape[-1]),
            rejected_seq_len=int(rejected_masking.masked_ids.shape[-1]),
        )

        logits = self.cfg.beta * (chosen_log_pi - rejected_log_pi - chosen_log_ref + rejected_log_ref)
        return -F.logsigmoid(logits).mean()

    def _record_batch_flops(self, *, phase: str, batch_size: int, chosen_seq_len: int, rejected_seq_len: int) -> None:
        """Book the four passes of one DPO batch on the ledger (plan §5.2).

        Per batch: 2 policy forward+backward (LoRA convention, ``4 N T``) and 2 reference
        forwards (``2 N T``), each over ``batch_size`` sequences of the padded length actually fed
        to the model. Chosen and rejected are booked separately because their padded lengths
        differ; summed they reproduce the plan's ``2 * B * pass_flops(N, T, ...)`` when they agree.
        """
        for side, seq_len in (("chosen", chosen_seq_len), ("rejected", rejected_seq_len)):
            self.ledger.add(
                component="attacker_policy",
                phase=phase,
                model_id=self._model_id,
                n_params=self._n_params,
                n_tokens_in=seq_len,
                pass_type="forward_and_backward",
                n_passes=batch_size,
                lora=True,
                detail={"side": side, "batch_size": batch_size, "seq_len": seq_len},
            )
        for side, seq_len in (("chosen", chosen_seq_len), ("rejected", rejected_seq_len)):
            self.ledger.add(
                component="attacker_reference",
                phase=phase,
                model_id=self._model_id,
                n_tokens_in=seq_len,
                n_params=self._n_params,
                pass_type="forward",
                n_passes=batch_size,
                lora=False,  # the adapter is disabled, so this really is the dense base model
                detail={"side": side, "batch_size": batch_size, "seq_len": seq_len},
            )

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(self, chosen: list[str], rejected: list[str], *, phase: str) -> float:
        """One optimizer step on a single preference batch; returns the scalar loss."""
        self.optimizer.zero_grad()
        loss = self.compute_dpo_loss(chosen, rejected, phase=phase)
        loss.backward()
        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.max_grad_norm)
        self.optimizer.step()
        self.global_step += 1
        loss_value = float(loss.item())
        self.logger.log({"train/step_loss": loss_value}, step=self.global_step)
        return loss_value

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------

    def train_cycle(
        self,
        dataset: PreferenceDataset,
        *,
        cycle_id: int,
        run_dir: str | Path,
        eval_fn: Callable[[str], pd.DataFrame] | None = None,
    ) -> CycleResult:
        """Run one DPO cycle: ``cfg.epochs`` epochs over ``dataset`` with periodic evaluation.

        Every ``cfg.checkpoint_every`` epochs (and never, if that is ``0`` or ``eval_fn`` is
        ``None``) the attacker is switched to eval mode and ``eval_fn(phase_id)`` resamples and
        scores it. The resulting frame drives the selection metric: an improvement saves the
        adapter to ``<run_dir>/checkpoints/best_cycle_<cycle_id>``, a non-improvement increments
        ``bad_checks``, and the cycle stops once ``bad_checks > cfg.patience``.

        ``eval_fn`` receives only a phase id and returns a scored-sample DataFrame; writing that
        frame to ``dpo_samples/`` is the caller's job, so the trainer does no sample file I/O.
        """
        if len(dataset) == 0:
            raise ValueError(f"train_cycle called with an empty PreferenceDataset (cycle {cycle_id})")

        run_dir = str(run_dir)
        checkpoint_root = os.path.join(run_dir, "checkpoints")

        if self.cfg.load_optimizer_state and cycle_id > 0:
            self.load_optimizer_state(os.path.join(checkpoint_root, f"optimizer_cycle_{cycle_id - 1}"))

        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        scheduler = self._make_warmup_scheduler(self.cfg.warmup_epochs)
        evaluating = eval_fn is not None and self.cfg.checkpoint_every > 0

        result = CycleResult()
        bad_checks = 0
        self.attacker.train()

        for epoch in range(self.cfg.epochs):
            phase = dpo_phase_id(cycle_id, epoch + 1)
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            epoch_loss = 0.0
            n_steps = 0
            for batch in dataloader:
                epoch_loss += self.train_step(batch["chosen"], batch["rejected"], phase=phase)
                n_steps += 1

            mean_loss = epoch_loss / max(n_steps, 1)
            result.losses.append(mean_loss)
            result.epochs_run = epoch + 1
            self.logger.log(
                {"train/epoch_loss": mean_loss, "train/lr": current_lr, "train/epoch": epoch + 1,
                 "train/cycle": cycle_id, "train/n_steps": n_steps},
                step=self.global_step,
            )
            scheduler.step()

            if not evaluating or epoch % self.cfg.checkpoint_every != 0:
                continue

            self.attacker.eval()
            metric_value = self._evaluate(eval_fn, phase=phase, cycle_id=cycle_id, epoch=epoch + 1)

            if result.best_metric is None or metric_value > result.best_metric:
                result.best_metric = metric_value
                bad_checks = 0
                best_path = os.path.join(checkpoint_root, f"best_cycle_{cycle_id}")
                os.makedirs(best_path, exist_ok=True)
                self.attacker.save_lora(best_path)
                result.best_checkpoint = best_path
            else:
                bad_checks += 1
                if bad_checks > self.cfg.patience:
                    logger.info(
                        "Early stopping cycle %d after epoch %d: %d checks without improvement over %.4f",
                        cycle_id, epoch + 1, bad_checks, result.best_metric,
                    )
                    result.stopped_early = True
                    break

            self.attacker.train()

        if self.cfg.load_optimizer_state:
            self.save_optimizer_state(os.path.join(checkpoint_root, f"optimizer_cycle_{cycle_id}"))

        self.attacker.eval()
        return result

    def _evaluate(
        self, eval_fn: Callable[[str], pd.DataFrame], *, phase: str, cycle_id: int, epoch: int
    ) -> float:
        """Resample via ``eval_fn``, log the summary metrics, and return the selection metric."""
        df = eval_fn(phase)
        if df is None or len(df) == 0:
            raise ValueError(
                f"eval_fn returned no rows for phase {phase!r}. An empty sample frame is a silent "
                "failure upstream, not a valid evaluation."
            )

        metric_value = select_metric(df, self.score_col, self.cfg.selection_metric)
        percentiles = judge_percentiles_avg_by_jb_index(df, self.score_col)

        metrics: dict[str, float | int | str] = {
            f"dpo_data/early_stop_metric_{self.cfg.selection_metric}": metric_value,
            f"dpo_data/std_{self.score_col}": float(df[self.score_col].std()),
            "dpo_data/n_samples": int(len(df)),
            "dpo_data/epoch": epoch,
            "dpo_data/cycle": cycle_id,
            "dpo_data/phase": phase,
        }
        metrics.update({f"dpo_data/avg_{name}_{self.score_col}": value for name, value in percentiles.items()})

        # Validation judges are logged only; they never influence checkpoint selection (plan §4.5).
        validation_col = "judge_score_validation"
        if validation_col in df.columns:
            val_percentiles = judge_percentiles_avg_by_jb_index(df, validation_col)
            metrics[f"dpo_data/std_{validation_col}"] = float(df[validation_col].std())
            metrics.update({f"dpo_data/avg_{n}_{validation_col}": v for n, v in val_percentiles.items()})

        metrics.update(self._flops_metrics())
        self.logger.log(metrics, step=self.global_step)
        return metric_value

    def _flops_metrics(self) -> dict[str, int]:
        """Cumulative FLOPs of the whole run so far, total and per component."""
        metrics: dict[str, int] = {"flops/total": self.ledger.total()}
        for (component,), value in self.ledger.by("component").items():
            metrics[f"flops/{component}"] = value
        return metrics
