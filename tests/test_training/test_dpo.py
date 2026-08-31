"""CPU-only tests for :mod:`adversariallm.training.dpo`.

No real model and no network: the trainer is exercised against a tiny fake attacker that
implements the plan §3.1 interface with a real trainable parameter, so gradients actually flow
and the optimizer actually steps. The fake also spies on every ``compute_log_likelihood`` call
(grad mode, adapter state, tensor shapes), which is how the reference-forward contract is tested.
"""

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adversariallm.training.dpo import (
    DEFAULT_SCORE_COL,
    CycleResult,
    DiffusionDPOTrainer,
    DPOConfig,
    dpo_phase_id,
    judge_percentiles_avg_by_jb_index,
    select_metric,
)
from adversariallm.training.flops import FlopsLedger, pass_flops
from adversariallm.training.preference import PreferenceDataset, PreferencePair
from adversariallm.training.run_logging import LoggingConfig, RunLogger

VOCAB_SIZE = 16
PAD_ID = 0
MASK_ID = 1
N_PARAMS = 1000
PROMPT_LEN = 3


# ----------------------------------------------------------------------------------------------
# Fake attacker (plan §3.1 interface)
# ----------------------------------------------------------------------------------------------


@dataclass
class FakeMaskingResult:
    masked_ids: torch.Tensor
    mask_positions: torch.Tensor
    prompt_positions: torch.Tensor


class _TinyModel(nn.Module):
    """One trainable "adapter" matrix on top of a frozen "base" matrix.

    ``logits`` is a one-hot lookup, i.e. a linear map from token id to a logit row, which is
    enough for real gradients without any of the cost of a transformer.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.base = nn.Parameter(torch.randn(vocab_size, vocab_size), requires_grad=False)
        self.adapter = nn.Parameter(0.01 * torch.randn(vocab_size, vocab_size), requires_grad=True)
        self.name_or_path = "fake/attacker"

    def logits(self, token_ids: torch.Tensor, *, use_adapter: bool) -> torch.Tensor:
        one_hot = F.one_hot(token_ids, self.base.shape[0]).float()
        weight = self.base + self.adapter if use_adapter else self.base
        return one_hot @ weight


class FakeAttacker:
    """Minimal duck-typed attacker: encode / mask / log-likelihood / adapter control."""

    def __init__(self, prompt_len: int = PROMPT_LEN, n_params: int = N_PARAMS) -> None:
        self.model = _TinyModel()
        self.device = torch.device("cpu")
        self.has_lora = True
        self.model_id = "fake/attacker"
        self.prompt_len = prompt_len
        self._n_params = n_params

        self.adapter_disabled = False
        self.tag = ""  # written into the saved "adapter" so tests can identify a checkpoint
        self.ll_calls: list[dict] = []
        self.mask_calls: list[torch.Tensor] = []
        self.saved_paths: list[str] = []

    @property
    def n_params_no_embed(self) -> int:
        return self._n_params

    def train(self) -> "FakeAttacker":
        self.model.train()
        return self

    def eval(self) -> "FakeAttacker":
        self.model.eval()
        return self

    def to(self, device) -> "FakeAttacker":
        return self

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False

    def save_lora(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        Path(path, "adapter.txt").write_text(self.tag)
        self.saved_paths.append(path)

    def encode(self, texts: list[str]) -> torch.Tensor:
        sequences = [[(ord(c) % (VOCAB_SIZE - 2)) + 2 for c in text] for text in texts]
        max_len = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), max_len), PAD_ID, dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return ids

    def mask_tokens(self, token_ids: torch.Tensor, masking_mode: str, mask_all: bool = False) -> FakeMaskingResult:
        self.mask_calls.append(token_ids.clone())
        batch, length = token_ids.shape
        if masking_mode == "all":
            eligible = torch.ones((batch, length), dtype=torch.bool)
        elif masking_mode == "prompt":
            eligible = torch.zeros((batch, length), dtype=torch.bool)
            eligible[:, : self.prompt_len] = True
        elif masking_mode == "attack":
            block = token_ids.new_full((batch, self.prompt_len), MASK_ID)
            token_ids = torch.cat([block, token_ids], dim=1)
            eligible = torch.zeros(token_ids.shape, dtype=torch.bool)
            eligible[:, : self.prompt_len] = True
        else:
            raise ValueError(f"Unknown masking_mode: {masking_mode}")

        if mask_all:
            mask_positions = eligible.clone()
        else:
            p = torch.rand(batch).view(batch, 1)
            mask_positions = (torch.rand(token_ids.shape) < p) & eligible

        masked_ids = token_ids.clone()
        masked_ids[mask_positions] = MASK_ID
        return FakeMaskingResult(masked_ids=masked_ids, mask_positions=mask_positions, prompt_positions=eligible)

    def compute_log_likelihood(
        self,
        masked_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        target_ids: torch.Tensor,
        use_base_model: bool = False,
    ) -> torch.Tensor:
        self.ll_calls.append(
            {
                "use_base_model": use_base_model,
                "grad_enabled": torch.is_grad_enabled(),
                "adapter_disabled": self.adapter_disabled,
                "shape": tuple(masked_ids.shape),
            }
        )
        log_probs = F.log_softmax(self.model.logits(masked_ids, use_adapter=not self.adapter_disabled), dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return (target_log_probs * mask_positions).sum(dim=1)


# ----------------------------------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------------------------------


@pytest.fixture
def attacker() -> FakeAttacker:
    return FakeAttacker()


@pytest.fixture
def ledger() -> FlopsLedger:
    return FlopsLedger()


@pytest.fixture
def run_logger(tmp_path) -> RunLogger:
    return RunLogger(LoggingConfig(mode="disabled"), tmp_path / "run", {})


def make_trainer(attacker, ledger, run_logger, **cfg_kwargs) -> DiffusionDPOTrainer:
    cfg = DPOConfig(**cfg_kwargs)
    return DiffusionDPOTrainer(attacker, cfg, ledger=ledger, logger=run_logger)


def make_dataset(n_pairs: int = 4) -> PreferenceDataset:
    return PreferenceDataset(
        [
            PreferencePair(
                behavior_idx=i % 2,
                prompt=f"goal {i % 2}",
                chosen=f"chosen text {i}",
                rejected=f"rejected {i}",
                chosen_score=0.9,
                rejected_score=0.1,
                chosen_cycle=0,
                rejected_cycle=0,
            )
            for i in range(n_pairs)
        ]
    )


def scored_df(scores_by_behavior: dict[int, list[float]], score_col: str = DEFAULT_SCORE_COL) -> pd.DataFrame:
    rows = [{"jb_index": jb, score_col: s} for jb, scores in scores_by_behavior.items() for s in scores]
    return pd.DataFrame(rows)


def constant_eval_fn(values: list[float], seen: list[str] | None = None, attacker: FakeAttacker | None = None):
    """``eval_fn`` returning a frame whose mean/max is ``values[i]`` on the i-th call."""
    state = {"i": 0}

    def eval_fn(phase: str) -> pd.DataFrame:
        i = state["i"]
        state["i"] += 1
        if seen is not None:
            seen.append(phase)
        if attacker is not None:
            attacker.tag = str(i)
        value = values[min(i, len(values) - 1)]
        return scored_df({0: [value, value], 1: [value, value]})

    return eval_fn


# ----------------------------------------------------------------------------------------------
# DPO loss
# ----------------------------------------------------------------------------------------------


def test_dpo_loss_matches_hand_computed(attacker, ledger, run_logger, monkeypatch):
    """The core contract: -logsigmoid(beta * (c_pi - r_pi - c_ref + r_ref)).mean()."""
    chosen_pi = torch.tensor([-1.0, -2.0])
    rejected_pi = torch.tensor([-3.0, -1.0])
    chosen_ref = torch.tensor([-2.0, -2.5])
    rejected_ref = torch.tensor([-2.5, -1.5])

    calls = {"policy": 0, "reference": 0}

    def fake_ll(masked_ids, mask_positions, target_ids, use_base_model=False):
        if use_base_model:
            out = chosen_ref if calls["reference"] == 0 else rejected_ref
            calls["reference"] += 1
        else:
            out = chosen_pi if calls["policy"] == 0 else rejected_pi
            calls["policy"] += 1
        return out

    monkeypatch.setattr(attacker, "compute_log_likelihood", fake_ll)
    trainer = make_trainer(attacker, ledger, run_logger, beta=0.25)
    attacker.train()

    loss = trainer.compute_dpo_loss(["a", "bb"], ["ccc", "d"], phase="cycle0:dpo_epoch1")

    logits = 0.25 * (chosen_pi - rejected_pi - chosen_ref + rejected_ref)
    expected = float(-torch.log(torch.sigmoid(logits)).mean())
    assert loss.item() == pytest.approx(expected, rel=1e-6)

    # Hand-computed, independent of torch: logits = 0.25 * ([2, -1] - [0.5, -1]) = [0.375, 0.0]
    manual = -(math.log(1 / (1 + math.exp(-0.375))) + math.log(0.5)) / 2
    assert loss.item() == pytest.approx(manual, rel=1e-6)
    assert calls == {"policy": 2, "reference": 2}


def test_dpo_loss_is_zero_advantage_at_log_two(attacker, ledger, run_logger, monkeypatch):
    """Identical policy and reference log-likelihoods give logits 0, i.e. loss log(2)."""
    monkeypatch.setattr(
        attacker, "compute_log_likelihood", lambda *a, **k: torch.tensor([-1.0, -1.0])
    )
    trainer = make_trainer(attacker, ledger, run_logger, beta=0.9)
    attacker.train()
    loss = trainer.compute_dpo_loss(["a", "b"], ["c", "d"], phase="cycle0:dpo_epoch1")
    assert loss.item() == pytest.approx(math.log(2.0), rel=1e-6)


def test_reference_forwards_run_without_grad_and_with_adapter_disabled(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger)
    attacker.train()
    trainer.compute_dpo_loss(["chosen a", "chosen b"], ["rejected a", "rejected b"], phase="cycle0:dpo_epoch1")

    assert len(attacker.ll_calls) == 4
    policy_calls = [c for c in attacker.ll_calls if not c["use_base_model"]]
    reference_calls = [c for c in attacker.ll_calls if c["use_base_model"]]
    assert len(policy_calls) == 2 and len(reference_calls) == 2

    assert all(c["grad_enabled"] and not c["adapter_disabled"] for c in policy_calls)
    assert all((not c["grad_enabled"]) and c["adapter_disabled"] for c in reference_calls)
    # The context manager must be exited again, not left dangling.
    assert attacker.adapter_disabled is False


def test_chosen_and_rejected_are_masked_independently(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger)
    attacker.train()
    trainer.compute_dpo_loss(["chosen text"], ["rejected"], phase="cycle0:dpo_epoch1")

    assert len(attacker.mask_calls) == 2  # one call per side, so the noise draws differ
    assert attacker.mask_calls[0].shape != attacker.mask_calls[1].shape


def test_loss_requires_training_mode(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger)
    attacker.eval()
    with pytest.raises(RuntimeError, match="training mode"):
        trainer.compute_dpo_loss(["a"], ["b"], phase="cycle0:dpo_epoch1")


def test_loss_rejects_mismatched_or_empty_batches(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger)
    attacker.train()
    with pytest.raises(ValueError, match="equal length"):
        trainer.compute_dpo_loss(["a", "b"], ["c"], phase="p")
    with pytest.raises(ValueError, match="empty batch"):
        trainer.compute_dpo_loss([], [], phase="p")


def test_loss_backward_produces_gradients_on_the_adapter_only(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger, mask_all=True)
    attacker.train()
    loss = trainer.compute_dpo_loss(["chosen a", "chosen b"], ["rejected a", "rejected b"], phase="p")
    loss.backward()
    assert attacker.model.adapter.grad is not None
    assert torch.any(attacker.model.adapter.grad != 0)
    assert attacker.model.base.grad is None


# ----------------------------------------------------------------------------------------------
# Construction / optimizer
# ----------------------------------------------------------------------------------------------


def test_optimizer_only_contains_trainable_parameters(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger)
    optimized = [p for group in trainer.optimizer.param_groups for p in group["params"]]
    assert len(optimized) == 1
    assert optimized[0] is attacker.model.adapter
    assert all(p.requires_grad for p in optimized)
    assert not any(p is attacker.model.base for p in optimized)


def test_requires_lora(ledger, run_logger):
    attacker = FakeAttacker()
    attacker.has_lora = False
    with pytest.raises(RuntimeError, match="LoRA"):
        make_trainer(attacker, ledger, run_logger)


def test_requires_at_least_one_trainable_parameter(ledger, run_logger):
    attacker = FakeAttacker()
    attacker.model.adapter.requires_grad_(False)
    with pytest.raises(RuntimeError, match="requires_grad"):
        make_trainer(attacker, ledger, run_logger)


def test_learning_rate_is_taken_from_config(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger, learning_rate=3e-4)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


# ----------------------------------------------------------------------------------------------
# Warmup schedule
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("warmup_epochs", "expected"),
    [
        (0, [1.0, 1.0, 1.0, 1.0, 1.0]),
        (1, [0.0, 1.0, 1.0, 1.0, 1.0]),
        (2, [0.0, 0.5, 1.0, 1.0, 1.0]),
        (4, [0.0, 0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_warmup_scheduler_lr_sequence(attacker, ledger, run_logger, warmup_epochs, expected):
    base_lr = 1e-3
    trainer = make_trainer(attacker, ledger, run_logger, learning_rate=base_lr, warmup_epochs=warmup_epochs)
    scheduler = trainer._make_warmup_scheduler(warmup_epochs)

    observed = []
    for _ in expected:
        observed.append(trainer.optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert observed == pytest.approx([base_lr * f for f in expected])


def test_warmup_is_stepped_once_per_epoch(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(
        attacker, ledger, run_logger, learning_rate=1e-3, epochs=3, warmup_epochs=2, batch_size=4, checkpoint_every=0
    )
    trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=None)
    # 3 epochs -> 3 scheduler steps -> lr_lambda(3) == 1.0
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


# ----------------------------------------------------------------------------------------------
# Selection metrics
# ----------------------------------------------------------------------------------------------


def test_select_metric_mean():
    df = scored_df({0: [0.0, 1.0, 0.5], 1: [0.2, 0.2]})
    assert select_metric(df, DEFAULT_SCORE_COL, "mean") == pytest.approx((0.0 + 1.0 + 0.5 + 0.2 + 0.2) / 5)


def test_select_metric_max_is_mean_of_per_behavior_maxima():
    df = scored_df({0: [0.0, 1.0, 0.5], 1: [0.2, 0.2]})
    assert select_metric(df, DEFAULT_SCORE_COL, "max") == pytest.approx((1.0 + 0.2) / 2)


def test_select_metric_weighted_is_the_average_of_the_two():
    df = scored_df({0: [0.0, 1.0, 0.5], 1: [0.2, 0.2]})
    mean_value = 1.9 / 5
    max_value = 1.2 / 2
    assert select_metric(df, DEFAULT_SCORE_COL, "weighted") == pytest.approx((mean_value + max_value) / 2)


def test_select_metric_rejects_unknown_metric():
    df = scored_df({0: [0.5]})
    with pytest.raises(ValueError, match="Unknown selection metric"):
        select_metric(df, DEFAULT_SCORE_COL, "median")


def test_select_metric_requires_the_score_and_behavior_columns():
    df = pd.DataFrame({"jb_index": [0, 1]})
    with pytest.raises(KeyError, match=DEFAULT_SCORE_COL):
        select_metric(df, DEFAULT_SCORE_COL, "mean")


def test_judge_percentiles_avg_by_jb_index():
    # Behavior 0: 0..1 in 0.25 steps; behavior 1: constant 0.5.
    df = scored_df({0: [0.0, 0.25, 0.5, 0.75, 1.0], 1: [0.5] * 5})
    percentiles = judge_percentiles_avg_by_jb_index(df, DEFAULT_SCORE_COL)
    assert set(percentiles) == {"p50", "p90", "p95", "p99", "max"}
    assert percentiles["p50"] == pytest.approx((0.5 + 0.5) / 2)
    assert percentiles["max"] == pytest.approx((1.0 + 0.5) / 2)
    # Linear interpolation on the 5-point grid: q=0.9 -> 0.9 * 4 = index 3.6 -> 0.9.
    assert percentiles["p90"] == pytest.approx((0.9 + 0.5) / 2)


def test_judge_percentiles_averages_behaviors_not_rows():
    """A behavior with many samples must not outweigh one with few."""
    df = scored_df({0: [1.0] * 100, 1: [0.0]})
    assert judge_percentiles_avg_by_jb_index(df, DEFAULT_SCORE_COL)["max"] == pytest.approx(0.5)


def test_dpo_phase_id_format():
    assert dpo_phase_id(0, 3) == "cycle0:dpo_epoch3"


# ----------------------------------------------------------------------------------------------
# FLOPs accounting
# ----------------------------------------------------------------------------------------------


def test_flops_entries_for_a_single_batch(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger, masking_mode="prompt")
    attacker.train()
    chosen = ["abcdefgh", "ijkl"]  # padded length 8
    rejected = ["mno", "p"]  # padded length 3
    trainer.compute_dpo_loss(chosen, rejected, phase="cycle0:dpo_epoch1")

    entries = ledger.entries
    assert len(entries) == 4
    assert [e.component for e in entries] == [
        "attacker_policy",
        "attacker_policy",
        "attacker_reference",
        "attacker_reference",
    ]
    assert [e.detail["side"] for e in entries] == ["chosen", "rejected", "chosen", "rejected"]
    assert [e.n_tokens_in for e in entries] == [8, 3, 8, 3]
    assert all(e.n_tokens_out == 0 for e in entries)
    assert all(e.n_passes == 2 for e in entries)  # batch_size sequences per pass entry
    assert all(e.phase == "cycle0:dpo_epoch1" for e in entries)
    assert all(e.n_params == N_PARAMS for e in entries)

    policy, reference = entries[:2], entries[2:]
    assert all(e.pass_type == "forward_and_backward" and e.lora for e in policy)
    assert all(e.pass_type == "forward" and not e.lora for e in reference)

    # LoRA forward+backward is 4NT, dense forward is 2NT (plan §5.2).
    expected = (
        2 * pass_flops(N_PARAMS, 8, "forward_and_backward", lora=True)
        + 2 * pass_flops(N_PARAMS, 3, "forward_and_backward", lora=True)
        + 2 * pass_flops(N_PARAMS, 8, "forward", lora=False)
        + 2 * pass_flops(N_PARAMS, 3, "forward", lora=False)
    )
    assert ledger.total() == expected
    by_component = ledger.by("component")
    assert by_component[("attacker_policy",)] == 4 * N_PARAMS * 2 * (8 + 3)
    assert by_component[("attacker_reference",)] == 2 * N_PARAMS * 2 * (8 + 3)


def test_flops_totals_over_a_full_epoch(attacker, ledger, run_logger, tmp_path):
    """4 pairs, batch_size 2, 2 epochs -> 2 batches/epoch -> 4 entries * 2 * 2 = 16 entries."""
    trainer = make_trainer(attacker, ledger, run_logger, epochs=2, batch_size=2, checkpoint_every=0)
    trainer.train_cycle(make_dataset(4), cycle_id=1, run_dir=tmp_path, eval_fn=None)

    entries = ledger.entries
    assert len(entries) == 16
    assert {e.phase for e in entries} == {"cycle1:dpo_epoch1", "cycle1:dpo_epoch2"}
    assert sum(e.n_passes for e in entries if e.component == "attacker_policy") == 2 * 2 * 2 * 2
    per_phase = ledger.by("phase", "component")
    assert per_phase[("cycle1:dpo_epoch1", "attacker_policy")] > 0
    assert per_phase[("cycle1:dpo_epoch2", "attacker_reference")] > 0


def test_attack_masking_mode_lengthens_the_accounted_sequence(attacker, ledger, run_logger):
    """`attack` mode prepends prompt_len mask tokens, and the ledger must see the fed length."""
    trainer = make_trainer(attacker, ledger, run_logger, masking_mode="attack")
    attacker.train()
    trainer.compute_dpo_loss(["abcd"], ["abcd"], phase="p")
    assert all(e.n_tokens_in == 4 + PROMPT_LEN for e in ledger.entries)


# ----------------------------------------------------------------------------------------------
# Train step / cycle
# ----------------------------------------------------------------------------------------------


def test_train_step_updates_the_adapter(attacker, ledger, run_logger):
    trainer = make_trainer(attacker, ledger, run_logger, learning_rate=1e-2, mask_all=True)
    attacker.train()
    before = attacker.model.adapter.detach().clone()
    base_before = attacker.model.base.detach().clone()
    loss = trainer.train_step(["chosen a", "chosen b"], ["rejected a", "rejected b"], phase="p")

    assert isinstance(loss, float)
    assert trainer.global_step == 1
    assert not torch.allclose(before, attacker.model.adapter)
    assert torch.equal(base_before, attacker.model.base)


def test_train_cycle_without_eval_runs_all_epochs(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger, epochs=3, batch_size=4, checkpoint_every=0)
    result = trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=None)

    assert isinstance(result, CycleResult)
    assert result.epochs_run == 3
    assert len(result.losses) == 3
    assert result.best_metric is None and result.best_checkpoint is None
    assert result.stopped_early is False
    assert attacker.saved_paths == []
    assert attacker.model.training is False  # the cycle leaves the attacker in eval mode


def test_train_cycle_rejects_an_empty_dataset(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger)
    with pytest.raises(ValueError, match="empty PreferenceDataset"):
        trainer.train_cycle(PreferenceDataset([]), cycle_id=0, run_dir=tmp_path, eval_fn=None)


def test_eval_fn_receives_phase_ids(attacker, ledger, run_logger, tmp_path):
    seen: list[str] = []
    trainer = make_trainer(attacker, ledger, run_logger, epochs=3, batch_size=4, checkpoint_every=1, patience=10)
    trainer.train_cycle(
        make_dataset(4), cycle_id=2, run_dir=tmp_path, eval_fn=constant_eval_fn([0.1, 0.2, 0.3], seen=seen)
    )
    assert seen == ["cycle2:dpo_epoch1", "cycle2:dpo_epoch2", "cycle2:dpo_epoch3"]


def test_checkpoint_every_controls_eval_frequency(attacker, ledger, run_logger, tmp_path):
    seen: list[str] = []
    trainer = make_trainer(attacker, ledger, run_logger, epochs=5, batch_size=4, checkpoint_every=2, patience=10)
    trainer.train_cycle(
        make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=constant_eval_fn([0.1, 0.2, 0.3], seen=seen)
    )
    # AAPL evaluates when epoch % checkpoint_every == 0 on the 0-based epoch index.
    assert seen == ["cycle0:dpo_epoch1", "cycle0:dpo_epoch3", "cycle0:dpo_epoch5"]


# ----------------------------------------------------------------------------------------------
# Early stopping
# ----------------------------------------------------------------------------------------------


def test_improving_eval_never_stops_early(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger, epochs=4, batch_size=4, checkpoint_every=1, patience=1)
    result = trainer.train_cycle(
        make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=constant_eval_fn([0.1, 0.2, 0.3, 0.4])
    )
    assert result.epochs_run == 4
    assert result.stopped_early is False
    assert result.best_metric == pytest.approx(0.4)
    assert len(attacker.saved_paths) == 4


def test_non_improving_eval_stops_after_patience_is_exceeded(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger, epochs=10, batch_size=4, checkpoint_every=1, patience=1)
    result = trainer.train_cycle(
        make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=constant_eval_fn([0.1, 0.2, 0.15, 0.14, 0.13])
    )
    # epoch1 best, epoch2 best, epoch3 bad=1 (<= patience), epoch4 bad=2 > patience -> break
    assert result.epochs_run == 4
    assert result.stopped_early is True
    assert result.best_metric == pytest.approx(0.2)


def test_patience_zero_stops_at_the_first_non_improvement(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger, epochs=10, batch_size=4, checkpoint_every=1, patience=0)
    result = trainer.train_cycle(
        make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=constant_eval_fn([0.1, 0.2, 0.15])
    )
    assert result.epochs_run == 3
    assert result.stopped_early is True
    assert result.best_metric == pytest.approx(0.2)


def test_best_checkpoint_corresponds_to_the_best_metric(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger, epochs=5, batch_size=4, checkpoint_every=1, patience=10)
    result = trainer.train_cycle(
        make_dataset(4),
        cycle_id=3,
        run_dir=tmp_path,
        eval_fn=constant_eval_fn([0.1, 0.9, 0.5, 0.4, 0.3], attacker=attacker),
    )
    assert result.best_metric == pytest.approx(0.9)
    assert result.best_checkpoint == str(tmp_path / "checkpoints" / "best_cycle_3")
    # The fake writes its eval index into the adapter, so the file identifies the winning eval.
    assert Path(result.best_checkpoint, "adapter.txt").read_text() == "1"
    assert len(attacker.saved_paths) == 2  # only evals 0 and 1 improved


def test_eval_switches_the_model_between_eval_and_train_mode(attacker, ledger, run_logger, tmp_path):
    modes: list[bool] = []

    def eval_fn(phase: str) -> pd.DataFrame:
        modes.append(attacker.model.training)
        return scored_df({0: [0.1 * len(modes)]})

    trainer = make_trainer(attacker, ledger, run_logger, epochs=3, batch_size=4, checkpoint_every=1, patience=10)
    trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=eval_fn)
    assert modes == [False, False, False]


def test_empty_eval_frame_is_an_error(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger, epochs=1, batch_size=4, checkpoint_every=1)
    with pytest.raises(ValueError, match="no rows"):
        trainer.train_cycle(
            make_dataset(4),
            cycle_id=0,
            run_dir=tmp_path,
            eval_fn=lambda phase: pd.DataFrame(columns=["jb_index", DEFAULT_SCORE_COL]),
        )


def test_custom_score_column_is_honoured(attacker, ledger, run_logger, tmp_path):
    trainer = DiffusionDPOTrainer(
        attacker,
        DPOConfig(epochs=1, batch_size=4, checkpoint_every=1),
        ledger=ledger,
        logger=run_logger,
        score_col="judge_score_custom",
    )
    result = trainer.train_cycle(
        make_dataset(4),
        cycle_id=0,
        run_dir=tmp_path,
        eval_fn=lambda phase: scored_df({0: [0.75, 0.25]}, score_col="judge_score_custom"),
    )
    assert result.best_metric == pytest.approx(0.5)


# ----------------------------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------------------------


def test_metrics_are_written_to_metrics_jsonl(attacker, ledger, tmp_path):
    import json

    run_dir = tmp_path / "run"
    logger_obj = RunLogger(LoggingConfig(mode="disabled"), run_dir, {})
    trainer = make_trainer(attacker, ledger, logger_obj, epochs=2, batch_size=4, checkpoint_every=1, patience=10)
    trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=constant_eval_fn([0.1, 0.2]))

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    keys = {k for record in records for k in record}
    assert "train/step_loss" in keys
    assert "train/epoch_loss" in keys
    assert "train/lr" in keys
    assert "dpo_data/early_stop_metric_mean" in keys
    assert "dpo_data/avg_p90_judge_score_training" in keys
    assert "flops/total" in keys
    assert "flops/attacker_policy" in keys
    assert "flops/attacker_reference" in keys

    eval_records = [r for r in records if "flops/total" in r]
    assert eval_records[-1]["flops/total"] == ledger.total()


def test_validation_scores_are_logged_when_present(attacker, ledger, tmp_path):
    import json

    run_dir = tmp_path / "run"
    logger_obj = RunLogger(LoggingConfig(mode="disabled"), run_dir, {})
    trainer = make_trainer(attacker, ledger, logger_obj, epochs=1, batch_size=4, checkpoint_every=1)

    def eval_fn(phase: str) -> pd.DataFrame:
        df = scored_df({0: [0.1, 0.9]})
        df["judge_score_validation"] = [0.2, 0.4]
        return df

    trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=eval_fn)
    keys = {k for line in (run_dir / "metrics.jsonl").read_text().splitlines() for k in json.loads(line)}
    assert "dpo_data/avg_max_judge_score_validation" in keys


def test_global_step_does_not_reset_between_cycles(attacker, ledger, run_logger, tmp_path):
    """RunLogger is shared across cycles, and wandb rejects decreasing step numbers."""
    trainer = make_trainer(attacker, ledger, run_logger, epochs=1, batch_size=4, checkpoint_every=0)
    trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=None)
    first = trainer.global_step
    trainer.train_cycle(make_dataset(4), cycle_id=1, run_dir=tmp_path, eval_fn=None)
    assert trainer.global_step == 2 * first > 0


# ----------------------------------------------------------------------------------------------
# Optimizer state persistence
# ----------------------------------------------------------------------------------------------


def test_optimizer_state_roundtrip(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger, learning_rate=1e-2, mask_all=True)
    attacker.train()
    trainer.train_step(["chosen a", "chosen b"], ["rejected a", "rejected b"], phase="p")

    path = tmp_path / "checkpoints" / "optimizer_cycle_0"
    trainer.save_optimizer_state(path)
    assert (path / "optimizer.pt").exists()

    saved_state = trainer.optimizer.state_dict()
    fresh = make_trainer(FakeAttacker(), ledger, run_logger, learning_rate=1e-2)
    fresh.load_optimizer_state(path)
    loaded_state = fresh.optimizer.state_dict()

    assert loaded_state["param_groups"] == saved_state["param_groups"]
    assert torch.allclose(loaded_state["state"][0]["exp_avg"], saved_state["state"][0]["exp_avg"])


def test_load_optimizer_state_missing_file_raises(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(attacker, ledger, run_logger)
    with pytest.raises(FileNotFoundError):
        trainer.load_optimizer_state(tmp_path / "nowhere")


def test_train_cycle_saves_and_reloads_optimizer_state_across_cycles(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(
        attacker, ledger, run_logger, epochs=1, batch_size=4, checkpoint_every=0, load_optimizer_state=True
    )
    trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=None)
    assert (tmp_path / "checkpoints" / "optimizer_cycle_0" / "optimizer.pt").exists()

    # Cycle 1 must load cycle 0's state; the exp_avg buffer proves it was not reinitialised.
    trainer.train_cycle(make_dataset(4), cycle_id=1, run_dir=tmp_path, eval_fn=None)
    assert (tmp_path / "checkpoints" / "optimizer_cycle_1" / "optimizer.pt").exists()
    assert trainer.optimizer.state_dict()["state"][0]["step"] > 1


def test_train_cycle_does_not_touch_optimizer_state_when_disabled(attacker, ledger, run_logger, tmp_path):
    trainer = make_trainer(
        attacker, ledger, run_logger, epochs=1, batch_size=4, checkpoint_every=0, load_optimizer_state=False
    )
    trainer.train_cycle(make_dataset(4), cycle_id=0, run_dir=tmp_path, eval_fn=None)
    assert not (tmp_path / "checkpoints" / "optimizer_cycle_0").exists()
