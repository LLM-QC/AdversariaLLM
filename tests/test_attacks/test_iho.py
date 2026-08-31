"""CPU tests for the IHO sampler and transfer attack.

No real LLaDA weights are loaded: a tiny fake ``nn.Module`` + stub tokenizer are injected into
``LLaDADiffusionAttacker`` (the same trick as ``tests/test_attackers/test_diffusion.py``), and the
``TargetSystem`` is faked like ``tests/test_attacks/test_inpainting.py``. Everything runs on CPU.

``gpu01`` has GTX 1080 Ti cards, so ``torch.cuda.is_available()`` is True on the login node; these
tests therefore never touch CUDA rather than relying on a bare cuda guard (AGENTS.md).
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from adversariallm.attackers import DiffusionAttackerConfig, LLaDADiffusionAttacker
from adversariallm.attacks import iho as iho_module
from adversariallm.io_utils import get_training_manifest
from adversariallm.attacks.iho import (
    IHO_DATAFRAME_COLUMNS,
    IHOAttack,
    IHOConfig,
    IHOSample,
    IHOSampler,
)
from adversariallm.lm_utils.text_generation import GenerationResult
from adversariallm.training.flops import FlopsLedger, pass_flops

VOCAB_SIZE = 64
MASK_TOKEN_ID = 63
PROMPT_LEN = 6
DENOISE_STEPS = 4
N_PARAMS_ATT = 1_000  # RandomLogitsModel.num_parameters(exclude_embeddings=True)
N_PARAMS_TGT = 5_000


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class StubTokenizer:
    """Character-level tokenizer with right padding (copied from the diffusion attacker tests)."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token = None

    def _encode_one(self, text: str, add_special_tokens: bool) -> list[int]:
        body = [3 + (ord(c) % (self.vocab_size - 4)) for c in text]
        if add_special_tokens:
            return [self.bos_token_id] + body + [self.eos_token_id]
        return body

    def __call__(self, texts, padding=True, return_tensors="pt", add_special_tokens=True):
        if isinstance(texts, str):
            return {"input_ids": self._encode_one(texts, add_special_tokens)}
        seqs = [self._encode_one(t, add_special_tokens) for t in texts]
        width = max(len(s) for s in seqs)
        ids = [s + [self.pad_token_id] * (width - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False) -> str:
        values = [int(i) for i in ids]
        if skip_special_tokens:
            specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            values = [i for i in values if i not in specials]
        return " ".join(f"<{i}>" for i in values)

    def batch_decode(self, sequences, skip_special_tokens=False) -> list[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]


class RandomLogitsModel(nn.Module):
    """Ignores its input; seed-reproducible logits. Same shape as the diffusion attacker test."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=torch.randn(*input_ids.shape, self.vocab_size))

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        return N_PARAMS_ATT if exclude_embeddings else 2_000


class FakeTargetModel:
    """Records .cpu()/.to() so the ping-pong can be spied on; a stand-in for the target model."""

    name_or_path = "fake/target"

    def __init__(self):
        self.device = torch.device("cpu")
        self.moves: list[str] = []

    def cpu(self):
        self.moves.append("cpu")
        self.device = torch.device("cpu")
        return self

    def to(self, device):
        self.moves.append(f"to:{device}")
        self.device = torch.device(device)
        return self

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        return N_PARAMS_TGT if exclude_embeddings else 9_000


class FakeTarget:
    """Fake TargetSystem: deterministic completions, constant loss; records generate/loss calls."""

    def __init__(self, completion="harmful text here", *, with_defense=False, with_raw=False):
        self.model = FakeTargetModel()
        self.tokenizer = StubTokenizer()
        self.completion = completion
        self.with_defense = with_defense
        self.with_raw = with_raw
        self.generate_calls = 0
        self.loss_calls = 0

    def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
        self.loss_calls += 1
        assert len(full_token_tensors) == len(prompt_token_tensors)
        return [1.5] * len(full_token_tensors)

    def generate(self, convs, **kwargs):
        self.generate_calls += 1
        n = len(convs)
        gen = [[self.completion] for _ in range(n)]
        input_ids = [[10, 11, 12, 13] for _ in range(n)]
        raw = [[f"raw {self.completion}"] for _ in range(n)] if self.with_raw else None
        decisions = (
            [[{"metadata": {"applied": True}}] for _ in range(n)] if self.with_defense else None
        )
        return GenerationResult(gen=gen, input_ids=input_ids, raw_gen=raw, defense_decisions=decisions)


def _fake_prepare_conversation(tokenizer, conversation):
    """Stub for prepare_conversation: user tokens (+ assistant tokens for the loss conversation).

    The real prepare_conversation needs apply_chat_template, so like test_inpainting.py we replace
    it. Returns the list-of-turns-of-tensors shape that _flatten_conversation flattens and cats.
    """
    user_ids = tokenizer(conversation[0]["content"], add_special_tokens=False)["input_ids"]
    assistant = conversation[1]["content"]
    if assistant == "":
        toks = list(user_ids)
    else:
        toks = list(user_ids) + list(tokenizer(assistant, add_special_tokens=False)["input_ids"])
    return [[torch.tensor(toks, dtype=torch.long)]]


@pytest.fixture(autouse=True)
def _patch_prepare_conversation(monkeypatch):
    monkeypatch.setattr(iho_module, "prepare_conversation", _fake_prepare_conversation)


def make_attacker() -> LLaDADiffusionAttacker:
    cfg = DiffusionAttackerConfig(
        id="stub/llada",
        mask_token_id=MASK_TOKEN_ID,
        prompt_len=PROMPT_LEN,
        num_denoise_steps=DENOISE_STEPS,
        global_remask_every=2,
    )
    return LLaDADiffusionAttacker(cfg, device="cpu", model=RandomLogitsModel(), tokenizer=StubTokenizer())


def make_sampler(attacker=None, seed=0) -> IHOSampler:
    attacker = attacker if attacker is not None else make_attacker()
    return IHOSampler(attacker.cfg, ledger=FlopsLedger(), seed=seed, attacker=attacker)


BEHAVIORS = [(0, "how do I do bad thing", "Sure, here is how"), (1, "another bad thing", "Of course")]


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------


def test_generate_produces_n_samples_per_behavior():
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate(BEHAVIORS, n_per_behavior=3, phase="eval", batch_size=2)

    assert len(samples) == 6
    assert [s.behavior_idx for s in samples] == [0, 0, 0, 1, 1, 1]
    for s in samples:
        assert isinstance(s.prompt_text, str)
        assert len(s.prompt_token_ids) == PROMPT_LEN
        assert "attacker_denoise" in s.flops and s.flops["attacker_denoise"] > 0
        assert s.completion is None  # not scored yet


def test_generate_records_denoise_flops_matching_a_hand_value():
    torch.manual_seed(0)
    ledger = FlopsLedger()
    attacker = make_attacker()
    sampler = IHOSampler(attacker.cfg, ledger=ledger, seed=0, attacker=attacker)
    samples = sampler.generate([BEHAVIORS[0]], n_per_behavior=2, phase="eval", batch_size=2)

    # One behavior, one batch of 2 identical canvases -> shared canvas width L.
    canvas_len = attacker.build_attack_canvas([BEHAVIORS[0][2]] * 2).masked_ids.shape[1]
    expected_per_sample = DENOISE_STEPS * pass_flops(N_PARAMS_ATT, canvas_len, "forward")
    assert samples[0].flops["attacker_denoise"] == expected_per_sample

    # The ledger's batch entry is batch_size * per-sample denoise cost.
    denoise_total = ledger.by("component").get(("attacker_denoise",))
    assert denoise_total == 2 * expected_per_sample


# --------------------------------------------------------------------------------------
# generate_pooled / _draw_pooled_counts (AAPL's total-budget semantics)
# --------------------------------------------------------------------------------------


def test_draw_pooled_counts_zero_budget_or_behaviors_returns_zeros():
    sampler = make_sampler()
    assert sampler._draw_pooled_counts(4, 0, 8) == [0, 0, 0, 0]
    assert sampler._draw_pooled_counts(0, 10, 8) == []


def test_draw_pooled_counts_rejects_nonpositive_batch_size():
    sampler = make_sampler()
    with pytest.raises(ValueError):
        sampler._draw_pooled_counts(4, 10, 0)


def test_draw_pooled_counts_non_replacement_branch_is_balanced():
    """n_behaviors >= batch_size: each epoch is one full shuffle (AAPL's
    RandomSampler(replacement=False)), so a budget that is an exact multiple of n_behaviors
    always yields an exactly even split, deterministically regardless of the RNG draw."""
    sampler = make_sampler()
    counts = sampler._draw_pooled_counts(n_behaviors=5, total_budget=15, batch_size=5)
    assert counts == [3, 3, 3, 3, 3]
    assert sum(counts) == 15


def test_draw_pooled_counts_replacement_branch_totals_the_budget_and_spreads_unevenly():
    """n_behaviors < batch_size (the common case: a handful of training behaviors against a much
    larger generation batch) forces AAPL's replacement branch: i.i.d. draws in batch_size chunks
    until the running count reaches total_budget. 512 over 10 behaviors (the port's defaults) must
    total exactly 512 -- not 512 * 10 -- and need not split evenly."""
    sampler = make_sampler()
    counts = sampler._draw_pooled_counts(n_behaviors=10, total_budget=512, batch_size=128)
    assert sum(counts) == 512  # 512 is an exact multiple of batch_size=128, so no overshoot here
    assert len(counts) == 10
    assert all(c >= 0 for c in counts)
    assert len(set(counts)) > 1  # i.i.d. draws over 10 categories should not land exactly even


def test_draw_pooled_counts_last_chunk_is_not_trimmed_and_can_overshoot():
    """AAPL keeps the batch that pushes the running count past the budget whole, rather than
    trimming it -- so the total can exceed total_budget by up to batch_size - 1."""
    sampler = make_sampler()
    counts = sampler._draw_pooled_counts(n_behaviors=3, total_budget=10, batch_size=4)
    # replacement branch (3 < 4); chunks of 4 until >= 10 -> 3 chunks = 12, not trimmed to 10.
    assert sum(counts) == 12


def test_draw_pooled_counts_reproducible_under_the_same_seed():
    s1 = make_sampler(seed=7)
    s2 = make_sampler(seed=7)
    assert s1._draw_pooled_counts(10, 512, 128) == s2._draw_pooled_counts(10, 512, 128)


def test_draw_pooled_counts_differs_across_seeds():
    s1 = make_sampler(seed=1)
    s2 = make_sampler(seed=2)
    assert s1._draw_pooled_counts(10, 512, 128) != s2._draw_pooled_counts(10, 512, 128)


def test_generate_pooled_total_is_the_budget_not_budget_times_n_behaviors():
    """The whole point of the port change: training.n_samples.{train,eval,val} is a TOTAL budget
    pooled across behaviors (AAPL's num_sampled_attacks semantics), not applied to each one."""
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate_pooled(BEHAVIORS, total_budget=6, phase="cycle0:pre", batch_size=2)

    # 2 behaviors, batch_size=2 -> non-replacement (2 < 2 is False), exact multiple -> exactly 6
    # samples total, never 6 * 2 = 12 (which is what the old per-behavior generate() would give).
    assert len(samples) == 6
    counts = Counter(s.behavior_idx for s in samples)
    assert set(counts) <= {0, 1}
    assert sum(counts.values()) == 6
    for s in samples:
        assert isinstance(s.prompt_text, str)
        assert s.completion is None  # not scored yet


def test_generate_pooled_with_replacement_can_spread_the_budget_unevenly():
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate_pooled(BEHAVIORS, total_budget=20, phase="p", batch_size=8)

    # batch_size(8) > n_behaviors(2) -> replacement; chunks of 8 until >= 20 -> exactly 24.
    assert len(samples) == 24
    counts = Counter(s.behavior_idx for s in samples)
    assert sum(counts.values()) == 24


def test_generate_unaffected_by_the_pooled_addition():
    """generate() must still mean 'n_per_behavior EACH behavior', unchanged by generate_pooled."""
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate(BEHAVIORS, n_per_behavior=3, phase="eval", batch_size=2)
    assert len(samples) == 6  # 3 * 2 behaviors, exactly as before
    assert [s.behavior_idx for s in samples] == [0, 0, 0, 1, 1, 1]


# --------------------------------------------------------------------------------------
# cpu/to ping-pong
# --------------------------------------------------------------------------------------


def test_generate_parks_target_on_cpu_and_restores_it():
    torch.manual_seed(0)
    target = FakeTarget()
    target.model.to("meta")  # pretend the target starts on a non-CPU device
    target.model.moves.clear()

    sampler = make_sampler()
    sampler.target = target
    sampler.generate([BEHAVIORS[0]], n_per_behavior=1, phase="eval", batch_size=1)

    # cpu() before loading the attacker, then restored to the original device afterwards.
    assert target.model.moves[0] == "cpu"
    assert target.model.moves[-1] == "to:meta"


def test_generate_skips_ping_pong_when_target_is_none():
    torch.manual_seed(0)
    sampler = make_sampler()
    assert sampler.target is None
    # Must not raise despite there being no target to move.
    samples = sampler.generate([BEHAVIORS[0]], n_per_behavior=1, phase="eval", batch_size=1)
    assert len(samples) == 1


# --------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------


def test_score_fills_completions_loss_and_target_flops():
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate([BEHAVIORS[0]], n_per_behavior=2, phase="eval", batch_size=2)
    target = FakeTarget(with_raw=True, with_defense=True)

    sampler.score(samples, target, judges=None, phase="eval", compute_target_loss=True)

    assert target.generate_calls == 1 and target.loss_calls == 1
    for s in samples:
        assert s.completion == "harmful text here"
        assert s.completion_raw == "raw harmful text here"
        assert s.target_loss == 1.5
        assert s.defense_metadata == [{"applied": True}]
        assert s.input_token_ids == [10, 11, 12, 13]

        p = len(s.input_token_ids)
        g = len(target.tokenizer("harmful text here", add_special_tokens=False)["input_ids"])
        assert s.flops["target_generate"] == pass_flops(N_PARAMS_TGT, p + g, "forward")
        assert s.flops["target_loss"] > 0


def test_score_can_skip_target_loss():
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate([BEHAVIORS[0]], n_per_behavior=1, phase="eval", batch_size=1)
    target = FakeTarget()

    sampler.score(samples, target, judges=None, phase="eval", compute_target_loss=False)

    assert target.loss_calls == 0
    assert samples[0].target_loss is None
    assert "target_loss" not in samples[0].flops


def test_score_with_judges_writes_scores_by_role():
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate([BEHAVIORS[0]], n_per_behavior=2, phase="eval", batch_size=2)
    target = FakeTarget()

    class FakeJudge:
        def __call__(self, chats):
            return {"p_harmful": [0.9] * len(chats)}

    judges = {"training": FakeJudge(), "validation": FakeJudge()}
    sampler.score(samples, target, judges=judges, phase="eval", compute_target_loss=True)

    for s in samples:
        assert s.judge_scores == {"training": 0.9, "validation": 0.9}


# --------------------------------------------------------------------------------------
# dataframe column contract (plan §4.6)
# --------------------------------------------------------------------------------------


def test_to_dataframe_has_exactly_the_contract_columns():
    torch.manual_seed(0)
    sampler = make_sampler()
    samples = sampler.generate([BEHAVIORS[0]], n_per_behavior=2, phase="cycle3:pre", batch_size=2)
    target = FakeTarget()
    sampler.score(samples, target, judges=None, phase="cycle3:pre", compute_target_loss=True)

    df = sampler.to_dataframe(samples)

    assert list(df.columns) == list(IHO_DATAFRAME_COLUMNS)
    assert len(df) == 2
    # Load-bearing columns for build_iho_phase_table.
    assert "jb_index" in df.columns and "judge_score_training" in df.columns
    assert df["jb_index"].tolist() == [0, 0]
    assert df["cycle_id"].tolist() == [3, 3]  # parsed from "cycle3:pre"
    assert df["phase_id"].tolist() == ["cycle3:pre", "cycle3:pre"]
    assert (df["flops_total"] > 0).all()


# --------------------------------------------------------------------------------------
# IHOAttack.run
# --------------------------------------------------------------------------------------


def _patch_build_attacker(monkeypatch):
    """Make the attack build a fake CPU attacker instead of loading real LLaDA weights."""

    def fake_build(cfg, **kwargs):
        return make_attacker()

    monkeypatch.setattr(iho_module, "build_attacker", fake_build)


def _dataset():
    return [
        [{"role": "user", "content": "goal one"}, {"role": "assistant", "content": "Sure, here is how"}],
        [{"role": "user", "content": "goal two"}, {"role": "assistant", "content": "Of course"}],
    ]


def test_attack_run_shape_and_fields(monkeypatch):
    torch.manual_seed(0)
    _patch_build_attacker(monkeypatch)

    cfg = IHOConfig(num_samples_per_behavior=3, attacker_batch_size=2, train=False)
    cfg.attacker.mask_token_id = MASK_TOKEN_ID
    cfg.attacker.prompt_len = PROMPT_LEN
    attack = IHOAttack(cfg)

    target = FakeTarget(with_raw=True)
    result = attack.run(target, _dataset())

    assert len(result.runs) == 2
    for run in result.runs:
        assert len(run.steps) == 3
        for j, step in enumerate(run.steps):
            assert step.step == j
            assert step.model_completions == ["harmful text here"]
            assert step.model_completions_raw == ["raw harmful text here"]
            assert step.loss == 1.5
            assert step.flops > 0
            assert step.model_input[0]["role"] == "user"
            assert step.model_input_tokens == [10, 11, 12, 13]


def test_attack_per_step_flops_equals_component_sum(monkeypatch):
    torch.manual_seed(0)
    _patch_build_attacker(monkeypatch)

    cfg = IHOConfig(num_samples_per_behavior=1, attacker_batch_size=1, train=False)
    cfg.attacker.mask_token_id = MASK_TOKEN_ID
    cfg.attacker.prompt_len = PROMPT_LEN
    attack = IHOAttack(cfg)

    target = FakeTarget()
    result = attack.run(target, _dataset())

    step = result.runs[0].steps[0]
    # Reconstruct the three components by hand.
    canvas_len = make_attacker().build_attack_canvas(["Sure, here is how"]).masked_ids.shape[1]
    denoise = DENOISE_STEPS * pass_flops(N_PARAMS_ATT, canvas_len, "forward")
    p = 4  # FakeTarget input_ids length
    g = len(target.tokenizer("harmful text here", add_special_tokens=False)["input_ids"])
    gen = pass_flops(N_PARAMS_TGT, p + g, "forward")
    full_conv = [
        {"role": "user", "content": step.model_input[0]["content"]},
        {"role": "assistant", "content": "Sure, here is how"},
    ]
    full_len = iho_module._flatten_conversation(target.tokenizer, full_conv).size(0)
    loss = pass_flops(N_PARAMS_TGT, full_len, "forward")

    assert step.flops == denoise + gen + loss


def test_train_default_trains_and_links_provenance(monkeypatch):
    """attack=iho trains by default (train=True, no lora_checkpoint), and the training provenance
    is attached to every behavior's run as resume_metadata."""
    torch.manual_seed(0)
    _patch_build_attacker(monkeypatch)

    calls = {"n": 0}
    sentinel = {"iho_training": {"training_samples": {"phases": [], "score_columns": {}}}}

    def fake_train(self, target, dataset):
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(IHOAttack, "_train_before_attack", fake_train)

    cfg = IHOConfig(num_samples_per_behavior=2, attacker_batch_size=2)  # train defaults True
    cfg.attacker.mask_token_id = MASK_TOKEN_ID
    cfg.attacker.prompt_len = PROMPT_LEN
    cfg.training = {"train_judge": "strong_reject"}
    attack = IHOAttack(cfg)

    result = attack.run(FakeTarget(), _dataset())

    assert calls["n"] == 1
    # eval_after_train defaults True: fresh eval steps present AND provenance linked per behavior.
    assert all(len(r.steps) == 2 for r in result.runs)
    assert all(r.resume_metadata is sentinel for r in result.runs)


def test_lora_checkpoint_forces_transfer_no_training(monkeypatch):
    """A lora_checkpoint means 'evaluate this trained adapter' -> transfer, so training is skipped
    even though train defaults True."""
    torch.manual_seed(0)
    _patch_build_attacker(monkeypatch)

    calls = {"n": 0}

    def fake_train(self, target, dataset):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(IHOAttack, "_train_before_attack", fake_train)

    cfg = IHOConfig(num_samples_per_behavior=2, attacker_batch_size=2)  # train defaults True
    cfg.attacker.mask_token_id = MASK_TOKEN_ID
    cfg.attacker.prompt_len = PROMPT_LEN
    cfg.attacker.lora_checkpoint = "/some/best_cycle_3"  # => transfer
    attack = IHOAttack(cfg)

    result = attack.run(FakeTarget(), _dataset())

    assert calls["n"] == 0  # training was NOT run
    assert all(len(r.steps) == 2 for r in result.runs)
    assert all(r.resume_metadata is None for r in result.runs)


def test_eval_after_train_false_is_training_only(monkeypatch):
    """eval_after_train=False => no fresh eval steps, but each run.json still links the training
    manifest (the practical adaptive case: every training attempt is the logged attack)."""
    torch.manual_seed(0)
    _patch_build_attacker(monkeypatch)

    sentinel = {
        "iho_training": {
            "training_samples": {
                "run_dir": "/somewhere",
                "score_columns": {"strong_reject": "judge_score_training"},
                "phases": [{"phase": "pre_dpo", "cycle": 0, "path": "samples/cycle_0.parquet"}],
            }
        }
    }
    monkeypatch.setattr(IHOAttack, "_train_before_attack", lambda self, target, dataset: sentinel)

    cfg = IHOConfig(num_samples_per_behavior=3, attacker_batch_size=2)
    cfg.attacker.mask_token_id = MASK_TOKEN_ID
    cfg.attacker.prompt_len = PROMPT_LEN
    cfg.training = {"train_judge": "strong_reject"}
    cfg.eval_after_train = False
    attack = IHOAttack(cfg)

    result = attack.run(FakeTarget(), _dataset())

    assert len(result.runs) == 2
    for r in result.runs:
        assert r.steps == []                     # no fresh eval logged
        assert r.resume_metadata is sentinel     # training manifest is linked
        assert get_training_manifest(r.resume_metadata) is not None


def test_attack_default_path_leaves_scores_empty(monkeypatch):
    """Footgun §6.1: the transfer path must not write in-loop scores, so run_judges.py can score."""
    torch.manual_seed(0)
    _patch_build_attacker(monkeypatch)

    cfg = IHOConfig(num_samples_per_behavior=2, attacker_batch_size=2, train=False)
    cfg.attacker.mask_token_id = MASK_TOKEN_ID
    cfg.attacker.prompt_len = PROMPT_LEN
    attack = IHOAttack(cfg)

    result = attack.run(FakeTarget(), _dataset())
    for run in result.runs:
        for step in run.steps:
            assert step.scores == {}


# --------------------------------------------------------------------------------------
# construction: dataclass and DictConfig
# --------------------------------------------------------------------------------------


def test_construct_from_dataclass_config():
    attack = IHOAttack(IHOConfig(num_samples_per_behavior=8))
    assert attack.config.num_samples_per_behavior == 8
    assert attack.config.attacker.backend == "llada"


def test_construct_and_run_from_dictconfig(monkeypatch):
    torch.manual_seed(0)
    _patch_build_attacker(monkeypatch)

    cfg = OmegaConf.create(
        {
            "name": "iho",
            "type": "discrete",
            "version": "0.1.0",
            "generation_config": {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "num_return_sequences": 1,
            },
            "seed": 0,
            "attacker": {"backend": "llada", "mask_token_id": MASK_TOKEN_ID, "prompt_len": PROMPT_LEN},
            "num_samples_per_behavior": 2,
            "attacker_batch_size": 2,
            "compute_target_loss": True,
            "train": False,
            "training": None,
            "attacker_dir": None,
        }
    )
    attack = IHOAttack(cfg)
    result = attack.run(FakeTarget(), _dataset())
    assert len(result.runs) == 2 and len(result.runs[0].steps) == 2


def test_train_requires_training_config(monkeypatch):
    # train defaults True with no checkpoint; with no `training` block it must fail clearly
    # rather than silently running the transfer path.
    _patch_build_attacker(monkeypatch)
    cfg = IHOConfig(train=True)  # training is None
    attack = IHOAttack(cfg)
    with pytest.raises(ValueError, match="training"):
        attack.run(FakeTarget(), _dataset())
