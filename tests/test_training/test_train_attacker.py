"""CPU-only tests for the phase-3 training entrypoint (``train_attacker.py``).

No real weights, GPU, network, or slurm: a single combined fake attacker implements both the
generation side the :class:`IHOSampler` needs (``build_attack_canvas`` / ``denoise`` / decode) and
the DPO side the :class:`DiffusionDPOTrainer` needs (``encode`` / ``mask_tokens`` /
``compute_log_likelihood`` / ``disable_adapter`` with a real trainable parameter). The target and
judge are faked like ``tests/test_attacks/test_iho.py`` and ``tests/test_training/test_dpo.py``.

The whole cycle loop is driven with tiny idx lists, so a real cycle runs end to end (sample ->
pair -> DPO -> checkpoint) without any of the cost of real models.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# The training engine now lives in the library; alias keeps `train_attacker.X` patch targets valid.
from adversariallm.training import attacker_loop as train_attacker
from adversariallm.attackers.base import DenoiseResult, MaskingResult
from adversariallm.attacks import iho as iho_module
from adversariallm.attacks.iho import IHOAttack, IHOConfig
from adversariallm.lm_utils.text_generation import GenerationResult
from adversariallm.training.attacker_loop import resolve_attacker_id, train_attacker_run

VOCAB = 32
MASK_ID = 31
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
PROMPT_LEN = 4
DENOISE_STEPS = 3
N_PARAMS = 1_000


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class StubTokenizer:
    def __init__(self, vocab_size: int = VOCAB):
        self.vocab_size = vocab_size
        self.pad_token_id = PAD_ID
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID
        self.mask_token = None

    def _encode_one(self, text: str, add_special_tokens: bool) -> list[int]:
        body = [3 + (ord(c) % (self.vocab_size - 5)) for c in text]
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


class _TinyModel(nn.Module):
    """Frozen base + trainable adapter one-hot lookup, so real gradients flow to the adapter."""

    def __init__(self, vocab_size: int = VOCAB) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.base = nn.Parameter(torch.randn(vocab_size, vocab_size), requires_grad=False)
        self.adapter = nn.Parameter(0.01 * torch.randn(vocab_size, vocab_size), requires_grad=True)
        self.name_or_path = "fake/attacker"

    def logits(self, token_ids: torch.Tensor, *, use_adapter: bool) -> torch.Tensor:
        one_hot = F.one_hot(token_ids.clamp(max=self.base.shape[0] - 1), self.base.shape[0]).float()
        weight = self.base + self.adapter if use_adapter else self.base
        return one_hot @ weight

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        return N_PARAMS if exclude_embeddings else 2 * N_PARAMS


class FakeTrainableAttacker:
    """Full attacker: generation side (denoise/decode) + DPO side (encode/mask/log-likelihood)."""

    def __init__(self, prompt_len: int = PROMPT_LEN, steps: int = DENOISE_STEPS):
        self.model = _TinyModel()
        self.tokenizer = StubTokenizer()
        self.device = torch.device("cpu")
        self.has_lora = True
        self.model_id = "fake/attacker"
        self.prompt_len = prompt_len
        self.steps = steps
        self.mask_token_id = MASK_ID
        self.adapter_disabled = False
        self.tag = ""
        self.saved_paths: list[str] = []
        self.loaded_paths: list[str] = []

    @property
    def n_params_no_embed(self) -> int:
        return N_PARAMS

    def train(self) -> "FakeTrainableAttacker":
        self.model.train()
        return self

    def eval(self) -> "FakeTrainableAttacker":
        self.model.eval()
        return self

    def to(self, device) -> "FakeTrainableAttacker":
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

    def load_lora(self, path: str) -> None:
        self.tag = Path(path, "adapter.txt").read_text()
        self.loaded_paths.append(path)

    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts, add_special_tokens=True)["input_ids"]

    def build_attack_canvas(self, target_texts: list[str]) -> MaskingResult:
        ids = self.encode(target_texts)
        return self.mask_tokens(ids, masking_mode="attack", mask_all=True)

    def mask_tokens(self, token_ids: torch.Tensor, masking_mode: str, mask_all: bool = False) -> MaskingResult:
        batch, length = token_ids.shape
        K = self.prompt_len
        if masking_mode == "all":
            eligible = torch.ones((batch, length), dtype=torch.bool)
        elif masking_mode == "prompt":
            eligible = torch.zeros((batch, length), dtype=torch.bool)
            eligible[:, :K] = True
        elif masking_mode == "attack":
            block = token_ids.new_full((batch, K), self.mask_token_id)
            token_ids = torch.cat([block, token_ids], dim=1)
            eligible = torch.zeros(token_ids.shape, dtype=torch.bool)
            eligible[:, :K] = True
        else:
            raise ValueError(masking_mode)

        if mask_all:
            mask_positions = eligible.clone()
        else:
            p = torch.rand(batch).view(batch, 1)
            mask_positions = (torch.rand(token_ids.shape) < p) & eligible

        masked = token_ids.clone()
        masked[mask_positions] = self.mask_token_id
        prompt_positions = torch.zeros(eligible.shape, dtype=torch.bool)
        prompt_positions[:, :K] = True
        return MaskingResult(masked_ids=masked, mask_positions=mask_positions, prompt_positions=prompt_positions)

    def extract_prompt_ids(self, denoised_ids: torch.Tensor, prompt_positions: torch.Tensor) -> torch.Tensor:
        return denoised_ids[prompt_positions].view(denoised_ids.size(0), -1)

    def decode_prompt(self, denoised_ids: torch.Tensor, prompt_positions: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(self.extract_prompt_ids(denoised_ids, prompt_positions))

    def decode_full(self, denoised_ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(denoised_ids)

    @torch.no_grad()
    def denoise(self, masked_ids: torch.Tensor) -> DenoiseResult:
        x = masked_ids.clone()
        mask = x == self.mask_token_id
        logits = self.model.logits(x, use_adapter=not self.adapter_disabled)
        # Avoid re-emitting the mask id so decoded prompts look like content, not masks.
        logits[..., self.mask_token_id] = -float("inf")
        filled = torch.argmax(logits, dim=-1)
        x = torch.where(mask, filled, x)
        return DenoiseResult(
            token_ids=x,
            loglikelihood=torch.zeros(x.size(0)),
            n_forward_passes=self.steps,
            n_tokens_per_pass=x.size(1),
        )

    def compute_log_likelihood(self, masked_ids, mask_positions, target_ids, use_base_model=False):
        log_probs = F.log_softmax(self.model.logits(masked_ids, use_adapter=not self.adapter_disabled), dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return (gathered * mask_positions).sum(dim=1)


class FakeTargetModel:
    name_or_path = "fake/target"

    def __init__(self):
        self.device = torch.device("cpu")

    def cpu(self):
        self.device = torch.device("cpu")
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        return 5_000 if exclude_embeddings else 9_000


class FakeTarget:
    NAME = "none"

    def __init__(self, completion="harmful text"):
        self.model = FakeTargetModel()
        self.tokenizer = StubTokenizer()
        self.completion = completion

    def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
        return [1.5] * len(full_token_tensors)

    def generate(self, convs, **kwargs):
        n = len(convs)
        gen = [[self.completion] for _ in range(n)]
        input_ids = [[10, 11, 12] for _ in range(n)]
        return GenerationResult(gen=gen, input_ids=input_ids, raw_gen=None, defense_decisions=None)


class FakeJudge:
    """Constant harmfulness >= min_score so preference pairs are always produced."""

    def __init__(self, value: float = 0.9):
        self.value = value

    def __call__(self, chats):
        return {"p_harmful": [self.value] * len(chats)}


class FakeDataset:
    def __init__(self, n: int = 4):
        self._data = [
            [{"role": "user", "content": f"goal {i}"}, {"role": "assistant", "content": f"Sure {i}"}]
            for i in range(n)
        ]
        self.config = SimpleNamespace(name="fake_ds", seed=0, shuffle=True, idx=None)
        self.idx = torch.arange(n)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int):
        return self._data[i]


def _fake_prepare_conversation(tokenizer, conversation):
    user_ids = tokenizer(conversation[0]["content"], add_special_tokens=False)["input_ids"]
    assistant = conversation[1]["content"]
    toks = list(user_ids)
    if assistant:
        toks += list(tokenizer(assistant, add_special_tokens=False)["input_ids"])
    return [[torch.tensor(toks, dtype=torch.long)]]


@pytest.fixture(autouse=True)
def _patch_prepare_conversation(monkeypatch):
    monkeypatch.setattr(iho_module, "prepare_conversation", _fake_prepare_conversation)


# --------------------------------------------------------------------------------------
# Config / helpers
# --------------------------------------------------------------------------------------


def training_block(**over) -> dict:
    block = {
        "attacker": {"backend": "llada", "id": "stub/llada", "prompt_len": PROMPT_LEN, "num_denoise_steps": DENOISE_STEPS},
        "dpo": {"epochs": 1, "batch_size": 4, "checkpoint_every": 1, "patience": 10, "masking_mode": "prompt"},
        "pairing": {"min_score": 0.2, "percent_chosen": 0.125, "expanding": False},
        "logging": {"mode": "disabled"},
        "n_cycles": 1,
        "attacker_batch_size": 4,
        "train_idx": None,
        "val_idx": None,
        "eval_idx": None,
        # TOTAL budgets pooled across FakeDataset's 4 behaviors (IHOSampler.generate_pooled).
        # attacker_batch_size == 4 == n_behaviors below, so the pooled draw is non-replacement
        # (full-dataset shuffles), and 32 is an exact multiple of 4 -> deterministically 8
        # samples/behavior, matching what the old per-behavior "8" used to produce, so
        # percent_chosen=0.125 still yields k=1 pair per behavior group.
        "n_samples": {"train": 32, "eval": 32, "val": 32},
        "train_judge": "fake_judge",
        "val_judges": [],
        "seed": 0,
        "attacker_id": "auto",
        "save_cycle_samples": True,
        "save_dpo_samples": True,
        "resume": True,
    }
    block.update(over)
    return block


GEN_CFG = {"max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "num_return_sequences": 1}


def run_training(tmp_path, *, attacker=None, training=None, **run_over):
    return train_attacker_run(
        target=FakeTarget(),
        dataset=FakeDataset(),
        training=training or training_block(),
        model_id="fake/model",
        dataset_name="fake_ds",
        defense_name="none",
        defense_params=None,
        attacker_dir=str(tmp_path),
        generation_config=GEN_CFG,
        attacker=attacker or FakeTrainableAttacker(),
        judge_loader=lambda name: FakeJudge(),
        **run_over,
    )


# --------------------------------------------------------------------------------------
# attacker_id (plan §4.4)
# --------------------------------------------------------------------------------------


def _resolve(training: dict, **over) -> str:
    kwargs = {
        "model_id": "fake/model",
        "model_short": "model",
        "dataset_name": "fake_ds",
        "defense_name": "none",
        "defense_params": None,
    }
    kwargs.update(over)
    return resolve_attacker_id(training.get("attacker_id", "auto"), training=training, **kwargs)


def test_attacker_id_is_deterministic():
    assert _resolve(training_block()) == _resolve(training_block())


def test_attacker_id_format():
    aid = _resolve(training_block())
    assert aid.startswith("model_fake_ds_none_")
    assert len(aid.split("_")[-1]) == 12


def test_attacker_id_changes_with_hash_relevant_fields():
    base = _resolve(training_block())
    assert _resolve(training_block(seed=1)) != base
    assert _resolve(training_block(n_cycles=2)) != base
    assert _resolve(training_block(train_judge="other")) != base
    assert _resolve(training_block(n_samples={"train": 16, "eval": 8, "val": 8})) != base
    assert _resolve(training_block(dpo={"epochs": 2})) != base
    assert _resolve(training_block(pairing={"min_score": 0.5})) != base
    assert _resolve(training_block(train_idx=[0, 1])) != base
    assert _resolve(base_training := training_block(), model_id="other/model") != base


def test_attacker_id_ignores_logging_and_save_flags():
    base = _resolve(training_block())
    assert _resolve(training_block(logging={"mode": "online", "project": "x"})) == base
    assert _resolve(training_block(save_cycle_samples=False)) == base
    assert _resolve(training_block(save_dpo_samples=False)) == base
    assert _resolve(training_block(resume=False)) == base
    assert _resolve(training_block(val_idx=[0], val_judges=["harmbench"])) == base


def test_explicit_attacker_id_is_used_verbatim():
    assert _resolve(training_block(attacker_id="my_run")) == "my_run"


# --------------------------------------------------------------------------------------
# Run-directory layout (plan §4.3)
# --------------------------------------------------------------------------------------


def test_run_dir_layout_matches_the_contract(tmp_path):
    result = run_training(tmp_path)
    run = Path(result.run_dir)

    assert run.name == result.attacker_id
    assert (run / "train_config.yaml").exists()
    assert (run / "flops.json").exists()
    assert (run / "metrics.jsonl").exists()
    assert (run / "samples" / "cycle_0.parquet").exists()
    assert (run / "dpo_samples" / "cycle_0_epoch_1.parquet").exists()
    assert (run / "prefs" / "cycle_0.parquet").exists()
    assert (run / "checkpoints" / "best_cycle_0").is_dir()
    assert (run / "best").is_symlink()
    # No validation split was configured, so its directory must not appear.
    assert not (run / "samples_validation").exists()

    assert result.best_checkpoint == str(run / "checkpoints" / "best_cycle_0")
    assert result.best_metric == pytest.approx(0.9)
    assert result.winning_cycle == 0
    assert result.n_behaviors_amortised == 4
    assert result.total_training_flops > 0


def test_validation_split_is_written_when_configured(tmp_path):
    result = run_training(tmp_path, training=training_block(val_idx=[0, 1]))
    assert (Path(result.run_dir) / "samples_validation" / "cycle_0.parquet").exists()


def test_sample_parquet_has_the_column_contract(tmp_path):
    import pandas as pd

    result = run_training(tmp_path)
    df = pd.read_parquet(Path(result.run_dir) / "samples" / "cycle_0.parquet")
    assert list(df.columns) == list(iho_module.IHO_DATAFRAME_COLUMNS)
    assert df["judge_score_training"].notna().all()
    assert set(df["jb_index"]) == {0, 1, 2, 3}


def test_n_samples_train_is_a_pooled_total_not_a_per_behavior_count(tmp_path):
    """training.n_samples.train=32 over FakeDataset's 4 behaviors must produce 32 rows total
    (AAPL's num_sampled_attacks semantics), not 32 * 4 = 128 (the port's old per-behavior meaning).
    """
    import pandas as pd

    result = run_training(tmp_path)
    df = pd.read_parquet(Path(result.run_dir) / "samples" / "cycle_0.parquet")
    assert len(df) == 32
    # attacker_batch_size(4) == n_behaviors(4): non-replacement branch, so a budget that is an
    # exact multiple of n_behaviors is deterministically split evenly.
    assert df["jb_index"].value_counts().to_dict() == {0: 8, 1: 8, 2: 8, 3: 8}


# --------------------------------------------------------------------------------------
# Resume (plan §4.4)
# --------------------------------------------------------------------------------------


def test_resume_continues_from_the_last_valid_cycle(tmp_path):
    import shutil

    # A full 2-cycle run, then simulate an interruption after cycle 0 (drop cycle 1's artifacts).
    first = run_training(tmp_path, training=training_block(n_cycles=2))
    run = Path(first.run_dir)
    assert (run / "checkpoints" / "best_cycle_1").is_dir()

    shutil.rmtree(run / "checkpoints" / "best_cycle_1")
    (run / "samples" / "cycle_1.parquet").unlink()
    # Also drop cycle 0's pre-samples so a correct resume proves it does NOT regenerate them.
    (run / "samples" / "cycle_0.parquet").unlink()

    second = run_training(tmp_path, training=training_block(n_cycles=2))
    assert second.run_dir == first.run_dir  # identical config -> same dir
    assert not (run / "samples" / "cycle_0.parquet").exists()  # cycle 0 was skipped
    assert (run / "samples" / "cycle_1.parquet").exists()  # cycle 1 was re-run
    assert (run / "checkpoints" / "best_cycle_1").is_dir()


def test_each_cycle_reloads_the_previous_cycles_best_adapter(tmp_path):
    """AAPL parity: before cycle i>0 the attacker is reset to best_cycle_{i-1}, discarding the
    early-stop-overshoot epochs. Cycle 0 must NOT reload (it holds the freshly injected LoRA)."""
    attacker = FakeTrainableAttacker()
    result = run_training(tmp_path, attacker=attacker, training=training_block(n_cycles=3))
    run = Path(result.run_dir)

    # One reload per transition into cycles 1 and 2, each from the prior cycle's best checkpoint.
    assert attacker.loaded_paths == [
        str(run / "checkpoints" / "best_cycle_0"),
        str(run / "checkpoints" / "best_cycle_1"),
    ]


def test_resume_reloads_from_the_resumed_checkpoint(tmp_path):
    """On resume, the newly built attacker already holds best_cycle_{start-1}; the first resumed
    cycle must not reload, and the next transition reloads from the just-produced best."""
    import shutil

    run_training(tmp_path, training=training_block(n_cycles=3))
    first_run = run_training(tmp_path, training=training_block(n_cycles=3))  # settle dir
    run = Path(first_run.run_dir)
    # Interrupt after cycle 0: drop cycles 1 and 2 so resume restarts at cycle 1.
    for c in (1, 2):
        shutil.rmtree(run / "checkpoints" / f"best_cycle_{c}", ignore_errors=True)
        (run / "samples" / f"cycle_{c}.parquet").unlink(missing_ok=True)

    attacker = FakeTrainableAttacker()
    run_training(tmp_path, attacker=attacker, training=training_block(n_cycles=3))
    # start_cycle == 1: no reload entering cycle 1 (attacker built from resume_ckpt=best_cycle_0),
    # exactly one reload entering cycle 2 from best_cycle_1.
    assert attacker.loaded_paths == [str(run / "checkpoints" / "best_cycle_1")]


def test_resume_disabled_reruns_from_scratch(tmp_path):
    run_training(tmp_path, training=training_block(n_cycles=1))
    result = run_training(tmp_path, training=training_block(n_cycles=1, resume=False))
    # Same id (resume flag is excluded from the hash), cycle 0 regenerated.
    assert (Path(result.run_dir) / "samples" / "cycle_0.parquet").exists()


# --------------------------------------------------------------------------------------
# train_before_attack wiring (plan §4.1 / §5.3)
# --------------------------------------------------------------------------------------


def test_train_before_attack_sets_checkpoint_and_records_provenance(tmp_path, monkeypatch):
    torch.manual_seed(0)
    # Both the training loop and the evaluation phase must build a fake attacker, not real LLaDA.
    monkeypatch.setattr(train_attacker, "build_attacker", lambda cfg, **k: FakeTrainableAttacker())
    monkeypatch.setattr(train_attacker, "_default_judge_loader", lambda name: FakeJudge())
    monkeypatch.setattr(iho_module, "build_attacker", lambda cfg, **k: FakeTrainableAttacker())

    cfg = IHOConfig(num_samples_per_behavior=2, attacker_batch_size=2, train=True)
    cfg.attacker.prompt_len = PROMPT_LEN
    cfg.attacker.num_denoise_steps = DENOISE_STEPS
    cfg.attacker_dir = str(tmp_path)
    cfg.training = training_block(n_cycles=1)

    attack = IHOAttack(cfg)
    result = attack.run(FakeTarget(), FakeDataset())

    # The evaluation attacker now points at the winning adapter.
    assert cfg.attacker.lora_checkpoint is not None
    assert cfg.attacker.lora is None
    assert "checkpoints/best_cycle_0" in cfg.attacker.lora_checkpoint

    # Every run carries the training provenance, with training FLOPs kept separable (§5.3).
    assert len(result.runs) == 4
    for run in result.runs:
        prov = run.resume_metadata["iho_training"]
        assert prov["checkpoint"] == cfg.attacker.lora_checkpoint
        assert prov["total_training_flops"] > 0
        assert prov["n_behaviors_amortised"] == 4
        assert prov["winning_cycle"] == 0


def test_transfer_path_still_has_no_training_provenance(tmp_path, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(iho_module, "build_attacker", lambda cfg, **k: FakeTrainableAttacker())

    cfg = IHOConfig(num_samples_per_behavior=2, attacker_batch_size=2, train=False)
    cfg.attacker.prompt_len = PROMPT_LEN
    cfg.attacker.num_denoise_steps = DENOISE_STEPS

    result = IHOAttack(cfg).run(FakeTarget(), FakeDataset())
    for run in result.runs:
        assert run.resume_metadata is None
