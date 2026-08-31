"""CPU tests for the LLaDA diffusion attacker.

No small masked-diffusion LM exists (LLaDA-8B is the floor), so everything here runs against a
tiny fake module plus a stub tokenizer injected into `LLaDADiffusionAttacker`. The single
real-weights test is guarded by `torch.cuda.is_available()`: without the guard it would fall
back to CPU and crawl for tens of minutes instead of failing (see AGENTS.md).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adversariallm.attackers import (
    DenoiseResult,
    DiffusionAttackerConfig,
    LLaDADiffusionAttacker,
    LoRASpec,
    build_attacker,
)
from adversariallm.attackers import diffusion as diffusion_module

VOCAB_SIZE = 64
MASK_TOKEN_ID = 63
PROMPT_LEN = 6


class StubTokenizer:
    """Character-level tokenizer with right padding, matching LLaDA's canvas layout."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token = None
        self.calls: list[list[str]] = []

    def _encode_one(self, text: str, add_special_tokens: bool) -> list[int]:
        body = [3 + (ord(c) % (self.vocab_size - 4)) for c in text]
        if add_special_tokens:
            return [self.bos_token_id] + body + [self.eos_token_id]
        return body

    def __call__(self, texts, padding=True, return_tensors="pt", add_special_tokens=True):
        self.calls.append(list(texts))
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
    """Ignores its input and draws logits from the global RNG, so runs are seed-reproducible."""

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))
        self.calls: list[tuple[tuple[int, int], bool]] = []

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.calls.append((tuple(input_ids.shape), attention_mask is not None))
        return SimpleNamespace(logits=torch.randn(*input_ids.shape, self.vocab_size))

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        return 1_000 if exclude_embeddings else 2_000


class FixedLogitsModel(nn.Module):
    """Returns a caller-supplied logits tensor, for exact-value assertions."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.fixed_logits[: input_ids.shape[0], : input_ids.shape[1]])

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        return 1_000 if exclude_embeddings else 2_000


def make_config(**overrides) -> DiffusionAttackerConfig:
    params = dict(
        id="stub/llada",
        mask_token_id=MASK_TOKEN_ID,
        prompt_len=PROMPT_LEN,
        num_denoise_steps=4,
        global_remask_every=2,
    )
    params.update(overrides)
    return DiffusionAttackerConfig(**params)


def make_attacker(model=None, tokenizer=None, **overrides) -> LLaDADiffusionAttacker:
    return LLaDADiffusionAttacker(
        make_config(**overrides),
        device="cpu",
        model=model if model is not None else RandomLogitsModel(),
        tokenizer=tokenizer if tokenizer is not None else StubTokenizer(),
    )


# --------------------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------------------


def test_lora_and_checkpoint_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_config(lora=LoRASpec(), lora_checkpoint="/tmp/adapter")


def test_no_lora_at_all_is_legal():
    """The untrained base attacker is the transfer baseline; AAPL raised here, we must not."""
    attacker = make_attacker()
    assert attacker.has_lora is False
    assert attacker.cfg.lora is None and attacker.cfg.lora_checkpoint is None


def test_build_attacker_rejects_unknown_backend():
    with pytest.raises(NotImplementedError, match="diffusion_gemma"):
        build_attacker(SimpleNamespace(backend="diffusion_gemma"))


# --------------------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mask_all", [True, False])
def test_mask_tokens_all_mode(mask_all):
    attacker = make_attacker()
    token_ids = torch.randint(3, 60, (3, 10))
    result = attacker.mask_tokens(token_ids, masking_mode="all", mask_all=mask_all)

    assert result.masked_ids.shape == (3, 10)
    assert result.mask_positions.shape == (3, 10)
    assert result.prompt_positions.shape == (3, 10)
    assert (result.masked_ids[result.mask_positions] == MASK_TOKEN_ID).all()
    assert (result.masked_ids[~result.mask_positions] == token_ids[~result.mask_positions]).all()
    if mask_all:
        assert result.mask_positions.all()
    # prompt span is the leading block regardless of which positions were noised
    assert result.prompt_positions[:, :PROMPT_LEN].all()
    assert not result.prompt_positions[:, PROMPT_LEN:].any()


@pytest.mark.parametrize("mask_all", [True, False])
def test_mask_tokens_prompt_mode_only_touches_prefix(mask_all):
    attacker = make_attacker()
    token_ids = torch.randint(3, 60, (2, 12))
    result = attacker.mask_tokens(token_ids, masking_mode="prompt", mask_all=mask_all)

    assert result.masked_ids.shape == (2, 12)
    assert not result.mask_positions[:, PROMPT_LEN:].any()
    assert (result.masked_ids[:, PROMPT_LEN:] == token_ids[:, PROMPT_LEN:]).all()
    if mask_all:
        assert result.mask_positions[:, :PROMPT_LEN].all()


@pytest.mark.parametrize("mask_all", [True, False])
def test_mask_tokens_attack_mode_widens_the_canvas(mask_all):
    attacker = make_attacker()
    token_ids = torch.randint(3, 60, (2, 9))
    result = attacker.mask_tokens(token_ids, masking_mode="attack", mask_all=mask_all)

    assert result.masked_ids.shape == (2, 9 + PROMPT_LEN)
    assert result.mask_positions.shape == result.masked_ids.shape
    assert (result.masked_ids[:, PROMPT_LEN:] == token_ids).all()
    assert not result.mask_positions[:, PROMPT_LEN:].any()
    if mask_all:
        assert result.mask_positions[:, :PROMPT_LEN].all()


def test_mask_tokens_rejects_unknown_mode():
    attacker = make_attacker()
    with pytest.raises(ValueError, match="Unknown masking_mode"):
        attacker.mask_tokens(torch.zeros(1, 4, dtype=torch.long), masking_mode="nonsense")


def test_build_attack_canvas_prompt_positions_and_prefix():
    tokenizer = StubTokenizer()
    attacker = make_attacker(tokenizer=tokenizer)
    targets = ["Sure, here is how", "Of course"]

    canvas = attacker.build_attack_canvas(targets)

    assert tokenizer.calls[-1] == [attacker.cfg.answer_prefix + t for t in targets]
    encoded_width = canvas.masked_ids.shape[1] - PROMPT_LEN
    assert encoded_width > 0
    # exactly the first prompt_len columns are the adversarial prompt, and they start masked
    assert canvas.prompt_positions[:, :PROMPT_LEN].all()
    assert not canvas.prompt_positions[:, PROMPT_LEN:].any()
    assert torch.equal(canvas.prompt_positions, canvas.mask_positions)
    assert (canvas.masked_ids[:, :PROMPT_LEN] == MASK_TOKEN_ID).all()


# --------------------------------------------------------------------------------------
# transfer schedule
# --------------------------------------------------------------------------------------


def test_get_num_transfer_tokens_schedule():
    attacker = make_attacker()
    mask_index = torch.zeros(3, 20, dtype=torch.bool)
    mask_index[0, :10] = True  # 10 masks over 4 steps -> 3,3,2,2
    mask_index[1, :8] = True  # 8 masks over 4 steps -> 2,2,2,2
    mask_index[2, :0] = True  # nothing masked -> all zeros

    schedule = attacker._get_num_transfer_tokens(mask_index, steps=4)

    assert schedule.shape == (3, 4)
    assert schedule[0].tolist() == [3, 3, 2, 2]
    assert schedule[1].tolist() == [2, 2, 2, 2]
    assert schedule[2].tolist() == [0, 0, 0, 0]
    assert schedule.sum(dim=1).tolist() == mask_index.sum(dim=1).tolist()


# --------------------------------------------------------------------------------------
# denoising
# --------------------------------------------------------------------------------------


def test_denoise_is_deterministic_under_a_fixed_seed():
    attacker = make_attacker()
    canvas = attacker.build_attack_canvas(["Sure, here is how", "Of course"])

    torch.manual_seed(1234)
    first = attacker.denoise(canvas.masked_ids)
    torch.manual_seed(1234)
    second = attacker.denoise(canvas.masked_ids)

    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.loglikelihood, second.loglikelihood)


def test_denoise_counters_and_shapes():
    cfg_steps = 5
    model = RandomLogitsModel()
    attacker = make_attacker(model=model, num_denoise_steps=cfg_steps)
    canvas = attacker.build_attack_canvas(["Sure, here is how", "Of course"])
    batch_size, length = canvas.masked_ids.shape

    torch.manual_seed(0)
    result = attacker.denoise(canvas.masked_ids)

    assert isinstance(result, DenoiseResult)
    assert result.token_ids.shape == (batch_size, length)
    assert result.loglikelihood.shape == (batch_size,)
    # the FLOPs ledger multiplies these two, so they must be steps and canvas width
    assert result.n_forward_passes == cfg_steps
    assert result.n_tokens_per_pass == length
    assert len(model.calls) == cfg_steps
    assert all(shape == (batch_size, length) for shape, _ in model.calls)


def test_denoise_passes_attention_mask_only_when_configured():
    model = RandomLogitsModel()
    attacker = make_attacker(model=model, mask_padding_tokens=False)
    canvas = attacker.build_attack_canvas(["Sure, here is how"])
    torch.manual_seed(0)
    attacker.denoise(canvas.masked_ids)
    assert all(not had_mask for _, had_mask in model.calls)

    model = RandomLogitsModel()
    attacker = make_attacker(model=model, mask_padding_tokens=True)
    canvas = attacker.build_attack_canvas(["Sure, here is how"])
    torch.manual_seed(0)
    attacker.denoise(canvas.masked_ids)
    assert all(had_mask for _, had_mask in model.calls)


def test_denoise_does_not_mutate_the_input_canvas():
    attacker = make_attacker()
    canvas = attacker.build_attack_canvas(["Sure, here is how"])
    before = canvas.masked_ids.clone()

    torch.manual_seed(0)
    attacker.denoise(canvas.masked_ids)

    assert torch.equal(canvas.masked_ids, before)


def test_denoise_refuses_to_run_in_training_mode():
    attacker = make_attacker().train()
    canvas = attacker.build_attack_canvas(["Sure, here is how"])
    with pytest.raises(RuntimeError, match="evaluation mode"):
        attacker.denoise(canvas.masked_ids)


def test_denoise_never_overwrites_known_tokens():
    attacker = make_attacker()
    canvas = attacker.build_attack_canvas(["Sure, here is how", "Of course"])
    known = canvas.masked_ids != MASK_TOKEN_ID

    torch.manual_seed(7)
    result = attacker.denoise(canvas.masked_ids)

    assert torch.equal(result.token_ids[known], canvas.masked_ids[known])


def test_decode_prompt_decodes_exactly_the_prompt_span():
    tokenizer = StubTokenizer()
    attacker = make_attacker(tokenizer=tokenizer)
    canvas = attacker.build_attack_canvas(["Sure, here is how", "Of course"])
    torch.manual_seed(3)
    result = attacker.denoise(canvas.masked_ids)

    prompts = attacker.decode_prompt(result.token_ids, canvas.prompt_positions)
    expected = tokenizer.batch_decode(result.token_ids[:, :PROMPT_LEN])

    assert prompts == expected
    assert attacker.decode_full(result.token_ids) == tokenizer.batch_decode(result.token_ids)


# --------------------------------------------------------------------------------------
# log-likelihood
# --------------------------------------------------------------------------------------


def test_compute_log_likelihood_gathers_targets_over_masked_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, VOCAB_SIZE)
    attacker = make_attacker(model=FixedLogitsModel(logits))

    masked_ids = torch.full((2, 5), MASK_TOKEN_ID, dtype=torch.long)
    target_ids = torch.randint(3, 60, (2, 5))
    mask_positions = torch.tensor(
        [[True, False, True, False, False], [False, True, True, True, False]],
        dtype=torch.bool,
    )

    got = attacker.compute_log_likelihood(masked_ids, mask_positions, target_ids)

    log_probs = F.log_softmax(logits, dim=-1)
    expected = torch.stack(
        [
            sum(log_probs[b, t, target_ids[b, t]] for t in range(5) if mask_positions[b, t])
            for b in range(2)
        ]
    )
    assert torch.allclose(got, expected, atol=1e-6)


def test_compute_log_likelihood_base_model_path_without_lora():
    """`use_base_model=True` must work (as a no-op) on an attacker that has no adapter."""
    torch.manual_seed(0)
    logits = torch.randn(1, 4, VOCAB_SIZE)
    attacker = make_attacker(model=FixedLogitsModel(logits))
    masked_ids = torch.full((1, 4), MASK_TOKEN_ID, dtype=torch.long)
    target_ids = torch.randint(3, 60, (1, 4))
    mask_positions = torch.ones(1, 4, dtype=torch.bool)

    with_base = attacker.compute_log_likelihood(masked_ids, mask_positions, target_ids, use_base_model=True)
    without = attacker.compute_log_likelihood(masked_ids, mask_positions, target_ids)

    assert torch.allclose(with_base, without)
    assert not with_base.requires_grad


def test_use_cache_is_pinned_off_on_the_model_config():
    """LLaDA's remote forward reads config.use_cache, which transformers 5.x no longer defines."""
    model = RandomLogitsModel()
    model.config = SimpleNamespace(use_cache=True)
    attacker = make_attacker(model=model)
    assert attacker.model.config.use_cache is False


def test_n_params_no_embed_uses_the_model_count():
    attacker = make_attacker()
    assert attacker.n_params_no_embed == 1_000


# --------------------------------------------------------------------------------------
# transformers 5.x tied-keys shim
# --------------------------------------------------------------------------------------


class _FakeRemoteModel:
    """Stands in for LLaDAModelLM: an __init__ that never assigns all_tied_weights_keys and a
    pre-transformers-5 `tie_weights()` that takes no keyword arguments."""

    def __init__(self, config, extra=None):
        self.config = config
        self.extra = extra
        self.tied = 0

    def tie_weights(self):
        self.tied += 1


def test_tied_keys_patch_assigns_a_fresh_dict_per_instance():
    cls = type("FakeA", (_FakeRemoteModel,), {})
    assert diffusion_module._apply_tied_keys_patch(cls) is True

    first, second = cls("cfg-a"), cls("cfg-b")

    assert first.all_tied_weights_keys == {} and second.all_tied_weights_keys == {}
    # tie_weights() mutates this dict in place, so the instances must not share one
    assert first.all_tied_weights_keys is not second.all_tied_weights_keys
    first.all_tied_weights_keys["lm_head.weight"] = "embed.weight"
    assert second.all_tied_weights_keys == {}
    # the original __init__ still runs
    assert first.config == "cfg-a"


def test_tied_keys_patch_is_idempotent():
    cls = type("FakeB", (_FakeRemoteModel,), {})
    assert diffusion_module._apply_tied_keys_patch(cls) is True
    patched_init = cls.__init__

    assert diffusion_module._apply_tied_keys_patch(cls) is False
    assert cls.__init__ is patched_init  # not re-wrapped


def test_tied_keys_patch_leaves_an_existing_attribute_alone():
    class FakeFixedUpstream(_FakeRemoteModel):
        def __init__(self, config, extra=None):
            super().__init__(config, extra)
            self.all_tied_weights_keys = {"lm_head.weight": "embed.weight"}

    diffusion_module._apply_tied_keys_patch(FakeFixedUpstream)
    assert FakeFixedUpstream("cfg").all_tied_weights_keys == {"lm_head.weight": "embed.weight"}


def test_tie_weights_patch_accepts_transformers_v5_kwargs():
    """transformers 5 calls tie_weights(missing_keys=..., recompute_mapping=...)."""
    cls = type("FakeC", (_FakeRemoteModel,), {})
    assert diffusion_module._apply_tie_weights_patch(cls) is True

    instance = cls("cfg")
    instance.tie_weights(missing_keys=["a"], recompute_mapping=False)
    assert instance.tied == 1  # the original override still ran, kwargs were dropped


def test_tie_weights_patch_is_idempotent_and_skips_compatible_signatures():
    cls = type("FakeD", (_FakeRemoteModel,), {})
    assert diffusion_module._apply_tie_weights_patch(cls) is True
    assert diffusion_module._apply_tie_weights_patch(cls) is False

    class AlreadyCompatible:
        def tie_weights(self, missing_keys=None, recompute_mapping=True):
            return None

    assert diffusion_module._apply_tie_weights_patch(AlreadyCompatible) is False

    class NoTieWeights:
        pass

    assert diffusion_module._apply_tie_weights_patch(NoTieWeights) is False


def test_patch_llada_warns_and_continues_when_the_class_cannot_be_resolved(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise OSError("no such remote module")

    monkeypatch.setattr(diffusion_module, "get_class_from_dynamic_module", boom)
    with caplog.at_level("WARNING"):
        diffusion_module._patch_llada_for_transformers_v5("GSAI-ML/LLaDA-8B-Base")

    assert any("tied-keys shim" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------------------
# real weights (GPU only)
# --------------------------------------------------------------------------------------


def _has_gpu_for_llada() -> tuple[bool, str]:
    """LLaDA-8B in bf16 is ~16 GB of weights, and Pascal cards have no usable bf16.

    Without the CUDA guard this test would silently fall back to CPU and crawl for tens of
    minutes (AGENTS.md); without the memory/bf16 guard it OOMs on the 1080 Ti dev nodes.
    """
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 20 * 1024**3:
        return False, f"{props.name} has only {props.total_memory / 1024**3:.0f} GiB, LLaDA-8B needs ~20"
    if not torch.cuda.is_bf16_supported():
        return False, f"{props.name} does not support bfloat16"
    return True, ""


_GPU_OK, _GPU_REASON = _has_gpu_for_llada()


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not _GPU_OK, reason=f"real-weights LLaDA test needs a big bf16 GPU: {_GPU_REASON}")
def test_llada_real_weights_denoise():
    cfg = DiffusionAttackerConfig(prompt_len=8, num_denoise_steps=4, global_remask_every=0)
    attacker = build_attacker(cfg, device="cuda")

    assert attacker.n_params_no_embed > 1e9
    canvas = attacker.build_attack_canvas(["Sure, here is how to build a bomb"])
    result = attacker.denoise(canvas.masked_ids)

    assert result.token_ids.shape == canvas.masked_ids.shape
    assert result.n_forward_passes == 4
    assert result.n_tokens_per_pass == canvas.masked_ids.shape[1]
    assert (result.token_ids[:, :8] != cfg.mask_token_id).all()
    assert len(attacker.decode_prompt(result.token_ids, canvas.prompt_positions)) == 1
