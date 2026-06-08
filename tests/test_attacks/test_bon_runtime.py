from __future__ import annotations

import torch

from adversariallm.attacks.bon import BonAttack, BonConfig
from adversariallm.lm_utils.text_generation import GenerationResult


def _fake_prepare_conversation(_tokenizer, _conversation):
    # Two-turn conversation: prompt tokens then assistant target tokens.
    return [
        [torch.tensor([10, 11], dtype=torch.long)],
        [torch.tensor([12, 13], dtype=torch.long)],
    ]


def _dataset():
    return [[{"role": "user", "content": "prompt"}, {"role": "assistant", "content": "target"}]]


def test_bon_uses_runtime_generator_without_defense(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.bon.prepare_conversation", _fake_prepare_conversation)

    loss_calls = {"count": 0}

    def _fake_get_losses_batched(_model, targets, token_list, initial_batch_size):
        loss_calls["count"] += 1
        assert len(targets) == 2
        assert len(token_list) == 2
        assert initial_batch_size == 128
        return [torch.arange(t.numel(), dtype=torch.float32) for t in token_list]

    monkeypatch.setattr("adversariallm.attacks.bon.get_losses_batched", _fake_get_losses_batched)

    gen_calls = {"count": 0}

    class FakeLocalTextGenerator:
        def __init__(self, _model, _tokenizer):
            pass

        def generate(self, convs, **kwargs):
            gen_calls["count"] += 1
            assert len(convs) == 2
            assert kwargs["initial_batch_size"] == 1024
            assert kwargs["verbose"] is True
            assert kwargs["top_k"] == 0
            return GenerationResult(
                gen=[["c0"], ["c1"]],
                input_ids=[[100, 101], [200, 201]],
            )

    monkeypatch.setattr("adversariallm.attacks.bon.LocalTextGenerator", FakeLocalTextGenerator)
    monkeypatch.setattr(
        "adversariallm.attacks.bon.create_defended_text_generator",
        lambda cfg, base_generator: base_generator,
    )

    cfg = BonConfig(num_steps=2)
    attack = BonAttack(cfg)
    res = attack.run(model=object(), tokenizer=object(), dataset=_dataset())

    assert loss_calls["count"] == 1
    assert gen_calls["count"] == 1
    assert len(res.runs) == 1
    assert len(res.runs[0].steps) == 2
    assert res.runs[0].steps[0].model_input_tokens == [100, 101]
    assert res.runs[0].steps[0].loss == 1.5


def test_bon_uses_runtime_generator_with_defense_and_skips_loss(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.bon.prepare_conversation", _fake_prepare_conversation)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("get_losses_batched should be skipped when runtime defense is active")

    monkeypatch.setattr("adversariallm.attacks.bon.get_losses_batched", _must_not_be_called)

    class FakeLocalTextGenerator:
        def __init__(self, _model, _tokenizer):
            pass

        def generate(self, convs, **kwargs):
            return GenerationResult(gen=[["base0"], ["base1"]], input_ids=[[1], [2]])

    class FakeDefendedGenerator:
        def generate(self, convs, **kwargs):
            assert len(convs) == 2
            assert kwargs["initial_batch_size"] == 1024
            assert kwargs["verbose"] is True
            return GenerationResult(
                gen=[["d0"], ["d1"]],
                input_ids=[[111], [222]],
                raw_gen=[["r0"], ["r1"]],
                defense_decisions=[[{"applied": True}], [{"applied": False}]],
            )

    monkeypatch.setattr("adversariallm.attacks.bon.LocalTextGenerator", FakeLocalTextGenerator)
    monkeypatch.setattr(
        "adversariallm.attacks.bon.create_defended_text_generator",
        lambda cfg, base_generator: FakeDefendedGenerator(),
    )

    cfg = BonConfig(num_steps=2)
    cfg.defense = {"type": "polyguard"}
    attack = BonAttack(cfg)
    res = attack.run(model=object(), tokenizer=object(), dataset=_dataset())

    s0, s1 = res.runs[0].steps
    assert s0.loss is None and s1.loss is None
    assert s0.model_input_tokens == [111]
    assert s0.model_completions_raw == ["r0"]
    assert s0.defense_metadata == [{"applied": True}]


def test_bon_requires_input_ids(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.bon.prepare_conversation", _fake_prepare_conversation)
    monkeypatch.setattr(
        "adversariallm.attacks.bon.get_losses_batched",
        lambda _model, targets, token_list, initial_batch_size: [torch.arange(t.numel(), dtype=torch.float32) for t in token_list],
    )

    class FakeLocalTextGenerator:
        def __init__(self, _model, _tokenizer):
            pass

        def generate(self, convs, **kwargs):
            return GenerationResult(gen=[["c0"], ["c1"]], input_ids=None)

    monkeypatch.setattr("adversariallm.attacks.bon.LocalTextGenerator", FakeLocalTextGenerator)
    monkeypatch.setattr(
        "adversariallm.attacks.bon.create_defended_text_generator",
        lambda cfg, base_generator: base_generator,
    )

    cfg = BonConfig(num_steps=2)
    attack = BonAttack(cfg)

    import pytest

    with pytest.raises(ValueError, match="BON requires `generation_result.input_ids`"):
        attack.run(model=object(), tokenizer=object(), dataset=_dataset())
