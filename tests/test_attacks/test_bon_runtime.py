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

    gen_calls = {"count": 0}

    class FakeDefense:
        model = object()
        tokenizer = object()

        def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
            loss_calls["count"] += 1
            assert len(full_token_tensors) == 2
            assert len(prompt_token_tensors) == 2
            assert initial_batch_size == 128
            assert verbose is False
            return [1.5, 2.5]

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

    cfg = BonConfig(num_steps=2)
    attack = BonAttack(cfg)
    res = attack.run(FakeDefense(), _dataset())

    assert loss_calls["count"] == 1
    assert gen_calls["count"] == 1
    assert len(res.runs) == 1
    assert len(res.runs[0].steps) == 2
    assert res.runs[0].steps[0].model_input_tokens == [100, 101]
    assert res.runs[0].steps[0].loss == 1.5


def test_bon_uses_runtime_generator_with_defense_and_computes_loss(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.bon.prepare_conversation", _fake_prepare_conversation)

    class FakeDefense:
        model = object()
        tokenizer = object()

        def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
            assert len(full_token_tensors) == 2
            assert len(prompt_token_tensors) == 2
            assert initial_batch_size == 128
            assert verbose is False
            return [3.5, 4.5]

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

    cfg = BonConfig(num_steps=2)
    attack = BonAttack(cfg)
    res = attack.run(FakeDefense(), _dataset())

    s0, s1 = res.runs[0].steps
    assert s0.loss == 3.5 and s1.loss == 4.5
    assert s0.model_input_tokens == [111]
    assert s0.model_completions_raw == ["r0"]
    assert s0.defense_metadata == [{"applied": True}]


def test_bon_requires_input_ids(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.bon.prepare_conversation", _fake_prepare_conversation)

    class FakeDefense:
        model = object()
        tokenizer = object()

        def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
            return [1.5, 2.5]

        def generate(self, convs, **kwargs):
            return GenerationResult(gen=[["c0"], ["c1"]], input_ids=None)

    cfg = BonConfig(num_steps=2)
    attack = BonAttack(cfg)

    import pytest

    with pytest.raises(ValueError, match="BON requires `generation_result.input_ids`"):
        attack.run(FakeDefense(), _dataset())
