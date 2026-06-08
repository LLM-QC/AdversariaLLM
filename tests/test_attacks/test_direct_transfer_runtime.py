from __future__ import annotations

import torch

from adversariallm.attacks.direct import DirectAttack, DirectConfig
from adversariallm.lm_utils.text_generation import GenerationResult


def _fake_prepare_conversation(_tokenizer, conversation):
    # Generation conversation has empty assistant content.
    if conversation[-1]["role"] == "assistant" and conversation[-1]["content"] == "":
        return [[torch.tensor([10, 11], dtype=torch.long)]]
    # Full conversation for loss includes assistant target tokens.
    return [
        [torch.tensor([10, 11], dtype=torch.long)],
        [torch.tensor([12, 13], dtype=torch.long)],
    ]


def _fake_get_losses_batched(_model, targets, token_list, initial_batch_size, **kwargs):
    assert len(targets) == len(token_list)
    return [torch.arange(t.numel(), dtype=torch.float32) for t in token_list]


def test_direct_runtime_generator(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.direct.prepare_conversation", _fake_prepare_conversation)
    monkeypatch.setattr("adversariallm.attacks.direct.get_losses_batched", _fake_get_losses_batched)

    class FakeLocalTextGenerator:
        def __init__(self, _model, _tokenizer):
            pass

        def generate(self, convs, **kwargs):
            assert len(convs) == 1
            assert kwargs["initial_batch_size"] == 1
            return GenerationResult(gen=[["direct completion"]], input_ids=[[1, 2]])

    monkeypatch.setattr("adversariallm.attacks.direct.LocalTextGenerator", FakeLocalTextGenerator)
    monkeypatch.setattr(
        "adversariallm.attacks.direct.create_defended_text_generator",
        lambda cfg, base_generator: base_generator,
    )

    dataset = [[{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]]
    attack = DirectAttack(DirectConfig())
    res = attack.run(model=object(), tokenizer=object(), dataset=dataset)

    step = res.runs[0].steps[0]
    assert step.model_completions == ["direct completion"]
    assert step.model_input_tokens == [1, 2]
    assert step.loss == 1.5
