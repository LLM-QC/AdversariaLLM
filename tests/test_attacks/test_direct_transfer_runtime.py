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


def test_direct_runtime_generator(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.direct.prepare_conversation", _fake_prepare_conversation)

    class FakeDefense:
        model = object()
        tokenizer = object()

        def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
            assert len(full_token_tensors) == 1
            assert len(prompt_token_tensors) == 1
            assert initial_batch_size == 1
            assert verbose is False
            return [1.5]

        def generate(self, convs, **kwargs):
            assert len(convs) == 1
            assert kwargs["initial_batch_size"] == 1
            return GenerationResult(gen=[["direct completion"]], input_ids=[[1, 2]])

    dataset = [[{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]]
    attack = DirectAttack(DirectConfig())
    res = attack.run(FakeDefense(), dataset)

    step = res.runs[0].steps[0]
    assert step.model_completions == ["direct completion"]
    assert step.model_input_tokens == [1, 2]
    assert step.loss == 1.5
