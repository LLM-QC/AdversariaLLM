from __future__ import annotations

import csv
from pathlib import Path

import torch

from adversariallm.attacks.inpainting import InpaintingAttack, InpaintingConfig
from adversariallm.lm_utils.text_generation import GenerationResult


def _write_inpainting_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original_prompt", "inpainted_prompt"])
        writer.writeheader()
        writer.writerow({"original_prompt": "orig prompt", "inpainted_prompt": "inpainted prompt"})


def _fake_prepare_conversation(_tokenizer, conversation):
    # In inference mode, assistant content is empty.
    if conversation[-1]["role"] == "assistant" and conversation[-1]["content"] == "":
        return [[torch.tensor([10, 11, 12], dtype=torch.long)]]
    # Full conversation (used for loss) includes a non-empty assistant target.
    return [[torch.tensor([10, 11, 12, 13], dtype=torch.long)]]


def test_inpainting_uses_runtime_generator_without_defense(monkeypatch, tmp_path):
    csv_path = tmp_path / "inpainting.csv"
    _write_inpainting_csv(csv_path)

    monkeypatch.setattr("adversariallm.attacks.inpainting.prepare_conversation", _fake_prepare_conversation)

    calls = {"generator_called": 0}

    class FakeDefense:
        model = object()
        tokenizer = object()

        def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
            assert len(full_token_tensors) == 1
            assert len(prompt_token_tensors) == 1
            assert initial_batch_size == 1
            assert verbose is True
            return [2.0]

        def generate(self, convs, **kwargs):
            calls["generator_called"] += 1
            assert len(convs) == 1
            assert kwargs["initial_batch_size"] == 1
            assert kwargs["verbose"] is True
            assert kwargs["top_k"] == 0
            return GenerationResult(gen=[["plain completion"]], input_ids=[[10, 11, 12]])

    cfg = InpaintingConfig(custom_data_path=str(csv_path), num_samples_per_behavior=1)
    attack = InpaintingAttack(cfg)

    dataset = [[{"role": "user", "content": "orig prompt"}, {"role": "assistant", "content": "target answer"}]]
    res = attack.run(FakeDefense(), dataset)

    assert calls["generator_called"] == 1
    assert len(res.runs) == 1
    assert len(res.runs[0].steps) == 1
    step = res.runs[0].steps[0]
    assert step.model_completions == ["plain completion"]
    assert step.model_completions_raw is None
    assert step.defense_metadata is None
    assert step.model_input_tokens == [10, 11, 12]
    assert step.loss == 2.0


def test_inpainting_uses_runtime_generator_with_defense_and_computes_loss(monkeypatch, tmp_path):
    csv_path = tmp_path / "inpainting.csv"
    _write_inpainting_csv(csv_path)

    monkeypatch.setattr("adversariallm.attacks.inpainting.prepare_conversation", _fake_prepare_conversation)

    class FakeDefense:
        model = object()
        tokenizer = object()

        def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
            assert len(full_token_tensors) == 1
            assert len(prompt_token_tensors) == 1
            assert initial_batch_size == 1
            assert verbose is True
            return [3.0]

        def generate(self, convs, **kwargs):
            assert len(convs) == 1
            assert kwargs["initial_batch_size"] == 1
            assert kwargs["verbose"] is True
            return GenerationResult(
                gen=[["defended completion"]],
                input_ids=[[1, 2, 3]],
                raw_gen=[["raw completion"]],
                defense_decisions=[[{"applied": True, "harmful_request": True}]],
            )

    cfg = InpaintingConfig(custom_data_path=str(csv_path), num_samples_per_behavior=1)
    attack = InpaintingAttack(cfg)

    dataset = [[{"role": "user", "content": "orig prompt"}, {"role": "assistant", "content": "target answer"}]]
    res = attack.run(FakeDefense(), dataset)

    step = res.runs[0].steps[0]
    assert step.model_completions == ["defended completion"]
    assert step.model_completions_raw == ["raw completion"]
    assert step.model_input_tokens == [1, 2, 3]
    assert step.loss == 3.0
    assert step.defense_metadata == [{"applied": True, "harmful_request": True}]


def test_inpainting_requires_input_ids_from_runtime_generator(monkeypatch, tmp_path):
    csv_path = tmp_path / "inpainting.csv"
    _write_inpainting_csv(csv_path)

    monkeypatch.setattr("adversariallm.attacks.inpainting.prepare_conversation", _fake_prepare_conversation)

    class FakeDefense:
        model = object()
        tokenizer = object()

        def loss(self, full_token_tensors, prompt_token_tensors, *, initial_batch_size, verbose=False):
            assert len(full_token_tensors) == 1
            assert len(prompt_token_tensors) == 1
            return [2.0]

        def generate(self, convs, **kwargs):
            return GenerationResult(gen=[["plain completion"]], input_ids=None)

    cfg = InpaintingConfig(custom_data_path=str(csv_path), num_samples_per_behavior=1)
    attack = InpaintingAttack(cfg)
    dataset = [[{"role": "user", "content": "orig prompt"}, {"role": "assistant", "content": "target answer"}]]

    import pytest

    with pytest.raises(ValueError, match="requires `generation_result.input_ids`"):
        attack.run(FakeDefense(), dataset)
