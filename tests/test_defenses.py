from __future__ import annotations

import pytest
import torch

from adversariallm.defenses import TargetSystem, UndefendedTarget, build_target_system, parse_polyguard_output
from adversariallm.lm_utils.text_generation import GenerationResult, TextGenerator


class _EchoGenerator(TextGenerator):
    def generate(self, convs, **kwargs):  # type: ignore[override]
        outputs = []
        for conv in convs:
            user = ""
            for msg in reversed(conv):
                if msg["role"] == "user":
                    user = msg["content"]
                    break
            outputs.append([f"raw:{user}"])
        return GenerationResult(gen=outputs, input_ids=[[1] for _ in convs])


def test_parse_polyguard_output_expected_fields():
    text = "Harmful request: yes\nResponse refusal: no\nHarmful response: no"
    parsed = parse_polyguard_output(text)
    assert parsed == {
        "harmful_request": True,
        "response_refusal": False,
        "harmful_response": False,
        "parse_success": True,
    }


def test_build_target_system_none_returns_undefended_target():
    assert isinstance(build_target_system(None, model=object(), tokenizer=object()), UndefendedTarget)
    assert isinstance(
        build_target_system({"name": "none"}, model=object(), tokenizer=object()),
        UndefendedTarget,
    )


def test_build_target_system_unknown_defense():
    with pytest.raises(ValueError, match="Unknown defense"):
        build_target_system({"name": "not_a_real_defense"}, model=object(), tokenizer=object())


def test_defense_loss_matches_previous_target_token_slicing(monkeypatch):
    from adversariallm.defenses import base as base_mod

    class _LossOnlyTarget(TargetSystem):
        @classmethod
        def from_config(cls, cfg, *, model, tokenizer, default_generate_kwargs=None):
            return cls(model, tokenizer)

        def generate(self, convs, **kwargs):
            raise NotImplementedError

    def _fake_get_losses_batched(model, targets, token_list, initial_batch_size, verbose):
        assert len(targets) == len(token_list) == 3
        return [
            torch.arange(tokens.numel() - 1, dtype=torch.float32) + batch_idx * 10
            for batch_idx, tokens in enumerate(token_list)
        ]

    monkeypatch.setattr(base_mod, "get_losses_batched", _fake_get_losses_batched)

    full_tokens = [torch.arange(6), torch.arange(8), torch.arange(4)]
    prompt_tokens = [full_tokens[0][:3], full_tokens[1][:2], full_tokens[2]]
    losses = _LossOnlyTarget(object(), object()).loss(
        full_tokens,
        prompt_tokens,
        initial_batch_size=1,
    )

    expected = [
        torch.arange(5, dtype=torch.float32)[2:5].mean().item(),
        (torch.arange(7, dtype=torch.float32) + 10)[1:7].mean().item(),
        None,
    ]
    assert losses == expected


def test_polyguard_uses_repo_model_loader_and_adaptive_batching(monkeypatch):
    from adversariallm.defenses import polyguard as polyguard_mod

    class _DummyTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, _message, tokenize=False, add_generation_prompt=True):
            assert not tokenize
            assert add_generation_prompt
            return "formatted"

        def __call__(self, batch, return_tensors="pt", padding=True, add_special_tokens=False):
            assert return_tensors == "pt"
            assert padding
            assert not add_special_tokens
            B = len(batch)
            return {
                "input_ids": torch.ones((B, 3), dtype=torch.long),
                "attention_mask": torch.ones((B, 3), dtype=torch.long),
            }

        def batch_decode(self, _generated, skip_special_tokens=True):
            assert skip_special_tokens
            return ["Harmful request: no\nResponse refusal: no\nHarmful response: no"]

    class _DummyModel:
        def __init__(self):
            self._param = torch.zeros(1)

        def parameters(self):
            yield self._param

        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, do_sample):
            assert input_ids.shape == attention_mask.shape
            assert max_new_tokens == 8
            assert pad_token_id == 0
            assert do_sample is False
            B = input_ids.shape[0]
            return torch.ones((B, input_ids.shape[1] + 1), dtype=torch.long)

    calls = {"loader": 0, "batching": 0}

    def _fake_with_max_batchsize(function, formatted, initial_batch_size=None, verbose=False):
        calls["batching"] += 1
        assert initial_batch_size == min(4, len(formatted))
        assert verbose is False
        return function(formatted)

    def _fake_loader_config(model_cfg):
        calls["loader"] += 1
        assert model_cfg["id"] == "ToxicityPrompts/PolyGuard-Qwen-Smol"
        assert model_cfg["dtype"] == "bfloat16"
        return _DummyModel(), _DummyTokenizer()

    monkeypatch.setattr(polyguard_mod, "load_model_and_tokenizer", _fake_loader_config)
    monkeypatch.setattr(polyguard_mod, "with_max_batchsize", _fake_with_max_batchsize)
    monkeypatch.setattr(polyguard_mod, "LocalTextGenerator", lambda _model, _tokenizer, default_generate_kwargs=None: _EchoGenerator())

    defended = build_target_system(
        {
            "name": "polyguard",
            "batch_size": 4,
            "max_new_tokens": 8,
            "overwrite_mode": "harmful_content",
            "refusal_text": "Sorry, I can't assist you with that.",
            "model": {
                "id": "ToxicityPrompts/PolyGuard-Qwen-Smol",
                "tokenizer_id": "ToxicityPrompts/PolyGuard-Qwen-Smol",
                "short_name": "PolyGuard",
                "developer_name": "ToxicityPrompts",
                "compile": False,
                "trust_remote_code": True,
                "chat_template": None,
                "dtype": "bfloat16",
            },
        },
        model=object(),
        tokenizer=object(),
    )

    res = defended.generate([[{"role": "user", "content": "hello"}]])
    assert calls["loader"] == 1
    assert calls["batching"] == 1
    assert res.gen == [["raw:hello"]]


def test_polyguard_preserves_batched_generation_order():
    from adversariallm.defenses.base import DefenseDecision
    from adversariallm.defenses.polyguard import PolyGuardDefense

    class _RaggedGenerator:
        def generate(self, convs, **kwargs):
            assert len(convs) == 2
            return GenerationResult(
                gen=[["first-0", "first-1"], ["second-0"]],
                input_ids=[[10], [20]],
            )

    defense = object.__new__(PolyGuardDefense)
    defense.text_generator = _RaggedGenerator()

    def _apply_batch(prompts, responses):
        assert prompts == ["prompt-0", "prompt-0", "prompt-1"]
        assert responses == ["first-0", "first-1", "second-0"]
        return [
            DefenseDecision(output_text=f"defended:{response}", metadata={"index": index})
            for index, response in enumerate(responses)
        ]

    defense.apply_batch = _apply_batch
    result = defense.generate(
        [
            [{"role": "user", "content": "prompt-0"}],
            [{"role": "user", "content": "prompt-1"}],
        ],
        num_return_sequences=2,
    )

    assert result.gen == [
        ["defended:first-0", "defended:first-1"],
        ["defended:second-0"],
    ]
    assert result.raw_gen == [["first-0", "first-1"], ["second-0"]]
    assert result.defense_decisions == [
        [{"index": 0}, {"index": 1}],
        [{"index": 2}],
    ]
    assert result.input_ids == [[10], [20]]


def test_parse_polyguard_output_fails_closed_and_marks_parse_failure():
    parsed = parse_polyguard_output("not a valid PolyGuard response")

    assert parsed == {
        "harmful_request": True,
        "response_refusal": True,
        "harmful_response": True,
        "parse_success": False,
    }
