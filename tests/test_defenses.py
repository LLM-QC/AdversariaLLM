from __future__ import annotations

import pytest
import torch

from adversariallm.defenses import create_defended_text_generator, parse_polyguard_output
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
    }


def test_create_defended_text_generator_none_returns_base():
    base = _EchoGenerator()
    assert create_defended_text_generator(None, base_generator=base) is base
    assert create_defended_text_generator({"type": "none"}, base_generator=base) is base


def test_create_defended_text_generator_unknown_type():
    with pytest.raises(ValueError, match="Unknown defense type"):
        create_defended_text_generator({"type": "not_a_real_defense"}, base_generator=_EchoGenerator())


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

    base = _EchoGenerator()
    defended = create_defended_text_generator(
        {
            "type": "polyguard",
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
                "device": "cpu",
                "cache_dir": None,
            },
        },
        base_generator=base,
    )

    res = defended.generate([[{"role": "user", "content": "hello"}]])
    assert calls["loader"] == 1
    assert calls["batching"] == 1
    assert res.gen == [["raw:hello"]]
