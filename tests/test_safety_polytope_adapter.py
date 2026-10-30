from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import adversariallm.defenses.safety_polytope as safety_polytope_mod
from adversariallm.defenses.safety_polytope import (
    SafetyPolytopeDefense,
    _last_user_text,
    _resolve_torch_dtype,
    _render_conversation,
)


class FakeTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return "PROMPT:" + "|".join(m["role"] + ":" + m["content"] for m in messages) + "|assistant:"


class RecorderTokenizer:
    chat_template = "template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "continue_final_message": continue_final_message,
            }
        )
        rendered = "|".join(m["role"] + ":" + m["content"] for m in messages)
        return f"PROMPT:{rendered}"


def test_last_user_text_uses_last_user_message():
    assert _last_user_text([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "last"},
    ]) == "last"


def test_render_uses_checkpoint_template():
    out = _render_conversation(FakeTokenizer(), [{"role": "user", "content": "hello"}])
    assert out == "PROMPT:user:hello|assistant:"


def test_render_without_template_falls_back_to_explicit_roles():
    out = _render_conversation(object(), [{"role": "user", "content": "hello"}])
    assert out == "USER: hello\nASSISTANT:"


def test_render_strips_empty_trailing_assistant_placeholder():
    tokenizer = RecorderTokenizer()
    out = _render_conversation(
        tokenizer,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
        ],
    )
    assert out == "PROMPT:user:hello"
    assert tokenizer.calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    assert tokenizer.calls[0]["continue_final_message"] is False


def test_render_continues_nonempty_assistant_prefill():
    tokenizer = RecorderTokenizer()
    out = _render_conversation(
        tokenizer,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prefill"},
        ],
    )
    assert out == "PROMPT:user:hello|assistant:prefill"
    assert tokenizer.calls[0]["add_generation_prompt"] is False
    assert tokenizer.calls[0]["continue_final_message"] is True


def test_render_without_template_continues_assistant_prefill():
    out = _render_conversation(
        object(),
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prefill"},
        ],
    )
    assert out == "USER: hello\nASSISTANT: prefill"


def test_missing_model_id_fails_clearly():
    with pytest.raises(ValueError, match="model_id"):
        SafetyPolytopeDefense.from_config({}, model=cast(Any, object()), tokenizer=cast(Any, object()))


def test_missing_weights_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights not found"):
        SafetyPolytopeDefense.from_config(
            {"model_id": "example/base", "weights_path": str(tmp_path / "missing.pth")},
            model=object(), tokenizer=object(),
        )


def test_from_config_rejects_mismatched_runner_model():
    with pytest.raises(ValueError, match="model mismatch"):
        SafetyPolytopeDefense.from_config(
            {"model_id": "Qwen/Qwen2-1.5B-Instruct", "weights_path": "/tmp/weights.pth"},
            model=SimpleNamespace(name_or_path="qwen/Qwen2-7B-Instruct"),
            tokenizer=SimpleNamespace(name_or_path="qwen/Qwen2-7B-Instruct"),
        )


def test_resolve_torch_dtype_handles_string_and_rejects_unknown():
    assert _resolve_torch_dtype("float16", fallback=torch.float32) is torch.float16
    assert _resolve_torch_dtype("auto", fallback=torch.float32) == "auto"
    with pytest.raises(ValueError, match="Unsupported SafetyPolytope torch_dtype"):
        _resolve_torch_dtype("not_a_real_dtype", fallback=torch.float32)


def test_from_config_calls_upstream_loader_with_expected_kwargs(tmp_path, monkeypatch):
    weights_path = tmp_path / "polytope.pth"
    weights_path.write_text("stub")

    captured: dict[str, Any] = {}

    class _FakeSafeRepModel:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, polytope_weight_path, **kwargs):
            captured["model_id"] = pretrained_model_name_or_path
            captured["weights_path"] = polytope_weight_path
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                tokenizer=FakeTokenizer(),
                eval=lambda: None,
                generate=lambda **_g: torch.zeros((1, 2), dtype=torch.long),
                model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1))),
                parameters=lambda: iter([torch.zeros(1)]),
                model_name_or_path=pretrained_model_name_or_path,
            )

    monkeypatch.setattr(
        safety_polytope_mod,
        "import_module",
        lambda module_name: SimpleNamespace(SafeRepModel=_FakeSafeRepModel)
        if module_name == "safety_polytope.polytope.safe_rep_model"
        else None,
    )

    defense = SafetyPolytopeDefense.from_config(
        {
            "model_id": "Qwen/Qwen2-1.5B-Instruct",
            "weights_path": str(weights_path),
            "model_kwargs": {"torch_dtype": "float16", "device_map": "cpu"},
            "steer_layer": 20,
            "max_new_tokens": 33,
        },
        model=SimpleNamespace(dtype=torch.bfloat16),
        tokenizer=FakeTokenizer(),
    )

    assert isinstance(defense, SafetyPolytopeDefense)
    assert captured["model_id"] == "Qwen/Qwen2-1.5B-Instruct"
    assert captured["weights_path"] == str(weights_path.resolve())
    assert captured["kwargs"]["torch_dtype"] is torch.float16
    assert captured["kwargs"]["device_map"] == "cpu"
    assert defense.weights_path == str(weights_path.resolve())
    assert defense.model_id == "Qwen/Qwen2-1.5B-Instruct"
    assert defense.safe_tokenizer.padding_side == "left"


def test_generate_groups_outputs_and_sets_metadata():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        chat_template = None

        def __call__(self, prompts, return_tensors="pt", padding=True, truncation=False, max_length=None, add_special_tokens=False):
            del return_tensors, padding, truncation, max_length, add_special_tokens
            assert len(prompts) == 2
            return {
                "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            }

        def batch_decode(self, token_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return [f"out-{i}" for i in range(token_ids.shape[0])]

    class _SafeModel:
        model_name_or_path = "Qwen/Qwen2-1.5B-Instruct"

        def __init__(self):
            self.model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1)))
            self._param = torch.zeros(1)
            self.kwargs = None

        def parameters(self):
            yield self._param

        def generate(self, input_ids, attention_mask, **kwargs):
            del attention_mask
            self.kwargs = kwargs
            rows = []
            for idx in range(input_ids.shape[0] * kwargs["num_return_sequences"]):
                rows.append(torch.cat([input_ids[idx % input_ids.shape[0]], torch.tensor([100 + idx])]))
            return torch.stack(rows, dim=0)

    safe_model = _SafeModel()
    defense = SafetyPolytopeDefense(
        model=cast(Any, object()),
        tokenizer=cast(Any, object()),
        safe_model=safe_model,
        model_id="Qwen/Qwen2-1.5B-Instruct",
        weights_path="/tmp/polytope.pth",
    )
    defense.safe_tokenizer = _Tokenizer()

    result = defense.generate(
        [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        num_return_sequences=2,
        max_new_tokens=11,
        temperature=0.8,
        initial_batch_size=8,
        verbose=True,
    )

    assert result.gen == [["out-0", "out-1"], ["out-2", "out-3"]]
    assert result.raw_gen == [["out-0", "out-1"], ["out-2", "out-3"]]
    assert result.input_ids == [[1, 2], [3, 4]]
    assert result.defense_decisions is not None
    assert result.defense_decisions[0][0]["weights_path"] == "/tmp/polytope.pth"
    assert safe_model.kwargs["max_new_tokens"] == 11
    assert safe_model.kwargs["num_return_sequences"] == 2
    assert safe_model.kwargs["do_sample"] is True
    assert "initial_batch_size" not in safe_model.kwargs
    assert "verbose" not in safe_model.kwargs


def test_generate_expands_prompts_for_multi_greedy_returns():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        chat_template = None

        def __call__(self, prompts, return_tensors="pt", padding=True, truncation=False, max_length=None, add_special_tokens=False):
            del return_tensors, padding, truncation, max_length, add_special_tokens
            rows = [[index + 1, index + 11] for index in range(len(prompts))]
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1] for _ in rows], dtype=torch.long),
            }

        def batch_decode(self, token_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return [f"out-{i}" for i in range(token_ids.shape[0])]

    class _SafeModel:
        model_name_or_path = "Qwen/Qwen2-1.5B-Instruct"

        def __init__(self):
            self.model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1)))
            self._param = torch.zeros(1)
            self.kwargs = None

        def parameters(self):
            yield self._param

        def generate(self, input_ids, attention_mask, **kwargs):
            del attention_mask
            self.kwargs = kwargs
            assert kwargs["num_return_sequences"] == 1
            rows = []
            for idx in range(input_ids.shape[0]):
                rows.append(torch.cat([input_ids[idx], torch.tensor([200 + idx])]))
            return torch.stack(rows, dim=0)

    safe_model = _SafeModel()
    defense = SafetyPolytopeDefense(
        model=cast(Any, object()),
        tokenizer=cast(Any, object()),
        safe_model=safe_model,
        model_id="Qwen/Qwen2-1.5B-Instruct",
        weights_path="/tmp/polytope.pth",
    )
    defense.safe_tokenizer = _Tokenizer()

    result = defense.generate(
        [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        num_return_sequences=2,
        temperature=0.0,
    )

    assert result.gen == [["out-0", "out-1"], ["out-2", "out-3"]]
    assert result.input_ids == [[1, 11], [2, 12]]


def test_generate_rejects_filters():
    defense = SafetyPolytopeDefense(
        model=cast(Any, object()),
        tokenizer=cast(Any, object()),
        safe_model=SimpleNamespace(
            tokenizer=FakeTokenizer(),
            generate=lambda **kwargs: kwargs,
            model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1))),
            parameters=lambda: iter([torch.zeros(1)]),
        ),
        model_id="Qwen/Qwen2-1.5B-Instruct",
        weights_path="/tmp/polytope.pth",
    )
    with pytest.raises(NotImplementedError, match="does not yet translate"):
        defense.generate([[{"role": "user", "content": "x"}]], filters=[{"type": "json"}])
