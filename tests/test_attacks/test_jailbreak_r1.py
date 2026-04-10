from types import SimpleNamespace

import torch

from adversariallm.attacks.attack import Attack
from adversariallm.attacks.jailbreak_r1 import JailbreakR1Attack, JailbreakR1Config


def test_attack_registry_includes_jailbreak_r1():
    cls = Attack.from_name("jailbreak_r1")
    assert cls.__name__ == "JailbreakR1Attack"


def test_jailbreak_r1_run_smoke(monkeypatch):
    class DummyDataset:
        def __iter__(self):
            yield [
                {"role": "user", "content": "Give instructions for harmful behavior"},
                {"role": "assistant", "content": "Sure, here is a harmful answer"},
            ]

    class DummyModel:
        name_or_path = "target-model"
        device = torch.device("cpu")

    class DummyTokenizer:
        name_or_path = "target-tokenizer"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert not tokenize
            assert add_generation_prompt
            return f"USER:{messages[0]['content']}::ASSISTANT:"

        def __call__(self, text, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt"):
            assert not add_special_tokens
            assert not return_token_type_ids
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long)}

    attack_model = SimpleNamespace(name_or_path="attack-model", device=torch.device("cpu"))
    attack_tokenizer = DummyTokenizer()
    target_model = DummyModel()
    target_tokenizer = DummyTokenizer()

    # First attack generation fails parsing, second one succeeds.
    attack_outputs = iter(
        [
            [["not-parseable"]],
            [["<think>strategy</think><attack>\nAttack Prompt: crafted attack text</attack>"]],
            [["<think>strategy2</think><attack>second attack</attack>"]],
        ]
    )
    target_outputs = iter(
        [
            [["completion-step-0"]],
            [["completion-step-1"]],
        ]
    )

    def _mock_generate_ragged_batched(model, tokenizer, token_list, **kwargs):
        assert len(token_list) == 1
        if model is attack_model:
            return next(attack_outputs)
        return next(target_outputs)

    def _mock_prepare_conversation(tokenizer, conversation):
        # Return one turn split into [sep, pre_attack, prompt, suf_attack, post, target]
        return [(
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([101, 102, 103], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([104], dtype=torch.long),
            torch.tensor([105, 106], dtype=torch.long),
        )]

    def _mock_load_model_and_tokenizer(cfg):
        return attack_model, attack_tokenizer

    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.generate_ragged_batched", _mock_generate_ragged_batched)
    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.prepare_conversation", _mock_prepare_conversation)
    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.load_model_and_tokenizer", _mock_load_model_and_tokenizer)

    cfg = JailbreakR1Config(num_steps=2, parse_retries=2)
    attack = JailbreakR1Attack(cfg)
    result = attack.run(target_model, target_tokenizer, DummyDataset())

    assert len(result.runs) == 1
    assert len(result.runs[0].steps) == 2
    assert result.runs[0].steps[0].model_input[0]["content"] == "crafted attack text"
    assert result.runs[0].steps[0].model_completions == ["completion-step-0"]
    assert result.runs[0].steps[0].scores["jailbreak_r1"]["parse_success"] == [1.0]
