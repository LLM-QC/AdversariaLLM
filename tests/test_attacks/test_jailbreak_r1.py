from dataclasses import asdict
from types import SimpleNamespace
import json

import torch

from adversariallm.attacks.attack import Attack
from adversariallm.attacks.jailbreak_r1 import JailbreakR1Attack, JailbreakR1Config
from adversariallm.lm_utils.text_generation import GenerationResult


def _patch_runtime_generator(monkeypatch, outputs: list[str]):
    class _FakeDefense:
        def generate(self, convs, **kwargs):
            assert len(convs) == len(outputs)
            return GenerationResult(
                gen=[[out] for out in outputs],
                input_ids=[[1, 2, 3] for _ in outputs],
            )

    return _FakeDefense()


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

    attack_call = {"n": 0}

    def _mock_generate_ragged_batched(model, tokenizer, token_list, **kwargs):
        if model is attack_model:
            attack_call["n"] += 1
            # Retry 1 for two steps: step0 fails, step1 succeeds.
            if attack_call["n"] == 1:
                assert len(token_list) == 2
                return [
                    ["not-parseable"],
                    ["<think>strategy2</think><attack>second attack</attack>"],
                ]
            # Retry 2 for pending step0 only.
            assert len(token_list) == 1
            return [["<think>strategy</think><attack>\nAttack Prompt: crafted attack text</attack>"]]
        # One batched target generation call for both steps.
        assert len(token_list) == 2
        return [["completion-step-0"], ["completion-step-1"]]

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
    defense = _patch_runtime_generator(monkeypatch, ["completion-step-0", "completion-step-1"])
    defense.model = target_model
    defense.tokenizer = target_tokenizer

    cfg = JailbreakR1Config(num_steps=2, parse_retries=2)
    attack = JailbreakR1Attack(cfg)
    result = attack.run(defense, DummyDataset())

    assert len(result.runs) == 1
    assert len(result.runs[0].steps) == 2
    assert result.runs[0].steps[0].model_input[0]["content"] == "crafted attack text"
    assert result.runs[0].steps[0].model_completions == ["completion-step-0"]
    think = result.runs[0].steps[0].think
    assert think.endswith("strategy")
    assert asdict(result.runs[0].steps[0])["think"] == think
    assert result.runs[0].steps[0].defense_metadata is None
    assert result.runs[0].steps[0].scores["jailbreak_r1"]["parse_success"] == [1.0]
    assert result.runs[0].steps[1].model_input[0]["content"] == "second attack"


def test_jailbreak_r1_allow_untagged_fallback(monkeypatch):
    class DummyDataset:
        def __iter__(self):
            yield [
                {"role": "user", "content": "goal"},
                {"role": "assistant", "content": "target"},
            ]

    class DummyModel:
        name_or_path = "target-model"
        device = torch.device("cpu")

    class DummyTokenizer:
        name_or_path = "target-tokenizer"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "prompt"

        def __call__(self, text, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    attack_model = SimpleNamespace(name_or_path="attack-model", device=torch.device("cpu"))
    attack_tokenizer = DummyTokenizer()
    target_model = DummyModel()
    target_tokenizer = DummyTokenizer()

    def _mock_generate_ragged_batched(model, tokenizer, token_list, **kwargs):
        if model is attack_model:
            return [["untagged generated attack prompt"]]
        return [["target completion"]]

    def _mock_prepare_conversation(tokenizer, conversation):
        return [(
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([10, 11], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([12], dtype=torch.long),
        )]

    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.load_model_and_tokenizer",
        lambda cfg: (attack_model, attack_tokenizer),
    )
    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.generate_ragged_batched", _mock_generate_ragged_batched)
    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.prepare_conversation", _mock_prepare_conversation)
    defense = _patch_runtime_generator(monkeypatch, ["target completion"])
    defense.model = target_model
    defense.tokenizer = target_tokenizer

    cfg = JailbreakR1Config(num_steps=1, parse_retries=1, allow_untagged_fallback=True)
    attack = JailbreakR1Attack(cfg)
    result = attack.run(defense, DummyDataset())

    assert result.runs[0].steps[0].model_input[0]["content"] == "untagged generated attack prompt"
    assert result.runs[0].steps[0].scores["jailbreak_r1"]["parse_success"] == [0.0]


def test_jailbreak_r1_reads_cache_and_skips_attacker_loading(monkeypatch, tmp_path):
    class DummyDataset:
        def __iter__(self):
            yield [
                {"role": "user", "content": "goal-1"},
                {"role": "assistant", "content": "target-1"},
            ]

    class DummyModel:
        name_or_path = "target-model"
        device = torch.device("cpu")

    class DummyTokenizer:
        name_or_path = "target-tokenizer"

    def _mock_prepare_conversation(tokenizer, conversation):
        return [(
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([13], dtype=torch.long),
        )]

    def _mock_generate_ragged_batched(model, tokenizer, token_list, **kwargs):
        assert len(token_list) == 1
        return [["target completion from cache path"]]

    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.prepare_conversation", _mock_prepare_conversation)
    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.generate_ragged_batched", _mock_generate_ragged_batched)
    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.load_model_and_tokenizer",
        lambda cfg: (_ for _ in ()).throw(AssertionError("attacker model should not be loaded in cache read mode")),
    )
    defense = _patch_runtime_generator(monkeypatch, ["target completion from cache path"])

    cfg = JailbreakR1Config(
        num_steps=1,
        parse_retries=1,
        prompt_cache_mode="read",
        prompt_cache_subset_strategy="first_n",
        prompt_cache_path=str(tmp_path / "jbr1_cache.json"),
    )

    cache_payload = {
        "format_version": 1,
        "metadata": {
            "generation_fingerprint": JailbreakR1Attack(cfg)._generation_fingerprint(),
            "generation_seed": 0,
            "prompts_per_behavior": 2,
            "dataset_signature": JailbreakR1Attack(cfg)._dataset_signature(list(DummyDataset())),
            "dataset_size": 1,
            "created_at_unix": 0.0,
        },
        "records": [
            {
                "original_prompt": "goal-1",
                "target": "target-1",
                "attacks": [
                    {"think": "t1", "attack_prompt": "cached-attack-1", "parse_success": True},
                    {"think": "t2", "attack_prompt": "cached-attack-2", "parse_success": False},
                ],
            }
        ],
    }
    with open(cfg.prompt_cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f)

    attack = JailbreakR1Attack(cfg)
    defense.model = DummyModel()
    defense.tokenizer = DummyTokenizer()
    result = attack.run(defense, DummyDataset())

    assert len(result.runs) == 1
    assert len(result.runs[0].steps) == 1
    assert result.runs[0].steps[0].model_input[0]["content"] == "cached-attack-1"
    assert result.runs[0].steps[0].scores["jailbreak_r1"]["parse_success"] == [1.0]


def test_jailbreak_r1_reads_subset_by_behavior_id(monkeypatch, tmp_path):
    class DummyDataset:
        idx = torch.tensor([1], dtype=torch.long)

        def __iter__(self):
            yield [
                {"role": "user", "content": "goal-b"},
                {"role": "assistant", "content": "target-b"},
            ]

    class DummyModel:
        name_or_path = "target-model"
        device = torch.device("cpu")

    class DummyTokenizer:
        name_or_path = "target-tokenizer"

    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.load_model_and_tokenizer",
        lambda cfg: (_ for _ in ()).throw(AssertionError("attacker model should not be loaded in cache read mode")),
    )
    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.prepare_conversation",
        lambda tokenizer, conversation: [(
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([21, 22], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([23], dtype=torch.long),
        )],
    )
    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.generate_ragged_batched",
        lambda model, tokenizer, token_list, **kwargs: [["subset completion"]],
    )
    defense = _patch_runtime_generator(monkeypatch, ["subset completion"])

    cfg = JailbreakR1Config(
        num_steps=1,
        parse_retries=1,
        prompt_cache_mode="read",
        prompt_cache_path=str(tmp_path / "jbr1_cache_subset.json"),
    )
    attack_for_meta = JailbreakR1Attack(cfg)
    cache_payload = {
        "format_version": 1,
        "metadata": {
            "generation_fingerprint": attack_for_meta._generation_fingerprint(),
            "generation_seed": 0,
            "prompts_per_behavior": 2,
            "created_at_unix": 0.0,
        },
        "records": [
            {
                "behavior_id": 0,
                "original_prompt": "goal-a",
                "target": "target-a",
                "attacks": [{"think": "ta", "attack_prompt": "cached-a", "parse_success": True}],
            },
            {
                "behavior_id": 1,
                "original_prompt": "goal-b",
                "target": "target-b",
                "attacks": [{"think": "tb", "attack_prompt": "cached-b", "parse_success": True}],
            },
        ],
    }
    with open(cfg.prompt_cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f)

    attack = JailbreakR1Attack(cfg)
    defense.model = DummyModel()
    defense.tokenizer = DummyTokenizer()
    result = attack.run(defense, DummyDataset())

    assert len(result.runs) == 1
    assert result.runs[0].steps[0].model_input[0]["content"] == "cached-b"


def test_jailbreak_r1_frees_attacker_vram_when_models_differ(monkeypatch, tmp_path):
    class DummyDataset:
        idx = torch.tensor([0], dtype=torch.long)

        def __iter__(self):
            yield [
                {"role": "user", "content": "goal-x"},
                {"role": "assistant", "content": "target-x"},
            ]

    class DummyModel:
        def __init__(self, name):
            self.name_or_path = name
            self.device = torch.device("cpu")

    class DummyTokenizer:
        name_or_path = "dummy-tokenizer"

    target_model = DummyModel("target-model")
    attack_model = DummyModel("attack-model")
    target_tokenizer = DummyTokenizer()
    attack_tokenizer = DummyTokenizer()

    free_calls = {"n": 0}
    monkeypatch.setattr("adversariallm.attacks.jailbreak_r1.free_vram", lambda: free_calls.__setitem__("n", free_calls["n"] + 1))
    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.load_model_and_tokenizer",
        lambda cfg: (attack_model, attack_tokenizer),
    )
    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.JailbreakR1Attack._generate_attack_pool_parallel",
        lambda self, attack_model, attack_tokenizer, goals, prompts_per_behavior: [
            [("t", "cached-ish attack", True)] for _ in goals
        ],
    )
    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.prepare_conversation",
        lambda tokenizer, conversation: [(
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.long),
            torch.tensor([3], dtype=torch.long),
        )],
    )
    monkeypatch.setattr(
        "adversariallm.attacks.jailbreak_r1.generate_ragged_batched",
        lambda model, tokenizer, token_list, **kwargs: [["target completion"]],
    )
    defense = _patch_runtime_generator(monkeypatch, ["target completion"])
    defense.model = target_model
    defense.tokenizer = target_tokenizer

    cfg = JailbreakR1Config(
        num_steps=1,
        prompt_cache_mode="write",
        prompt_cache_path=str(tmp_path / "cache.json"),
    )
    attack = JailbreakR1Attack(cfg)
    _ = attack.run(defense, DummyDataset())

    assert free_calls["n"] == 1
