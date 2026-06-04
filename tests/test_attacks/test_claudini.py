from types import SimpleNamespace

import pytest
import torch

from adversariallm.attacks.attack import Attack
from adversariallm.attacks.claudini import ClaudiniAttack, ClaudiniConfig


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    name_or_path = "dummy-tokenizer"

    def __len__(self):
        return 64

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(f"t{int(x)}" for x in token_ids)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=64, dim=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, dim)
        self.lm_head = torch.nn.Linear(dim, vocab_size, bias=False)
        self.name_or_path = "dummy-model"
        self.config = SimpleNamespace(name_or_path="dummy-model")

    @property
    def device(self):
        return self.emb.weight.device

    def get_input_embeddings(self):
        return self.emb

    def num_parameters(self, exclude_embeddings=True):
        if exclude_embeddings:
            return sum(p.numel() for name, p in self.named_parameters() if not name.startswith("emb."))
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.emb(input_ids)
        logits = self.lm_head(inputs_embeds)
        return SimpleNamespace(logits=logits)


class DummyDataset:
    def __iter__(self):
        yield [
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "target"},
        ]


def _mock_prepare_conversation(tokenizer, conversation, conversation_opt=None):
    # [pre, atk_prefix, prompt, atk_suffix, post, target]
    return [(
        torch.tensor([10], dtype=torch.long),
        torch.tensor([], dtype=torch.long),
        torch.tensor([11], dtype=torch.long),
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([12], dtype=torch.long),
        torch.tensor([13, 14], dtype=torch.long),
    )]


def _mock_generate_ragged_batched(model, tokenizer, token_list=None, **kwargs):
    n = kwargs.get("num_return_sequences", 1)
    return [[f"mock completion {i}" for i in range(n)] for _ in token_list]


def test_attack_registry_includes_claudini():
    cls = Attack.from_name("claudini")
    assert cls.__name__ == "ClaudiniAttack"


def test_claudini_unknown_version_rejected():
    cfg = ClaudiniConfig(variant="unknown")
    with pytest.raises(ValueError, match="Unsupported claudini variant"):
        ClaudiniAttack(cfg)


def test_claudini_v63_run_smoke(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.claudini.prepare_conversation", _mock_prepare_conversation)
    monkeypatch.setattr("adversariallm.attacks.claudini.generate_ragged_batched", _mock_generate_ragged_batched)

    cfg = ClaudiniConfig(variant="claude_v63", num_steps=2, num_starts=2, init_mode="manual", optim_str_init="! ! !")
    attack = ClaudiniAttack(cfg)

    model = DummyModel()
    tokenizer = DummyTokenizer()
    dataset = DummyDataset()

    result = attack.run(model, tokenizer, dataset)
    assert len(result.runs) == 1
    run = result.runs[0]
    assert len(run.steps) == 2
    assert len(run.steps[0].model_completions) == 1
    assert run.steps[0].scores["claudini"]["soft_loss"]
    assert run.steps[0].loss is not None


@pytest.mark.parametrize("variant", ["claude_v82", "claude_oss_v53"])
def test_claudini_other_versions_smoke(monkeypatch, variant):
    monkeypatch.setattr("adversariallm.attacks.claudini.prepare_conversation", _mock_prepare_conversation)
    monkeypatch.setattr("adversariallm.attacks.claudini.generate_ragged_batched", _mock_generate_ragged_batched)

    cfg = ClaudiniConfig(variant=variant, num_steps=1, num_starts=2, init_mode="manual", optim_str_init="! ! !")
    attack = ClaudiniAttack(cfg)

    model = DummyModel()
    tokenizer = DummyTokenizer()
    dataset = DummyDataset()

    result = attack.run(model, tokenizer, dataset)
    assert len(result.runs) == 1
    run = result.runs[0]
    assert len(run.steps) == 1
    assert len(run.steps[0].model_completions) == 1
    assert run.steps[0].loss is not None


def test_claudini_last_step_sampling_override(monkeypatch):
    monkeypatch.setattr("adversariallm.attacks.claudini.prepare_conversation", _mock_prepare_conversation)
    monkeypatch.setattr("adversariallm.attacks.claudini.generate_ragged_batched", _mock_generate_ragged_batched)

    cfg = ClaudiniConfig(
        variant="claude_v63",
        num_steps=2,
        num_starts=2,
        init_mode="manual",
        optim_str_init="! ! !",
        last_step_num_return_sequences=5,
    )
    attack = ClaudiniAttack(cfg)

    model = DummyModel()
    tokenizer = DummyTokenizer()
    dataset = DummyDataset()
    result = attack.run(model, tokenizer, dataset)
    run = result.runs[0]
    assert len(run.steps) == 2
    assert len(run.steps[0].model_completions) == 1
    assert len(run.steps[1].model_completions) == 5
