from __future__ import annotations

from omegaconf import OmegaConf
import pytest

import run_attacks


def test_collect_configs_includes_defense(monkeypatch):
    class _DummyDataset:
        config_idx = [0]

        def __len__(self):
            return 1

    class _DummyPromptDataset:
        @staticmethod
        def from_name(_name):
            return lambda _cfg: _DummyDataset()

    monkeypatch.setattr(run_attacks, "PromptDataset", _DummyPromptDataset)
    monkeypatch.setattr(run_attacks, "filter_config", lambda rc, _dset_len, overwrite=False: rc)

    cfg = OmegaConf.create(
        {
            "models": {"m": {"id": "m"}},
            "datasets": {"d": {"idx": [0]}},
            "attacks": {"actor": {"name": "actor"}},
            "defenses": {
                "none": {"type": "none"},
                "polyguard": {"type": "polyguard"},
            },
            "model": "m",
            "dataset": "d",
            "attack": "actor",
            "defense": "polyguard",
            "overwrite": True,
        }
    )

    run_configs = run_attacks.collect_configs(cfg)
    assert len(run_configs) == 1
    assert run_configs[0].defense == "polyguard"
    assert run_configs[0].attack_params["defense"]["type"] == "polyguard"


def test_runtime_defense_capability_validation(monkeypatch):
    class _DummyAttack:
        def __init__(self, _cfg):
            pass

        def run(self, _model, _tokenizer, _dataset):
            from adversariallm.attacks.attack import AttackResult

            return AttackResult(runs=[])

    monkeypatch.setattr(run_attacks, "load_model_and_tokenizer", lambda _params: ("m", "t"))
    monkeypatch.setattr(run_attacks, "PromptDataset", type("PD", (), {"from_name": staticmethod(lambda _n: (lambda _p: []))}))
    monkeypatch.setattr(run_attacks.Attack, "from_name", classmethod(lambda _cls, _n: _DummyAttack))
    monkeypatch.setattr(run_attacks, "log_attack", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_attacks, "get_defense_capabilities", lambda _cfg: set())

    from adversariallm.io_utils.config import RunConfig

    rc = RunConfig(
        model="m",
        dataset="d",
        attack="actor",
        defense="polyguard",
        model_params={},
        dataset_params={},
        attack_params={"defense": {"type": "polyguard"}},
        defense_params={"type": "polyguard"},
    )
    cfg = OmegaConf.create({"save_dir": "/tmp", "embed_dir": "/tmp"})
    with pytest.raises(ValueError, match="lacks required capabilities"):
        run_attacks.run_attacks([rc], cfg, "2026-04-06/00-00-00")


def test_collect_configs_resolves_interpolations_with_defense(monkeypatch):
    class _DummyDataset:
        config_idx = [0]

        def __len__(self):
            return 1

    class _DummyPromptDataset:
        @staticmethod
        def from_name(_name):
            return lambda _cfg: _DummyDataset()

    def _filter_and_resolve(rc, _dset_len, overwrite=False):
        del overwrite
        OmegaConf.resolve(rc.attack_params)
        return rc

    monkeypatch.setattr(run_attacks, "PromptDataset", _DummyPromptDataset)
    monkeypatch.setattr(run_attacks, "filter_config", _filter_and_resolve)

    cfg = OmegaConf.create(
        {
            "models": {"m": {"id": "m"}},
            "datasets": {"d": {"idx": [0]}},
            "attacks": {
                "_default": {"generation_config": {"max_new_tokens": 42}},
                "inpainting": {"generation_config": "${attacks._default.generation_config}"},
            },
            "defenses": {"polyguard": {"type": "polyguard"}, "none": {"type": "none"}},
            "model": "m",
            "dataset": "d",
            "attack": "inpainting",
            "defense": "polyguard",
            "overwrite": True,
        }
    )

    run_configs = run_attacks.collect_configs(cfg)
    assert len(run_configs) == 1
    assert run_configs[0].attack_params["generation_config"]["max_new_tokens"] == 42
