from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
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
                "none": {"name": "none"},
                "polyguard": {"name": "polyguard"},
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
    assert isinstance(run_configs[0].attack_params, DictConfig)
    assert isinstance(run_configs[0].defense_params, DictConfig)
    assert "defense" not in run_configs[0].attack_params
    assert run_configs[0].defense_params["name"] == "polyguard"


def test_collect_configs_rejects_unsupported_attack_before_loading_dataset(monkeypatch):
    def _unexpected_dataset_lookup(_name):
        pytest.fail("Compatibility validation must happen before loading the dataset")

    monkeypatch.setattr(run_attacks.PromptDataset, "from_name", _unexpected_dataset_lookup)
    cfg = OmegaConf.create(
        {
            "models": {"m": {"id": "m"}},
            "datasets": {"d": {"idx": [0]}},
            "attacks": {"gcg": {"name": "gcg"}},
            "defenses": {"polyguard": {"name": "polyguard"}},
            "model": "m",
            "dataset": "d",
            "attack": "gcg",
            "defense": "polyguard",
            "overwrite": True,
        }
    )

    with pytest.raises(ValueError, match="incompatible with runtime defenses"):
        run_attacks.collect_configs(cfg)


def test_runner_builds_and_passes_target_system(monkeypatch):
    from adversariallm.io_utils.config import RunConfig

    sentinel_target = object()
    calls = {}
    events = []

    class _DummyAttack:
        def __init__(self, attack_params):
            events.append("attack")
            calls["attack_params"] = attack_params

        def run(self, target, dataset):
            calls["run_args"] = (target, dataset)
            return "results"

    monkeypatch.setattr(run_attacks, "load_model_and_tokenizer", lambda _params: ("model", "tokenizer"))
    monkeypatch.setattr(run_attacks, "PromptDataset", type("PD", (), {"from_name": staticmethod(lambda _n: (lambda _p: []))}))
    def _build_target_system(cfg, *, model, tokenizer):
        events.append("target")
        return sentinel_target

    monkeypatch.setattr(run_attacks, "build_target_system", _build_target_system)
    monkeypatch.setattr(run_attacks.Attack, "from_name", classmethod(lambda _cls, _name: _DummyAttack))
    monkeypatch.setattr(run_attacks, "log_attack", lambda *_args, **_kwargs: None)

    rc = RunConfig(
        model="m",
        dataset="d",
        attack="direct",
        defense=None,
        model_params={},
        dataset_params={},
        attack_params={"name": "direct"},
        defense_params=None,
    )
    cfg = OmegaConf.create({"save_dir": "/tmp", "embed_dir": "/tmp"})

    run_attacks.run_attacks([rc], cfg, "2026-04-06/00-00-00")

    assert events == ["attack", "target"]
    assert calls["run_args"] == (sentinel_target, [])


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
            "defenses": {"polyguard": {"name": "polyguard"}, "none": {"name": "none"}},
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
