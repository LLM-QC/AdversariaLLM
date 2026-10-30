from __future__ import annotations

from omegaconf import OmegaConf

import adversariallm.io_utils.config as config_mod
from adversariallm.io_utils.config import RunConfig, filter_config
from adversariallm.io_utils.database import mongodb_is_configured


def _sample_run_config() -> RunConfig:
    return RunConfig(
        model="m",
        dataset="d",
        attack="direct",
        defense=None,
        model_params=OmegaConf.create({"id": "m", "dtype": None}),
        dataset_params=OmegaConf.create({"name": "d", "idx": [0, 1, 2]}),
        attack_params=OmegaConf.create({"name": "direct"}),
        defense_params=None,
    )


def test_filter_config_overwrite_true_skips_mongodb(monkeypatch):
    def _unexpected_mongo():
        raise AssertionError("MongoDB must not be touched when overwrite=True")

    monkeypatch.setattr(config_mod, "get_mongodb_connection", _unexpected_mongo)

    run_config = _sample_run_config()
    filtered = filter_config(run_config, dset_len=10, overwrite=True)

    assert filtered is not None
    assert list(filtered.dataset_params.idx) == [0, 1, 2]


def test_mongodb_is_configured_with_uri(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://example")
    monkeypatch.delenv("MONGODB_USER", raising=False)
    monkeypatch.delenv("MONGODB_PASSWORD", raising=False)
    monkeypatch.delenv("MONGODB_HOST", raising=False)
    monkeypatch.delenv("MONGODB_DB", raising=False)

    assert mongodb_is_configured() is True


def test_mongodb_is_configured_uses_username_db_fallback_without_uri(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.setenv("MONGODB_USER", "user")
    monkeypatch.setenv("MONGODB_PASSWORD", "pw")
    monkeypatch.setenv("MONGODB_HOST", "localhost")
    monkeypatch.setenv("MONGODB_DB", "db")
    assert mongodb_is_configured() is True

    monkeypatch.delenv("MONGODB_DB", raising=False)
    assert mongodb_is_configured() is True

    monkeypatch.delenv("MONGODB_USER", raising=False)
    assert mongodb_is_configured() is False
