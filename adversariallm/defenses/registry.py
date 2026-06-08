from __future__ import annotations

from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf

from ..lm_utils import TextGenerator
from .polyguard import PolyGuardDefense

class DefenseFactory(Protocol):
    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        base_generator: TextGenerator,
        default_cache_dir: str | None = None,
    ) -> TextGenerator:
        ...

    @classmethod
    def capabilities(cls) -> set[str]:
        ...


_DEFENSE_REGISTRY: dict[str, type[DefenseFactory]] = {
    PolyGuardDefense.DEFENSE_TYPE: PolyGuardDefense,
}


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def create_defended_text_generator(
    defense_cfg: DictConfig | dict[str, Any] | None,
    *,
    base_generator: TextGenerator,
    default_cache_dir: str | None = None,
) -> TextGenerator:
    if defense_cfg is None:
        return base_generator
    cfg = _as_dict(defense_cfg)
    defense_type = cfg.get("type", "none")
    if defense_type in ("none", None):
        return base_generator

    defense_cls = _DEFENSE_REGISTRY.get(defense_type)
    if defense_cls is None:
        raise ValueError(f"Unknown defense type: {defense_type}")
    return defense_cls.from_config(
        cfg,
        base_generator=base_generator,
        default_cache_dir=default_cache_dir,
    )


def get_defense_capabilities(defense_cfg: DictConfig | dict[str, Any] | None) -> set[str]:
    if defense_cfg is None:
        return set()
    cfg = _as_dict(defense_cfg)
    defense_type = cfg.get("type", "none")
    if defense_type in ("none", None):
        return set()

    defense_cls = _DEFENSE_REGISTRY.get(defense_type)
    if defense_cls is None:
        raise ValueError(f"Unknown defense type: {defense_type}")
    return defense_cls.capabilities()
