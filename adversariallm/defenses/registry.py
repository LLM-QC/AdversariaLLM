from __future__ import annotations

from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base import Defense, NoDefense
from .polyguard import PolyGuardDefense

class DefenseFactory(Protocol):
    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        default_cache_dir: str | None = None,
        default_generate_kwargs: dict[str, Any] | None = None,
    ) -> Defense:
        ...


_DEFENSE_REGISTRY: dict[str, type[DefenseFactory]] = {
    NoDefense.DEFENSE_TYPE: NoDefense,
    PolyGuardDefense.DEFENSE_TYPE: PolyGuardDefense,
}

DEFENSE_COMPATIBLE_ATTACKS = frozenset(
    {
        "actor",
        "ample_gcg",
        "bon",
        "crescendo",
        "direct",
        "inpainting",
        "jailbreak_r1",
        "pair",
    }
)


def _as_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return cfg


def create_defense(
    defense_cfg: DictConfig | dict[str, Any] | None,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    default_cache_dir: str | None = None,
    default_generate_kwargs: dict[str, Any] | None = None,
) -> Defense:
    if defense_cfg is None:
        defense_cfg = {"type": "none"}
    cfg = _as_dict(defense_cfg)
    defense_type = cfg.get("type", "none")
    if defense_type is None:
        defense_type = "none"
    cfg = {**cfg, "type": defense_type}

    defense_cls = _DEFENSE_REGISTRY.get(defense_type)
    if defense_cls is None:
        raise ValueError(f"Unknown defense type: {defense_type}")
    return defense_cls.from_config(
        cfg,
        model=model,
        tokenizer=tokenizer,
        default_cache_dir=default_cache_dir,
        default_generate_kwargs=default_generate_kwargs,
    )


def validate_defense_compatibility(
    attack_name: str,
    defense_cfg: DictConfig | dict[str, Any] | None,
) -> None:
    if defense_cfg is None:
        return
    cfg = _as_dict(defense_cfg)
    defense_type = cfg.get("type", "none")
    if defense_type in {None, "none"}:
        return
    if attack_name not in DEFENSE_COMPATIBLE_ATTACKS:
        raise ValueError(
            f"Attack '{attack_name}' is incompatible with runtime defenses. "
            "Switch to an attack that supports runtime defenses."
        )
