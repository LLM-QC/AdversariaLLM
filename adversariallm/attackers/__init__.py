"""Attacker models: trainable prompt generators, as opposed to the search procedures in `attacks/`.

`build_attacker(cfg)` is the only entry point callers should need; the backend seam exists so a
second diffusion backend can be added without touching the trainer or the attack.
"""

from .base import Attacker, DenoiseResult, MaskingMode, MaskingResult, build_attacker
from .diffusion import DiffusionAttackerConfig, LLaDADiffusionAttacker, LoRASpec

__all__ = [
    "Attacker",
    "DenoiseResult",
    "DiffusionAttackerConfig",
    "LLaDADiffusionAttacker",
    "LoRASpec",
    "MaskingMode",
    "MaskingResult",
    "build_attacker",
]
