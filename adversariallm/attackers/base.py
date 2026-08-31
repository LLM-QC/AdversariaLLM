"""Attacker interface shared by the diffusion attackers.

An *attacker* is a model that produces adversarial prompts, as opposed to an *attack*, which
is the search procedure evaluated against a target system. The split exists because the IHO
attacker is trained (DPO on a LoRA adapter) independently of the attack that evaluates it, so
`adversariallm/training/dpo.py` only ever talks to this interface.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import torch
from beartype import beartype
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MaskingMode = Literal["all", "prompt", "attack"]


@beartype
@dataclass
class MaskingResult:
    """Output of :meth:`Attacker.mask_tokens` / :meth:`Attacker.build_attack_canvas`."""

    masked_ids: torch.Tensor  # (B, L) int64, mask_token_id at masked positions
    mask_positions: torch.Tensor  # (B, L) bool
    prompt_positions: torch.Tensor  # (B, L) bool -- the span that becomes the adversarial prompt


@beartype
@dataclass
class DenoiseResult:
    """Output of :meth:`Attacker.denoise`.

    ``n_forward_passes`` and ``n_tokens_per_pass`` are the inputs the FLOPs ledger needs
    (`adversariallm/training/flops.py::FlopsLedger.add_denoise`): diffusion sampling costs
    ``n_forward_passes`` full-canvas forwards per sequence, not one forward per emitted token.
    """

    token_ids: torch.Tensor  # (B, L) int64, fully denoised canvas
    loglikelihood: torch.Tensor  # (B,) float, sum of chosen-token logprobs over transferred positions
    n_forward_passes: int  # per sequence; == num_denoise_steps
    n_tokens_per_pass: int  # == L


class Attacker(ABC):
    """Interface an attacker model must satisfy to be trainable by ``DiffusionDPOTrainer``."""

    device: torch.device
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    has_lora: bool

    @property
    @abstractmethod
    def n_params_no_embed(self) -> int:
        """Non-embedding parameter count, the ``N`` of the Kaplan FLOPs model."""

    @abstractmethod
    def train(self) -> "Attacker": ...

    @abstractmethod
    def eval(self) -> "Attacker": ...

    @abstractmethod
    def to(self, device: str | torch.device) -> "Attacker": ...

    @contextmanager
    @abstractmethod
    def disable_adapter(self) -> Iterator[None]:
        """Run the wrapped block against the frozen base model (no-op without LoRA)."""

    @abstractmethod
    def save_lora(self, path: str) -> None: ...

    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor: ...

    @abstractmethod
    def build_attack_canvas(self, target_texts: list[str]) -> MaskingResult: ...

    @abstractmethod
    def mask_tokens(
        self,
        token_ids: torch.Tensor,
        masking_mode: MaskingMode,
        mask_all: bool = False,
    ) -> MaskingResult: ...

    @abstractmethod
    def compute_log_likelihood(
        self,
        masked_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        target_ids: torch.Tensor,
        use_base_model: bool = False,
    ) -> torch.Tensor: ...

    @abstractmethod
    def denoise(self, masked_ids: torch.Tensor) -> DenoiseResult: ...

    @abstractmethod
    def decode_prompt(self, denoised_ids: torch.Tensor, prompt_positions: torch.Tensor) -> list[str]: ...

    @abstractmethod
    def decode_full(self, denoised_ids: torch.Tensor) -> list[str]: ...


def build_attacker(cfg: Any, **kwargs: Any) -> Attacker:
    """Instantiate the attacker selected by ``cfg.backend``.

    Only ``"llada"`` exists today; the dispatch is the seam a second diffusion backend would
    plug into. ``kwargs`` are forwarded to the backend constructor (e.g. ``device=``).
    """
    backend = getattr(cfg, "backend", None)
    if backend == "llada":
        from .diffusion import LLaDADiffusionAttacker  # local import: diffusion.py imports this module

        return LLaDADiffusionAttacker(cfg, **kwargs)
    raise NotImplementedError(f"Unknown attacker backend: {backend!r}. Implemented backends: 'llada'.")
