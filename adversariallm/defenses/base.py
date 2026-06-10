from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..lm_utils import LocalTextGenerator, get_losses_batched
from ..lm_utils.text_generation import GenerationResult, RetryOverrides
from ..types import Conversation


@dataclass
class DefenseDecision:
    output_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TargetSystem(ABC):
    """Runtime interface attacks use for target-system interactions."""

    NAME: str

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        default_generate_kwargs: dict[str, Any] | None = None,
    ) -> "TargetSystem":
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        filters: list[dict] | None = None,
        retry_overrides: RetryOverrides | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        raise NotImplementedError

    @torch.no_grad()
    def loss(
        self,
        full_token_tensors: list[torch.Tensor],
        prompt_token_tensors: list[torch.Tensor],
        *,
        initial_batch_size: int,
        verbose: bool = False,
    ) -> list[float | None]:
        if len(full_token_tensors) != len(prompt_token_tensors):
            raise ValueError(
                "full_token_tensors and prompt_token_tensors must have the same length: "
                f"got {len(full_token_tensors)} and {len(prompt_token_tensors)}."
            )
        shifted_target_tensors = [t.roll(-1, 0) for t in full_token_tensors]
        all_losses_per_token = get_losses_batched(
            self.model,
            targets=shifted_target_tensors,
            token_list=full_token_tensors,
            initial_batch_size=initial_batch_size,
            verbose=verbose,
        )

        instance_losses: list[float | None] = []
        for full_tokens, prompt_tokens, losses_per_token in zip(
            full_token_tensors,
            prompt_token_tensors,
            all_losses_per_token,
        ):
            full_len = full_tokens.size(0)
            prompt_len = prompt_tokens.size(0)
            target_token_losses = losses_per_token[prompt_len - 1 : full_len - 1]
            if target_token_losses.numel() > 0:
                instance_losses.append(target_token_losses.mean().item())
            else:
                instance_losses.append(None)
        return instance_losses


class UndefendedTarget(TargetSystem):
    NAME = "none"

    @classmethod
    def from_config(
        cls,
        _cfg: dict[str, Any],
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        default_generate_kwargs: dict[str, Any] | None = None,
    ) -> "UndefendedTarget":
        return cls(model, tokenizer, default_generate_kwargs=default_generate_kwargs)

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        default_generate_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model, tokenizer)
        self.text_generator = LocalTextGenerator(
            model,
            tokenizer,
            default_generate_kwargs=default_generate_kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        filters: list[dict] | None = None,
        retry_overrides: RetryOverrides | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        return self.text_generator.generate(
            convs,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            filters=filters,
            retry_overrides=retry_overrides,
            **kwargs,
        )
