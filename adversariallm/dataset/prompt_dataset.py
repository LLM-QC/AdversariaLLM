import logging
from collections.abc import Sequence
from typing import ClassVar, Type

import torch
from ..types import Conversation


class PromptDataset(torch.utils.data.Dataset):
    """Base class for prompt datasets with simple registry support."""

    _registry: ClassVar[dict[str, Type["PromptDataset"]]] = {}

    def __init__(self, config):
        self.config = config

    @classmethod
    def register(cls, name: str):
        """Decorator used by subclasses to register themselves by name."""

        def decorator(subclass: Type["PromptDataset"]) -> Type["PromptDataset"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str) -> Type["PromptDataset"]:
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise ValueError(f"Unknown dataset: {name}") from exc

    def _select_idx(self, config, dataset_len: int):
        """Select dataset indices according to configuration metadata."""
        if getattr(config, "shuffle", False):
            torch.manual_seed(getattr(config, "seed", 0))
            idx = torch.randperm(dataset_len)
        else:
            idx = torch.arange(dataset_len)

        config_idx = getattr(config, "idx", None)

        if isinstance(config_idx, str):
            if config_idx.startswith("list(range("):
                try:
                    config_idx = eval(config_idx, {"__builtins__": None}, {"range": range, "list": list})
                except Exception as e:
                    raise ValueError(f"Could not parse idx string: {config_idx}\n{e}")
            else:
                raise ValueError(f"Could not parse idx string: {config_idx}\nDoes not start with 'list(range('.")

        if isinstance(config_idx, int):
            idx = idx[config_idx : config_idx + 1]
        elif isinstance(config_idx, Sequence):
            logging.info(f"Selecting indices: {config_idx}")
            idx = idx[config_idx]
        elif config_idx is not None:
            raise ValueError(f"Invalid idx: {config_idx}")

        return idx, config_idx

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Conversation:
        raise NotImplementedError
