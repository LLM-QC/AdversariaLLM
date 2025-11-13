"""
@article{rottger2023xstest,
  title={Xstest: A test suite for identifying exaggerated safety behaviours in large language models},
  author={R{\"o}ttger, Paul and Kirk, Hannah Rose and Vidgen, Bertie and Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  journal={arXiv preprint arXiv:2308.01263},
  year={2023}
}
"""
from dataclasses import dataclass
from typing import Sequence

from datasets import load_dataset

from src.types import Conversation
from .prompt_dataset import PromptDataset


@dataclass
class XSTestConfig:
    name: str = "xstest"
    seed: int = 0
    idx: list[int] | int | str | None = None
    categories: tuple[str, ...] = ("privacy_fictional", "homonyms", "safe_contexts", "safe_targets", "figurative_language", "privacy_public", "historical_events", "real_group_nons_discr", "definitions", "nons_group_real_discr")
    shuffle: bool = True


@PromptDataset.register("xs_test")
class XSTestDataset(PromptDataset):
    def __init__(self, config: XSTestConfig):
        super().__init__(config)
        dataset: Sequence[dict] = load_dataset("walledai/XSTest")["test"]
        dataset = [d["prompt"] for d in dataset if d["label"] == "safe" and d["type"] in config.categories]

        self.idx, self.config_idx = self._select_idx(config, len(dataset))

        self.messages = [dataset[i] for i in self.idx]
        self.targets = [""] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages[idx]
        conversation = [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": ""},
        ]
        return conversation
