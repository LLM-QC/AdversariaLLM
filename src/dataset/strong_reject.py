import os
from dataclasses import dataclass
from typing import Literal

from datasets import load_dataset

from src.types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class StrongRejectConfig:
    name: str = "strong_reject"
    path: str = "./data/strong_reject/"
    split: str = "train"
    categories: tuple[str, ...] = ("Disinformation and deception", "Hate, harassment and discrimination", "Illegal goods and services", "Non-violent crimes", "Sexual content", "Violence")
    seed: int = 0
    version: Literal["small", "full"] = "full"
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("strong_reject")
class StrongRejectDataset(PromptDataset):
    def __init__(self, config: StrongRejectConfig):
        super().__init__(config)
        full_url = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"

        self.dataset = load_dataset("csv", data_files=full_url)[config.split]
        self.targets = [line.rstrip() for line in open(os.path.join(config.path, "targets.txt"), "r").readlines()]
        assert len(self.dataset) == len(self.targets), f"Mismatched lengths, {len(self.dataset)} != {len(self.targets)}"

        self.dataset = self.dataset.add_column("target", self.targets)
        self.dataset = self.dataset.to_pandas()

        if config.version == "small":
            small_url = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv"
            self.small_dataset = load_dataset("csv", data_files=small_url)[config.split]
            small_dataset_df = self.small_dataset.to_pandas()
            small_prompts = set(small_dataset_df["forbidden_prompt"])
            self.dataset = self.dataset[self.dataset["forbidden_prompt"].isin(small_prompts)]

        self.idx, self.config_idx = self._select_idx(config, len(self.dataset))
        self.dataset = self.dataset.iloc[self.idx]

        self.dataset = self.dataset[self.dataset["category"].isin(config.categories)]
        self.messages = self.dataset["forbidden_prompt"]
        self.targets = self.dataset["target"]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages.iloc[idx]
        target = self.targets.iloc[idx]
        conversation = [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": target},
        ]
        return conversation
