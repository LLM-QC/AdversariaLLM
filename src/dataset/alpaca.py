import json
from dataclasses import dataclass

from src.types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class AlpacaConfig:
    name: str = "alpaca"
    messages_path: str = "./data/alpaca.json"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("alpaca")
class AlpacaDataset(PromptDataset):
    def __init__(self, config: AlpacaConfig):
        super().__init__(config)
        dataset = json.load(open(config.messages_path, "r"))
        dataset = [d["instruction"] for d in dataset]

        self.idx, self.config_idx = self._select_idx(config, len(dataset))

        self.messages = [dataset[i] for i in self.idx]
        self.targets = [""] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages[idx]
        conversation = [
            {"role": "user", "content": msg},
        ]
        return conversation
