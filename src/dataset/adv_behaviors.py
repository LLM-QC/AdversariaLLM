import logging
import os
from dataclasses import dataclass

import pandas as pd

from src.types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class AdvBehaviorsConfig:
    name: str = "adv_behaviors"
    messages_path: str = "./data/behavior_datasets/harmbench_behaviors_text_all.csv"
    targets_path: str = "./data/optimizer_targets/harmbench_targets_text.json"
    categories: tuple[str, ...] = ('chemical_biological', 'illegal', 'misinformation_disinformation', 'harmful', 'harassment_bullying', 'cybercrime_intrusion')
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("adv_behaviors")
class AdvBehaviorsDataset(PromptDataset):
    def __init__(self, config: AdvBehaviorsConfig):
        super().__init__(config)

        logging.info(f"Loading dataset from {config.messages_path}")
        logging.info(f"Current working directory: {os.getcwd()}")

        self.messages = pd.read_csv(config.messages_path, header=0)
        self.messages = self.messages[self.messages["SemanticCategory"].isin(config.categories)]
        targets = pd.read_json(config.targets_path, typ="series")
        self.targets = self.messages[self.messages.columns[-1]].map(targets)
        assert len(self.messages) == len(self.targets), "Mismatched lengths"

        self.idx, self.config_idx = self._select_idx(config, len(self.messages))

        self.messages = self.messages.iloc[self.idx].reset_index(drop=True)
        self.targets = self.targets.iloc[self.idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.messages.iloc[idx]
        target = self.targets.iloc[idx]
        if isinstance(msg["ContextString"], str):
            content = msg["ContextString"] + "\n\n" + msg["Behavior"]
        else:
            content = msg["Behavior"]
        conversation = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": target},
        ]
        return conversation
