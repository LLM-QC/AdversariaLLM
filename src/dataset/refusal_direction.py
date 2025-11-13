"""
@article{arditi2024refusal,
  title={Refusal in language models is mediated by a single direction},
  author={Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={136037--136083},
  year={2024}
}
"""

import json
import os
from dataclasses import dataclass
from typing import Literal

from src.types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class RefusalDirectionDataConfig:
    name: str = "refusal_direction"
    path: str = "./data/refusal_direction/"
    split: str = "train"
    type: Literal["harmful", "harmless"] = "harmful"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("refusal_direction")
class RefusalDirectionDataDataset(PromptDataset):
    def __init__(self, config: RefusalDirectionDataConfig):
        super().__init__(config)
        with open(os.path.join(config.path, f"{config.type}_{config.split}.json"), "r") as f:
            self.messages = json.load(f)

        self.idx, self.config_idx = self._select_idx(config, len(self.messages))

        self.messages = [self.messages[int(i)]["instruction"] for i in self.idx]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        conversation = [
            {"role": "user", "content": self.messages[idx]},
            {"role": "assistant", "content": ""},
        ]
        return conversation
