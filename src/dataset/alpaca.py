"""
@article{taori2023alpaca,
  title={Alpaca: A strong, replicable instruction-following model},
  author={Taori, Rohan and Gulrajani, Ishaan and Zhang, Tianyi and Dubois, Yann and Li, Xuechen and Guestrin, Carlos and Liang, Percy and Hashimoto, Tatsunori B},
  journal={Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html},
  volume={3},
  number={6},
  pages={7},
  year={2023}
}
"""

from dataclasses import dataclass

from datasets import load_dataset

from ..types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class AlpacaConfig:
    name: str = "alpaca"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("alpaca")
class AlpacaDataset(PromptDataset):
    def __init__(self, config: AlpacaConfig):
        super().__init__(config)
        dataset = load_dataset("tatsu-lab/alpaca")["train"].to_pandas()

        self.idx, self.config_idx = self._select_idx(config, len(dataset))
        self.data = [dataset.iloc[idx] for idx in self.idx.tolist()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Conversation:
        msg = self.data[idx]["instruction"]
        target = self.data[idx]["output"]
        conversation = [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": target}
        ]
        return conversation
