"""
@article{chao2024jailbreakbench,
  title={Jailbreakbench: An open robustness benchmark for jailbreaking large language models},
  author={Chao, Patrick and Debenedetti, Edoardo and Robey, Alexander and Andriushchenko, Maksym and Croce, Francesco and Sehwag, Vikash and Dobriban, Edgar and Flammarion, Nicolas and Pappas, George J and Tramer, Florian and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={55005--55029},
  year={2024}
}
"""

from dataclasses import dataclass

import jailbreakbench as jbb

from ..types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class JBBBehaviorsConfig:
    name: str = "jbb_behaviors"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("jbb_behaviors")
class JBBBehaviorsDataset(PromptDataset):
    def __init__(self, config: JBBBehaviorsConfig):
        super().__init__(config)
        dataset = jbb.read_dataset().as_dataframe()

        self.idx, self.config_idx = self._select_idx(config, len(dataset))

        self.messages = dataset.Goal.iloc[self.idx].reset_index(drop=True)
        self.targets = dataset.Target.iloc[self.idx].reset_index(drop=True)

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
