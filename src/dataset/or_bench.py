from dataclasses import dataclass

from datasets import load_dataset

from src.types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class ORBenchConfig:
    name: str = "or_bench"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("or_bench")
class ORBenchDataset(PromptDataset):
    def __init__(self, config: ORBenchConfig):
        super().__init__(config)
        dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]
        dataset = [d["prompt"] for d in dataset]

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
