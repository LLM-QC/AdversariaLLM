from dataclasses import dataclass
import logging
from typing import Sequence

import jailbreakbench as jbb
import pandas as pd
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _get_selected_idx(idx, batch: int = None, max_dataset_size: int = None):
    """Handles creating a sequence of indices for selecting a sequential batch of data

    Args:
        idx (int, list): If batching, the starting index. If not batching, a list of indices or single int.
        batch (int, optional): Number of elements to batch . Defaults to None.
        max_dataset_size (int, optional)
    """
    # handle batching inputs
    if batch is not None:
        if isinstance(idx, Sequence):
            raise ValueError("Cannot use config.batch with a sequence of indices")
        selected_idx = list(range(idx, min(idx + batch, max_dataset_size)))
        return selected_idx
    return idx

def _shuffle_and_select_idx(selected_idx: int|Sequence, seed: int, dataset_size: int):
    torch.manual_seed(seed)
    idx = torch.randperm(dataset_size)

    # We keep this shuffle for backwards compatibility
    if isinstance(selected_idx, int):
        idx = idx[selected_idx:selected_idx + 1]
    elif isinstance(selected_idx, Sequence):
        idx = idx[selected_idx]
        logger.info(f"Selected subset of data: {selected_idx}")
    elif selected_idx is not None:
        raise ValueError(f"Invalid idx: {selected_idx}")
    logger.info(f"Selected all data: {len(idx)}")
    return idx


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_name(cls, name):
        match name:
            case "adv_behaviors":
                return AdvBehaviorsDataset
            case "jbb_behaviors":
                return JBBBehaviorsDataset
            case "or_bench":
                return ORBenchDataset
            case "xs_test":
                return XSTestDataset
            case "rf_test":
                return RFTestDataset
            case _:
                raise ValueError(f"Unknown dataset: {name}")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        raise NotImplementedError


@dataclass(kw_only=True)
class BaseDataConfig:
    seed: int = 0
    idx: list[int]|int|None = None
    batch: int|None = None


@dataclass
class AdvBehaviorsConfig(BaseDataConfig):
    name: str
    messages_path: str
    targets_path: str
    categories: list[str]


class AdvBehaviorsDataset(Dataset):
    def __init__(self, config: AdvBehaviorsConfig):
        self.config = config
        import logging
        import os
        import sys
        logging.info(f"Loading dataset from {config.messages_path}")
        logging.info(f"Current working directory: {os.getcwd()}")

        self.messages = pd.read_csv(config.messages_path, header=0)
        # Ignore copyright-related rows
        self.messages = self.messages[
            self.messages["SemanticCategory"].isin(config.categories)
        ]
        targets = pd.read_json(config.targets_path, typ="series")
        # Merge the CSV data with the JSON data
        self.targets = self.messages[self.messages.columns[-1]].map(targets)
        assert len(self.messages) == len(self.targets), "Mismatched lengths"

        selected_idx = _get_selected_idx(config.idx, config.batch, len(self.messages))
        idx = _shuffle_and_select_idx(selected_idx, config.seed, len(self.messages))

        # cut
        self.messages = self.messages.iloc[idx].reset_index(drop=True)
        self.targets = self.targets.iloc[idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        msg = self.messages.iloc[idx]
        message = {"role": "user", "content": msg["Behavior"]}

        if isinstance(msg["ContextString"], str):
            message["content"] = msg["ContextString"] + "\n\n" + message["content"]
        target: str = self.targets.iloc[idx]
        return message, target


@dataclass
class RFTestConfig(BaseDataConfig):
    name: str
    path: str | None = None


class RFTestDataset(Dataset):
    def __init__(self, config: RFTestConfig):
        self.config = config
        _path = config.path if config.path else "redflag-tokens/harmful-harmless-eval"
        dataset = load_dataset(_path, "harmful")['test']
        messages = [d["Behavior"] for d in dataset]
        targets = [d["Prefilling"] for d in dataset]

        # shuffle
        selected_idx = _get_selected_idx(config.idx, config.batch, len(dataset))
        idx = _shuffle_and_select_idx(selected_idx, config.seed, len(dataset))

        self.messages = [messages[i] for i in idx]
        self.targets = [targets[i] for i in idx]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        msg = self.messages[idx]
        # messages = [[{"role": "user", "content": m}] for m in msg]  # TODO: is this safe?
        messages = {"role": "user", "content": msg}
        target = self.targets[idx] 
        return messages, target


@dataclass
class JBBBehaviorsConfig(BaseDataConfig):
    name: str


class JBBBehaviorsDataset(Dataset):
    def __init__(self, config: JBBBehaviorsConfig):
        self.config = config
        dataset = jbb.read_dataset().as_dataframe()

        # shuffle
        selected_idx = _get_selected_idx(config.idx, config.batch, len(dataset))
        idx = _shuffle_and_select_idx(selected_idx, config.seed, len(dataset))

        self.messages = dataset.Goal.iloc[idx].reset_index(drop=True)
        self.targets = dataset.Target.iloc[idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        msg = self.messages.iloc[idx]
        message = {"role": "user", "content": msg}
        target: str = self.targets.iloc[idx]
        return message, target


@dataclass
class ORBenchConfig(BaseDataConfig):
    name: str


class ORBenchDataset(Dataset):
    def __init__(self, config: ORBenchConfig):
        self.config = config
        dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]
        dataset = [d["prompt"] for d in dataset]

        # shuffle
        selected_idx = _get_selected_idx(config.idx, config.batch, len(dataset))
        idx = _shuffle_and_select_idx(selected_idx, config.seed, len(dataset))

        self.messages = [dataset[i] for i in idx]
        self.targets = [""] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        msg = self.messages[idx]
        message = {"role": "user", "content": msg}
        target: str = "ZZZZZ"  # dummy, shouldn't be empty for prepare_tokens
        return message, target


@dataclass
class XSTestConfig(BaseDataConfig):
    name: str


class XSTestDataset(Dataset):
    def __init__(self, config: XSTestConfig):
        self.config = config
        dataset = load_dataset("walledai/XSTest")["test"]
        dataset = [d["prompt"] for d in dataset if d["label"] == "safe"]

        # shuffle
        selected_idx = _get_selected_idx(config.idx, config.batch, len(dataset))
        idx = _shuffle_and_select_idx(selected_idx, config.seed, len(dataset))

        self.messages = [dataset[i] for i in idx]
        self.targets = [""] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str], str]:
        msg = self.messages[idx]
        message = {"role": "user", "content": msg}
        target: str = "ZZZZZ"  # dummy, shouldn't be empty for prepare_tokens
        return message, target


if __name__ == "__main__":
    d = Dataset.from_name("adv_behaviors")(
        AdvBehaviorsConfig(
            "adv_behaviors",
            messages_path="data/behavior_datasets/harmbench_behaviors_text_all.csv",
            targets_path="data/optimizer_targets/harmbench_targets_text.json",
            seed=1,
            categories=[
                "chemical_biological",
                "illegal",
                "misinformation_disinformation",
                "harmful",
                "harassment_bullying",
                "cybercrime_intrusion",
            ],
        )
    )
    print(len(d))
    d = Dataset.from_name("jbb_behaviors")(
        JBBBehaviorsConfig("jbb_behaviors", seed=1)
    )
    print(len(d))
    d = Dataset.from_name("or_bench")(
        ORBenchConfig("or_bench", seed=1)
    )

    print(len(d))
    d = Dataset.from_name("xs_test")(
        ORBenchConfig("xs_test", seed=1)
    )
    print(len(d))
    for thing in d:
        print(thing)
