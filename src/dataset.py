import json
import logging
import os
from dataclasses import dataclass
from typing import Literal, Sequence

import jailbreakbench as jbb
import pandas as pd
import torch
from datasets import load_dataset


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
            case _:
                raise ValueError(f"Unknown dataset: {name}")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        raise NotImplementedError


@dataclass
class AdvBehaviorsConfig:
    name: str
    messages_path: str
    targets_path: str
    categories: list[str]
    seed: int = 0
    idx: list[int] | int | None = None


class AdvBehaviorsDataset(Dataset):
    def __init__(self, config: AdvBehaviorsConfig):
        self.config = config

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

        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(self.messages))
        # We keep this shuffle for backwards compatibility
        if isinstance(config.idx, int):
            idx = idx[config.idx:config.idx + 1]
        elif isinstance(config.idx, Sequence):
            idx = idx[config.idx]
        elif config.idx is not None:
            raise ValueError(f"Invalid idx: {config.idx}")
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
class RefusalDirectionDataConfig:
    name: str
    path: str
    split: str
    type: Literal["harmful", "harmless"]
    seed: int = 0
    idx: list[int] | int | None = None
    n_samples: int = 100

class RefusalDirectionDataDataset(Dataset):
    def __init__(self, config: RefusalDirectionDataConfig):
        self.config = config
        with open(os.path.join(config.path, f"{config.type}_{config.split}.json"), "r") as f:
            self.messages = json.load(f)[:config.n_samples]
        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(self.messages))
        # We keep this shuffle for backwards compatibility
        if isinstance(config.idx, int):
            idx = idx[config.idx:config.idx + 1]
        elif isinstance(config.idx, Sequence):
            idx = idx[config.idx]
        elif config.idx is not None:
            raise ValueError(f"Invalid idx: {config.idx}")
        # Convert tensor indices to integers before using them
        self.messages = [self.messages[int(i)]["instruction"] for i in idx]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> str:
        return self.messages[idx]


@dataclass
class JBBBehaviorsConfig:
    name: str
    seed: int = 0
    idx: list[int]|int|None = None


class JBBBehaviorsDataset(Dataset):
    def __init__(self, config: JBBBehaviorsConfig):
        self.config = config
        dataset = jbb.read_dataset().as_dataframe()

        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(dataset))
        if isinstance(config.idx, int):
            idx = idx[config.idx:config.idx + 1]
        elif isinstance(config.idx, Sequence):
            idx = idx[config.idx]
        elif config.idx is not None:
            raise ValueError(f"Invalid idx: {config.idx}")
        self.messages = dataset.Goal.iloc[idx].reset_index(drop=True)
        self.targets = dataset.Target.iloc[idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        msg = self.messages.iloc[idx]
        message = {"role": "user", "content": msg}
        target: str = self.targets.iloc[idx]
        return message, target


class StrongRejectConfig:
    name: str
    categories: list[str]
    split: str = "train"
    seed: int = 0
    version: Literal["small", "full"] = "small"
    idx: list[int] | int | None = None


class StrongRejectDataset(Dataset):
    def __init__(self, config: StrongRejectConfig):
        self.config = config
        url = "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/"
        if config.version == "small":
            url += "strongreject_small_dataset.csv"
        else:
            url += "strongreject_dataset.csv"

        self.dataset = load_dataset("csv", data_files=url)[config.split]

        # Ignore copyright-related rows
        self.messages = self.dataset[
            self.dataset["category"].isin(config.categories)
        ]
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(self.messages))
        # We keep this shuffle for backwards compatibility
        if isinstance(config.idx, int):
            idx = idx[config.idx:config.idx + 1]
        elif isinstance(config.idx, Sequence):
            idx = idx[config.idx]
        elif config.idx is not None:
            raise ValueError(f"Invalid idx: {config.idx}")
        # cut
        self.messages = self.messages.iloc[idx].reset_index(drop=True)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str, str], str]:
        msg = self.messages.iloc[idx]
        message = {"role": "user", "content": msg["forbidden_prompt"]}

        target: str = ""
        return message, target


@dataclass
class ORBenchConfig:
    name: str
    seed: int = 0
    idx: list[int]|int|None = None


class ORBenchDataset(Dataset):
    def __init__(self, config: ORBenchConfig):
        self.config = config
        dataset = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]
        dataset = [d["prompt"] for d in dataset]
        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(dataset))
        if isinstance(config.idx, int):
            idx = idx[config.idx: config.idx + 1]
        elif isinstance(config.idx, Sequence):
            idx = idx[config.idx]
        elif config.idx is not None:
            raise ValueError(f"Invalid idx: {config.idx}")
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
class XSTestConfig:
    name: str
    seed: int = 0
    idx: list[int]|int|None = None


class XSTestDataset(Dataset):
    def __init__(self, config: XSTestConfig):
        self.config = config
        dataset = load_dataset("walledai/XSTest")["test"]
        dataset = [d["prompt"] for d in dataset if d["label"] == "safe"]
        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(dataset))
        if isinstance(config.idx, int):
            idx = idx[config.idx: config.idx + 1]
        elif isinstance(config.idx, Sequence):
            idx = idx[config.idx]
        elif config.idx is not None:
            raise ValueError(f"Invalid idx: {config.idx}")
        self.messages = [dataset[i] for i in idx]
        self.targets = [""] * len(self.messages)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> tuple[dict[str], str]:
        msg = self.messages[idx]
        message = {"role": "user", "content": msg}
        target: str = "ZZZZZ"  # dummy, shouldn't be empty for prepare_tokens
        return message, target


class AlpacaDataset(Dataset):
    def __init__(self, config: XSTestConfig):
        self.config = config
        dataset = json.load(open(config.messages_path, "r"))
        dataset = [d["instruction"] for d in dataset]
        # shuffle
        torch.manual_seed(config.seed)
        idx = torch.randperm(len(dataset))
        if isinstance(config.idx, int):
            idx = idx[config.idx: config.idx + 1]
        elif isinstance(config.idx, Sequence):
            idx = idx[config.idx]
        elif config.idx is not None:
            raise ValueError(f"Invalid idx: {config.idx}")
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
        XSTestConfig("xs_test", seed=1)
    )
    print(len(d))
    for thing in d:
        print(thing)
