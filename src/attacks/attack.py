import random
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AttackResult:
    """A class to store the attack results for all instances in a datasets."""

    attacks: list[list[str]]
    losses: list[list[float]]
    prompts: list[str]
    completions: list[list[int | None]] = None
    times: list[list[float]] = None


class Attack:
    def __init__(self, config):
        self.config = config
        set_seed(config.seed)

    @classmethod
    def from_name(cls, name):
        match name:
            case "autodan":
                from .autodan import AutoDANAttack

                return AutoDANAttack
            case "ample_gcg":
                from .ample_gcg import AmpleGCGAttack

                return AmpleGCGAttack
            case "direct":
                from .direct import DirectAttack

                return DirectAttack
            case "gcg":
                from .gcg import GCGAttack

                return GCGAttack
            case "human_jailbreaks":
                from .human_jailbreaks import HumanJailbreaksAttack

                return HumanJailbreaksAttack
            case "pair":
                from .pair import PAIRAttack

                return PAIRAttack
            case "pgd":
                from .pgd import PGDAttack

                return PGDAttack
            case "pgd_one_hot":
                from .pgd_one_hot import PGDOneHotAttack

                return PGDOneHotAttack
            case _:
                raise ValueError(f"Unknown attack: {name}")

    def run(
        self,
        model: transformers.AutoModelForCausalLM,
        tokenizer: transformers.AutoTokenizer,
        dataset: Dataset,
    ) -> AttackResult:
        raise NotImplementedError
