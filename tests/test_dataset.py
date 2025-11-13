import json
from pathlib import Path

import pandas as pd

from src.types import Conversation


def _is_conversation(instance: Conversation) -> bool:
    return isinstance(instance, list) and all(isinstance(msg, dict) for msg in instance) and all("role" in msg and "content" in msg for msg in instance)


def test_adv_behaviors_dataset():
    from src.dataset import AdvBehaviorsConfig, AdvBehaviorsDataset

    config = AdvBehaviorsConfig(
        name="adv_behaviors",
        messages_path="./data/behavior_datasets/harmbench_behaviors_text_all.csv",
        targets_path="./data/optimizer_targets/harmbench_targets_text.json",
        # categories=("chemical_biological", "illegal"),
        seed=0,
        shuffle=False,
    )
    dataset = AdvBehaviorsDataset(config)

    assert len(dataset) == 300

    for conv in dataset:
        assert len(conv) == 2
        assert _is_conversation(conv)

def test_alpaca_dataset():
    from src.dataset import AlpacaConfig, AlpacaDataset

    config = AlpacaConfig(
        name="alpaca",
        seed=0,
        shuffle=False,
    )
    dataset = AlpacaDataset(config)

    assert len(dataset) == 52002
    for conv in dataset:
        assert len(conv) == 2, "Alpaca conversation length should be 2"
        assert _is_conversation(conv)


def test_jbb_behaviors_dataset():
    from src.dataset import JBBBehaviorsConfig, JBBBehaviorsDataset

    config = JBBBehaviorsConfig(
        name="jbb_behaviors",
        seed=0,
        shuffle=False,
    )
    dataset = JBBBehaviorsDataset(config)

    assert len(dataset) == 100
    for conv in dataset:
        assert len(conv) == 2, "JBBBehaviors conversation length should be 2"
        assert _is_conversation(conv)



def test_mmlu_dataset():
    from src.dataset import MMLUConfig, MMLUDataset

    config = MMLUConfig(
        name="mmlu",
        seed=0,
        shuffle=False,
    )
    dataset = MMLUDataset(config)

    assert len(dataset) == 14042
    for conv in dataset:
        assert len(conv) == 2, "MMLU conversation length should be 2"
        assert _is_conversation(conv)


def test_or_bench_dataset():
    from src.dataset import ORBenchConfig, ORBenchDataset

    config = ORBenchConfig(name="or_bench", shuffle=False)
    dataset = ORBenchDataset(config)

    assert len(dataset) == 1319
    for conv in dataset:
        assert len(conv) == 1, "ORBench conversation length should be 1"
        assert _is_conversation(conv)


def test_refusal_direction_dataset():
    from src.dataset import RefusalDirectionDataConfig, RefusalDirectionDataDataset

    config = RefusalDirectionDataConfig(
        name="refusal_direction",
        path="./data/refusal_direction/",
        split="train",
        type="harmless",
        n_samples=2,
        shuffle=False,
    )
    dataset = RefusalDirectionDataDataset(config)

    assert len(dataset) == 2
    for conv in dataset:
        assert len(conv) == 1
        assert _is_conversation(conv)


def test_strong_reject_dataset():
    from src.dataset import StrongRejectConfig, StrongRejectDataset

    config = StrongRejectConfig(
        name="strong_reject",
        path="./data/strong_reject/",
        split="train",
        seed=0,
        shuffle=False,
    )
    dataset = StrongRejectDataset(config)

    assert len(dataset) == 313
    for conv in dataset:
        assert len(conv) == 2, "StrongReject conversation length should be 2"
        assert _is_conversation(conv)

    # small version
    config = StrongRejectConfig(
        name="strong_reject",
        path="./data/strong_reject/",
        split="train",
        version="small",
        seed=0,
        shuffle=False,
    )
    dataset = StrongRejectDataset(config)

    assert len(dataset) == 60
    for conv in dataset:
        assert len(conv) == 2, "StrongReject conversation length should be 2"
        assert _is_conversation(conv)


def test_xstest_dataset():
    from src.dataset import XSTestConfig, XSTestDataset

    config = XSTestConfig(
        name="xstest",
        shuffle=False,
    )
    dataset = XSTestDataset(config)

    assert len(dataset) == 250
    for conv in dataset:
        assert _is_conversation(conv)

    config = XSTestConfig(
        name="xstest",
        seed=1,
        idx=list(range(200)),
        shuffle=True,
    )
    dataset = XSTestDataset(config)

    assert len(dataset) == 200
    for conv in dataset:
        assert _is_conversation(conv)
