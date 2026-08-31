"""Attacker-training building blocks: FLOPs accounting, run logging, preference pairing.

Kept free of attack- and model-specific imports so each piece can be unit-tested on CPU and
reused by other attacks.
"""

from .dpo import CycleResult, DiffusionDPOTrainer, DPOConfig, dpo_phase_id, select_metric
from .flops import COMPONENTS, FlopsEntry, FlopsLedger, PassType, format_group_key, pass_flops
from .preference import (
    PairingConfig,
    PairingStrategy,
    PreferenceDataset,
    PreferencePair,
    ScoredSample,
    ThresholdedTopKBottomKPairing,
    build_preference_pairs,
    dataframe_to_samples,
    pairs_to_dataframe,
)
from .run_logging import LoggingConfig, RunLogger, wandb_enabled

__all__ = [
    "COMPONENTS",
    "CycleResult",
    "DPOConfig",
    "DiffusionDPOTrainer",
    "FlopsEntry",
    "FlopsLedger",
    "LoggingConfig",
    "PairingConfig",
    "PairingStrategy",
    "PassType",
    "PreferenceDataset",
    "PreferencePair",
    "RunLogger",
    "ScoredSample",
    "ThresholdedTopKBottomKPairing",
    "build_preference_pairs",
    "dataframe_to_samples",
    "dpo_phase_id",
    "format_group_key",
    "pairs_to_dataframe",
    "pass_flops",
    "select_metric",
    "wandb_enabled",
]
