"""Preference-pair construction for diffusion DPO training.

Ported from ``aapl/datasets/PreferenceDataset.py``. The pairing *semantics* are preserved
character for character (see :class:`ThresholdedTopKBottomKPairing`); what changed is the
plumbing:

* pools come from in-memory DataFrames instead of a run directory, so the caller owns all file
  I/O and this module is trivially testable;
* the score column is a parameter rather than the hardcoded ``judge_score_training``;
* :class:`PairingStrategy` is genuinely polymorphic -- AAPL declared an ABC and then rejected
  every subclass but one with an ``isinstance`` check;
* no ``__main__`` block and no hardcoded absolute paths.

Input column names follow plan §4.6.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pandas as pd
from beartype import BeartypeConf, beartype
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)

#: beartype with the PEP 484 numeric tower enabled, so an ``int`` is accepted wherever a
#: ``float`` is annotated -- ``min_score: 0`` in a yaml config means ``0.0`` to everyone but a
#: strict type checker.
_typed = beartype(conf=BeartypeConf(is_pep484_tower=True))

# Input column contract (plan §4.6). `score_col` is a parameter; the rest are fixed.
BEHAVIOR_COL = "jb_index"
PROMPT_COL = "original_prompt_text"
FULL_TEXT_COL = "inpainted_prompt_text_full"
CYCLE_COL = "cycle_id"
DEFAULT_SCORE_COL = "judge_score_training"


@_typed
@dataclass(frozen=True)
class ScoredSample:
    """One scored attacker sample, i.e. one row of a ``samples/cycle_<i>.parquet``."""

    behavior_idx: int
    prompt: str
    full_text: str
    score: float
    cycle: int


@_typed
@dataclass(frozen=True)
class PreferencePair:
    behavior_idx: int
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    chosen_cycle: int
    rejected_cycle: int


@_typed
@dataclass
class PairingConfig:
    min_score: float = 0.2  # AAPL harmfulness_threshold
    percent_chosen: float = 0.125
    expanding: bool = False  # pool the chosen candidates over previous cycles too


PAIR_COLUMNS = tuple(f.name for f in fields(PreferencePair))


class PairingStrategy(ABC):
    """Turns a (chosen pool, rejected pool) for a single group into preference pairs.

    Implementations receive one ``(behavior_idx, prompt)`` group at a time and the group's ``k``.
    """

    @abstractmethod
    def generate(
        self,
        *,
        chosen_pool: list[ScoredSample],
        rejected_pool: list[ScoredSample],
        k: int,
    ) -> list[PreferencePair]: ...


class ThresholdedTopKBottomKPairing(PairingStrategy):
    """AAPL's pairing rule.

    Within one ``(behavior_idx, prompt)`` group, given ``k``:

    1. sort the *rejected* pool ascending by score and take the bottom ``k``;
    2. drop chosen candidates scoring below ``min_score``, sort ascending, take the top ``k``
       (i.e. the last ``k`` of the ascending list);
    3. with ``pair_k = min(len(top_k), len(bottom_k))``, zip ``reversed(top_k[-pair_k:])``
       against ``bottom_k[:pair_k]``.

    So the best chosen is paired with the worst rejected, the second best with the second worst,
    and so on. ``prompt`` and ``behavior_idx`` on the pair come from the *chosen* sample. Sorting
    is stable, so ties keep pool order -- which for equal scores makes the result depend on row
    order, exactly as in AAPL.
    """

    def __init__(self, min_score: float = 0.2):
        self.min_score = min_score

    def generate(
        self,
        *,
        chosen_pool: list[ScoredSample],
        rejected_pool: list[ScoredSample],
        k: int,
    ) -> list[PreferencePair]:
        if k <= 0 or not rejected_pool:
            return []

        rejected_sorted = sorted(rejected_pool, key=lambda s: s.score)
        bottom_k = rejected_sorted[:k]

        chosen_candidates = [s for s in chosen_pool if s.score >= self.min_score]
        if not chosen_candidates:
            return []

        chosen_sorted = sorted(chosen_candidates, key=lambda s: s.score)
        top_k = chosen_sorted[-k:]

        pair_k = min(len(top_k), len(bottom_k))
        if pair_k == 0:
            return []

        return [
            PreferencePair(
                behavior_idx=chosen.behavior_idx,
                prompt=chosen.prompt,
                chosen=chosen.full_text,
                rejected=rejected.full_text,
                chosen_score=chosen.score,
                rejected_score=rejected.score,
                chosen_cycle=chosen.cycle,
                rejected_cycle=rejected.cycle,
            )
            for chosen, rejected in zip(reversed(top_k[-pair_k:]), bottom_k[:pair_k])
        ]


def dataframe_to_samples(df: pd.DataFrame, *, score_col: str = DEFAULT_SCORE_COL) -> list[ScoredSample]:
    """Convert a sample DataFrame (plan §4.6 columns) into :class:`ScoredSample` records.

    The schema is checked even for an empty frame: a missing column is a caller bug, and letting
    it through would silently produce zero pairs instead of an error.
    """
    missing = [c for c in (BEHAVIOR_COL, PROMPT_COL, FULL_TEXT_COL, CYCLE_COL, score_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Sample DataFrame is missing required column(s) {missing}; has {list(df.columns)}")

    return [
        ScoredSample(
            behavior_idx=int(behavior_idx),
            prompt=str(prompt),
            full_text=str(full_text),
            score=float(score),
            cycle=int(cycle),
        )
        for behavior_idx, prompt, full_text, score, cycle in zip(
            df[BEHAVIOR_COL], df[PROMPT_COL], df[FULL_TEXT_COL], df[score_col], df[CYCLE_COL]
        )
    ]


def build_preference_pairs(
    current: pd.DataFrame,
    previous: list[pd.DataFrame],
    cfg: PairingConfig,
    *,
    score_col: str = DEFAULT_SCORE_COL,
    strategy: PairingStrategy | None = None,
) -> list[PreferencePair]:
    """Build DPO preference pairs from the current cycle's samples.

    The rejected pool is *always* the current cycle only. With ``cfg.expanding`` the chosen pool
    is every previous cycle followed by the current one; otherwise it is the current cycle only.
    Both pools are grouped by ``(behavior_idx, prompt)`` and ``k`` is computed **per group** from
    the size of that group's chosen pool: ``k = int(len(chosen_group) * cfg.percent_chosen)``.
    Groups are visited in the order they first appear in ``current``.

    ``strategy`` defaults to :class:`ThresholdedTopKBottomKPairing` built from ``cfg.min_score``;
    an explicit strategy is used as given and ``cfg.min_score`` is then ignored.
    """
    if strategy is None:
        strategy = ThresholdedTopKBottomKPairing(min_score=cfg.min_score)

    current_samples = dataframe_to_samples(current, score_col=score_col)

    if cfg.expanding:
        previous_samples: list[ScoredSample] = []
        for df in previous:
            previous_samples.extend(dataframe_to_samples(df, score_col=score_col))
        chosen_pool = previous_samples + current_samples
    else:
        chosen_pool = current_samples

    rejected_pool = current_samples

    grouped_chosen: defaultdict[tuple[int, str], list[ScoredSample]] = defaultdict(list)
    grouped_rejected: defaultdict[tuple[int, str], list[ScoredSample]] = defaultdict(list)
    for sample in chosen_pool:
        grouped_chosen[(sample.behavior_idx, sample.prompt)].append(sample)
    for sample in rejected_pool:
        grouped_rejected[(sample.behavior_idx, sample.prompt)].append(sample)

    pairs: list[PreferencePair] = []
    for key, rejected_samples in grouped_rejected.items():
        chosen_samples = grouped_chosen.get(key, [])
        k = int(len(chosen_samples) * cfg.percent_chosen)

        if not chosen_samples or not rejected_samples:
            continue

        pairs.extend(strategy.generate(chosen_pool=chosen_samples, rejected_pool=rejected_samples, k=k))

    if not pairs:
        logger.warning(
            "build_preference_pairs produced zero preference pairs (%d current samples, %d groups, "
            "min_score=%s, percent_chosen=%s, expanding=%s). DPO would train on nothing.",
            len(current_samples),
            len(grouped_rejected),
            cfg.min_score,
            cfg.percent_chosen,
            cfg.expanding,
        )

    return pairs


def pairs_to_dataframe(pairs: list[PreferencePair]) -> pd.DataFrame:
    """Preference pairs as a DataFrame with a stable column order, empty frames included."""
    if not pairs:
        return pd.DataFrame(columns=list(PAIR_COLUMNS))
    return pd.DataFrame([asdict(p) for p in pairs], columns=list(PAIR_COLUMNS))


class PreferenceDataset(TorchDataset):
    """Thin ``torch`` dataset over a list of :class:`PreferencePair`.

    Items are plain dicts, so the default collate yields ``list[str]`` for ``"chosen"`` and
    ``"rejected"`` -- what :class:`DiffusionDPOTrainer` consumes.
    """

    def __init__(self, pairs: list[PreferencePair]):
        self.pairs = list(pairs)
        if not self.pairs:
            logger.warning("PreferenceDataset initialized with zero preference pairs")

    @classmethod
    def from_dataframes(
        cls,
        current: pd.DataFrame,
        previous: list[pd.DataFrame] | None = None,
        cfg: PairingConfig | None = None,
        *,
        score_col: str = DEFAULT_SCORE_COL,
        strategy: PairingStrategy | None = None,
    ) -> "PreferenceDataset":
        pairs = build_preference_pairs(
            current,
            previous or [],
            cfg or PairingConfig(),
            score_col=score_col,
            strategy=strategy,
        )
        return cls(pairs)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "PreferenceDataset":
        df = pd.read_parquet(path)
        missing = [c for c in PAIR_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Preference parquet {path} is missing column(s) {missing}; has {list(df.columns)}")
        pairs = [
            PreferencePair(
                behavior_idx=int(row.behavior_idx),
                prompt=str(row.prompt),
                chosen=str(row.chosen),
                rejected=str(row.rejected),
                chosen_score=float(row.chosen_score),
                rejected_score=float(row.rejected_score),
                chosen_cycle=int(row.chosen_cycle),
                rejected_cycle=int(row.rejected_cycle),
            )
            for row in df.itertuples(index=False)
        ]
        return cls(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return asdict(self.pairs[idx])

    def to_dataframe(self) -> pd.DataFrame:
        return pairs_to_dataframe(self.pairs)

    def to_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
