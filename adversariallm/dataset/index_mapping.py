from collections.abc import Sequence
from typing import Any

import torch


def parse_config_idx(config_idx: int | list[int] | str | None) -> int | list[int] | None:
    """Parse and normalize `config.idx` into int/list[int]/None."""
    if isinstance(config_idx, str):
        if config_idx.startswith("list(range("):
            try:
                config_idx = eval(config_idx, {"__builtins__": None}, {"range": range, "list": list})
            except Exception as e:
                raise ValueError(f"Could not parse idx string: {config_idx}\n{e}")
        else:
            raise ValueError(f"Could not parse idx string: {config_idx}\nDoes not start with 'list(range('.")

    if isinstance(config_idx, Sequence) and not isinstance(config_idx, (str, bytes)):
        return [int(i) for i in config_idx]
    if isinstance(config_idx, int) or config_idx is None:
        return config_idx
    raise ValueError(f"Invalid idx: {config_idx}")


def shuffled_order(dataset_len: int, seed: int = 0, shuffle: bool = True) -> torch.Tensor:
    """Return original dataset indices in the deterministic (possibly shuffled) order."""
    if shuffle:
        torch.manual_seed(seed)
        return torch.randperm(dataset_len)
    return torch.arange(dataset_len)


def selected_original_indices(
    dataset_len: int,
    seed: int = 0,
    shuffle: bool = True,
    config_idx: int | list[int] | str | None = None,
) -> tuple[torch.Tensor, int | list[int] | None]:
    """Return original indices selected by dataset config and normalized config_idx."""
    order = shuffled_order(dataset_len=dataset_len, seed=seed, shuffle=shuffle)
    parsed_idx = parse_config_idx(config_idx)

    if isinstance(parsed_idx, int):
        order = order[parsed_idx : parsed_idx + 1]
    elif isinstance(parsed_idx, list):
        order = order[parsed_idx]

    return order, parsed_idx


def map_internal_to_original(
    internal_idx: int | Sequence[int],
    dataset_len: int,
    seed: int = 0,
    shuffle: bool = True,
    config_idx: int | list[int] | str | None = None,
) -> int | list[int]:
    """Map internal (post-shuffle/subset) indices to original dataset indices."""
    selected, _ = selected_original_indices(
        dataset_len=dataset_len,
        seed=seed,
        shuffle=shuffle,
        config_idx=config_idx,
    )

    def _single(i: int) -> int:
        if i < 0 or i >= len(selected):
            raise IndexError(f"Internal index {i} out of range for selected dataset of length {len(selected)}")
        return int(selected[i])

    if isinstance(internal_idx, int):
        return _single(internal_idx)
    return [_single(int(i)) for i in internal_idx]


def map_original_to_internal(
    original_idx: int | Sequence[int],
    dataset_len: int,
    seed: int = 0,
    shuffle: bool = True,
    config_idx: int | list[int] | str | None = None,
) -> int | list[int] | None:
    """Map original dataset index/indices to internal (post-shuffle/subset) indices."""
    selected, _ = selected_original_indices(
        dataset_len=dataset_len,
        seed=seed,
        shuffle=shuffle,
        config_idx=config_idx,
    )
    position_by_original = {int(original): pos for pos, original in enumerate(selected.tolist())}

    def _single(i: int) -> int | None:
        return position_by_original.get(i)

    if isinstance(original_idx, int):
        return _single(original_idx)
    return [_single(int(i)) for i in original_idx]


def map_config_internal_to_original(
    config: Any,
    internal_idx: int | Sequence[int],
    dataset_len: int,
) -> int | list[int]:
    """Map internal indices to original indices using dataset config object."""
    return map_internal_to_original(
        internal_idx=internal_idx,
        dataset_len=dataset_len,
        seed=getattr(config, "seed", 0),
        shuffle=getattr(config, "shuffle", True),
        config_idx=getattr(config, "idx", None),
    )


def map_config_original_to_internal(
    config: Any,
    original_idx: int | Sequence[int],
    dataset_len: int,
) -> int | list[int] | None:
    """Map original indices to internal indices using dataset config object."""
    return map_original_to_internal(
        original_idx=original_idx,
        dataset_len=dataset_len,
        seed=getattr(config, "seed", 0),
        shuffle=getattr(config, "shuffle", True),
        config_idx=getattr(config, "idx", None),
    )
