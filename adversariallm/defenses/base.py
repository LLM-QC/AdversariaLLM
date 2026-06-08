from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DefenseDecision:
    output_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
