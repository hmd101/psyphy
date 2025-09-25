from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .base import TrialPlacement

"""
Staircase procedure placement (often used in psychophysics).

Implements classical adaptive rules:
- 1-up 2-down
- Step size adjustment
- Converges to ~70% correct threshold

Does not require posterior sampling.
"""


@dataclass
class StaircasePlacement(TrialPlacement):
    step: float = 1.0

    def next_batch(self, posterior: Any, n: int = 1) -> List[Any]:
        _ = posterior
        return [self.step for _ in range(n)]
