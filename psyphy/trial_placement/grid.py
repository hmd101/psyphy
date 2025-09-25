from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .base import TrialPlacement

"""
Grid-based trial placement.

Non-adaptive baseline:
- Traverse a fixed grid of stimuli
- Useful for validation and debugging
"""


@dataclass
class GridPlacement(TrialPlacement):
    grid: List[Any]

    def next_batch(self, posterior: Any, n: int = 1) -> List[Any]:
        _ = posterior
        return self.grid[:n]
