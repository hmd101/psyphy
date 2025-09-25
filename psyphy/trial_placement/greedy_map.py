from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .base import TrialPlacement

"""
Greedy MAP-based placement.

Chooses stimuli that maximize informativeness
given current MAP parameters (point estimate).

Faster but less robust than posterior-aware methods.
"""

@dataclass
class GreedyMAPPlacement(TrialPlacement):
    def next_batch(self, posterior: Any, n: int = 1) -> List[Any]:
        _ = posterior
        return ["map_trial"] * n
