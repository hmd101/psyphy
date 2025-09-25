from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .base import TrialPlacement

"""
Sobol sequence placement.

Implements low-discrepancy sampling:
- Quasi-random exploration of stimulus space
- Typically used for initialization batches (as in Hong et al. 2025)
"""

@dataclass
class SobolPlacement(TrialPlacement):
    dim: int

    def next_batch(self, posterior: Any, n: int = 1) -> List[Any]:
        _ = posterior
        # Return n zero vectors as placeholders
        return [[0.0] * self.dim for _ in range(n)]
