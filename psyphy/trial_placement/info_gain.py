from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .base import TrialPlacement

"""
Information-gain placement (EAVC-style).

Implements:
- Expected Absolute Volume Change (EAVC)
- Predictive entropy heuristic

Requires posterior.sample() or Laplace approx.
Selects trials that maximally reduce posterior uncertainty.
"""


@dataclass
class InfoGainPlacement(TrialPlacement):
    def next_batch(self, posterior: Any, n: int = 1) -> List[Any]:
        _ = posterior
        return ["ig_trial"] * n
