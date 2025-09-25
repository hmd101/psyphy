from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

"""
Abstract base class for trial placement strategies.

Defines:
- propose(posterior, batch_size) -> TrialBatch

All adaptive designs (grid, staircase, info gain) subclass this.
"""


class TrialPlacement(ABC):
    @abstractmethod
    def next_batch(self, posterior: Any, n: int = 1) -> Any:  # pragma: no cover - interface
        ...
