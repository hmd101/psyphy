"""
inference/base.py
-----------------

Abstract base class for inference engines.

Defines interface:
- fit(model, data) -> Posterior

All inference engines (MAPOptimizer, LangevinSampler, LaplaceApproximation)
subclass from this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceEngine(ABC):
    """Abstract interface for fitting models to data."""

    @abstractmethod
    def fit(self, model: Any, data: Any) -> Any:
        """
        Fit model parameters to data.

        Returns
        -------
        Posterior
            A posterior object that wraps MAP params or samples.
        """
        ...
