"""
sobol.py
--------

Sobol quasi-random placement.

MVP implementation:
- Uses a Sobol engine to generate low-discrepancy points.
- Independent of posterior.

Full WPPM mode:
- Could combine Sobol exploration with InfoGain exploitation.
"""

import numpy as np
from scipy.stats.qmc import Sobol

from psyphy.data.dataset import TrialBatch


class SobolPlacement:
    """
    Sobol quasi-random placement.

    Parameters
    ----------
    dim : int
        Dimensionality of stimulus space.
    bounds : list of (low, high)
        Bounds per dimension.
    seed : int, optional
        RNG seed.
    """

    def __init__(self, dim: int, bounds, seed: int = 0):
        self.engine = Sobol(d=dim, scramble=True, seed=seed)
        self.bounds = bounds

    def propose(self, posterior, batch_size: int):
        """
        Propose Sobol points (ignores posterior).

        Notes
        -----
        MVP:
            Pure exploration.
        Future:
            Switch after burn-in to InfoGainPlacement for exploitation.
        """
        raw = self.engine.random(batch_size)
        scaled = [low + (high - low) * raw[:, i] for i, (low, high) in enumerate(self.bounds)]
        stimuli = list(zip(*scaled))
        return TrialBatch.from_stimuli(stimuli)
