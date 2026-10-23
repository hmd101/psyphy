"""
trial_placement
===============

Non-acquisition-based trial placement strategies.

NOTE: this modeule is currently untested and still in active development.

This module provides classical (non-Bayesian-optimization) placement strategies:

- GridPlacement: Fixed grid designs for systematic exploration
- SobolPlacement: Quasi-random low-discrepancy sequences (space-filling)

For acquisition-based adaptive designs (Bayesian optimization), use:
- psyphy.acquisition: Expected Improvement, UCB, Mutual Information
- See: psyphy.acquisition.optimize_acqf_discrete() for trial selection

Examples
--------
>>> # Fixed grid design: walks through the grid in order
>>> from psyphy.trial_placement import GridPlacement
>>> grid = [[x0, x1] for x0 in (-1.0, 0.0, 1.0) for x1 in (-1.0, 0.0, 1.0)]  # 9 points
>>> placement = GridPlacement(grid_points=grid)
>>> trials = placement.propose(posterior=None, batch_size=5)

>>> # Quasi-random exploration
>>> from psyphy.trial_placement import SobolPlacement
>>> placement = SobolPlacement(dim=2, bounds=[[-1, 1], [-1, 1]])
>>> trials = placement.propose(posterior=None, batch_size=8)
"""

from psyphy.trial_placement.grid import GridPlacement
from psyphy.trial_placement.sobol import SobolPlacement

__all__ = [
    "GridPlacement",
    "SobolPlacement",
]
