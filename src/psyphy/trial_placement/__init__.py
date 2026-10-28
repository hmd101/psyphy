"""
trial_placement
===============

Trial placement that does not depend on the posterior.

This module is untested and still in development.

- GridPlacement: proposes the points of a fixed grid, in order.
- SobolPlacement: proposes quasi-random points that fill the stimulus space evenly.

Both take a ``posterior`` argument in ``propose`` and ignore it. psyphy does
not yet place trials adaptively, from the posterior.

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
