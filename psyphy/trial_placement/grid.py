"""
grid.py
-------

Grid-based placement strategy.

MVP implementation:
- Returns a fixed list of grid points, sliced into batches.

Full WPPM mode:
- Could adaptively refine grid around posterior uncertainty.
"""

from psyphy.data.dataset import TrialBatch


class GridPlacement:
    """
    Fixed grid placement.

    Parameters
    ----------
    grid_points : list of (ref, probe)
        Predefined set of trial stimuli.

    Notes
    -----
    MVP:
        Just iterates through the provided grid points.
    Future:
        Implement adaptive grid refinement.
    """

    def __init__(self, grid_points):
        self.grid_points = list(grid_points)
        self._index = 0

    def propose(self, posterior, batch_size: int):
        """
        Return the next batch of trials from the grid.

        Parameters
        ----------
        posterior : Posterior
            Ignored in MVP (grid is non-adaptive).
        batch_size : int
            Number of trials to return.

        Returns
        -------
        TrialBatch
            Fixed batch of (ref, probe).
        """
        start, end = self._index, self._index + batch_size
        batch = self.grid_points[start:end]
        self._index = end
        return TrialBatch.from_stimuli(batch)
