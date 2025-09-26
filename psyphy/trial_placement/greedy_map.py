"""
greedy_map.py
-------------

Greedy placement using MAP estimate.

MVP implementation:
- Selects trials that maximize discriminability under MAP params.

Full WPPM mode:
- Could consider posterior variance (Laplace/MCMC).

Faster but less robust than posterior-aware methods.

"""

from psyphy.data.dataset import TrialBatch


class GreedyMAPPlacement:
    """
    Greedy adaptive placement (MAP-based).

    Parameters
    ----------
    candidate_pool : list of (ref, probe)
        Candidate stimuli.

    Notes
    -----
    MVP:
        Uses only MAP parameters.
    Future:
        Replace with posterior-aware scoring (variance reduction).
    """

    def __init__(self, candidate_pool):
        self.pool = candidate_pool

    def propose(self, posterior, batch_size: int):
        """
        Select top trials based on MAP discriminability.

        Notes
        -----
        MVP:
            Calls posterior.MAP_params() and scores candidates deterministically.
        Future:
            Use posterior.sample() for uncertainty-aware acquisition.
        """
        # TODO: implement scoring. For MVP, just return first N.
        return TrialBatch.from_stimuli(self.pool[:batch_size])
