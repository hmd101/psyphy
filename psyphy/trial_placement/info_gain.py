"""
info_gain.py
------------

Information-gain placement (e.g., expected Absolute Volume Change, entropy).

MVP implementation:
- Placeholder entropy scoring.

Full WPPM mode:
- Requires posterior.sample() (Laplace/MCMC).
- Compute expected reduction in posterior uncertainty per candidate.
"""

from psyphy.data.dataset import TrialBatch


class InfoGainPlacement:
    """
    Information-gain adaptive placement.

    Parameters
    ----------
    candidate_pool : list of (ref, probe)
        Candidate stimuli.

    Notes
    -----
    MVP:
        Uses placeholder entropy-based scores.
    Full WPPM mode:
        Requires posterior.sample() to evaluate uncertainty reduction.
    """

    def __init__(self, candidate_pool):
        self.pool = candidate_pool

    def propose(self, posterior, batch_size: int):
        """
        Propose trials that maximize information gain.

        Notes
        -----
        MVP:
            Placeholder: returns first N candidates.
        Full WPPM mode:
            For each candidate:
              - Draw posterior samples.
              - Compute predictive entropy (or EAVC).
              - Select top scoring candidates.
        """
        # TODO: replace with acquisition scoring.
        return TrialBatch.from_stimuli(self.pool[:batch_size])
