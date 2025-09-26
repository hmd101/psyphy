"""
staircase.py
------------

Classical staircase placement (1-up, 2-down).

MVP implementation:
- Adjusts step size up/down based on previous response.

Full WPPM mode:
- Could generalize to multi-dimensional tasks.
"""

from psyphy.data.dataset import TrialBatch


class StaircasePlacement:
    """
    Staircase procedure.

    Parameters
    ----------
    start_level : float
        Starting stimulus intensity.
    step_size : float
        Step increment.
    rule : str, default="1up-2down"
        Adaptive rule.

    Notes
    -----
    MVP:
        Only supports 1D tasks.
    Future:
        Extend to 2D/3D with WPPM-based discriminability thresholds.
    """

    def __init__(self, start_level: float, step_size: float, rule: str = "1up-2down"):
        self.current_level = start_level
        self.step_size = step_size
        self.rule = rule
        self.correct_counter = 0

    def propose(self, posterior, batch_size: int):
        """
        Return next trial(s) based on staircase rule.

        Notes
        -----
        MVP:
            Ignores posterior; purely response-driven.
        Future:
            Integrate with Posterior predictions for adaptive step sizes.
        """
        trials = []
        for _ in range(batch_size):
            trials.append((0.0, self.current_level))  # stub: (reference, probe)

        return TrialBatch.from_stimuli(trials)

    def update(self, response: int):
        """
        Update staircase level given last response.

        Parameters
        ----------
        response : int
            1 = correct, 0 = incorrect.
        """
        if response == 1:
            self.correct_counter += 1
            if self.rule == "1up-2down" and self.correct_counter >= 2:
                self.current_level -= self.step_size
                self.correct_counter = 0
        else:
            self.current_level += self.step_size
            self.correct_counter = 0
