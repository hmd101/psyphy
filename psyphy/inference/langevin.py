"""
inference/langevin.py
---------------------

Langevin samplers for posterior inference.

Implements:
- Overdamped (unadjusted) Langevin Algorithm (ULA)
- Underdamped Langevin (with BAOAB splitting scheme?)

Used for posterior-aware trial placement (InfoGain).


"""

from psyphy.posterior.posterior import Posterior


class LangevinSampler:
    def __init__(self, steps: int = 1000, step_size: float = 1e-3, temperature: float = 1.0):
        self.steps = steps
        self.step_size = step_size
        self.temperature = temperature

    def fit(self, model, data):
        # MVP: just return MAP-style Posterior
        # Full version: implement Langevin dynamics
        return Posterior(params=model.init_params(None), model=model)
