"""
posterior/posterior.py
----------------------

Posterior wrapper around model parameters and predictions.


- MAP_params: return point estimate
- sample(n): draw parameter samples (from sampler or Laplace approx)
- predict_prob(ref, probe): predict trial performance via Monte Carlo
- predict_thresholds(ref, criterion): compute threshold contours/ellipses


This is the central object passed into trial placement strategies.

"""

from __future__ import annotations

import jax.numpy as jnp


class Posterior:
    def __init__(self, params, model):
        self.params = params
        self.model = model

    def MAP_params(self):
        return self.params

    def sample(self, n: int = 1):
        # MVP: return repeated MAP params
        return [self.params] * n

    def predict_prob(self, stimulus):
        return self.model.predict_prob(self.params, stimulus)

    def predict_thresholds(self, reference, criterion: float = 0.667, directions: int = 16):
        # MVP stub: return unit circle ellipse
        angles = jnp.linspace(0, 2 * jnp.pi, directions, endpoint=False)
        return jnp.stack([reference + jnp.array([jnp.cos(a), jnp.sin(a)]) for a in angles])
