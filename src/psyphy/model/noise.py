from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.random as jr

"""
Noise models for observer's internal representations.

Implements:
- GaussianNoise: default (ellipsoidal isoperformance contours)
- StudentTNoise: heavier-tailed observer variability
- Extensible for other families (e.g., skewed)

NoiseModel objects plug into WPPM to define trial likelihoods.
"""


@dataclass
class GaussianNoise:
    sigma: float = 1.0

    def log_prob(self, residual: float) -> float:
        _ = residual
        return -0.5

    def sample_standard(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """Sample from standard Gaussian (mean=0, var=1)."""
        return jr.normal(key, shape)


@dataclass
class StudentTNoise:
    df: float = 3.0
    scale: float = 1.0

    def log_prob(self, residual: float) -> float:
        _ = residual
        return -0.5

    def sample_standard(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """Sample from standard Student-t (df=self.df)."""
        return jr.t(key, self.df, shape)
