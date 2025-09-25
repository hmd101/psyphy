"""
inference/laplace.py
--------------------

Laplace approximation to the posterior.

Approximates posterior with a Gaussian:
    N(mean = MAP, covariance = H^-1 at MAP)

Provides posterior.sample() cheaply.
Useful for InfoGainPlacement when only MAP fit is available.
"""

from psyphy.posterior.posterior import Posterior


class LaplaceApproximation:
    def from_map(self, map_posterior: Posterior):
        # MVP: just return the MAP posterior unchanged
        # Full version: compute covariance from Hessian
        return map_posterior
