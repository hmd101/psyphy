"""
inference
=========

Inference engines for WPPM.

This subpackage provides different strategies for fitting model parameters
to data and returning posterior objects.

Implementations
---------------
- MAPOptimizer : maximum a posteriori fit with Optax optimizers.
- NUTSSampler : full posterior sampling via NUTS (requires blackjax).
- LaplaceApproximation : approximate posterior covariance around MAP (stub).
- LangevinSampler : skeleton for Langevin-based sampling (stub).

Optional dependencies
---------------------
- NUTSSampler requires blackjax: pip install 'psyphy[sampling]'
"""

from .base import InferenceEngine
from .langevin import LangevinSampler
from .laplace import LaplaceApproximation
from .map_optimizer import MAPOptimizer

# NUTSSampler soft-imports blackjax at call time, but the class itself is
# always importable — missing blackjax only raises at .fit() time.
from .nuts import NUTSSampler

# Registry for string-based inference selection
INFERENCE_ENGINES = {
    "map": MAPOptimizer,
    "nuts": NUTSSampler,
    "laplace": LaplaceApproximation,
    "langevin": LangevinSampler,
}

__all__ = [
    "InferenceEngine",
    "MAPOptimizer",
    "NUTSSampler",
    "LangevinSampler",
    "LaplaceApproximation",
    "INFERENCE_ENGINES",
]
