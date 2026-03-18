"""
utils
=====

Shared utility functions and helpers for psyphy.

This subpackage provides:
- math : mathematical utilities (basis functions, distances, kernels).
- rng : random number handling for reproducibility.


"""

from .diagnostics import (
    estimate_threshold_contour_uncertainty,
    estimate_threshold_uncertainty,
    parameter_summary,
    print_parameter_summary,
)
from .math import chebyshev_basis

__all__ = [
    # candidates
    "grid_candidates",
    "sobol_candidates",
    "custom_candidates",
    # math
    "chebyshev_basis",
]
