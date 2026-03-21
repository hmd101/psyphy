"""
utils
=====

Shared utility functions and helpers for psyphy.

This subpackage provides:
- math : mathematical utilities
    (currently: basis functions, which may get their own module).


"""

from .math import chebyshev_basis

__all__ = [
    # math
    "chebyshev_basis",
]
