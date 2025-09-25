from __future__ import annotations

import random

"""
Random number utilities.

Provides:
- JAX/NumPy random seed handling
- Reproducible PRNG splitting for experiments
"""

def set_seed(seed: int) -> None:
    random.seed(seed)
