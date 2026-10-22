"""
psyphy.model
============

Model-layer API: everything model-related in one place.

Includes
--------
- WPPM (core model)
- Priors (Prior)
- Tasks (TaskLikelihood base, OddityTask)
- Noise models (GaussianNoise, StudentTNoise)

All functions/classes use JAX arrays (jax.numpy as jnp) for autodiff
and optimization with Optax.

Typical usage
-------------
    from psyphy.model import WPPM, Prior, OddityTask, GaussianNoise
"""

from .base import Model
from .covariance_field import (
    CovarianceField,
    PosteriorMeanCovarianceField,
    ThresholdBand,
    WPPMCovarianceField,
    covariance_to_arviz,
    threshold_band,
    threshold_radii,
)
from .likelihood import OddityTask, OddityTaskConfig, TaskLikelihood
from .noise import GaussianNoise, StudentTNoise
from .prior import Prior
from .wppm import WPPM

__all__ = [
    # Base
    "Model",
    # Covariance fields
    "CovarianceField",
    "WPPMCovarianceField",
    "PosteriorMeanCovarianceField",
    "covariance_to_arviz",
    "ThresholdBand",
    "threshold_radii",
    "threshold_band",
    # Models
    "WPPM",
    # Priors
    "Prior",
    # Tasks
    "TaskLikelihood",
    "OddityTask",
    "OddityTaskConfig",
    # Noise models
    "GaussianNoise",
    "StudentTNoise",
]
