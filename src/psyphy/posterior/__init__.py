"""
posterior
=========

Posterior representations.

This subpackage provides:
- ParameterPosterior: protocol for posteriors over model parameters p(θ | data)
- MAPPosterior: delta distribution at θ_MAP (point estimate)

Two-tier design
---------------
- ParameterPosterior: represents p(θ | data)
- PredictivePosterior: represents p(f(X*) | data)

Future extensions
-----------------
- LaplacePosterior: Gaussian approximation N(θ_MAP, Σ)
- NumpyroPosterior/BlackjaxPosterior: MCMC samples
"""

from .parameter_posterior import ParameterPosterior
from .posterior import MAPPosterior
from .predictive_posterior import PredictivePosterior, WPPMPredictivePosterior

__all__ = [
    # Core protocols
    "ParameterPosterior",
    "PredictivePosterior",
    # Parameter posterior implementations
    "MAPPosterior",
    # Predictive posterior implementations
    "WPPMPredictivePosterior",
]
