"""
base.py
-------

Base class for psychophysical models.

Design
------
Models in PsyPhy are **stateless configuration objects**.
They define:
1. The parameter space (init_params)
2. The probabilistic rules (log_likelihood, forward)

They do NOT hold data or fitted parameters.
Inference is handled by external engines (e.g. MAPOptimizer) which
return a Posterior object containing the results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from psyphy.data import ResponseData


class Model(ABC):
    """
    Abstract base class for psychophysical models.

    Subclasses must implement:
    - init_params(key) --> sample initial parameters
    - log_likelihood_from_data(params, data) --> compute likelihood
    - _forward(X, comparisons, params) --> compute predictions
    """

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def init_params(self, key: Any) -> dict:  # jax.random.KeyArray
        """
        Sample initial parameters from prior.

        Parameters
        ----------
        key : jax.random.KeyArray
            PRNG key

        Returns
        -------
        dict
            Parameter PyTree
        """
        ...

    @abstractmethod
    def log_likelihood_from_data(self, params: dict, data: ResponseData) -> jnp.ndarray:
        """
        Compute log p(data | params).

        Parameters
        ----------
        params : dict
            Model parameters
        data : ResponseData
            Observed trials

        Returns
        -------
        jnp.ndarray
            Log-likelihood (scalar)
        """
        ...

    @abstractmethod
    def _forward(
        self,
        X: jnp.ndarray,
        comparisons: jnp.ndarray | None,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Model-specific forward pass with given parameters.

        Subclasses must implement this to support predict_with_params().

        Parameters
        ----------
        X : jnp.ndarray, shape (n_test, input_dim)
            Test stimuli
        comparisons : jnp.ndarray | None, shape (n_test, input_dim)
            Probe stimuli (None for detection tasks)
        params : dict[str, jnp.ndarray]
            Model parameters

        Returns
        -------
        jnp.ndarray, shape (n_test,)
            Predicted response probabilities
        """
        pass

    # ------------------------------------------------------------------
    # Functional Helpers
    # ------------------------------------------------------------------

    def predict_with_params(
        self,
        X: jnp.ndarray,
        comparisons: jnp.ndarray | None,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Evaluate model at specific parameter values (no marginalization).

        This is useful for:
        - Threshold uncertainty estimation (evaluate at sampled parameters)
        - Parameter sensitivity analysis
        - Debugging and diagnostics

        Parameters
        ----------
        X : jnp.ndarray, shape (n_test, input_dim)
            Test stimuli (references)
        comparisons : jnp.ndarray, shape (n_test, input_dim), optional
            Probe stimuli (for discrimination tasks)
        params : dict[str, jnp.ndarray]
            Specific parameter values to evaluate at.

        Returns
        -------
        predictions : jnp.ndarray, shape (n_test,)
            Predicted probabilities at each test point
        """
        return self._forward(X, comparisons, params)
