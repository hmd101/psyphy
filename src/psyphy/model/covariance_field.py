"""
covariance_field.py
-------------------

Covariance field abstraction for spatially-varying Σ(x).

A covariance field represents a function from stimulus locations to covariance
matrices, encapsulating the spatially-varying perceptual uncertainty in WPPM.

Design
------
This module provides:
- CovarianceField: Protocol defining the interface
- WPPMCovarianceField: Concrete implementation for WPPM models
- construction from priors, posteriors, or arbitrary parameters

Mathematical Background
-----------------------
In the Wishart Process Psychophysical Model:
    Σ(x) = U(x) @ U(x)^T + λI

where:
    U(x) = Σ_ij W_ij * φ_ij(x)  (basis expansion)
    φ_ij(x) are Chebyshev basis functions
    W_ij are learned coefficients
    λ is a numerical stabilizer (diag_term)

In MVP mode, Σ(x) is constant: Σ(x) = diag(exp(log_diag)).

Usage Examples
--------------
# Sample from prior
>>> model = WPPM(input_dim=2, prior=Prior(basis_degree=5), ...)
>>> field = WPPMCovarianceField.from_prior(model, key)
>>> Sigma = field.cov(x)

# From fitted posterior
>>> posterior = model.fit(data, optimizer=MAPOptimizer())
>>> field = posterior.get_covariance_field()
>>> Sigmas = field.cov_batch(X_grid)

# Access square root
>>> U = field.sqrt_cov(x)  # Only in Wishart mode
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    import jax.random as jr


@runtime_checkable
class CovarianceField(Protocol):
    """
    Protocol for spatially-varying covariance fields Σ(x).

    A covariance field maps stimulus locations x ∈ R^d to
    covariance matrices Σ(x) ∈ R^{dxd}.

    Methods
    -------
    __call__(x)
        Evaluate field at one or more locations. Supports both single points
        and arbitrary batch dimensions.
    cov(x)
        Evaluate Σ(x) at stimulus location x (deprecated, use __call__).
    sqrt_cov(x)
        Evaluate U(x) such that Σ(x) = U(x) @ U(x)^T + λI.
    cov_batch(X)
        Vectorized evaluation at multiple locations (deprecated, use __call__).

    Notes
    -----
    This protocol enables polymorphic use of covariance fields from different
    sources (prior samples, fitted posteriors, custom parameterizations).

    The field is callable for mathematical elegance and JAX compatibility:
        Sigma = field(x)  # Single point or batch
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate covariance field at one or more stimulus locations."""
        ...

    def cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate covariance matrix Σ(x) at stimulus location x (deprecated)."""
        ...

    def sqrt_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate "square root" matrix U(x) such that Σ(x) = U(x) @ U(x)^T + λI."""
        ...

    def cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate covariance at multiple locations (deprecated)."""
        ...

    def cov_stimulus(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract stimulus-relevant covariance block."""
        ...


class WPPMCovarianceField:
    """
    Covariance field for WPPM with Wishart process or MVP mode.

    Encapsulates model + parameters to provide clean evaluation interface
    for Σ(x) and U(x).

    Parameters
    ----------
    model : WPPM
        Model providing evaluation logic (local_covariance, _compute_U)
    params : dict
        Model parameters:
        - MVP: {"log_diag": (input_dim,)}
        - Wishart: {"W": (degree+1, degree+1, embedding_dim, embedding_dim+extra_dims)}

    Attributes
    ----------
    model : WPPM
        Associated model instance
    params : dict
        Parameter dictionary

    Examples
    --------
    >>> # From prior
    >>> model = WPPM(input_dim=2, prior=Prior(basis_degree=5), ...)
    >>> field = WPPMCovarianceField.from_prior(model, key)
    >>> Sigma = field.cov(jnp.array([0.5, 0.3]))
    >>>
    >>> # From posterior
    >>> posterior = model.fit(data, optimizer=MAPOptimizer())
    >>> field = posterior.get_covariance_field()
    >>> Sigmas = field.cov_batch(X_grid)
    >>>
    >>> # Access square root (Wishart mode only)
    >>> U = field.sqrt_cov(jnp.array([0.5, 0.3]))

    Notes
    -----
    Implements the CovarianceField protocol for polymorphic use.
    """

    def __init__(self, model, params: dict):
        """
        Construct covariance field from model and parameters.

        Parameters
        ----------
        model : WPPM
            Model providing evaluation logic
        params : dict
            Parameter dictionary
        """
        self.model = model
        self.params = params
        # Pre-compile JIT batch evaluation path for performance
        self._eval_batch_jitted = jax.jit(self._eval_batch_impl)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance field at one or more stimulus locations.

        Automatically dispatches to single-point or batch evaluation based on input shape.
        Last axis must have size input_dim. Any number of batch dimensions supported.

        Parameters
        ----------
        x : jnp.ndarray
            - Single point: shape (input_dim,)
            - Batch: shape (..., input_dim) where ... are arbitrary batch dimensions

        Returns
        -------
        jnp.ndarray
            - Single point: shape (input_dim, input_dim)
            - Batch: shape (..., input_dim, input_dim)

        Raises
        ------
        ValueError
            If last axis size != input_dim

        Examples
        --------
        >>> field = WPPMCovarianceField.from_prior(model, key)
        >>>
        >>> # Single point
        >>> x = jnp.array([0.5, 0.3])  # Shape (2,) for input_dim=2
        >>> Sigma = field(x)  # Shape (2, 2)
        >>>
        >>> # 1D batch
        >>> X = jnp.array([[0.1, 0.2], [0.5, 0.5]])  # Shape (2, 2)
        >>> Sigmas = field(X)  # Shape (2, 2, 2)
        >>>
        >>> # 2D grid
        >>> X_grid = jnp.ones((10, 10, 2))  # Shape (10, 10, 2)
        >>> Sigmas_grid = field(X_grid)  # Shape (10, 10, 2, 2)
        >>>
        >>> # 3D+ batches work automatically
        >>> X_3d = jnp.ones((5, 10, 10, 2))
        >>> Sigmas_3d = field(X_3d)  # Shape (5, 10, 10, 2, 2)

        Notes
        -----
        Dispatch logic:
        - x.ndim == 1: Single point -> call _eval_single()
        - x.ndim >= 2: Batch -> flatten, vmap, reshape

        For input_dim=1, single points must have shape (1,) not (). Use x[None]
        if needed to convert scalar to shape (1,).
        """
        # Validate input dimension
        if x.shape[-1] != self.model.input_dim:
            raise ValueError(
                f"Last axis must be input_dim={self.model.input_dim}, got {x.shape[-1]}"
            )

        # Dispatch based on shape
        if x.ndim == 1:
            # Single point: shape (input_dim,) -> (input_dim, input_dim)
            return self._eval_single(x)
        else:
            # Batch: shape (..., input_dim) -> (..., input_dim, input_dim)
            return self._eval_batch_jitted(x)

    def _eval_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance at a single stimulus location.

        This is the base evaluation method that all other methods build upon.
        Called directly for single points, vmapped for batches.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Single stimulus location

        Returns
        -------
        jnp.ndarray, shape (input_dim, input_dim)
            Covariance matrix Σ(x)
        """
        return self.model.local_covariance(self.params, x)

    def _eval_batch_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Batch evaluation implementation (vmapped single evaluation).

        Strategy: Flatten -> vmap -> reshape
        This handles arbitrary batch dimensions with a single vmap.

        Parameters
        ----------
        X : jnp.ndarray, shape (..., input_dim)
            Batch of stimulus locations with arbitrary batch dimensions

        Returns
        -------
        jnp.ndarray, shape (..., input_dim, input_dim)
            Batch of covariance matrices
        """
        input_dim = self.model.input_dim
        batch_shape = X.shape[:-1]  # Store original batch structure

        # Flatten all batch dimensions: (..., input_dim) -> (n_total, input_dim)
        X_flat = X.reshape(-1, input_dim)

        # Vectorized evaluation: (n_total, input_dim) -> (n_total, input_dim, input_dim)
        Sigmas_flat = jax.vmap(self._eval_single)(X_flat)

        # Restore batch structure: (n_total, input_dim, input_dim) -> (..., input_dim, input_dim)
        return Sigmas_flat.reshape(batch_shape + (input_dim, input_dim))

    def cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance matrix Σ(x) at stimulus location x.

        .. deprecated::
            Use `field(x)` instead for unified single/batch API.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location in [0, 1]^d

        Returns
        -------
        jnp.ndarray, shape (input_dim, input_dim)
            Covariance matrix Σ(x) in stimulus space

        Notes
        -----
        With the rectangular U design, this always returns stimulus-space
        covariance (input_dim, input_dim), regardless of extra_dims.
        """
        warnings.warn(
            "cov() is deprecated. Use field(x) instead for unified single/batch API.",
            DeprecationWarning,
            stacklevel=2,
        )
        if x.ndim != 1:
            raise ValueError(
                f"cov() only accepts single points with shape (input_dim,), got {x.shape}. "
                f"Use field(X) for batches."
            )
        return self._eval_single(x)

    def cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate covariance at multiple locations (vectorized).

        .. deprecated::
            Use `field(X)` instead for unified single/batch API.

        Parameters
        ----------
        X : jnp.ndarray, shape (n_points, input_dim)
            Multiple stimulus locations

        Returns
        -------
        jnp.ndarray, shape (n_points, dim, dim)
            Covariance matrices at each location
        """
        warnings.warn(
            "cov_batch() is deprecated. Use field(X) instead for unified single/batch API.",
            DeprecationWarning,
            stacklevel=2,
        )
        if X.ndim < 2:
            raise ValueError(
                f"cov_batch() expects batch with shape (n_points, input_dim), got {X.shape}. "
                f"Use field(x) for single points."
            )
        return self._eval_batch_jitted(X)

    @classmethod
    def from_prior(cls, model, key: jr.KeyArray) -> WPPMCovarianceField:
        """
        Sample a covariance field from the prior.

        Parameters
        ----------
        model : WPPM
            Model defining prior distribution
        key : jax.random.KeyArray
            PRNG key for sampling

        Returns
        -------
        WPPMCovarianceField
            Field sampled from p(Σ(x))

        Examples
        --------
        >>> model = WPPM(input_dim=2, prior=Prior(basis_degree=5), ...)
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(42))
        >>> Sigma = field.cov(jnp.array([0.5, 0.5]))
        """
        params = model.init_params(key)
        return cls(model, params)

    @classmethod
    def from_posterior(cls, posterior) -> WPPMCovarianceField:
        """
        Create covariance field from fitted posterior.

        Parameters
        ----------
        posterior : ParameterPosterior
            Fitted posterior (e.g., from model.fit())

        Returns
        -------
        WPPMCovarianceField
            Field representing posterior estimate of Σ(x)

        Notes
        -----
        For MAP posteriors, uses θ_MAP.
        For variational posteriors, could use posterior mean or sample.

        Examples
        --------
        >>> posterior = model.fit(data, optimizer=MAPOptimizer())
        >>> field = WPPMCovarianceField.from_posterior(posterior)
        >>> Sigma = field.cov(x)
        """
        return cls(posterior._model, posterior.params)

    @classmethod
    def from_params(cls, model, params: dict) -> WPPMCovarianceField:
        """
        Create field from arbitrary parameters.

        Useful for:
        - Custom initialization
        - Posterior samples
        - Intermediate optimization checkpoints
        - Testing

        Parameters
        ----------
        model : WPPM
            Model providing evaluation logic
        params : dict
            Parameter dictionary

        Returns
        -------
        WPPMCovarianceField

        Examples
        --------
        >>> params = {"log_diag": jnp.array([0.1, 0.2])}
        >>> field = WPPMCovarianceField.from_params(model, params)
        """
        return cls(model, params)

    def sqrt_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate U(x) such that Σ(x) = U(x) @ U(x)^T + diag_term*I.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        jnp.ndarray, shape (input_dim, embedding_dim)
            Rectangular square root matrix U(x).
            embedding_dim = input_dim + extra_dims

        Raises
        ------
        ValueError
            If in MVP mode (basis_degree=None).

        Notes
        -----
        Only available in Wishart mode. MVP mode uses diagonal parameterization
        without explicit U matrices.

        In the rectangular design (Hong et al.), U is (input_dim, embedding_dim).

        Examples
        --------
        >>> # Wishart mode
        >>> model = WPPM(input_dim=2, basis_degree=5, extra_dims=1, ...)
        >>> field = WPPMCovarianceField.from_prior(model, key)
        >>> U = field.sqrt_cov(jnp.array([0.5, 0.3]))
        >>> print(U.shape)  # (2, 3) for input_dim=2, extra_dims=1
        >>>
        >>> # Verify: Σ = U @ U^T + λI
        >>> Sigma_from_U = U @ U.T + model.diag_term * jnp.eye(2)
        >>> Sigma_direct = field.cov(x)
        >>> assert jnp.allclose(Sigma_from_U, Sigma_direct)
        """
        if "W" not in self.params:
            raise ValueError(
                "sqrt_cov only available in Wishart mode. "
                "Set basis_degree when creating WPPM to use Wishart process."
            )
        return self.model._compute_U(self.params, x)

    def cov_stimulus(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Extract stimulus-relevant covariance block.

        With rectangular U design, this is just an alias for cov(x).

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        jnp.ndarray, shape (input_dim, input_dim)
            Covariance in observable stimulus subspace

        Examples
        --------
        >>> model = WPPM(input_dim=2, basis_degree=3, extra_dims=1, ...)
        >>> field = WPPMCovarianceField.from_prior(model, key)
        >>> x = jnp.array([0.5, 0.5])
        >>>
        >>> # Both return the same thing now
        >>> Sigma = field.cov(x)  # (2, 2)
        >>> Sigma_stim = field.cov_stimulus(x)  # (2, 2)
        >>>
        >>> assert jnp.allclose(Sigma, Sigma_stim)

        Notes
        -----
        Kept for backward compatibility. With rectangular U design,
        cov(x) already returns stimulus covariance, so no extraction needed.
        """
        # With rectangular U, cov already returns (input_dim, input_dim)
        warnings.warn(
            "cov_stimulus() is deprecated. Use field(x) or field.cov(x) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._eval_single(x)

    def sqrt_cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized evaluation of U(x) at multiple locations.

        Parameters
        ----------
        X : jnp.ndarray, shape (n_points, input_dim)
            Multiple stimulus locations

        Returns
        -------
        jnp.ndarray, shape (n_points, input_dim, embedding_dim)
            Rectangular square root matrices at each location.
            embedding_dim = input_dim + extra_dims

        Raises
        ------
        ValueError
            If in MVP mode.

        Notes
        -----
        In the rectangular design (Hong et al.), U is (input_dim, embedding_dim).

        Examples
        --------
        >>> X_grid = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        >>> U_batch = field.sqrt_cov_batch(X_grid)
        >>> print(U_batch.shape)  # (3, 2, 3) for input_dim=2, extra_dims=1
        """
        if "W" not in self.params:
            raise ValueError("sqrt_cov_batch only available in Wishart mode")
        return jax.vmap(self.sqrt_cov)(X)

    @property
    def is_wishart_mode(self) -> bool:
        """
        Check if using Wishart process (spatially-varying covariance).

        Returns
        -------
        bool
            True if params contain "W" (Wishart mode), False otherwise (MVP).

        Examples
        --------
        >>> field_wishart = WPPMCovarianceField.from_prior(wishart_model, key)
        >>> assert field_wishart.is_wishart_mode
        >>>
        >>> field_mvp = WPPMCovarianceField.from_prior(mvp_model, key)
        >>> assert not field_mvp.is_wishart_mode
        """
        return "W" in self.params

    @property
    def is_mvp_mode(self) -> bool:
        """
        Check if using MVP diagonal mode (constant covariance).

        Returns
        -------
        bool
            True if params contain "log_diag" (MVP), False otherwise (Wishart).

        Examples
        --------
        >>> field_mvp = WPPMCovarianceField.from_prior(mvp_model, key)
        >>> assert field_mvp.is_mvp_mode
        >>>
        >>> field_wishart = WPPMCovarianceField.from_prior(wishart_model, key)
        >>> assert not field_wishart.is_mvp_mode
        """
        return "log_diag" in self.params
