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
- PosteriorMeanCovarianceField: E[Σ(x)] over posterior samples -- the correct
  "posterior mean" field for sample-based posteriors (see its docstring for
  why averaging W instead gives too-small ellipses)
- covariance_to_arviz: per-sample Σ(x) at chosen points as an ArviZ DataTree,
  for convergence diagnostics that are not fooled by the non-identifiable
  orientation of W (see its docstring)
- contour_radii / contour_band / ContourBand: the Mahalanobis-1 contour of a
  covariance as a radius per direction, and pointwise credible bands over it.
  For the oddity task that radius is the discrimination threshold

Mathematical Background
-----------------------
In the Wishart Process Psychophysical Model:
    Σ(x) = U(x) @ U(x)^T + λI

where:
    U(x) = Σ_ij W_ij * φ_ij(x)  (basis expansion)
    φ_ij(x) are Chebyshev basis functions
    W_ij are learned coefficients
    λ is a numerical stabilizer (diag_term)


Usage Examples
--------------
>>> import jax.numpy as jnp
>>> import jax.random as jr
>>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
>>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
>>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
>>> field(jnp.array([0.5, 0.3])).shape
(2, 2)
>>> field(jnp.zeros((10, 2))).shape
(10, 2, 2)
>>> field.sqrt_cov(jnp.array([0.5, 0.3])).shape  # (input_dim, embedding_dim)
(2, 3)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

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


class WPPMCovarianceField:
    """
    Covariance field for WPPM with Wishart process
    Encapsulates model + parameters to provide clean evaluation interface
    for Σ(x) and U(x).

    Parameters
    ----------
    model : WPPM
        Model providing evaluation logic (local_covariance, _compute_sqrt)
    params : dict
        Model parameters:
        - MVP: {"log_diag": (input_dim,)}
        - Wishart: {"W": (degree+1, degree+1, input_dim, embedding_dim)}
          where embedding_dim = input_dim + extra_embedding_dims

          Note: The 3rd dimension is input_dim (output/stimulus space), not
          embedding_dim. This matches the einsum in _compute_sqrt where U(x) has
          shape (input_dim, embedding_dim).

    Attributes
    ----------
    model : WPPM
        Associated model instance
    params : dict
        Parameter dictionary

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
    >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
    >>> # from the prior
    >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
    >>> field(jnp.array([0.5, 0.3])).shape
    (2, 2)
    >>> # from a MAP fit
    >>> from psyphy.inference import MAPOptimizer
    >>> map_posterior = MAPOptimizer().fit(model, data)  # doctest: +SKIP
    >>> field = WPPMCovarianceField.from_posterior(map_posterior)  # doctest: +SKIP

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
            Square covariance matrix/matrices. Shape adapts to stimulus dimensionality:
            - Single point: shape (d, d) where d = model.input_dim
            - Batch: shape (..., d, d) where d = model.input_dim

            For input_dim=2: returns (2, 2) or (..., 2, 2)
            For input_dim=3: returns (3, 3) or (..., 3, 3)

        Raises
        ------
        ValueError
            If last axis size != input_dim

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> # 2D stimulus space (e.g., isoluminant plane)
        >>> model_2d = WPPM(
        ...     Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2
        ... )
        >>> field_2d = WPPMCovarianceField.from_prior(model_2d, jr.PRNGKey(0))
        >>>
        >>> # Single 2D point
        >>> x = jnp.array([0.5, 0.3])  # Shape (2,)
        >>> Sigma = field_2d(x)  # Shape (2, 2)
        >>>
        >>> # Batch of 2D points
        >>> X = jnp.array([[0.1, 0.2], [0.5, 0.5]])  # Shape (2, 2)
        >>> Sigmas = field_2d(X)  # Shape (2, 2, 2)
        >>>
        >>> # 2D spatial grid in 2D stimulus space
        >>> X_grid = jnp.ones((10, 10, 2))  # Shape (10, 10, 2)
        >>> Sigmas_grid = field_2d(X_grid)  # Shape (10, 10, 2, 2)
        >>>
        >>> # 3D stimulus space (e.g., RGB)
        >>> model_3d = WPPM(
        ...     Prior(input_dim=3, basis_degree=2), OddityTask(), input_dim=3
        ... )
        >>> field_3d = WPPMCovarianceField.from_prior(model_3d, jr.PRNGKey(0))
        >>>
        >>> # Single 3D point
        >>> x_3d = jnp.array([0.5, 0.3, 0.2])  # Shape (3,)
        >>> Sigma_3d = field_3d(x_3d)  # Shape (3, 3)
        >>>
        >>> # Batch of 3D points
        >>> X_3d = jnp.ones((5, 10, 3))  # Shape (5, 10, 3)
        >>> Sigmas_3d = field_3d(X_3d)  # Shape (5, 10, 3, 3)

        Notes
        -----
        For input_dim=1, single points must have shape (1,) not (). Use x[None]
        if needed to convert scalar to shape (1,).
        """
        # Validate input dimension
        if x.shape[-1] != self.model.input_dim:
            raise ValueError(
                f"Last axis must be input_dim={self.model.input_dim}, got {x.shape[-1]}"
            )

        # Dispatch logic:
        #   - x.ndim == 1: Single point -> call _eval_single()
        #   - x.ndim >= 2: Batch -> flatten, vmap, reshape

        # Dispatch based on shape
        if x.ndim == 1:
            # Single point: shape (d,) -> (d, d) where d = model.input_dim
            # E.g., input_dim=2: (2,) -> (2,2), input_dim=3: (3,) -> (3,3)
            return self._eval_single(x)
        else:
            # Batch: shape (..., d) -> (..., d, d) where d = model.input_dim
            # Returns square covariance matrices for each point in the batch
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
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(42))
        >>> field(jnp.array([0.5, 0.5])).shape
        (2, 2)
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
        Builds the field from ``posterior.params``, i.e. a single parameter set.

        - MAP posteriors: ``params`` is θ_MAP, so this is Σ(θ_MAP).
        - Sample-based posteriors (e.g. ``MCMCPosterior``): ``params`` is the
          mean parameter E[W], and Σ(E[W]) is NOT the posterior mean of Σ. It
          is systematically too small and can collapse toward λI (see
          ``PosteriorMeanCovarianceField`` for why). A ``UserWarning`` is
          raised in that case. Use ``PosteriorMeanCovarianceField(posterior)``
          for E[Σ], or ``posterior.sample(n)`` for individual samples.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
        >>> from psyphy.inference import MAPOptimizer
        >>> map_posterior = MAPOptimizer().fit(model, data)  # doctest: +SKIP
        >>> field = WPPMCovarianceField.from_posterior(map_posterior)  # doctest: +SKIP
        >>> field(jnp.array([0.5, 0.5])).shape  # doctest: +SKIP
        (2, 2)
        """
        # Sample-based posteriors are recognized by duck typing (they expose a
        # sample count), the same criterion PosteriorMeanCovarianceField uses.
        # isinstance(posterior, MCMCPosterior) would need an import from
        # psyphy.posterior, which already imports psyphy.model -> cycle.
        #
        # This warns instead of switching to the posterior-mean field because
        # that would change the return type (and drop sqrt_cov) for existing
        # callers.
        if getattr(posterior, "n_draws", None) is not None:
            warnings.warn(
                f"from_posterior() on a {type(posterior).__name__} builds Σ from the "
                "mean parameter E[W]. Σ is nonlinear in W, so Σ(E[W]) is not the "
                "posterior mean of Σ and comes out systematically too small. Use "
                "PosteriorMeanCovarianceField(posterior) for E[Σ].",
                UserWarning,
                stacklevel=2,
            )
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
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
        >>> params = model.init_params(jr.PRNGKey(0))  # {"W": (3, 3, 2, 3)}
        >>> field = WPPMCovarianceField.from_params(model, params)
        >>> field(jnp.array([0.5, 0.5])).shape
        (2, 2)
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


        Notes
        -----
        Only available in Wishart mode. MVP mode uses diagonal parameterization
        without explicit U matrices.

        In the rectangular design (Hong et al.), U is (input_dim, embedding_dim).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
        >>> x = jnp.array([0.5, 0.3])
        >>> U = field.sqrt_cov(x)
        >>> U.shape  # (input_dim, embedding_dim) for input_dim=2, extra_dims=1
        (2, 3)
        >>> # Σ = U @ U^T + λI
        >>> bool(jnp.allclose(U @ U.T + model.diag_term * jnp.eye(2), field(x)))
        True
        """
        if "W" not in self.params:
            raise ValueError(
                "sqrt_cov only available in Wishart mode. "
                "Set basis_degree when creating WPPM to use Wishart process."
            )
        return self.model._compute_sqrt(self.params, x)

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
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
        >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
        >>> field = WPPMCovarianceField.from_prior(model, jr.PRNGKey(0))
        >>> X_grid = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])  # (3, 2)
        >>> field.sqrt_cov_batch(X_grid).shape  # (n_points, input_dim, embedding_dim)
        (3, 2, 3)
        """
        if "W" not in self.params:
            raise ValueError("sqrt_cov_batch only available in Wishart mode")
        return jax.vmap(self.sqrt_cov)(X)


class PosteriorMeanCovarianceField:
    r"""
    Posterior mean of the covariance field, E[Σ(x) | data].

    Evaluates Σ(x) separately for every posterior sample W_k and averages the
    resulting matrices:

        Σ̂(x) = (1/N) Σ_k [ U_k(x) U_k(x)^T + λI ]

    Use this, not ``WPPMCovarianceField(model, posterior.params)``, to draw a
    "posterior mean" covariance field from a sample-based posterior such as
    ``MCMCPosterior``.

    Why average Σ and not W
    -----------------------
    ``posterior.params`` is the mean parameter E[W]. A field built from it,
    Σ(E[W]), comes out too small:

        E[Σ(x)] = Σ(x; E[W]) + Cov[U(x)]

    This is the definition of covariance rearranged, using Σ = U Uᵀ + λI with
    U linear in W. Cov[U] is a covariance matrix, so it is positive
    semi-definite. Averaging W first therefore gives ellipses that are too
    small.

    Two things follow:

    1. The gap is there for any posterior with spread. No symmetry is needed.
    2. W is not identifiable. For orthogonal R on the embedding axis,
       W -> W R leaves U Uᵀ unchanged, and so leaves Σ and the prior
       unchanged. Samples at different orientations, in practice different
       chains cancel when W is averaged. At the extreme: W and -W describe
       the same Σ and average to E[W] = 0, collapsing Σ(E[W]) to λI.

    Averaging Σ avoids both. Every Σ(x; W_k) is unchanged by W_k -> W_k R, and
    nothing is squared after averaging. E[Σ(x)] is also the Bayes estimate of
    Σ(x) under squared (Frobenius) loss, which is what "posterior mean of Σ"
    means.

    Parameters
    ----------
    posterior : ParameterPosterior
        Any posterior with ``.model`` and ``.sample(n, key=...)``, e.g.
        ``MCMCPosterior`` or ``MAPPosterior``.
    n_samples : int | None, default=None
        Number of samples to average over. If None, uses every stored sample,
        which requires the posterior to expose ``n_chains`` and ``n_draws``
        (``MCMCPosterior`` does). Posteriors without a fixed sample count
        (``MAPPosterior``, future variational posteriors) must pass it.
    key : jax.random.KeyArray | None, default=None
        PRNG key forwarded to ``posterior.sample``. Only matters when
        ``n_samples`` is smaller than the number of stored samples.

    Notes
    -----
    - Implements the ``CovarianceField`` protocol, so it can be passed
      wherever a field is expected (e.g. ``psyphy.viz.plot_ellipses`` after
      evaluating it). ``sqrt_cov`` raises: see its docstring.
    - Deliberately not a subclass of ``WPPMCovarianceField``: that class is
      "one parameter set W", and its ``params``, ``from_params`` and
      ``sqrt_cov`` have no meaning for an average over many W.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
    >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
    >>> from psyphy.posterior import MCMCPosterior
    >>> # stand-in for NUTSSampler().fit(model, data): 2 chains of 50 prior samples
    >>> keys = jr.split(jr.PRNGKey(1), 100)
    >>> W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (100, 3, 3, 2, 3)
    >>> nuts_posterior = MCMCPosterior({"W": W.reshape(2, 50, *W.shape[1:])}, model)
    >>> ref_points = jnp.array([[-0.5, -0.5], [0.5, 0.5]])  # (n_points, 2)
    >>> from psyphy.model import PosteriorMeanCovarianceField
    >>> mean_field = PosteriorMeanCovarianceField(nuts_posterior)
    >>> mean_field(ref_points).shape  # (n_points, 2, 2)
    (2, 2, 2)
    """

    def __init__(self, posterior, n_samples: int | None = None, key=None):
        self.model = posterior.model

        if n_samples is None:
            # "All samples" is only well defined for posteriors that store a
            # fixed set of samples. MCMCPosterior.sample(n) samples WITHOUT
            # replacement when n <= n_chains * n_draws, so asking for exactly
            # that many returns every stored sample once (in shuffled order,
            # which does not affect a mean). This stays on the public
            # ParameterPosterior interface instead of reading _samples.
            n_chains = getattr(posterior, "n_chains", None)
            n_draws = getattr(posterior, "n_draws", None)
            if n_chains is None or n_draws is None:
                raise ValueError(
                    f"{type(posterior).__name__} does not expose n_chains/n_draws; "
                    "pass n_samples explicitly."
                )
            n_samples = int(n_chains) * int(n_draws)

        # Drawn ONCE here, so repeated evaluations of the field are
        # deterministic and consistent with each other.
        # self._draws: PyTree with leading axis n_samples, e.g. {"W": (N, *W_shape)}
        self._draws = posterior.sample(int(n_samples), key=key)
        self.n_samples = int(n_samples)

        self._eval_batch_jitted = jax.jit(self._eval_batch_impl)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate E[Σ(x)] at one or more stimulus locations.

        Same shape contract as ``WPPMCovarianceField.__call__``: ``(d,)`` ->
        ``(d, d)`` and ``(..., d)`` -> ``(..., d, d)``.
        """
        if x.shape[-1] != self.model.input_dim:
            raise ValueError(
                f"Last axis must be input_dim={self.model.input_dim}, got {x.shape[-1]}"
            )
        if x.ndim == 1:
            return self._eval_single(x)
        return self._eval_batch_jitted(x)

    def _eval_single(self, x: jnp.ndarray) -> jnp.ndarray:
        """E[Σ(x)] at one location: Σ for every sample, THEN the mean.

        The order is the whole point of this class -- see the class
        docstring. Do not "optimize" this into Σ(x; mean of samples).
        """
        # vmap over the leading sample axis of every leaf of self._draws:
        # (N, d, d) -- one covariance matrix per posterior sample
        per_draw = jax.vmap(lambda params: self.model.local_covariance(params, x))(
            self._draws
        )
        return jnp.mean(per_draw, axis=0)  # (d, d)

    def _eval_batch_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """Flatten -> vmap -> reshape, as in ``WPPMCovarianceField``.

        Memory note: the intermediate is (n_points, n_draws, d, d), e.g.
        25 x 4000 x 2 x 2 floats = 1.6 MB; dense grids with many samples scale
        as their product.
        """
        input_dim = self.model.input_dim
        batch_shape = X.shape[:-1]
        X_flat = X.reshape(-1, input_dim)
        Sigmas_flat = jax.vmap(self._eval_single)(X_flat)
        return Sigmas_flat.reshape(batch_shape + (input_dim, input_dim))

    def cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate E[Σ(x)] at a single location.

        .. deprecated::
            Use `field(x)` instead, as for ``WPPMCovarianceField``.
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
        return self(x)

    def cov_batch(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate E[Σ(x)] at multiple locations.

        .. deprecated::
            Use `field(X)` instead, as for ``WPPMCovarianceField``.
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
        return self(X)

    def sqrt_cov(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Not available: a posterior mean of Σ has no WPPM square root U(x).

        Each sample has its own U_k(x), but the average of U_k U_k^T is not
        U U^T for any single U from the model (and averaging the U_k instead
        reintroduces exactly the cancellation this class exists to avoid).
        Any matrix returned here would be a different object under the same
        name, so this raises instead. For a factor of the mean matrix itself,
        use ``jnp.linalg.cholesky(field(x))``.
        """
        raise NotImplementedError(
            "PosteriorMeanCovarianceField has no square root U(x): an average "
            "of U U^T over samples is not U U^T for any single U. Use "
            "jnp.linalg.cholesky(field(x)) for a factor of the mean matrix."
        )


def covariance_to_arviz(posterior, points):
    r"""
    Per-sample covariance Σ(x) at chosen points, as an ArviZ DataTree.

    Use this, not ``posterior.to_arviz()``, for convergence diagnostics
    (R-hat, ESS, trace plots) of a WPPM fit.

    Why Σ and not W
    ---------------
    W is only identified up to an orthogonal rotation R of its embedding
    axis: W -> W R gives U(x) -> U(x) R and leaves Σ(x) = U Uᵀ + λI, and the
    prior, unchanged. Chains can therefore settle at different orientations
    that describe exactly the same covariance field. R-hat on W reports that
    as non-convergence, although no ellipse, threshold or prediction depends
    on the orientation. Σ(x) is invariant, so its diagnostics measure what the
    fit is actually used for.

    Why points must be chosen
    -------------------------
    Unlike W, Σ is a function of the stimulus location x, so a diagnostic of Σ
    is a diagnostic *at some x*. This function deliberately has no default for
    ``points``: which locations to check is a modeling decision. Sensible
    choices are the locations you report (e.g. the reference points of the
    plotted ellipses), optionally plus a grid covering the stimulus space,
    since away from the data the posterior relies more on the prior. Report
    the worst value over all points. A modest number of points suffices: Σ is
    a low-degree polynomial in x, so dense grids add mostly near-duplicates.

    Quantities
    ----------
    For each sample and point, the d x d matrix Σ is summarized by its d log
    variances and d(d-1)/2 correlations, which together determine Σ:

    - ``log_var_i``  = log Σ_ii
    - ``corr_i_j``   = Σ_ij / sqrt(Σ_ii Σ_jj), for i < j

    (2D: ``log_var_0``, ``log_var_1``, ``corr_0_1``.) The off-diagonal is not
    counted twice. Logs make the variances less skewed; note that ArviZ's
    default rank-normalized R-hat is unchanged by such monotone transforms
    anyway.

    Parameters
    ----------
    posterior : MCMCPosterior
        A sample-based posterior with ``to_arviz()`` (chain structure is
        needed for R-hat; ``sample()`` would pool the chains).
    points : array_like, shape (n_points, input_dim)
        Stimulus locations at which to evaluate Σ.

    Returns
    -------
    xarray.DataTree
        ``posterior`` group with variable ``Sigma`` of dims
        ``(chain, draw, point, quantity)``, coordinates ``quantity`` (names as
        above) and ``location_0 ... location_{d-1}`` (the evaluation points, one
        coordinate per stimulus axis, along ``point``). The posterior's
        ``sample_stats`` group (divergences, energy, ...) is carried over.

    Raises
    ------
    TypeError
        If the posterior has no ``to_arviz()`` (e.g. a MAP posterior: no
        chains, nothing to diagnose).
    ValueError
        If ``points`` is not of shape ``(n_points, input_dim)``.
    ImportError
        If ArviZ is not installed (``pip install 'psyphy[diagnostics]'``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
    >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
    >>> from psyphy.posterior import MCMCPosterior
    >>> # stand-in for NUTSSampler().fit(model, data): 2 chains of 50 prior samples
    >>> keys = jr.split(jr.PRNGKey(1), 100)
    >>> W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (100, 3, 3, 2, 3)
    >>> nuts_posterior = MCMCPosterior({"W": W.reshape(2, 50, *W.shape[1:])}, model)
    >>> ref_points = jnp.array([[-0.5, -0.5], [0.5, 0.5]])  # (n_points, 2)
    >>> import arviz as az
    >>> from psyphy.model import covariance_to_arviz
    >>> dt = covariance_to_arviz(nuts_posterior, ref_points)
    >>> worst = az.rhat(dt)["Sigma"].max()  # worst point and quantity
    >>> az.plot_trace(dt, var_names=["Sigma"], coords={"point": 0})  # doctest: +SKIP
    """
    to_arviz = getattr(posterior, "to_arviz", None)
    if to_arviz is None:
        raise TypeError(
            f"{type(posterior).__name__} has no to_arviz(); convergence diagnostics "
            "need a sample-based posterior with chains (e.g. MCMCPosterior)."
        )

    model = posterior.model
    d = model.input_dim
    points = jnp.asarray(points)
    if points.ndim != 2:
        raise ValueError(
            f"points must have shape (n_points, input_dim), got {points.shape}. "
            "For a single location use points[None, :]."
        )
    if points.shape[-1] != d:
        raise ValueError(
            f"Last axis of points must be input_dim={d}, got {points.shape[-1]}"
        )

    # Chain-structured samples via the public API (raises ImportError without
    # ArviZ). Every posterior variable keeps its leading (chain, draw) axes,
    # e.g. {"W": (C, S, *W_shape)}.
    idata = to_arviz()
    import arviz as az
    import numpy as np

    params = {
        name: jnp.asarray(var.values)
        for name, var in idata["posterior"].data_vars.items()
    }

    def sigma_one_draw(p):
        # Σ at every point for ONE draw: (n_points, d, d)
        return jax.vmap(lambda x: model.local_covariance(p, x))(points)

    # vmap over samples, then chains: (C, S, n_points, d, d)
    Sig = jax.vmap(jax.vmap(sigma_one_draw))(params)

    # Identifiable summary of each Σ: log variances, then correlations (i < j).
    variances = jnp.diagonal(Sig, axis1=-2, axis2=-1)  # (C, S, P, d)
    columns = [jnp.log(variances[..., i]) for i in range(d)]
    names = [f"log_var_{i}" for i in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            columns.append(
                Sig[..., i, j] / jnp.sqrt(variances[..., i] * variances[..., j])
            )
            names.append(f"corr_{i}_{j}")
    feats = np.asarray(jnp.stack(columns, axis=-1))  # (C, S, P, n_quantities)

    data = {"posterior": {"Sigma": feats}}
    if "sample_stats" in idata.children:
        data["sample_stats"] = {
            k: v.values for k, v in idata["sample_stats"].data_vars.items()
        }
    dt = az.from_dict(
        data, dims={"Sigma": ["point", "quantity"]}, coords={"quantity": names}
    )
    # Record where Σ was evaluated, one 1-D coordinate per stimulus axis along
    # `point`. (A single (point, axis) coordinate would not attach to Sigma,
    # which has no `axis` dimension.)
    pts = np.asarray(points)
    dt["posterior"] = (
        dt["posterior"]
        .to_dataset()
        .assign_coords({f"location_{i}": ("point", pts[:, i]) for i in range(d)})
    )
    return dt


class ContourBand(NamedTuple):
    """A pointwise credible band on the contour radii, with its provenance.

    Attributes
    ----------
    points : jnp.ndarray, shape (n_points, d)
        Stimulus locations the band was evaluated at.
    directions : jnp.ndarray, shape (n_directions, d)
        Unit directions the radii are measured along, in order. Closed, so the
        last direction repeats the first and a filled polygon has no seam.
    lower, median, upper : jnp.ndarray
        Interval endpoints and the posterior median radius, shape
        (n_points, n_directions) for a single ``ci_prob``, or
        (n_levels, n_points, n_directions) when a sequence was requested. All
        three carry the same shape, so a loop over levels can index all of
        them. The median does not depend on the interval probability, so its
        levels are identical copies.
    ci_prob : float | tuple[float, ...]
        Nominal probability each interval carries. A tuple when several levels
        were requested, in the order the arrays are stacked.
    ci_kind : str
        ``"eti"`` for equal-tailed, ``"hdi"`` for highest-density.

    Notes
    -----
    The interval is **pointwise**: it covers the true radius in each direction
    separately with probability ``ci_prob``. It does NOT cover the whole
    contour jointly at that probability, which is a stronger (simultaneous)
    statement requiring a different construction. ``ci_prob`` and ``ci_kind``
    travel with the arrays so a figure can state which interval it draws.
    """

    points: jnp.ndarray
    directions: jnp.ndarray
    lower: jnp.ndarray
    median: jnp.ndarray
    upper: jnp.ndarray
    ci_prob: float | tuple[float, ...]
    ci_kind: str


def contour_radii(covs: jnp.ndarray, directions: jnp.ndarray) -> jnp.ndarray:
    r"""Distance from the center to the Mahalanobis-1 contour, per direction.

    For a covariance :math:`\Sigma`, the contour
    :math:`\{v : v^\top \Sigma^{-1} v = 1\}` meets the ray along a unit
    direction :math:`u` at distance

    .. math:: r(u) = 1 / \sqrt{u^\top \Sigma^{-1} u}

    Writing a point on the ray as :math:`v = r\,u` turns the contour condition
    into :math:`r^2 u^\top \Sigma^{-1} u = 1`, which is the expression above.
    Along an eigenvector this is :math:`\sqrt{\lambda}`, i.e. exactly the
    semi-axis :func:`psyphy.viz.ellipse_segments` draws at ``scale=1``. For the
    oddity task this radius is the discrimination threshold, the just
    noticeable difference, in direction ``u``.

    Parameters
    ----------
    covs : jnp.ndarray, shape (..., d, d)
        One or more covariance matrices. Leading axes are broadcast over.
    directions : jnp.ndarray, shape (n_directions, d)
        Unit directions. We do not normalize them, and a non-unit direction
        rescales the radius.

    Returns
    -------
    jnp.ndarray, shape (..., n_directions)

    Notes
    -----
    This is a *polar* parameterization of the same ellipse that
    ``ellipse_segments`` produces, and the two are not interchangeable. That
    function maps a circle through the Cholesky factor, so its parameter angle
    is not the direction from the center. Any statistic taken across samples at
    fixed angle, a quantile or a mean, must use this parameterization, or it
    compares points lying in different directions.
    """
    prec = jnp.linalg.inv(covs)
    quad = jnp.einsum("ki,...ij,kj->...k", directions, prec, directions)
    return 1.0 / jnp.sqrt(quad)


def _eti(radii: jnp.ndarray, prob: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Equal-tailed interval over axis 0: the (1-p)/2 and 1-(1-p)/2 quantiles."""
    tail = (1.0 - prob) / 2.0
    lo, hi = jnp.quantile(radii, jnp.array([tail, 1.0 - tail]), axis=0)
    return lo, hi


def _hdi(radii: jnp.ndarray, prob: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Highest-density interval over axis 0: the narrowest window holding p mass.

    Sorts the samples and takes the narrowest window of ``ceil(prob * n)`` of
    them. This is the standard sample HDI. ArviZ computes the same quantity
    with a slightly different window convention, so the two agree closely but
    not bit for bit.
    """
    n = radii.shape[0]
    width = max(int(jnp.ceil(prob * n)), 2)  # samples inside the interval
    ordered = jnp.sort(radii, axis=0)
    los = ordered[: n - width + 1]  # (n_windows, ...)
    his = ordered[width - 1 :]  # (n_windows, ...)
    best = jnp.argmin(his - los, axis=0)[None]  # narrowest window per element
    return (
        jnp.take_along_axis(los, best, axis=0)[0],
        jnp.take_along_axis(his, best, axis=0)[0],
    )


_CI_KINDS = {"eti": _eti, "hdi": _hdi}


def contour_band(
    posterior,
    points: jnp.ndarray,
    *,
    n_directions: int = 181,
    ci_prob: float | Sequence[float] = 0.89,
    ci_kind: str = "eti",
    n_samples: int | None = None,
    key=None,
) -> ContourBand:
    """Pointwise credible band on the Mahalanobis-1 contour at each point.

    The posterior-interval counterpart to
    :class:`PosteriorMeanCovarianceField` (a posterior *mean*) and
    :func:`covariance_to_arviz` (posterior *samples*, for diagnostics).

    Every sample gives one covariance field, hence one contour per point.
    Writing each contour as a radius per direction
    (:func:`contour_radii`) makes it a scalar per sample, so an interval
    follows from quantiles over samples. For a :math:`1-\alpha` interval, at
    each direction separately, the band runs between
    :math:`q_{\alpha/2}\\{r_i\\}` and :math:`q_{1-\alpha/2}\\{r_i\\}` over the
    :math:`N` samples.

    Parameters
    ----------
    posterior : ParameterPosterior
        A fitted posterior, typically an ``MCMCPosterior``. A ``MAPPosterior``
        is a point mass, so its band has zero width.
    points : jnp.ndarray, shape (n_points, 2)
        Stimulus locations, usually the reference points we report.
    n_directions : int, default=181
        Directions per contour. The grid is closed, so the last direction
        repeats the first, and 181 gives 180 distinct directions, one every two
        degrees. The number is a magic default with nothing behind it: large
        enough that a filled band looks smooth, small enough to stay cheap.
        Nothing in the result depends on the choice.
    ci_prob : float or sequence of float, default=0.89
        How much posterior mass the interval holds. At 0.89, 89% of the samples
        of the radius lie between ``lower`` and ``upper``, in each direction
        separately. 0.89 is ArviZ's default too, so a band and ``az.summary``
        agree. Pass a sequence for nested bands, and each level gets its own
        interval stacked on a leading axis.
    ci_kind : {"eti", "hdi"}, default="eti"
        Which interval of that mass to take, since many of them hold it.
        ``"eti"`` is equal-tailed: it cuts the same probability off each end,
        so ``lower`` and ``upper`` are quantiles. ``"hdi"`` is
        highest-density: it takes the narrowest interval holding the mass. The
        two differ when the samples are skewed. Radii usually are, because a
        radius cannot go below zero.
    n_samples : int, optional
        Samples to use. Defaults to every stored sample, which needs the posterior
        to expose ``n_chains`` and ``n_draws``. A point-mass posterior such as
        ``MAPPosterior`` has to pass this.
    key : jax.Array, optional
        PRNG key forwarded to ``posterior.sample``. Only matters when
        ``n_samples`` is smaller than the number of stored samples.

    Returns
    -------
    ContourBand

    Raises
    ------
    NotImplementedError
        If the model is not 2-D. The direction grid is a circle, and there is
        no canonical equivalent in higher dimensions.
    ValueError
        If ``ci_kind`` is unknown, if any ``ci_prob`` is outside (0, 1), or if
        ``n_samples`` is needed and cannot be inferred.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax.random as jr
    >>> from psyphy.model import WPPM, OddityTask, Prior, WPPMCovarianceField
    >>> model = WPPM(Prior(input_dim=2, basis_degree=2), OddityTask(), input_dim=2)
    >>> from psyphy.posterior import MCMCPosterior
    >>> # stand-in for NUTSSampler().fit(model, data): 2 chains of 50 prior samples
    >>> keys = jr.split(jr.PRNGKey(1), 100)
    >>> W = jnp.stack([model.init_params(k)["W"] for k in keys])  # (100, 3, 3, 2, 3)
    >>> nuts_posterior = MCMCPosterior({"W": W.reshape(2, 50, *W.shape[1:])}, model)
    >>> ref_points = jnp.array([[-0.5, -0.5], [0.5, 0.5]])  # (n_points, 2)
    >>> from psyphy.model import contour_band
    >>> band = contour_band(nuts_posterior, ref_points)
    >>> band.lower.shape  # (n_points, n_directions)
    (2, 181)
    >>> nested = contour_band(nuts_posterior, ref_points, ci_prob=[0.5, 0.8, 0.95])
    >>> nested.upper.shape  # (n_levels, n_points, n_directions)
    (3, 2, 181)
    """
    if ci_kind not in _CI_KINDS:
        raise ValueError(
            f"ci_kind must be one of {sorted(_CI_KINDS)}, got {ci_kind!r}."
        )

    if isinstance(ci_prob, (int, float)):
        scalar_level = True
        probs: tuple[float, ...] = (float(ci_prob),)
    else:
        scalar_level = False
        probs = tuple(float(p) for p in ci_prob)
    if not probs:
        raise ValueError("ci_prob must hold at least one probability.")
    for p in probs:
        if not 0.0 < p < 1.0:
            raise ValueError(f"ci_prob must lie in (0, 1), got {p}.")

    model = posterior.model
    if model.input_dim != 2:
        raise NotImplementedError(
            "contour_band supports 2-D stimulus spaces only, got "
            f"input_dim={model.input_dim}. contour_radii is d-dimensional, "
            "so call it with explicit directions instead."
        )

    if n_samples is None:
        n_chains = getattr(posterior, "n_chains", None)
        n_draws = getattr(posterior, "n_draws", None)
        if n_chains is None or n_draws is None:
            raise ValueError(
                f"{type(posterior).__name__} does not expose n_chains/n_draws, "
                "so n_samples cannot be inferred. Pass n_samples explicitly."
            )
        n_samples = int(n_chains) * int(n_draws)

    # Closed grid, so the last angle repeats the first. n_directions is a magic
    # default (181 -> one direction every two degrees); nothing depends on it.
    phi = jnp.linspace(0.0, 2 * jnp.pi, n_directions)  # radians
    directions = jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)  # (n_dir, 2)

    draws = posterior.sample(int(n_samples), key=key)

    def radii_one_draw(params):
        covs = WPPMCovarianceField(model, params)(points)  # (n_points, 2, 2)
        return contour_radii(covs, directions)  # (n_points, n_directions)

    # (n_samples, n_points, n_directions)
    radii = jax.vmap(radii_one_draw)({"W": draws["W"]})

    reduce = _CI_KINDS[ci_kind]
    bounds = [reduce(radii, p) for p in probs]
    lower = jnp.stack([lo for lo, _ in bounds])  # (n_levels, n_pts, n_dir)
    upper = jnp.stack([hi for _, hi in bounds])
    # One median whatever ci_prob holds, repeated per level so that lower,
    # median and upper always share a shape. Indexing median[i] in a loop over
    # levels would otherwise reach a reference point instead.
    median = jnp.median(radii, axis=0)  # (n_points, n_directions)
    median = jnp.broadcast_to(median, lower.shape)  # (n_levels, n_pts, n_dir)

    return ContourBand(
        points=points,
        directions=directions,
        lower=lower[0] if scalar_level else lower,
        median=median[0] if scalar_level else median,
        upper=upper[0] if scalar_level else upper,
        ci_prob=probs[0] if scalar_level else probs,
        ci_kind=ci_kind,
    )
