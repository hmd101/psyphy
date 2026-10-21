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
- PosteriorMeanCovarianceField: E[Σ(x)] over posterior draws -- the correct
  "posterior mean" field for sample-based posteriors (see its docstring for
  why averaging W instead gives too-small ellipses)
- covariance_to_arviz: per-draw Σ(x) at chosen points as an ArviZ DataTree,
  for convergence diagnostics that are not fooled by the non-identifiable
  orientation of W (see its docstring)

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
# Sample from prior
>>> model = WPPM(input_dim=2, prior=Prior(basis_degree=5), ...)
>>> field = WPPMCovarianceField.from_prior(model, key)
>>> Sigma = field.cov(x)

# From fitted posterior
>>> posterior = model.fit(data, optimizer=MAPOptimizer())
>>> field = posterior.get_covariance_field()
>>> Sigmas = field.cov_batch(X_grid)

# Access square root
>>> U = field.sqrt_cov(x)
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
        >>> # 2D stimulus space (e.g., isoluminant plane)
        >>> model_2d = WPPM(input_dim=2, ...)
        >>> field_2d = WPPMCovarianceField.from_prior(model_2d, key)
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
        >>> model_3d = WPPM(input_dim=3, ...)
        >>> field_3d = WPPMCovarianceField.from_prior(model_3d, key)
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
        Builds the field from ``posterior.params``, i.e. a single parameter set.

        - MAP posteriors: ``params`` is θ_MAP, so this is Σ(θ_MAP). Correct.
        - Sample-based posteriors (e.g. ``MCMCPosterior``): ``params`` is the
          mean parameter E[W], and Σ(E[W]) is NOT the posterior mean of Σ. It
          is systematically too small and can collapse toward λI (see
          ``PosteriorMeanCovarianceField`` for why). A ``UserWarning`` is
          raised in that case. Use ``PosteriorMeanCovarianceField(posterior)``
          for E[Σ], or ``posterior.sample(n)`` for individual draws.

        Examples
        --------
        >>> posterior = model.fit(data, optimizer=MAPOptimizer())
        >>> field = WPPMCovarianceField.from_posterior(posterior)
        >>> Sigma = field.cov(x)
        """
        # Sample-based posteriors are recognized by duck typing (they expose a
        # draw count), the same criterion PosteriorMeanCovarianceField uses.
        # isinstance(posterior, MCMCPosterior) would need an import from
        # psyphy.posterior, which already imports psyphy.model -> cycle.
        #
        # This warns instead of switching to the posterior-mean field because
        # that would change the return type (and drop sqrt_cov) for existing
        # callers; the behavior is unchanged, only made loud.
        if getattr(posterior, "n_draws", None) is not None:
            warnings.warn(
                f"from_posterior() on a {type(posterior).__name__} builds Σ from the "
                "mean parameter E[W]. For the WPPM, Σ(E[W]) is not the posterior "
                "mean of Σ: it is systematically too small, because Σ is quadratic "
                "in W and W is only identified up to an embedding rotation. Use "
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
        >>> X_grid = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        >>> U_batch = field.sqrt_cov_batch(X_grid)
        >>> print(U_batch.shape)  # (3, 2, 3) for input_dim=2, extra_dims=1
        """
        if "W" not in self.params:
            raise ValueError("sqrt_cov_batch only available in Wishart mode")
        return jax.vmap(self.sqrt_cov)(X)


class PosteriorMeanCovarianceField:
    r"""
    Posterior mean of the covariance field, E[Σ(x) | data].

    Evaluates Σ(x) separately for every posterior draw W_k and averages the
    resulting matrices:

        Σ̂(x) = (1/N) Σ_k [ U_k(x) U_k(x)^T + λI ]

    Use this, not ``WPPMCovarianceField(model, posterior.params)``, to draw a
    "posterior mean" covariance field from a sample-based posterior such as
    ``MCMCPosterior``.

    Why average Σ and not W
    -----------------------
    ``posterior.params`` is the mean *parameter* E[W]. Building a field from
    it gives Σ(E[W]), which is the wrong summary for the WPPM, for two reasons
    that compound:

    1. Σ is quadratic in W (through U U^T, with U linear in W), so averaging
       before squaring drops the spread term:

           E[Σ(x)] = Σ(x; E[W]) + Cov[U(x)],   with Cov[U(x)] ⪰ 0.

       Σ(E[W]) is therefore never larger than E[Σ], in any direction.

    2. W is not identifiable. For any orthogonal R acting on the embedding
       axis, W -> W R gives U -> U R and leaves U U^T, and hence Σ, unchanged;
       the prior is invariant too. Different draws (in practice, different
       chains) can sit at different orientations of the same Σ, and averaging
       W makes them cancel, pulling E[W] toward 0 and Σ(E[W]) toward λI. In
       the extreme, the draws W and -W average to E[W] = 0.

    Averaging Σ itself avoids both: every term Σ(x; W_k) is unchanged by
    W_k -> W_k R, and nothing is squared after averaging. E[Σ(x)] is also the
    Bayes estimate of Σ(x) under squared (Frobenius) loss, i.e. what
    "posterior mean of Σ" means.

    Parameters
    ----------
    posterior : ParameterPosterior
        Any posterior with ``.model`` and ``.sample(n, key=...)``, e.g.
        ``MCMCPosterior`` or ``MAPPosterior``.
    n_samples : int | None, default=None
        Number of draws to average over. If None, uses every stored draw,
        which requires the posterior to expose ``n_chains`` and ``n_draws``
        (``MCMCPosterior`` does). Posteriors without a fixed draw count
        (``MAPPosterior``, future variational posteriors) must pass it.
    key : jax.random.KeyArray | None, default=None
        PRNG key forwarded to ``posterior.sample``. Only matters when
        ``n_samples`` is smaller than the number of stored draws.

    Notes
    -----
    - Implements the ``CovarianceField`` protocol, so it can be passed
      wherever a field is expected (e.g. ``psyphy.viz.plot_ellipses`` after
      evaluating it). ``sqrt_cov`` raises: see its docstring.
    - The arithmetic mean of covariance matrices is somewhat larger than a
      typical single draw, because spread in ellipse size adds to the mean.
      That is a property of the mean, not an error.
    - Deliberately not a subclass of ``WPPMCovarianceField``: that class is
      "one parameter set W", and its ``params``, ``from_params`` and
      ``sqrt_cov`` have no meaning for an average over many W.

    Examples
    --------
    >>> nuts_posterior = NUTSSampler(...).fit(model, data)
    >>> mean_field = PosteriorMeanCovarianceField(nuts_posterior)
    >>> Sigmas = mean_field(ref_points)  # (n_points, 2, 2)
    """

    def __init__(self, posterior, n_samples: int | None = None, key=None):
        self.model = posterior.model

        if n_samples is None:
            # "All draws" is only well defined for posteriors that store a
            # fixed set of draws. MCMCPosterior.sample(n) samples WITHOUT
            # replacement when n <= n_chains * n_draws, so asking for exactly
            # that many returns every stored draw once (in shuffled order,
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
        """E[Σ(x)] at one location: Σ for every draw, THEN the mean.

        The order is the whole point of this class -- see the class
        docstring. Do not "optimize" this into Σ(x; mean of draws).
        """
        # vmap over the leading draw axis of every leaf of self._draws:
        # (N, d, d) -- one covariance matrix per posterior draw
        per_draw = jax.vmap(lambda params: self.model.local_covariance(params, x))(
            self._draws
        )
        return jnp.mean(per_draw, axis=0)  # (d, d)

    def _eval_batch_impl(self, X: jnp.ndarray) -> jnp.ndarray:
        """Flatten -> vmap -> reshape, as in ``WPPMCovarianceField``.

        Memory note: the intermediate is (n_points, n_draws, d, d), e.g.
        25 x 4000 x 2 x 2 floats = 1.6 MB; dense grids with many draws scale
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

        Each draw has its own U_k(x), but the average of U_k U_k^T is not
        U U^T for any single U from the model (and averaging the U_k instead
        reintroduces exactly the cancellation this class exists to avoid).
        Any matrix returned here would be a different object under the same
        name, so this raises instead. For a factor of the mean matrix itself,
        use ``jnp.linalg.cholesky(field(x))``.
        """
        raise NotImplementedError(
            "PosteriorMeanCovarianceField has no square root U(x): an average "
            "of U U^T over draws is not U U^T for any single U. Use "
            "jnp.linalg.cholesky(field(x)) for a factor of the mean matrix."
        )


def covariance_to_arviz(posterior, points):
    r"""
    Per-draw covariance Σ(x) at chosen points, as an ArviZ DataTree.

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
    For each draw and point, the d x d matrix Σ is summarized by its d log
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
    >>> import arviz as az
    >>> dt = covariance_to_arviz(nuts_posterior, ref_points)
    >>> az.rhat(dt)["Sigma"].max()  # worst point and quantity
    >>> az.plot_trace(dt, var_names=["Sigma"], coords={"point": 0})
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

    # Chain-structured draws via the public API (raises ImportError without
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

    # vmap over draws, then chains: (C, S, n_points, d, d)
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
