"""
wppm.py
-------

Wishart Process Psychophysical Model (WPPM) — MVP-style implementation with
forward-compatible hooks for the full WPPM model.

Goals
-----
1) MVP that runs today:
   - Local covariance Σ(x) is diagonal and *constant* across the space.
   - Discriminability is Mahalanobis distance under Σ(reference).
   - Task mapping (e.g., Oddity, 2AFC) converts discriminability -> p(correct).
   - Likelihood is delegated to the TaskLikelihood (no Bernoulli code here).

2) Forward compatibility with full WPPM model:
   - Expose hyperparameters needed to for example use Model config used in Hong et al.:
       * extra_dims: embedding size for basis expansions (unused in MVP)
       * variance_scale: global covariance scale (unused in MVP)
       * decay_rate: smoothness/length-scale for covariance field (unused in MVP)
       * diag_term: numerical stabilizer added to covariance diagonals (used in MVP)
   - Later, replace `local_covariance` with a basis-expansion Wishart process
     and swap discriminability/likelihood with MC observer simulation.

All numerics use JAX (jax.numpy as jnp) to support autodiff and optax optimizers
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from .base import Model
from .prior import Prior
from .task import TaskLikelihood

# Type aliases for readability
Params = dict[str, jnp.ndarray]
# A "stimulus" is a pair (reference, probe) in model space (shape: (input_dim,))
Stimulus = tuple[jnp.ndarray, jnp.ndarray]


class WPPM(Model):
    """
    Wishart Process Psychophysical Model (WPPM).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the *input stimulus space* (e.g., 2 for isoluminant plane,
        3 for RGB). Both reference and probe live in R^{input_dim}.
    prior : Prior
        Prior distribution over model parameters. Controls basis_degree for Wishart
        mode (basis expansion) vs MVP mode (diagonal covariance). The WPPM delegates
        to prior.basis_degree to ensure consistency between parameter sampling and
        basis evaluation.
    task : TaskLikelihood
        Psychophysical task mapping that defines how discriminability translates
        to p(correct) and how log-likelihood of responses is computed.
        (e.g., OddityTask, TwoAFC)
    noise : Any, optional
        Noise model describing internal representation noise (e.g., GaussianNoise).
        Not used in MVP mapping but passed to the task interface for future MC sims.

    Forward-compatible hyperparameters
    -----------------------------------
    extra_dims : int, default=0
        Additional embedding dimensions for basis expansions (beyond input_dim).
        In Wishart mode, embedding_dim = input_dim + extra_dims.
    variance_scale : float, default=1.0
        Global scaling factor for covariance magnitude (unused in MVP).
    decay_rate : float, default=1.0
        Smoothness/length-scale for spatial covariance variation (unused in MVP).
        (formerly "decay_rate")
    diag_term : float, default=1e-6
        Small positive value added to the covariance diagonal for numerical stability.
        MVP uses this in matrix solves; the research model will also use it.
    """

    def __init__(
        self,
        input_dim: int,
        prior: Prior,
        task: TaskLikelihood,
        noise: Any | None = None,
        *,  # everything after here is keyword-only
        extra_dims: int = 0,
        variance_scale: float = 1.0,
        decay_rate: float = 1.0,
        diag_term: float = 1e-6,
        **kwargs,  # Accept online_config from model base
    ) -> None:
        # Initialize Model base class
        super().__init__(**kwargs)

        # --- core components ---
        self.input_dim = int(input_dim)  # stimulus-space dimensionality
        self.prior = prior  # prior over parameter PyTree

        if self.prior.input_dim != self.input_dim:
            raise ValueError(
                f"Dimension mismatch: Model initialized with input_dim={self.input_dim}, "
                f"but Prior expects input_dim={self.prior.input_dim}."
            )

        self.task = task  # task mapping and likelihood
        self.noise = noise  # noise model

        # --- forward-compatible hyperparameters (stubs in MVP) ---
        self.extra_dims = int(extra_dims)
        self.variance_scale = float(variance_scale)
        self.decay_rate = float(decay_rate)
        self.diag_term = float(diag_term)

    @property
    def basis_degree(self) -> int | None:
        """
        Chebyshev polynomial degree for Wishart process basis expansion.

        This property delegates to self.prior.basis_degree to ensure consistency
        between parameter sampling and basis evaluation.

        Returns
        -------
        int | None
            Degree of Chebyshev polynomial basis (0 = constant, 1 = linear, etc.)
            None indicates MVP mode (no basis expansion)

        Notes
        -----
        WPPM gets its basis_degree parameter from Prior.basis_degree.
        """
        return self.prior.basis_degree

    @property
    def embedding_dim(self) -> int:
        """
        Dimension of the embedding space (perceptual space).

        embedding_dim = input_dim + extra_dims.
        this represents the full perceptual space where:
        - First input_dim dimensions correspond to observable stimulus features
        - Remaining extra_dims are latent  dimensions

        Returns
        -------
        int
            input_dim + extra_dims (in Wishart mode)
            input_dim (in MVP mode, extra_dims ignored)

        Notes
        -----
        This is a computed property, not a constructor parameter.
        """
        if self.basis_degree is None:
            # MVP mode: no extra dimensions make sense
            return self.input_dim
        # Wishart mode: full perceptual space
        return self.input_dim + self.extra_dims

    def _normalize_stimulus(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize stimulus coordinates to [-1, 1] for Chebyshev basis.

        Assumes input stimuli are in [0, 1] range (standard for psychophysics).
        Maps [0, 1] -> [-1, 1] via x_norm = 2*x - 1.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Raw stimulus coordinates

        Returns
        -------
        x_norm : jnp.ndarray, shape (input_dim,)
            Normalized coordinates in [-1, 1]
        """
        return 2.0 * x - 1.0

    def _embed_stimulus(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Transform stimulus to embedding space via Chebyshev basis expansion.

        Hong et al. (2025) uses degree-5 Chebyshev polynomials for dimensionality
        reduction and better numerical conditioning.

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Raw stimulus coordinates (assumed in [0, 1])

        Returns
        -------
        x_embed : jnp.ndarray, shape (embedding_dim,)
            Embedded stimulus representation

        Notes
        -----
        If basis_degree is None (MVP mode), returns input unchanged.
        Otherwise, applies Chebyshev basis separately to each input dimension
        and concatenates the results.
        """
        from psyphy.utils.math import chebyshev_basis

        # MVP mode: no embedding
        if self.basis_degree is None:
            return x

        # Normalize to [-1, 1] for numerical stability
        x_norm = self._normalize_stimulus(x)

        # Apply Chebyshev basis to each dimension
        embeddings = []
        for i in range(self.input_dim):
            # chebyshev_basis expects shape (N,) and returns (N, degree+1)
            # We have a single point, so add/remove batch dimension
            x_i = x_norm[i : i + 1]  # shape (1,)
            cheb_i = chebyshev_basis(
                x_i, degree=self.basis_degree
            )  # shape (1, degree+1)
            embeddings.append(cheb_i.ravel())  # shape (degree+1,)

        # Concatenate all dimensions
        x_embed = jnp.concatenate(embeddings)  # shape (input_dim * (degree+1),)
        return x_embed

    # ----------------------------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------------------------
    def init_params(self, key: jr.KeyArray) -> Params:
        """
        Sample initial parameters from the prior.

        MVP parameters:
            {"log_diag": shape (input_dim,)}
        which defines a constant diagonal covariance across the space.

        Returns
        -------
        params : dict[str, jnp.ndarray]
        """
        return self.prior.sample_params(key)

    # ----------------------------------------------------------------------
    # WISHART PROCESS COVARIANCE
    # ----------------------------------------------------------------------
    def _evaluate_basis_at_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate all Chebyshev basis functions at point x, keeping structure for einsum.

        For 2D: returns φ_ij(x) = T_i(x_1) * T_j(x_2) with shape (degree+1, degree+1)
        For 3D: returns φ_ijk(x) = T_i(x_1) * T_j(x_2) * T_k(x_3) with shape (degree+1, degree+1, degree+1)

        Note: chebyshev_basis(x, degree=d) returns (degree+1) basis functions [T_0, ..., T_d].

        Parameters
        ----------
        x : jnp.ndarray, shape (input_dim,)
            Stimulus coordinates (assumed in [0, 1])

        Returns
        -------
        phi : jnp.ndarray
            Basis function values with structured shape for efficient einsum.
            Shape is (degree+1, degree+1) for 2D or (degree+1, degree+1, degree+1) for 3D.
        """
        from psyphy.utils.math import chebyshev_basis

        if self.basis_degree is None:
            raise ValueError(
                "Cannot evaluate basis: basis_degree is None (MVP mode). "
                "Set basis_degree to use Wishart process."
            )

        # Normalize to [-1, 1]
        x_norm = self._normalize_stimulus(x)

        if self.input_dim == 2:
            # Evaluate basis functions: φ_ij(x) = T_i(x_1) * T_j(x_2)
            # chebyshev_basis returns (1, degree+1) for each dimension
            cheb_0 = chebyshev_basis(x_norm[0:1], degree=self.basis_degree)[
                0, :
            ]  # (degree+1,)
            cheb_1 = chebyshev_basis(x_norm[1:2], degree=self.basis_degree)[
                0, :
            ]  # (degree+1,)
            phi = cheb_0[:, None] * cheb_1[None, :]  # (degree+1, degree+1)

        elif self.input_dim == 3:
            # 3D case: φ_ijk(x) = T_i(x_1) * T_j(x_2) * T_k(x_3)
            # phi.shape = (degree+1, degree+1, degree+1)
            cheb_0 = chebyshev_basis(x_norm[0:1], degree=self.basis_degree)[0, :]
            cheb_1 = chebyshev_basis(x_norm[1:2], degree=self.basis_degree)[0, :]
            cheb_2 = chebyshev_basis(x_norm[2:3], degree=self.basis_degree)[0, :]
            phi = cheb_0[:, None, None] * cheb_1[None, :, None] * cheb_2[None, None, :]

        else:
            raise NotImplementedError(
                f"Wishart process currently only supports 2D and 3D. Got input_dim={self.input_dim}"
            )

        return phi

    def _compute_sqrt(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute "square root" matrix U(x) from basis expansion.

        This is the core of the Wishart process: U(x) = Σ_ij W_ij * φ_ij(x)
        where W_ij are learned coefficients and φ_ij are Chebyshev basis functions.

        The covariance is then Σ(x) = U(x) @ U(x)^T + diag_term * I, which is
        guaranteed to be positive definite.

        Parameters
        ----------
        params : dict
            Model parameters. Must contain "W" for Wishart mode.
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        U : jnp.ndarray, shape (input_dim, embedding_dim)
            Rectangular square root matrix if extra_dims > 0.
            embedding_dim = input_dim + extra_dims

        Raises
        ------
        ValueError
            If params doesn't contain "W" (not in Wishart mode)

        Notes
        -----
        U is rectangular when extra_dims > 0.
        When multiplied U @ U^T, this produces covariance in stimulus space:
        Σ(x) \in R^(input_dim x input_dim)
        """
        if "W" not in params:
            raise ValueError(
                "Cannot compute U(x): params missing 'W'. "
                "Use Wishart mode (basis_degree set) or call local_covariance() in MVP mode."
            )

        W = params["W"]
        phi = self._evaluate_basis_at_point(x)

        # Linear combination: U(x) = Σ_ij W_ij * phi_ij(x)
        # Einstein summation over basis function indices
        if self.input_dim == 2:
            # W[i,j,d,e] * phi[i,j] -> U[d,e]
            # W is (degree+1, degree+1, input_dim, embedding_dim)
            # U is rectangular if extra_dims > 0: (input_dim, embedding_dim)
            U = jnp.einsum("ijde,ij->de", W, phi)
        elif self.input_dim == 3:
            # W[i,j,k,d,e] * phi[i,j,k] -> U[d,e]
            # W is (degree+1, degree+1, degree+1, input_dim, embedding_dim)
            # U is rectangular if extra_dims > 0: (input_dim, embedding_dim)
            U = jnp.einsum("ijkde,ijk->de", W, phi)
        else:
            raise NotImplementedError(
                f"Wishart process only supports 2D and 3D. Got input_dim={self.input_dim}"
            )

        return U

    # ----------------------------------------------------------------------
    # LOCAL COVARIANCE (Σ(x))
    # ----------------------------------------------------------------------
    def local_covariance(self, params: Params, x: jnp.ndarray) -> jnp.ndarray:
        """
        Return local covariance Σ(x) at stimulus location x.

        MVP mode (basis_degree=None):
            Σ(x) = diag(exp(log_diag)), constant across x.
            - Positive-definite because exp(log_diag) > 0.

        Wishart mode (basis_degree set):
            Σ(x) = U(x) @ U(x)^T + diag_term * I
            where U(x) is rectangular (input_dim, embedding_dim) if extra_dims > 0.
            - Varies smoothly with x
            - Guaranteed positive-definite
            - Returns stimulus covariance directly (input_dim, input_dim)

        Parameters
        ----------
        params : dict
            Model parameters:
            - MVP: {"log_diag": (input_dim,)}
            - Wishart: {"W": (degree+1, ..., input_dim, embedding_dim)}
        x : jnp.ndarray, shape (input_dim,)
            Stimulus location

        Returns
        -------
        Σ : jnp.ndarray, shape (input_dim, input_dim)
            Covariance matrix in stimulus space.
        """
        # MVP mode: constant diagonal covariance
        if "log_diag" in params:
            log_diag = params["log_diag"]
            diag = jnp.exp(log_diag)
            return jnp.diag(diag)

        # Wishart mode: spatially-varying covariance
        if "W" in params:
            U = self._compute_sqrt(params, x)  # (input_dim, embedding_dim)
            # Σ(x) = U(x) @ U(x)^T + diag_term * I
            # Result is (input_dim, input_dim)
            Sigma = U @ U.T + self.diag_term * jnp.eye(self.input_dim)
            return Sigma

        raise ValueError("params must contain either 'log_diag' (MVP) or 'W' (Wishart)")

    # ----------------------------------------------------------------------
    # DISCRIMINABILITY (d), later implemented via MC simulation
    # ----------------------------------------------------------------------
    def discriminability(self, params: Params, stimulus: Stimulus) -> jnp.ndarray:
        """
        Compute scalar discriminability d >= 0 for a (reference, probe) pair

        MVP mode:
            d = sqrt( (probe - ref)^T Σ(ref)^{-1} (probe - ref) )
            with Σ(ref) the local covariance at the reference in stimulus space.

        Wishart mode (rectangular U design) if extra_dims > 0:
            d = sqrt( (probe - ref)^T Σ(ref)^{-1} (probe - ref) )
            where Σ(ref) is directly computed in stimulus space (input_dim, input_dim)
            via U(x) @ U(x)^T with U rectangular.

        The discrimination task only depends on observable stimulus dimensions.
        The rectangular U design means local_covariance() already returns
        the stimulus covariance - no block extraction needed.

        Future (full WPPM mode):
            d is implicit via Monte Carlo simulation of internal noisy responses
            under the task's decision rule (no closed form). In that case, tasks
            will directly implement predict/loglik with MC, and this method may be
            used only for diagnostics.

        Parameters
        ----------
        params : dict
            Model parameters.
        stimulus : tuple
            (reference, probe) arrays of shape (input_dim,).

        Returns
        -------
        d : jnp.ndarray
            Nonnegative scalar discriminability.
        """
        ref, probe = stimulus

        # Delta is in stimulus space (input_dim)
        delta = probe - ref

        # Get stimulus covariance at reference
        # (rectangular U design: already returns (input_dim, input_dim))
        Sigma = self.local_covariance(params, ref)

        # Add jitter for stable solve; diag_term is configurable
        jitter = self.diag_term * jnp.eye(self.input_dim)

        # Solve (Σ + jitter)^{-1} delta using a PD-aware solver
        x = jax.scipy.linalg.solve(Sigma + jitter, delta, assume_a="pos")
        d2 = jnp.dot(delta, x)  # quadratic form

        # Guard against tiny negative values from numerical error
        return jnp.sqrt(jnp.maximum(d2, 0.0))

    # ----------------------------------------------------------------------
    # PREDICTION (delegates to task)
    # ----------------------------------------------------------------------
    def predict_prob(self, params: Params, stimulus: Stimulus) -> jnp.ndarray:
        """
        Predict probability of a correct response for a single stimulus.

        Design choice:
            WPPM computes discriminability & covariance; the TASK defines how
            that translates to performance. We therefore delegate to:
                task.predict(params, stimulus, model=self, noise=self.noise)

        Parameters
        ----------
        params : dict
        stimulus : (reference, probe)

        Returns
        -------
        p_correct : jnp.ndarray
        """
        return self.task.predict(params, stimulus, self, self.noise)

    # ----------------------------------------------------------------------
    # LIKELIHOOD (delegates to task)
    # ----------------------------------------------------------------------
    def log_likelihood(
        self,
        params: Params,
        refs: jnp.ndarray,
        probes: jnp.ndarray,
        responses: jnp.ndarray,
        **task_kwargs: Any,
    ) -> jnp.ndarray:
        """
        Compute the log-likelihood for arrays of trials.

        IMPORTANT:
            We delegate to the TaskLikelihood to avoid duplicating Bernoulli (MPV)
            or MC likelihood logic in multiple places.

        Parameters
        ----------
        params : dict
            Model parameters.
        refs : jnp.ndarray, shape (N, input_dim)
        probes : jnp.ndarray, shape (N, input_dim)
        responses : jnp.ndarray, shape (N,)
            Typically 0/1; task may support richer encodings.

        Returns
        -------
        loglik : jnp.ndarray
            Scalar log-likelihood (task-only; add prior outside if needed)
        """
        # We need a ResponseData-like object. To keep this method usable from
        # array inputs, we construct one on the fly. If you already have a
        # ResponseData instance, prefer `log_likelihood_from_data`.
        from psyphy.data.dataset import ResponseData  # local import to avoid cycles

        data = ResponseData()
        # ResponseData.add_trial(ref, probe, resp)
        for r, p, y in zip(refs, probes, responses):
            data.add_trial(r, p, int(y))
        return self.task.loglik(params, data, self, self.noise, **task_kwargs)

    def log_likelihood_from_data(
        self, params: Params, data: Any, **task_kwargs: Any
    ) -> jnp.ndarray:
        """Compute log-likelihood directly from a ResponseData object.

        Why delegate to the task?
            - The task knows the decision rule (oddity, 2AFC, ...).
            - The task can use the model (this WPPM) to fetch discriminabilities.
            - The task can use the noise model if it needs MC simulation.

        Parameters
        ----------
        params : dict
            Model parameters.
        data : ResponseData
            Collected trial data.

        Returns
        -------
        loglik : jnp.ndarray
            Scalar log-likelihood (task-only; add prior outside if needed).
        """
        return self.task.loglik(params, data, self, self.noise, **task_kwargs)

    # ----------------------------------------------------------------------
    # POSTERIOR-STYLE CONVENIENCE (OPTIONAL)
    # ----------------------------------------------------------------------
    def log_posterior_from_data(self, params: Params, data: Any) -> jnp.ndarray:
        """
        Convenience helper if you want log posterior in one call (MVP).

        This simply adds the prior log-probability to the task log-likelihood.
        Inference engines (e.g., MAP optimizer) typically optimize this quantity.

        Returns
        -------
        jnp.ndarray : scalar log posterior = loglik(params | data) + log_prior(params)
        """
        return self.log_likelihood_from_data(params, data) + self.prior.log_prob(params)

    # ----------------------------------------------------------------------
    # MODEL FORWARD PASS (for predict_with_params)
    # ----------------------------------------------------------------------
    def _forward(
        self,
        X: jnp.ndarray,
        probes: jnp.ndarray | None,
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Evaluate WPPM at specific parameter values (no marginalization).

        This is called by Model.predict_with_params() and is used for:
        - Threshold uncertainty estimation
        - Parameter sensitivity analysis
        - Debugging

        Parameters
        ----------
        X : jnp.ndarray, shape (n_test, input_dim)
            Test stimuli (references)
        probes : jnp.ndarray | None, shape (n_test, input_dim)
            Probe stimuli (None for detection tasks)
        params : dict[str, jnp.ndarray]
            Model parameters (e.g., {"log_diag": (input_dim,)})

        Returns
        -------
        predictions : jnp.ndarray, shape (n_test,)
            Predicted response probabilities at each test point

        Notes
        -----
        This evaluates the model deterministically at the given parameters.
        For proper predictions that account for parameter uncertainty,
        use model.posterior() instead.
        """
        if probes is None:
            # Detection task - not yet implemented in MVP
            raise NotImplementedError(
                "Detection tasks not yet supported in MVP. "
                "Use discrimination with probes."
            )

        # Vectorize over test points
        def predict_single(ref, probe):
            stimulus = (ref, probe)
            return self.predict_prob(params, stimulus)

        # Use vmap for efficient batch evaluation
        predictions = jax.vmap(predict_single)(X, probes)

        return predictions
