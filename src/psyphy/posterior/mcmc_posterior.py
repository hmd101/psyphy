"""
mcmc_posterior.py
-----------------

MCMCPosterior: backend-agnostic container for MCMC samples.

Satisfies the ParameterPosterior protocol -->  plugs directly into
WPPMPredictivePosterior and any future acquisition functions.

Design
------
MCMCPosterior stores raw chain draws as a dict of JAX arrays with shape
(n_chains, n_draws, *param_shape). It is produced by any MCMC inference
engine (NUTSSampler, future MALASampler, etc.) and consumed identically
to MAPPosterior by all downstream code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


class MCMCPosterior:
    """
    Backend-agnostic container for MCMC posterior samples.

    Satisfies the ParameterPosterior protocol:
    - ``params``: posterior mean over all chains and draws
    - ``model``: the fitted WPPM instance
    - ``sample(n, key)``: draw n parameter vectors from the chain pool

    Parameters
    ----------
    samples : dict
        Parameter samples. Each value has shape ``(n_chains, n_draws, *param_shape)``.
        For WPPM this is ``{"W": ndarray of shape (n_chains, n_draws, d0, d1, d2, d3)}``.
    model : WPPM
        The model instance used during inference.
    sampler_stats : dict | None, optional
        Per-chain diagnostics returned by the sampler, e.g.
        ``{"acceptance_rate": ndarray of shape (n_chains, n_draws)}``.

    Notes
    -----
    Implements the ParameterPosterior protocol (i.e., structural, not inherited).
    """

    def __init__(
        self,
        samples: dict,
        model,
        *,
        sampler_stats: dict | None = None,
    ):
        # samples: dict of JAX arrays, each shaped (n_chains, n_draws, *param_shape)
        # For WPPM: {"W": (n_chains, n_draws, degree+1, degree+1, input_dim, embedding_dim)}
        # e.g. 4 chains × 1000 draws, degree=1, input_dim=2, extra_dims=1:
        #      {"W": (4, 1000, 2, 2, 2, 3)}
        # This is the canonical shape that flows from NUTSSampler -> here -> ArviZ.
        self._samples = samples
        self._model = model
        # sampler_stats: optional dict of per-chain diagnostics from the sampler.
        # Typical keys from NUTSSampler: {"acceptance_rate": (n_chains, n_draws)}
        self.sampler_stats = sampler_stats or {}

    # ------------------------------------------------------------------
    # ParameterPosterior protocol
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict:
        """Posterior mean over all chains and draws.

        Returns
        -------
        dict
            Parameter PyTree with the same structure as a single draw.
            For WPPM: ``{"W": ndarray of shape (*W_shape)}``.
            e.g. {"W": (2, 2, 2, 3)} — mean over axes 0 (chains) and 1 (draws).
        """
        # _samples["W"]: (n_chains, n_draws, *param_shape)
        # mean over (0, 1)  ->  (*param_shape)
        return jax.tree.map(lambda x: jnp.mean(x, axis=(0, 1)), self._samples)

    @property
    def model(self):
        """The WPPM instance used during inference."""
        return self._model

    def sample(self, n: int, *, key=None) -> dict:
        """Draw n parameter vectors from the pooled chain samples.

        Pools all chains and draws into a flat collection of
        ``n_chains * n_draws`` samples, then randomly subsamples ``n``
        of them (without replacement if n <= total, with replacement otherwise).

        Parameters
        ----------
        n : int
            Number of samples to return.
        key : jax.random.KeyArray | None
            PRNG key. Defaults to ``jax.random.PRNGKey(0)`` when None.

        Returns
        -------
        dict
            Parameter PyTree with leading dimension ``n``.
            For WPPM: ``{"W": ndarray of shape (n, *W_shape)}``.
        """
        if key is None:
            key = jr.PRNGKey(0)

        def _subsample(x):
            # x     : (n_chains, n_draws, *param_shape)
            #           e.g. (4, 1000, 2, 2, 2, 3)
            # pool  : (n_chains * n_draws, *param_shape)
            #           e.g. (4000, 2, 2, 2, 3)  — all draws from all chains flattened
            # output: (n, *param_shape)
            #           e.g. (100, 2, 2, 2, 3)   — n draws subsampled from pool
            n_chains, n_draws = x.shape[:2]
            event_shape = x.shape[2:]
            pool = x.reshape(n_chains * n_draws, *event_shape)
            total = n_chains * n_draws
            replace = n > total
            indices = jr.choice(key, total, shape=(n,), replace=replace)
            return pool[indices]

        return jax.tree.map(_subsample, self._samples)

    # Convenience properties
    # -----------------------------------------------

    @property
    def n_chains(self) -> int:
        """Number of chains."""
        first = next(iter(self._samples.values()))
        return int(first.shape[0])

    @property
    def n_draws(self) -> int:
        """Number of draws per chain."""
        first = next(iter(self._samples.values()))
        return int(first.shape[1])

    # -------------------------------------------------------------
    # ArviZ conversion

    def to_arviz(self):
        """Convert MCMC samples to an ArviZ InferenceData object.

        Requires `arviz` to be installed::

            pip install 'psyphy[diagnostics]'

        Returns
        -------
        arviz.InferenceData
            Contains a ``posterior`` group with variables shaped
            ``(n_chains, n_draws, *param_shape)``.
            For WPPM: variable ``W`` of shape
            ``(n_chains, n_draws, d0, d1, d2, d3)``.

        Notes
        -----
        ArviZ handles arbitrary-dimensional variables natively — no
        flattening required. R-hat and ESS are computed per scalar element.

        If ``sampler_stats`` were provided at construction (e.g.
        ``acceptance_rate``), they are included in the ``sample_stats``
        group.

        Examples
        --------
        >>> idata = nuts_posterior.to_arviz()
        >>> import arviz as az
        >>> print(az.summary(idata))
        """
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "ArviZ is required for MCMCPosterior.to_arviz(). "
                "Install it with: pip install 'psyphy[diagnostics]'"
            ) from e

        # -- ArviZ 1.0 API boundary --------------------------------
        # az.from_dict(data) where data is a nested dict:
        #   {
        #     "posterior":    {var_name: np.ndarray of shape (n_chains, n_draws, *param_shape)},
        #     "sample_stats": {stat_name: np.ndarray of shape (n_chains, n_draws)},  # optional
        #   }
        #
        # For WPPM with 4 chains, 1000 draws, W shape (2,2,2,3):
        #   {"posterior": {"W": np.ndarray (4, 1000, 2, 2, 2, 3)}}
        #
        # Returns: xarray.DataTree  (ArviZ 1.0 — was InferenceData in ArviZ 0.x)
        #   idata["posterior"]["W"].values  -> np.ndarray (4, 1000, 2, 2, 2, 3)
        #   list(idata.children)            -> ["posterior", "sample_stats"]
        #   az.rhat(idata).ds.data_vars     -> {"W"}  (R-hat per scalar element)
        #
        # Note: must convert JAX arrays to numpy — ArviZ uses numpy/xarray internally.
        # ----------------------------------------------------------------
        data: dict = {"posterior": {k: np.asarray(v) for k, v in self._samples.items()}}

        if self.sampler_stats:
            # sample_stats values shape: (n_chains, n_draws)
            # e.g. {"acceptance_rate": np.ndarray (4, 1000)}
            data["sample_stats"] = {
                k: np.asarray(v) for k, v in self.sampler_stats.items()
            }

        return az.from_dict(data)
