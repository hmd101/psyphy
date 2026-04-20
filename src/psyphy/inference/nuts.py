r"""
nuts.py
-------

NUTS (No-U-Turn Sampler) inference engine backed by BlackJAX

Uses gradient-based MCMC to draw samples from the full posterior
p(θ| data), unlike MAP which returns only a point estimate.

Connections
-----------
-  BlackJax needs a log density and calls
    WPPM.log_posterior_from_data(params, data, key=fixed_key) as the
  log-density. A fixed key is baked in at construction time to make the
  stochastic MC likelihood deterministic for NUTS (see Notes).
- Returns an MCMCPosterior wrapping the sampled chains and model.

Notes on stochastic likelihood
-------------------------------
WPPM's current oddity-task-likelihood is estimated
via Monte Carlo inside OddityTask.loglik.
NUTS requires a (nearly) deterministic log-density to compute consistent
Hamiltonian dynamics. The chosen approach: a single fixed JAX key is baked
into the logdensity closure so every gradient evaluation uses the same MC
noise realization. This bias decreases as MC_SAMPLES -> \inf (>=200 recommended).

Alternatives (not implemented here):
- fresh key per NUTS *trajectory* (pseudo-marginal / noisy HMC)
- Neural surrogate likelihood
(exact gradients, see NeuralSurrogateTask, not yet implemented)

Requires
--------
blackjax >= 1.0  (install with: pip install 'psyphy[sampling]')
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp

from psyphy.inference.base import InferenceEngine
from psyphy.posterior.mcmc_posterior import MCMCPosterior


class NUTSSampler(InferenceEngine):
    """
    NUTS (No-U-Turn Sampler) using BlackJAX.

    Draws samples from the full posterior p(θ | data) via gradient-based
    MCMC. Returns an MCMCPosterior with shape
    (num_chains, num_samples, *param_shape).

    Parameters
    ----------
    num_warmup : int, default=500
        Number of warmup/burn-in steps per chain.
    num_samples : int, default=1000
        Number of posterior draws per chain (after warmup).
    num_chains : int, default=4
        Number of independent chains to run.
    logdensity_key_seed : int, default=0
        Seed for the fixed JAX key baked into the logdensity closure.
        Controls which MC noise realization is used throughout sampling.
        See module docstring for rationale.
    target_acceptance_rate : float, default=0.8
        Target Metropolis acceptance rate for window adaptation
        (Mode A only). Typical values: 0.-0.9.
    step_size : float | None, default=None
        If None (default), use blackjax.window_adaptation to
        automatically tune step_size and mass matrix (Mode A).
        If a float, skip adaptation and use this fixed step_size for all
        chains — warmup becomes plain burn-in and all chains are vmapped
        in parallel (Mode B). The user is responsible for choosing a
        suitable value.
    seed : int, default=0
        Master PRNG seed for chain initialization and sampling keys.
    show_progress : bool, default=True
        Display a tqdm progress bar during warmup and sampling.
        Silently skipped if tqdm is not installed.

    Notes
    -----
    **Two warmup modes:**

    *Mode A* (step_size=None, default): blackjax.window_adaptation
    adapts step_size and inverse_mass_matrix per chain. Adaptation uses
    Python-level averaging and is NOT vmappable — chains run
    sequentially in a Python for loop. After warmup, each chain's
    sampling is JIT-compiled via jax.lax.scan.

    *Mode B* (step_size=<float>): No adaptation. All chains share the
    same fixed step_size and a diagonal mass matrix of ones. Warmup is
    pure burn-in. All chains are fully vmapped -> single compiled call.
    Faster for many chains but requires manual step_size tuning.

    Examples
    --------
    >>> from psyphy.inference import NUTSSampler
    >>> sampler = NUTSSampler(num_warmup=200, num_samples=500, num_chains=4)
    >>> nuts_posterior = sampler.fit(model, data, init_params=map_posterior.params)
    >>> print(nuts_posterior.n_chains, nuts_posterior.n_draws)
    4 500
    """

    def __init__(
        self,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 4,
        *,
        logdensity_key_seed: int = 0,
        target_acceptance_rate: float = 0.8,
        step_size: float | None = None,
        seed: int = 0,
        show_progress: bool = True,
    ):
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.logdensity_key_seed = int(logdensity_key_seed)
        self.target_acceptance_rate = float(target_acceptance_rate)
        self.step_size = float(step_size) if step_size is not None else None
        self.seed = int(seed)
        self.show_progress = bool(show_progress)

    # ------------------------------------------------------------------
    # InferenceEngine protocol

    def fit(
        self,
        model,
        data,
        init_params: dict | None = None,
        seed: int | None = None,
    ) -> MCMCPosterior:
        """
        Run NUTS sampling and return a posterior over model parameters.

        Parameters
        ----------
        model : WPPM
            Model instance providing log_posterior_from_data.
        data : TrialData
            Observed trials.
        init_params : dict | None, optional
            Initial parameter PyTree. If None, samples from the model prior
            using the master seed. Providing MAP params as init_params
            dramatically reduces the warmup needed.
        seed : int | None, optional
            Override the master PRNG seed set at construction.

        Returns
        -------
        MCMCPosterior
            Samples shaped (num_chains, num_samples, *param_shape),
            sampler_stats with acceptance_rate.
        """
        try:
            import blackjax
        except ImportError as e:
            raise ImportError(
                "BlackJAX is required for NUTSSampler. "
                "Install it with: pip install 'psyphy[sampling]'"
            ) from e

        rng_seed = self.seed if seed is None else int(seed)
        master_key = jax.random.PRNGKey(rng_seed)
        init_key, chain_key = jax.random.split(master_key)

        # initial position: user-supplied or sampled from prior
        init_position = (
            init_params if init_params is not None else model.init_params(init_key)
        )

        # Fixed MC key baked into logdensity — makes the stochastic
        # likelihood deterministic for NUTS gradient evaluations.
        fixed_mc_key = jax.random.PRNGKey(self.logdensity_key_seed)

        def logdensity_fn(params: dict) -> jnp.ndarray:
            return model.log_posterior_from_data(params, data, key=fixed_mc_key)

        # Per-chain PRNG keys
        chain_keys = jax.random.split(chain_key, self.num_chains)

        if self.step_size is None:
            positions, acc_rates = self._run_adaptive(
                blackjax, logdensity_fn, init_position, chain_keys
            )
        else:
            positions, acc_rates = self._run_fixed_stepsize(
                blackjax, logdensity_fn, init_position, chain_keys
            )

        # positions: dict with each value shaped (n_chains, n_draws, *event)
        sampler_stats = {"acceptance_rate": acc_rates}
        return MCMCPosterior(positions, model, sampler_stats=sampler_stats)

    # ------------------------------------------------------------------
    # Mode A: adaptive warmup (window_adaptation), sequential chains
    # ------------------------------------------------------------------

    def _run_adaptive(self, blackjax, logdensity_fn, init_position, chain_keys):
        """Run adaptive NUTS — one chain at a time (window_adaptation not vmappable)."""
        all_positions = []
        all_acc_rates = []

        pbar = self._make_pbar(
            total=self.num_chains * (self.num_warmup + self.num_samples),
            desc="NUTS (adaptive)",
        )

        for i, chain_key in enumerate(chain_keys):
            warmup_key, sample_key = jax.random.split(chain_key)

            # --- Warmup: adapt step_size and inverse_mass_matrix ---
            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.set_description(f"NUTS warmup chain {i + 1}/{self.num_chains}")

            warmup = blackjax.window_adaptation(
                blackjax.nuts,
                logdensity_fn,
                target_acceptance_rate=self.target_acceptance_rate,
                progress_bar=self.show_progress,
            )
            (state, parameters), _ = warmup.run(
                warmup_key, init_position, num_steps=self.num_warmup
            )

            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.update(self.num_warmup)

            # --- Sampling: lax.scan over adapted kernel ---
            kernel = blackjax.nuts(logdensity_fn, **parameters).step

            def one_step(state, rng_key, _kernel=kernel):
                state, info = _kernel(rng_key, state)
                return state, (state.position, info.acceptance_rate)

            sample_keys = jax.random.split(sample_key, self.num_samples)
            _, (positions, acc_rates) = jax.lax.scan(one_step, state, sample_keys)

            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.update(self.num_samples)

            all_positions.append(positions)
            all_acc_rates.append(acc_rates)

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.close()

        # Stack chains: list of dicts -> dict of (n_chains, n_draws, *shape)
        stacked_positions = jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0), *all_positions
        )
        stacked_acc = jnp.stack(all_acc_rates, axis=0)  # (n_chains, n_draws)
        return stacked_positions, stacked_acc

    # ------------------------------------------------------------------
    # Mode B: fixed step_size, all chains vmapped
    # ------------------------------------------------------------------

    def _run_fixed_stepsize(self, blackjax, logdensity_fn, init_position, chain_keys):
        """Run NUTS with fixed step_size — all chains vmapped in one call."""
        # Determine parameter dimension for diagonal mass matrix
        flat_leaves = jax.tree.leaves(init_position)
        n_params = sum(x.size for x in flat_leaves)
        inverse_mass_matrix = jnp.ones(n_params)

        nuts = blackjax.nuts(
            logdensity_fn,
            step_size=self.step_size,
            inverse_mass_matrix=inverse_mass_matrix,
        )

        def run_one_chain(chain_key):
            warmup_key, sample_key = jax.random.split(chain_key)
            state = nuts.init(init_position)

            def one_step(state, rng_key):
                state, info = nuts.step(rng_key, state)
                return state, (state.position, info.acceptance_rate)

            # Burn-in (warmup as plain scan)
            burn_keys = jax.random.split(warmup_key, self.num_warmup)
            state, _ = jax.lax.scan(one_step, state, burn_keys)

            # Sampling
            s_keys = jax.random.split(sample_key, self.num_samples)
            _, (positions, acc_rates) = jax.lax.scan(one_step, state, s_keys)
            return positions, acc_rates

        pbar = self._make_pbar(total=1, desc="NUTS (fixed step_size, vmapped)")

        # vmap over chains — single compiled call
        all_positions, all_acc_rates = jax.vmap(run_one_chain)(chain_keys)

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.update(1)
                pbar.close()

        return all_positions, all_acc_rates

    # ------------------------------------------------------------------
    # Progress bar helper
    # ------------------------------------------------------------------

    def _make_pbar(self, *, total: int, desc: str):
        if not self.show_progress:
            return None
        try:
            from tqdm.auto import tqdm

            return tqdm(total=total, desc=desc, leave=False)
        except Exception:
            return None
