"""
map_optimizer.py
----------------

MAP (Maximum A Posteriori) optimizer using Optax.

MVP implementation:
- Uses gradient ascent on log posterior.
- Defaults to SGD with momentum, but any Optax optimizer can be passed in.

Connections
-----------
- Calls WPPM.log_posterior_from_data(params, data) as the objective.
- Returns a Posterior object wrapping the MAP estimate.
"""

from __future__ import annotations

import jax
import optax

from psyphy.inference.base import InferenceEngine
from psyphy.posterior.posterior import MAPPosterior


class MAPOptimizer(InferenceEngine):
    """
    MAP (Maximum A Posteriori) optimizer.

    Parameters
    ----------
    steps : int, default=500
        Number of optimization steps.
    optimizer : optax.GradientTransformation, optional
        Optax optimizer to use. Default: SGD with momentum.

    Notes
    -----
    - Loss function = negative log posterior.
    - Gradients computed with jax.grad.
    """

    def __init__(
        self,
        steps: int = 500,
        learning_rate: float = 5e-5,
        momentum: float = 0.9,
        optimizer: optax.GradientTransformation | None = None,
        *,
        track_history: bool = False,
        log_every: int = 1,
        max_grad_norm: float | None = 1.0,
    ):
        """Create a MAP optimizer.

        Parameters
        ----------
        steps : int
            Number of optimization steps.
        optimizer : optax.GradientTransformation | None
            Optax optimizer to use.
        learning_rate : float, optional
            Learning rate for the default optimizer (SGD with momentum).
        momentum : float, optional
            Momentum for the default optimizer (SGD with momentum).
        track_history : bool, optional
            When True, record loss history during fitting for plotting.
        log_every : int, optional
            Record every N steps (also records the last step).
        max_grad_norm : float | None, optional
            If set, clip gradients by global norm to this value before applying
            optimizer updates. This stabilizes optimization when gradients blow up.
        """
        self.steps = steps
        base_optimizer = optimizer or optax.sgd(
            learning_rate=learning_rate, momentum=momentum
        )
        if max_grad_norm is None:
            self.optimizer = base_optimizer
        else:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(float(max_grad_norm)),
                base_optimizer,
            )
        self.track_history = track_history
        self.log_every = max(1, int(log_every))
        self.max_grad_norm = max_grad_norm
        # Exposed after fit() when tracking is enabled
        self.loss_steps: list[int] = []
        self.loss_history: list[float] = []

    def fit(
        self, model, data, init_params: dict | None = None, seed: int | None = None
    ) -> MAPPosterior:
        """
        Fit model parameters with MAP optimization.

        Parameters
        ----------
        model : WPPM
            Model instance.
        data : ResponseData
            Observed trials.
        init_params : dict | None, optional
            Initial parameter PyTree to start optimization from. If provided,
            this takes precedence over the seed.
        seed : int | None, optional
            PRNG seed used to draw initial parameters from the model's prior
            when init_params is not provided. If None, defaults to 0.

        Returns
        -------
        MAPPosterior
            Posterior wrapper around MAP params and model.
        """

        def loss_fn(params):
            return -model.log_posterior_from_data(params, data)

        # Initialize parameters
        if init_params is not None:
            params = init_params
        else:
            rng_seed = 0 if seed is None else int(seed)
            params = model.init_params(jax.random.PRNGKey(rng_seed))
        opt_state = self.optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            # Ensure params and opt_state are JAX PyTrees for JIT compatibility
            loss, grads = jax.value_and_grad(loss_fn)(params)  # auto-diff
            updates, opt_state = self.optimizer.update(
                grads, opt_state, params
            )  # optimizer update
            params = optax.apply_updates(params, updates)  # apply updates
            # Only return JAX-compatible types (PyTrees of arrays, scalars)
            return params, opt_state, loss

        # clear any previous history
        if self.track_history:
            self.loss_steps.clear()
            self.loss_history.clear()

        for i in range(self.steps):
            params, opt_state, loss = step(params, opt_state)

            # Non-finite guard: if loss becomes NaN/Inf, optimization has diverged.
            # Stop early so downstream plots don’t look “truncated” due to NaNs.
            if not bool(jax.numpy.isfinite(loss)):
                if self.track_history:
                    try:
                        self.loss_steps.append(i)
                        self.loss_history.append(float(loss))
                    except Exception:
                        pass
                print(
                    f"[MAPOptimizer] Non-finite loss at step {i}: {loss}. "
                    "Stopping early."
                )
                break

            if self.track_history and (
                (i % self.log_every == 0) or (i == self.steps - 1)
            ):
                # Pull scalar to host and record
                try:
                    self.loss_steps.append(i)
                    self.loss_history.append(float(loss))
                except Exception:
                    # Best-effort; do not break fitting if logging fails
                    pass

        return MAPPosterior(params=params, model=model)

    # Optional helper
    def get_history(self) -> tuple[list[int], list[float]]:
        """Return (steps, losses) recorded during the last fit when tracking was enabled."""
        return self.loss_steps, self.loss_history
