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

import contextlib

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
        track_history: bool = True,
        log_every: int = 1,
        progress_every: int = 10,
        show_progress: bool = False,
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
        progress_every : int, optional
            Update the progress-bar loss display every N steps (and the last step)
            when show_progress=True.
            This is kept separate from log_every so you can record loss at high
            frequency for plotting (e.g. log_every=1) without forcing a device->host
            sync for the progress UI every step.
        show_progress : bool, optional
            When True, display a tqdm progress bar during fitting.
            This is a UI feature: if tqdm is not installed,
            fitting proceeds without a progress bar.
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
        self.progress_every = max(1, int(progress_every))
        self.show_progress = bool(show_progress)
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

        # Optional progress bar.
        #
        # Why we *manually* advance the bar:
        # - When JAX runs on GPU, the first `step(...)` call can spend a long time in
        #   compilation, and tqdm may not visibly advance if the underlying iterator
        #   doesn't get a chance to redraw.
        # - By keeping a normal `range(self.steps)` loop and calling `pbar.update(1)`
        #   ourselves, we ensure the bar advances exactly once per iteration.
        #
        # Performance note: *displaying the loss* requires transferring `loss` from
        # device -> host, which can add sync overhead. We therefore only attach a
        # loss postfix every `progress_every` steps.
        pbar = None
        if self.show_progress:
            try:
                from tqdm.auto import tqdm

                pbar = tqdm(total=self.steps, desc="MAP fit", leave=False)
            except Exception:
                # Soft dependency: tqdm not available (or terminal unsuitable).
                pbar = None

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
                if pbar is not None:
                    with contextlib.suppress(Exception):
                        pbar.update(1)
                break

            if self.track_history and (
                (i % self.log_every == 0) or (i == self.steps - 1)
            ):
                # Pull scalar to host and record
                try:
                    self.loss_steps.append(i)
                    self.loss_history.append(float(loss))
                except Exception:
                    #  do not break fitting if logging fails
                    pass

            #  progress bar loss display (avoid host sync every step)
            if pbar is not None and (
                (i % self.progress_every == 0) or (i == self.steps - 1)
            ):
                with contextlib.suppress(Exception):
                    pbar.set_postfix(loss=float(loss))

            if pbar is not None:
                with contextlib.suppress(Exception):
                    pbar.update(1)
                    # Encourage a redraw occasionally in environments with buffered/stale
                    # TTY updates.
                    if (i % self.progress_every == 0) or (i == self.steps - 1):
                        pbar.refresh()

        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.close()

        return MAPPosterior(params=params, model=model)

    # Optional helper
    def get_history(self) -> tuple[list[int], list[float]]:
        """Return (steps, losses) recorded during the last fit when tracking was enabled."""
        return self.loss_steps, self.loss_history
