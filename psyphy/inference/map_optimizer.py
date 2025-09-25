"""
inference/map_optimizer.py
--------------------------

MAP (Maximum A Posteriori) optimizer with Optax.

Provides:
- MAPOptimizer: gradient-based optimization of WPPM log posterior
- Supports explicit choice of optimizer (SGD, Adam, etc.)
- Steps argument = number of parameter update iterations

Returns a Posterior with MAP_params set.
"""

import jax
import jax.numpy as jnp
import optax

from psyphy.posterior.posterior import Posterior

from .base import InferenceEngine


class MAPOptimizer(InferenceEngine):
    def __init__(self, steps: int = 500, optimizer: optax.GradientTransformation | None = None):
        self.steps = steps
        # default to SGD with momentum
        self.optimizer = optimizer or optax.sgd(learning_rate=1e-2, momentum=0.9)

    def fit(self, model, data):
        """
        Optimize log posterior using gradient ascent.
        """

        def loss_fn(params):
            # negative log posterior (we want to maximize posterior)
            return -model.log_posterior_from_data(params, data)

        params = model.init_params(jax.random.PRNGKey(0))
        opt_state = self.optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for _ in range(self.steps):
            params, opt_state, loss = step(params, opt_state)

        return Posterior(params=params, model=model)
