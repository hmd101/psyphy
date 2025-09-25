"""
posterior.py
------------

Posterior representation for WPPM.

MVP implementation:
- Stores only a single parameter set (MAP estimate).
- Provides a consistent interface (MAP_params, sample, predict_prob,
  predict_thresholds).
- Delegates probability calculations back to WPPM.

Design note
-----------
This class is currently a *stub*. In the future it should evolve into
a BasePosterior abstract base class, with concrete subclasses for:

- MAPPosterior      : point estimate only
- LaplacePosterior  : Gaussian approx around MAP (mean + covariance)
- MCMCPosterior     : posterior samples from Langevin/HMC

Keeping the interface stable now ensures downstream code (Session,
TrialPlacement, Viz) can use the same methods regardless of inference method.
"""

from __future__ import annotations

import jax.numpy as jnp


class Posterior:
    """
    Posterior wrapper for WPPM (MVP version).

    Parameters
    ----------
    params : dict
        Parameter dictionary (MAP estimate in MVP).
    model : WPPM
        Model instance for predictions.

    Notes
    -----
    - MVP: only supports MAP estimates.
    - Full mode: this will become an abstract base class (BasePosterior),
      and real subclasses will implement sampling behavior.
    """

    def __init__(self, params, model):
        self.params = params
        self.model = model

    # ------------------------------------------------------------------
    # ACCESSORS (expose internal information)
    # ------------------------------------------------------------------
    def MAP_params(self):
        """
        Return the MAP parameter set.

        Returns
        -------
        dict
            Model parameters.

        Future hook
        -----------
        - In MCMCPosterior: could return posterior mean instead.
        - In LaplacePosterior: could return Gaussian mean (MAP).
        """
        return self.params

    def sample(self, n: int = 1):
        """
        Draw parameter samples from the posterior.

        Parameters
        ----------
        n : int, default=1
            Number of samples.

        Returns
        -------
        list of dict
            Parameter sets.

        MVP
        ---
        Returns MAP params repeated n times.

        Future hook
        -----------
        - LaplacePosterior: draw from N(mean, cov).
        - MCMCPosterior: return stored MCMC samples.
        """
        return [self.params] * n

    # ------------------------------------------------------------------
    # PREDICTION INTERFACE (delegates to model)
    # ------------------------------------------------------------------
    def predict_prob(self, stimulus):
        """
        Predict probability of correct response for a stimulus.

        Parameters
        ----------
        stimulus : tuple
            (reference, probe).

        Returns
        -------
        jnp.ndarray
            Probability of correct response.

        Notes
        -----
        This is not recursion: Posterior delegates to WPPM's predict_prob,
        automatically supplying self.params.
        """
        return self.model.predict_prob(self.params, stimulus)

    def predict_thresholds(self, reference, criterion: float = 0.667, directions: int = 16):
        """
        Predict discrimination threshold contour around a reference stimulus.

        Parameters
        ----------
        reference : jnp.ndarray
            Reference point in model space.
        criterion : float, default=0.667
            Target performance (e.g., 2/3 for oddity).
        directions : int, default=16
            Number of directions to probe.

        Returns
        -------
        jnp.ndarray
            Threshold contour points (MVP: unit circle).

        MVP
        ---
        Returns a simple unit circle, ignoring criterion.

        Future hook
        -----------
        - Use model + posterior samples to estimate where performance
          crosses criterion along each direction.
        """
        angles = jnp.linspace(0, 2 * jnp.pi, directions, endpoint=False)
        contour = jnp.stack([reference + jnp.array([jnp.cos(a), jnp.sin(a)]) for a in angles])
        return contour
