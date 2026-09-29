"""
test_quick_start_recovery.py
----------------------------

Regression test for the quick-start tutorial's headline claim: fitting a WPPM
to data simulated from a known WPPM recovers the ground-truth covariance.

This exists because the tutorial's committed figure once showed a good fit that
the committed code could not reproduce -- the fit had silently collapsed to zero
covariance and nothing caught it, because "the script ran without error" was the
only check. The failure mode is specific and worth naming: with all trials at a
*single* reference point the likelihood is weak, and below roughly 200 trials
the prior's shrinkage wins and drives the weights to zero. That produces a
plausible-looking run, a monotone learning curve, and a meaningless fit.

The settings here mirror ``docs/examples/wppm/quick_start.py``. If that script's
compute settings change, change them here too -- the point is to pin the
tutorial's result, not an arbitrary configuration.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data import TrialData
from psyphy.inference import MAPOptimizer
from psyphy.model import (
    WPPM,
    GaussianNoise,
    OddityTask,
    OddityTaskConfig,
    Prior,
    WPPMCovarianceField,
)

# Kept in sync with docs/examples/wppm/quick_start.py.
MC_SAMPLES = 50
NUM_TRIALS = 400
NUM_STEPS = 600
LEARNING_RATE = 1e-4

REF_POINT = jnp.array([[0.0, 0.0]])
MAHAL_RADIUS = 2.8
NOISE_SIGMA = 0.1

# Measured over five init seeds: area ratio 1.019 +/- 0.022, worst relative
# Frobenius error 0.33. The bounds below are wide enough for Monte Carlo and
# platform variation, but nowhere near wide enough to admit a collapse (which
# scores ~0.01) or a divergence.
AREA_RATIO_BOUNDS = (0.7, 1.4)
MAX_REL_FROBENIUS = 0.6


def _area_ratio(fitted, truth) -> float:
    """Ratio of ellipse areas, sqrt(det) based. ~0 when the fit collapses."""
    return float(
        jnp.sqrt(jnp.sqrt(jnp.linalg.det(fitted)) / jnp.sqrt(jnp.linalg.det(truth)))
    )


def _simulate(n_trials: int = NUM_TRIALS):
    """Ground-truth model plus trials drawn from it, as the tutorial does."""
    task = OddityTask(config=OddityTaskConfig(num_samples=MC_SAMPLES))
    noise = GaussianNoise(sigma=NOISE_SIGMA)
    truth_model = WPPM(prior=Prior(), likelihood=task, noise=noise)
    truth_params = truth_model.init_params(jr.PRNGKey(123))

    refs = jnp.repeat(REF_POINT, repeats=n_trials, axis=0)
    truth_field = WPPMCovarianceField(truth_model, truth_params)
    sigmas = truth_field(refs)

    k_dir, k_sim = jr.split(jr.PRNGKey(3))
    angles = jr.uniform(k_dir, (n_trials,), minval=0.0, maxval=2 * jnp.pi)
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    chol = jnp.linalg.cholesky(sigmas)
    comps = jnp.clip(
        refs + MAHAL_RADIUS * jnp.einsum("nij,nj->ni", chol, dirs), -1.0, 1.0
    )
    stimuli = jnp.stack([refs, comps], axis=1)
    responses, _ = task.simulate(truth_params, stimuli, truth_model, key=k_sim)
    return TrialData(stimuli=stimuli, responses=responses), task, noise, truth_field


@pytest.fixture(scope="module")
def fit():
    """Run the tutorial's fit once and share it across assertions (~4 s)."""
    data, task, noise, truth_field = _simulate()
    model = WPPM(prior=Prior(), likelihood=task, noise=noise)
    init_params = model.init_params(jr.PRNGKey(42))
    optimizer = MAPOptimizer(
        steps=NUM_STEPS,
        learning_rate=LEARNING_RATE,
        track_history=True,
        log_every=1,
    )
    estimate = optimizer.fit(model, data, init_params=init_params, seed=4)
    fitted = WPPMCovarianceField(model, estimate.params)(REF_POINT)[0]
    return {
        "truth": truth_field(REF_POINT)[0],
        "fitted": fitted,
        "init": WPPMCovarianceField(model, init_params)(REF_POINT)[0],
        "losses": optimizer.get_history()[1],
        "optimizer": optimizer,
    }


def test_fit_is_finite_and_positive_definite(fit):
    assert jnp.all(jnp.isfinite(fit["fitted"]))
    assert jnp.all(jnp.linalg.eigvalsh(fit["fitted"]) > 0)


def test_fitted_ellipse_has_the_right_area(fit):
    """sqrt(det) ratio: the statistic that catches a collapse to zero covariance."""
    ratio = _area_ratio(fit["fitted"], fit["truth"])
    low, high = AREA_RATIO_BOUNDS
    assert low < ratio < high, (
        f"fitted/true ellipse area ratio {ratio:.3f} outside [{low}, {high}]. "
        "A ratio near zero means the fit collapsed to zero covariance -- most "
        "likely NUM_TRIALS is too low for a single reference point."
    )


def test_fitted_covariance_matches_ground_truth(fit):
    rel = float(
        jnp.linalg.norm(fit["fitted"] - fit["truth"]) / jnp.linalg.norm(fit["truth"])
    )
    assert rel < MAX_REL_FROBENIUS, (
        f"relative Frobenius error {rel:.3f} exceeds {MAX_REL_FROBENIUS}"
    )


def test_fit_improves_on_its_initialization(fit):
    """Guards the vacuous pass where the prior draw already happened to be close."""

    def err(S):
        return float(jnp.linalg.norm(S - fit["truth"]))

    assert err(fit["fitted"]) < err(fit["init"])


def test_loss_decreases_and_stays_finite(fit):
    losses = fit["losses"]
    assert len(losses) == NUM_STEPS
    assert all(jnp.isfinite(jnp.asarray(x)) for x in losses)
    assert losses[-1] < losses[0]


def test_tutorial_settings_do_not_trigger_gradient_clipping(fit):
    """The tutorial relies on the default; if that changes, this test says so."""
    assert fit["optimizer"].max_grad_norm is None
    assert fit["optimizer"].clip_rate is None


def test_too_few_trials_collapses_the_fit():
    """Pins the failure mode itself, so the bound in AREA_RATIO_BOUNDS is meaningful.

    Not a desired behaviour -- a documented one. If this ever stops collapsing,
    the prior/likelihood balance has changed and the tutorial's trial count can
    be revisited.
    """
    data, task, noise, truth_field = _simulate(n_trials=100)
    model = WPPM(prior=Prior(), likelihood=task, noise=noise)
    init_params = model.init_params(jr.PRNGKey(42))
    opt = MAPOptimizer(steps=200, learning_rate=LEARNING_RATE)
    est = opt.fit(model, data, init_params=init_params, seed=4)
    ratio = _area_ratio(
        WPPMCovarianceField(model, est.params)(REF_POINT)[0], truth_field(REF_POINT)[0]
    )
    assert ratio < AREA_RATIO_BOUNDS[0]
