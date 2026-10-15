"""Tests for NeuralOddityTask: the surrogate wired in as a task likelihood.

Most of these guard against failures that are *silent*. A task likelihood that is
subtly wrong does not raise and does not produce NaN; it returns probabilities in
(0, 1) that are merely incorrect, and a fit built on them looks entirely normal.
Each test below names the specific way things go quiet.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from psyphy.data import TrialData
from psyphy.model import WPPM, GaussianNoise, OddityTask, OddityTaskConfig, Prior
from psyphy.model.likelihood import (
    BernoulliTaskLikelihood,
    NeuralOddityTask,
    NeuralSurrogate,
)
from psyphy.model.noise import StudentTNoise


def _model(likelihood, *, input_dim=2, basis_degree=2, noise=None):
    return WPPM(
        input_dim=input_dim,
        prior=Prior(input_dim=input_dim, basis_degree=basis_degree),
        likelihood=likelihood,
        noise=noise or GaussianNoise(),
    )


def _trials(key, n=24, spread=0.25):
    refs = jr.uniform(key, (n, 2), minval=-0.6, maxval=0.6)
    comps = refs + jr.uniform(jr.fold_in(key, 1), (n, 2), minval=-spread, maxval=spread)
    return jnp.stack([refs, comps], axis=1)


# --------------------------------------------------------------------------
# it loads with no configuration at all
# --------------------------------------------------------------------------


def test_constructs_with_no_arguments():
    """The shipped weights must be found without the user naming a path."""
    task = NeuralOddityTask()
    assert isinstance(task.surrogate, NeuralSurrogate)
    assert task.surrogate.meta["task"] == "oddity"
    assert task.surrogate.meta["input_dim"] == 2


def test_accepts_a_caller_supplied_surrogate():
    """Retraining on a wider domain should not require a psyphy change."""
    shipped = NeuralOddityTask().surrogate
    task = NeuralOddityTask(surrogate=shipped)
    assert task.surrogate is shipped


# --------------------------------------------------------------------------
# the contract with BernoulliTaskLikelihood
# --------------------------------------------------------------------------


def test_is_a_bernoulli_task_likelihood():
    """Two isinstance gates depend on this.

    WPPM.predict_prob and WPPMPredictivePosterior both require a
    BernoulliTaskLikelihood. A task that subclasses TaskLikelihood directly still
    fits without complaint, then raises on every prediction path -- so the
    relationship is pinned here rather than left to inheritance review.
    """
    assert isinstance(NeuralOddityTask(), BernoulliTaskLikelihood)


def test_predict_returns_a_one_tuple():
    """A bare array here silently collapses every trial onto trial 0.

    BernoulliTaskLikelihood.loglik vmaps predict and then indexes element 0 of
    the result. If predict returns a scalar rather than a tuple, that index
    selects the *first trial's* probability and broadcasts it across the batch:
    no error, a plausible log-likelihood, and a gradient that is wrong for every
    trial but the first.
    """
    task = NeuralOddityTask()
    model = _model(task)
    params = model.init_params(jr.PRNGKey(0))
    out = task.predict(params, _trials(jr.PRNGKey(1), n=1)[0], model)

    assert isinstance(out, tuple), (
        "predict must return a tuple of sufficient statistics"
    )
    assert len(out) == 1, "a Bernoulli task returns exactly (p_correct,)"
    assert jnp.ndim(out[0]) == 0
    assert 0.0 < float(out[0]) < 1.0


def test_loglik_uses_each_trial_separately():
    """The test a constant-probability fixture cannot perform.

    If every trial has the same p, reusing trial 0 is indistinguishable from
    doing it properly. These trials are built so the probabilities differ, and
    the assertion is that the total is the sum of the individual terms rather
    than a multiple of the first.
    """
    task = NeuralOddityTask()
    model = _model(task)
    params = model.init_params(jr.PRNGKey(0))

    stimuli = jnp.stack(
        [
            jnp.stack([jnp.array([0.0, 0.0]), jnp.array([0.02, 0.0])]),  # near: p low
            jnp.stack([jnp.array([0.0, 0.0]), jnp.array([0.60, 0.0])]),  # far:  p high
        ]
    )
    responses = jnp.array([[1], [1]])

    per_trial = [float(task.predict(params, s, model)[0]) for s in stimuli]
    assert abs(per_trial[0] - per_trial[1]) > 0.05, "fixture must have differing p"

    got = float(
        task.loglik(params, TrialData(stimuli=stimuli, responses=responses), model)
    )
    expected = float(np.log(per_trial[0]) + np.log(per_trial[1]))
    collapsed = float(2 * np.log(per_trial[0]))

    assert got == pytest.approx(expected, abs=1e-4), (
        f"loglik={got}, expected {expected}; {collapsed} would mean trial 0 is "
        "being reused for the whole batch"
    )


def test_predict_prob_works_end_to_end():
    """Exercises the isinstance gate rather than merely asserting the class."""
    model = _model(NeuralOddityTask())
    params = model.init_params(jr.PRNGKey(0))
    p = model.predict_prob(params, (jnp.array([0.0, 0.0]), jnp.array([0.3, 0.1])))
    assert 0.0 < float(p) < 1.0


# --------------------------------------------------------------------------
# guards: the places where the surrogate is simply not applicable
# --------------------------------------------------------------------------


def test_rejects_a_non_two_dimensional_model():
    """The feature map's rotation step is a 2-D construction.

    In 3-D the same argument gives six features, not four, and would need its own
    trained weights. Running the 2-D map on a 3-D model would not fail loudly, so
    it is refused explicitly.
    """
    model = _model(NeuralOddityTask(), input_dim=3)
    params = model.init_params(jr.PRNGKey(0))
    stim = jnp.stack([jnp.zeros(3), jnp.full(3, 0.2)])
    with pytest.raises(NotImplementedError, match="input_dim"):
        model.likelihood.predict(params, stim, model)


def test_rejects_non_gaussian_noise():
    """The network was fitted to Gaussian internal noise; nothing else is valid."""
    model = _model(NeuralOddityTask(), noise=StudentTNoise())
    params = model.init_params(jr.PRNGKey(0))
    with pytest.raises(NotImplementedError, match="[Gg]aussian"):
        model.likelihood.predict(params, _trials(jr.PRNGKey(1), n=1)[0], model)


# --------------------------------------------------------------------------
# behaviour under JAX transforms, and the determinism that is the point
# --------------------------------------------------------------------------


def test_is_declared_deterministic():
    """is_stochastic is what lets samplers stop guessing.

    An MC likelihood needs a PRNG key and gives a slightly different answer each
    call; this one does not. Declaring it lets NUTS skip its fixed-key workaround
    and lets a Laplace approximation refuse a likelihood whose Hessian would be
    noise.
    """
    assert NeuralOddityTask.is_stochastic is False


def test_is_deterministic_in_fact():
    task = NeuralOddityTask()
    model = _model(task)
    params = model.init_params(jr.PRNGKey(0))
    data = TrialData(
        stimuli=_trials(jr.PRNGKey(1)),
        responses=jr.bernoulli(jr.PRNGKey(2), 0.6, (24,)).astype(jnp.int32),
    )
    a = float(task.loglik(params, data, model, key=jr.PRNGKey(10)))
    b = float(task.loglik(params, data, model, key=jr.PRNGKey(99)))
    c = float(task.loglik(params, data, model))
    assert a == b == c, "different keys, and no key, must all give the same value"


def test_jit_and_grad_are_finite():
    import jax

    task = NeuralOddityTask()
    model = _model(task)
    params = model.init_params(jr.PRNGKey(0))
    data = TrialData(
        stimuli=_trials(jr.PRNGKey(1)),
        responses=jr.bernoulli(jr.PRNGKey(2), 0.6, (24,)).astype(jnp.int32),
    )
    value, grads = jax.jit(jax.value_and_grad(lambda p: task.loglik(p, data, model)))(
        params
    )
    assert jnp.isfinite(value)
    assert bool(jnp.all(jnp.isfinite(grads["W"])))
    assert float(jnp.max(jnp.abs(grads["W"]))) > 0.0, "a zero gradient would be useless"


# --------------------------------------------------------------------------
# the sub-chance regime: real, and not to be clamped
# --------------------------------------------------------------------------


def test_sub_chance_predictions_are_preserved():
    """Below 1/3 is a correct answer here, not a bug to be clipped away.

    The observer modelled by the oddity rule picks the two stimuli that are
    closest together and calls the third the odd one. That rule is not optimal.
    When the comparison's distribution is much tighter than the reference's, the
    two reference draws scatter while the comparison sits near the shared centre,
    so one reference is usually closer to the comparison than to its own twin and
    the rule names the wrong stimulus. Performance is then *reliably* wrong, which
    is worse than guessing.

    Confirmed against a two-million-sample simulation: cases like this one have a
    true P(correct) near 0.17. Clipping to the 1/3 guessing rate would replace a
    correct prediction with a wrong one, and would flatten the gradient exactly
    where the data is most informative about the comparison covariance.
    """
    from psyphy.model.likelihood.features import whitened_oddity_features

    surrogate = NeuralOddityTask().surrogate
    Sigma_ref = jnp.array([[5.74, 0.0], [0.0, 12.18]])
    Sigma_comp = jnp.array([[0.33, 0.0], [0.0, 1.00]])
    mu = jnp.zeros(2)

    x = whitened_oddity_features(Sigma_ref, Sigma_comp, mu, mu)
    p = float(surrogate(x))
    assert p < 1.0 / 3.0, f"expected a sub-chance probability, got {p}"
    assert p > 0.10, f"{p} is implausibly low even for this regime"


# --------------------------------------------------------------------------
# agreement with the simulation the network was fitted to
# --------------------------------------------------------------------------


def test_agrees_with_the_monte_carlo_task():
    """The claim that makes the surrogate usable: it reproduces OddityTask.

    The comparison is against a *high-sample* simulation on purpose. At the
    default sample count the simulation's own sampling error is larger than the
    surrogate's approximation error, so testing against it would mostly measure
    noise and would reward the wrong thing.

    Tolerance comes from measurement rather than taste: over these trials the
    median absolute difference is about 0.003 and the maximum about 0.009, and
    the published worst-case error for this network is roughly 0.02. The bound
    below sits above the observed maximum and below the published one, so it
    fails if the feature map, the weights or the standardization drift, without
    being so tight that Monte Carlo noise trips it.
    """
    nn_task = NeuralOddityTask()
    mc_task = OddityTask(OddityTaskConfig(num_samples=20_000))
    model_nn = _model(nn_task)
    model_mc = _model(mc_task)
    params = model_nn.init_params(jr.PRNGKey(0))

    n = 30
    refs = jr.uniform(jr.PRNGKey(1), (n, 2), minval=-0.6, maxval=0.6)
    # Clipped because the Chebyshev basis is only defined on [-1, 1].
    comps = jnp.clip(
        refs + jr.uniform(jr.PRNGKey(2), (n, 2), minval=-0.35, maxval=0.35), -1.0, 1.0
    )
    stimuli = jnp.stack([refs, comps], axis=1)
    keys = jr.split(jr.PRNGKey(3), n)

    p_nn = np.array([float(nn_task.predict(params, s, model_nn)[0]) for s in stimuli])
    p_mc = np.array(
        [
            float(mc_task.predict(params, s, model_mc, key=k)[0])
            for s, k in zip(stimuli, keys)
        ]
    )
    diff = np.abs(p_nn - p_mc)

    assert np.median(diff) < 0.01, f"median |diff| = {np.median(diff):.4f}"
    assert diff.max() < 0.02, f"max |diff| = {diff.max():.4f} at trial {diff.argmax()}"
