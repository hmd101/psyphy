"""Tests for FastOddityTask: the network wired in as a task likelihood.

Most of these pin failures that raise no error and return probabilities in (0, 1)
that are wrong. Each test says which one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from psyphy.data import TrialData
from psyphy.model import WPPM, GaussianNoise, OddityTask, OddityTaskConfig, Prior
from psyphy.model.likelihood import (
    BernoulliTaskLikelihood,
    FastOddityTask,
    TrainedNetwork,
)
from psyphy.model.noise import StudentTNoise


@pytest.fixture(autouse=True)
def _full_matmul_precision():
    """Full float32 matmuls: TF32 on Ampere+ GPUs would blur exact comparisons."""
    with jax.default_matmul_precision("float32"):
        yield


def _model(likelihood, *, input_dim=2, basis_degree=2, noise=None):
    return WPPM(
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
    """The shipped weights load without a path."""
    task = FastOddityTask()
    assert isinstance(task.network, TrainedNetwork)
    assert task.network.metadata["task"] == "oddity"
    assert task.network.metadata["input_dim"] == 2


def test_network_cannot_be_reassigned():
    """A jitted closure over the task would keep the first network after a reassignment."""
    task = FastOddityTask()
    with pytest.raises(AttributeError):
        task.network = lambda x: x


def test_accepts_a_caller_supplied_network():
    """Retraining on a wider domain should not require a psyphy change."""
    shipped = FastOddityTask().network
    task = FastOddityTask(network=shipped)
    assert task.network is shipped


def test_accepts_a_plain_callable():
    """A callable with no metadata, such as a network from another framework, is not checked."""

    def logit_fn(x):
        return jnp.zeros(x.shape[:-1])

    assert FastOddityTask(network=logit_fn).network is logit_fn


def _reload_with(tmp_path, **meta_changes):
    """The shipped weights re-saved with some metadata values replaced."""
    from psyphy.model.likelihood.neural import save_network

    shipped = FastOddityTask().network
    path = tmp_path / "other_task.npz"
    save_network(
        path,
        params=shipped.params,
        input_mean=shipped.input_mean,
        input_std=shipped.input_std,
        metadata=dict(shipped.metadata, **meta_changes),
    )
    return TrainedNetwork.from_npz(path)  # require=None: loads whatever the file says


def test_refuses_a_network_trained_for_another_task(tmp_path):
    """Oddity weights re-saved as 2AFC load and predict 0.3654 with no error.

    The constructor has to be where this raises, because a network reaches
    the task from `from_npz(path)` with no `require`.
    """
    other = _reload_with(tmp_path, task="2afc", pooling_weight=0.5)
    with pytest.raises(ValueError) as e:
        FastOddityTask(network=other)
    msg = str(e.value)
    assert "task" in msg and "2afc" in msg and "oddity" in msg, (
        f"the error must name the key and both values; got: {msg}"
    )


def test_refuses_a_network_trained_for_another_dimension(tmp_path):
    other = _reload_with(tmp_path, input_dim=3)
    with pytest.raises(ValueError, match="input_dim"):
        FastOddityTask(network=other)


# --------------------------------------------------------------------------
# the contract with BernoulliTaskLikelihood
# --------------------------------------------------------------------------


def test_is_a_bernoulli_task_likelihood():
    """Two isinstance gates depend on this.

    WPPM.predict_prob and WPPMPredictivePosterior both require a
    BernoulliTaskLikelihood. A task that subclasses TaskLikelihood directly still
    fits, then raises on every prediction path.
    """
    assert isinstance(FastOddityTask(), BernoulliTaskLikelihood)


def test_predict_returns_log_probabilities():
    """predict returns (log_p, log_1mp), which BernoulliTaskLikelihood.loglik unpacks."""
    task = FastOddityTask()
    model = _model(task)
    params = model.init_params(jr.PRNGKey(0))
    out = task.predict(params, _trials(jr.PRNGKey(1), n=1)[0], model)

    assert isinstance(out, tuple), "predict must return a tuple"
    assert len(out) == 2, "a Bernoulli task returns exactly (log_p, log_1mp)"
    log_p, log_1mp = out
    assert jnp.ndim(log_p) == 0 and jnp.ndim(log_1mp) == 0
    assert float(log_p) < 0.0 and float(log_1mp) < 0.0
    # complementary outcomes; float32 rounding is about 1e-7, so 1e-5 is 100x headroom
    assert float(jnp.exp(log_p) + jnp.exp(log_1mp)) == pytest.approx(1.0, abs=1e-5)


def test_loglik_uses_each_trial_separately():
    """The total is the sum of per-trial terms, not a multiple of the first.

    With the same p on every trial, reusing trial 0 gives the right total, so
    the 2 trials here have different p.
    """
    task = FastOddityTask()
    model = _model(task)
    params = model.init_params(jr.PRNGKey(0))

    stimuli = jnp.stack(
        [
            jnp.stack([jnp.array([0.0, 0.0]), jnp.array([0.02, 0.0])]),  # near: p low
            jnp.stack([jnp.array([0.0, 0.0]), jnp.array([0.60, 0.0])]),  # far:  p high
        ]
    )
    responses = jnp.array([[1], [1]])

    per_trial = [float(jnp.exp(task.predict(params, s, model)[0])) for s in stimuli]
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
    """Goes through the isinstance gate in WPPM.predict_prob."""
    model = _model(FastOddityTask())
    params = model.init_params(jr.PRNGKey(0))
    p = model.predict_prob(params, (jnp.array([0.0, 0.0]), jnp.array([0.3, 0.1])))
    assert 0.0 < float(p) < 1.0


# --------------------------------------------------------------------------
# guards: the places where the network is simply not applicable
# --------------------------------------------------------------------------


def test_rejects_a_non_two_dimensional_model():
    """The feature map's rotation step is a 2-D construction.

    In 3-D the same reduction gives 6 features and needs its own weights. The
    2-D map on a 3-D model would not raise by itself.
    """
    model = _model(FastOddityTask(), input_dim=3)
    params = model.init_params(jr.PRNGKey(0))
    stim = jnp.stack([jnp.zeros(3), jnp.full(3, 0.2)])
    with pytest.raises(NotImplementedError, match="input_dim"):
        model.likelihood.predict(params, stim, model)


def test_rejects_non_gaussian_noise():
    """The network was fitted to Gaussian internal noise; nothing else is valid."""
    model = _model(FastOddityTask(), noise=StudentTNoise())
    params = model.init_params(jr.PRNGKey(0))
    with pytest.raises(NotImplementedError, match="[Gg]aussian"):
        model.likelihood.predict(params, _trials(jr.PRNGKey(1), n=1)[0], model)


def test_accepts_a_gaussian_noise_subclass():
    """The check is isinstance, so a subclass of GaussianNoise is Gaussian noise."""

    class MyGaussian(GaussianNoise):
        pass

    model = _model(FastOddityTask(), noise=MyGaussian())
    params = model.init_params(jr.PRNGKey(0))
    log_p, _ = model.likelihood.predict(params, _trials(jr.PRNGKey(1), n=1)[0], model)
    assert bool(jnp.isfinite(log_p))


def test_rejects_an_unrelated_class_named_gaussian_noise():
    """The name is not what makes noise Gaussian."""
    impostor = type("GaussianNoise", (), {})()
    model = _model(FastOddityTask(), noise=impostor)
    with pytest.raises(NotImplementedError, match="[Gg]aussian"):
        model.likelihood._check_applicable(model)


# --------------------------------------------------------------------------
# behavior under JAX transforms, and determinism
# --------------------------------------------------------------------------


def test_is_declared_deterministic():
    """The class attribute says loglik ignores its key."""
    assert FastOddityTask.is_stochastic is False


def test_is_deterministic_in_fact():
    task = FastOddityTask()
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
    task = FastOddityTask()
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


def test_grad_is_finite_when_a_comparison_equals_its_reference():
    """One zero-offset trial, as in a design with catch trials, must not turn the
    whole gradient NaN."""
    task = FastOddityTask()
    model = _model(task)
    params = model.init_params(jr.PRNGKey(0))
    stimuli = _trials(jr.PRNGKey(1))
    # stimuli is (N, K=2, d): trial 0 gets comparison = reference
    stimuli = stimuli.at[0, 1].set(stimuli[0, 0])
    data = TrialData(
        stimuli=stimuli,
        responses=jr.bernoulli(jr.PRNGKey(2), 0.6, (24,)).astype(jnp.int32),
    )
    value, grads = jax.value_and_grad(lambda p: task.loglik(p, data, model))(params)
    assert jnp.isfinite(value)
    assert bool(jnp.all(jnp.isfinite(grads["W"])))


# --------------------------------------------------------------------------
# below chance, and not clipped
# --------------------------------------------------------------------------


def test_sub_chance_predictions_are_preserved():
    """Below 1/3 is the right answer here, so the clip must not floor it there.

    The oddity rule picks the two stimuli closest together and calls the third
    the odd one. When the comparison's distribution is much tighter than the
    reference's, the two reference samples scatter while the comparison sits
    near the shared center. One reference is then usually closer to the
    comparison than to the other reference, and the rule names the wrong
    stimulus more often than a guess would.

    A two-million-sample simulation gives P(correct) near 0.17 for cases like
    this one.
    """
    from psyphy.model.likelihood.features import oddity_features

    network = FastOddityTask().network
    cov_ref = jnp.array([[5.74, 0.0], [0.0, 12.18]])
    cov_comparison = jnp.array([[0.33, 0.0], [0.0, 1.00]])
    mu = jnp.zeros(2)

    x = oddity_features(cov_ref, cov_comparison, mu, mu)
    p = float(jax.nn.sigmoid(network(x)))  # the network returns a logit
    assert p < 1.0 / 3.0, f"expected a sub-chance probability, got {p}"
    assert p > 0.10, f"{p} is implausibly low even for this regime"


# --------------------------------------------------------------------------
# agreement with the simulation the network was fitted to
# --------------------------------------------------------------------------


def test_agrees_with_the_monte_carlo_task():
    """The network reproduces OddityTask at 20,000 samples.

    At the default sample count the simulation's own sampling error is larger
    than the network's error, so we compare against a high-sample run.

    Measured over these trials: median |diff| about 0.003, maximum about 0.009.
    The bound of 0.02 is above that maximum and fails if the feature map, the
    weights or the standardization change.
    """
    nn_task = FastOddityTask()
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

    # predict returns (log_p, log_1mp), so [0] is log P(correct)
    p_nn = np.array(
        [float(jnp.exp(nn_task.predict(params, s, model_nn)[0])) for s in stimuli]
    )
    p_mc = np.array(
        [
            float(jnp.exp(mc_task.predict(params, s, model_mc, key=k)[0]))
            for s, k in zip(stimuli, keys)
        ]
    )
    diff = np.abs(p_nn - p_mc)

    assert np.median(diff) < 0.01, f"median |diff| = {np.median(diff):.4f}"
    assert diff.max() < 0.02, f"max |diff| = {diff.max():.4f} at trial {diff.argmax()}"
