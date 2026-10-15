"""
test_posterior_mean_covariance_field.py
---------------------------------------

Tests for PosteriorMeanCovarianceField: the posterior mean of the covariance
field, E[Σ(x) | data], computed by averaging Σ over posterior draws.

Why this class exists (the bug it fixes): Σ(x) = U(x) U(x)^T + λI is quadratic
in the parameters W, so building Σ from the *averaged* W, Σ(E[W]), is not the
posterior mean of Σ. It is always smaller (a Jensen gap), and for the WPPM it
can collapse toward λI, because rotating or reflecting W along its embedding
axis leaves Σ unchanged; draws at different orientations cancel when W is
averaged. These tests pin both effects down with hand-built draws, so no
sampler has to run.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, Prior
from psyphy.model.covariance_field import (
    CovarianceField,
    PosteriorMeanCovarianceField,
    WPPMCovarianceField,
)
from psyphy.model.likelihood import OddityTask
from psyphy.model.noise import GaussianNoise
from psyphy.posterior.mcmc_posterior import MCMCPosterior
from psyphy.posterior.posterior import MAPPosterior

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def _full_matmul_precision():
    """Run every test here with full float32 matmul precision.

    On Ampere+ GPUs JAX's default float32 matmul uses TF32 (~10-bit mantissa),
    and a vmapped evaluation can round differently from an unbatched one, by
    ~5e-4 relative. Several tests below assert that two sides are *exactly*
    equal in math (W vs -W, rotated W), and loosening their tolerance to
    absorb TF32 noise would also let real regressions through. Pinning the
    precision keeps them strict. The JIT cache keys on this setting, so
    functions compiled inside a test use it.
    """
    with jax.default_matmul_precision("highest"):
        yield


@pytest.fixture
def model():
    """Small 2D WPPM: W has shape (3, 3, 2, 3) -- embedding_dim m = 3."""
    return WPPM(
        input_dim=2,
        prior=Prior(
            input_dim=2,
            basis_degree=2,
            variance_scale=4e-3,
            decay_rate=0.4,
            extra_embedding_dims=1,
        ),
        likelihood=OddityTask(),
        noise=GaussianNoise(),
        extra_dims=1,
        diag_term=1e-4,
    )


@pytest.fixture
def X():
    """A few evaluation points, shape (5, 2)."""
    return jr.uniform(jr.PRNGKey(7), (5, 2), minval=-0.8, maxval=0.8)


def _prior_draws(model, n_chains, n_draws, seed=0):
    """Stack independent prior draws into MCMC layout (n_chains, n_draws, *W)."""
    keys = jr.split(jr.PRNGKey(seed), n_chains * n_draws)
    W = jnp.stack([model.init_params(k)["W"] for k in keys])
    return {"W": W.reshape(n_chains, n_draws, *W.shape[1:])}


def _random_orthogonal(key, m):
    """A random m x m orthogonal matrix (rotation or reflection), via QR."""
    Q, _ = jnp.linalg.qr(jr.normal(key, (m, m)))
    return Q


def _sigma(model, W, X):
    """Σ(X) for a single parameter draw W."""
    return WPPMCovarianceField(model, {"W": W})(X)


# ==============================================================================
# Core definition: average of per-draw Σ
# ==============================================================================


def test_equals_mean_of_per_draw_sigma(model, X):
    """Default (all draws) equals the hand-computed mean of Σ(x; W_k)."""
    samples = _prior_draws(model, n_chains=2, n_draws=4)
    posterior = MCMCPosterior(samples, model)

    W_all = samples["W"].reshape(-1, *samples["W"].shape[2:])
    expected = jnp.mean(jnp.stack([_sigma(model, W, X) for W in W_all]), axis=0)

    field = PosteriorMeanCovarianceField(posterior)
    assert jnp.allclose(field(X), expected, rtol=1e-5, atol=1e-9)


def test_single_point_matches_batch(model, X):
    """Shape (d,) input gives the same matrix as the matching batch row."""
    posterior = MCMCPosterior(_prior_draws(model, 2, 3), model)
    field = PosteriorMeanCovarianceField(posterior)
    assert field(X[0]).shape == (2, 2)
    assert jnp.allclose(field(X[0]), field(X)[0], rtol=1e-5, atol=1e-9)


def test_map_posterior_gives_sigma_of_theta_map(model, X):
    """For a MAP posterior every 'draw' is θ_MAP, so the mean is Σ(θ_MAP)."""
    params = model.init_params(jr.PRNGKey(3))
    posterior = MAPPosterior(params, model)
    field = PosteriorMeanCovarianceField(posterior, n_samples=3)
    assert jnp.allclose(field(X), _sigma(model, params["W"], X), rtol=1e-5, atol=1e-9)


# ==============================================================================
# The two effects that make Σ(E[W]) wrong
# ==============================================================================


def test_jensen_gap_mean_sigma_dominates_sigma_of_mean_W(model, X):
    """E[Σ] - Σ(E[W]) = Cov(U) is positive semidefinite at every point."""
    samples = _prior_draws(model, n_chains=2, n_draws=10)
    posterior = MCMCPosterior(samples, model)

    mean_sigma = PosteriorMeanCovarianceField(posterior)(X)
    sigma_of_mean_W = _sigma(model, posterior.params["W"], X)

    gap_eigs = jnp.linalg.eigvalsh(mean_sigma - sigma_of_mean_W)
    assert jnp.all(gap_eigs >= -1e-8)


def test_sign_flipped_draws_do_not_cancel(model, X):
    """W and -W give the same Σ. Averaging W gives 0 -> λI; averaging Σ must not."""
    W = model.init_params(jr.PRNGKey(1))["W"]
    samples = {"W": jnp.stack([W, -W])[None]}  # (1 chain, 2 draws, *W)
    posterior = MCMCPosterior(samples, model)

    # The trap this class replaces: E[W] = 0, so Σ(E[W]) is just λI.
    assert jnp.allclose(posterior.params["W"], 0.0)
    lam_I = model.diag_term * jnp.eye(2)
    assert jnp.allclose(_sigma(model, posterior.params["W"], X), lam_I)

    field = PosteriorMeanCovarianceField(posterior)
    assert jnp.allclose(field(X), _sigma(model, W, X), rtol=1e-5, atol=1e-9)


def test_invariant_to_embedding_rotations(model, X):
    """Draws W R_k (R_k orthogonal on the embedding axis) all share Σ(W).

    This is the WPPM's non-identifiability: chains that settle at different
    orientations agree on Σ but not on W. The mean field must equal Σ(W).
    """
    W = model.init_params(jr.PRNGKey(2))["W"]
    m = W.shape[-1]
    Rs = [_random_orthogonal(k, m) for k in jr.split(jr.PRNGKey(9), 6)]
    # W @ R acts on the last (embedding) axis: U(x) -> U(x) R
    rotated = jnp.stack([W @ R for R in Rs]).reshape(2, 3, *W.shape)
    posterior = MCMCPosterior({"W": rotated}, model)

    field = PosteriorMeanCovarianceField(posterior)
    assert jnp.allclose(field(X), _sigma(model, W, X), rtol=1e-4, atol=1e-8)


# ==============================================================================
# Interface
# ==============================================================================


def test_satisfies_covariance_field_protocol(model):
    posterior = MCMCPosterior(_prior_draws(model, 1, 2), model)
    assert isinstance(PosteriorMeanCovarianceField(posterior), CovarianceField)


def test_sqrt_cov_raises(model, X):
    """An average of U U^T has no single U behind it; refuse rather than guess."""
    posterior = MCMCPosterior(_prior_draws(model, 1, 2), model)
    field = PosteriorMeanCovarianceField(posterior)
    with pytest.raises(NotImplementedError, match="square root"):
        field.sqrt_cov(X[0])


def test_requires_n_samples_when_posterior_has_no_draw_count(model):
    """MAP (or any posterior without n_chains/n_draws) must be told how many."""
    posterior = MAPPosterior(model.init_params(jr.PRNGKey(0)), model)
    with pytest.raises(ValueError, match="n_samples"):
        PosteriorMeanCovarianceField(posterior)


def test_rejects_wrong_input_dim(model):
    posterior = MCMCPosterior(_prior_draws(model, 1, 2), model)
    field = PosteriorMeanCovarianceField(posterior)
    with pytest.raises(ValueError, match="input_dim"):
        field(jnp.ones((4, 3)))


def test_exported_from_psyphy_model():
    from psyphy.model import PosteriorMeanCovarianceField as Exported

    assert Exported is PosteriorMeanCovarianceField
