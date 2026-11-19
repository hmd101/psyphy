"""Tests for psyphy.model.likelihood.features.

The assertions follow from the math of the feature map:

* normalizing by noise is defined by `M S_bar M^T = I`
* the pooled constraint `w*A + (1-w)*B = I` gives `B = (I - w*A)/(1-w)`, which
  removes 3 free numbers
* `P(correct)` does not change under an invertible affine change of stimulus
  coordinates, so the features must not either. A rotation alone would pass a
  wrong whitening, so we also test a shear

The reference values at the bottom come from Philip Greengard's implementation
(`whiten_inputs`), copied in as literals.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from psyphy.model.likelihood.features import (
    align_displacement_2d,
    normalize_by_noise,
    oddity_features,
)

ODDITY_W = 2.0 / 3.0  # two reference presentations, one comparison
TWOAFC_W = 0.5  # one each


@pytest.fixture(autouse=True)
def _full_matmul_precision():
    """Full float32 matmuls: TF32 on Ampere+ GPUs would blur exact comparisons."""
    with jax.default_matmul_precision("float32"):
        yield


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _spd(rng, d=2, scale=1.0):
    """A random symmetric positive-definite matrix."""
    a = rng.normal(size=(d, d)) * scale
    return jnp.asarray(a @ a.T + 0.3 * np.eye(d))


def _trial(rng, d=2):
    return (
        _spd(rng, d),
        _spd(rng, d),
        jnp.asarray(rng.normal(size=d)),
        jnp.asarray(rng.normal(size=d)),
    )


# --------------------------------------------------------------------------
# normalize_by_noise(): the defining property, and what it buys
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pooling_weight", [ODDITY_W, TWOAFC_W])
def test_normalize_by_noise_makes_the_pooled_covariance_the_identity(pooling_weight):
    """The definition of normalizing by noise."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        Sr, Sc, mr, mc = _trial(rng)
        A, _ = normalize_by_noise(Sr, Sc, mr, mc, pooling_weight=pooling_weight)
        B = (jnp.eye(2) - pooling_weight * A) / (1.0 - pooling_weight)
        # A and B are cov_ref and cov_comparison in noise units, so the
        # pooled combination of them must be exactly the identity.
        pooled = pooling_weight * A + (1.0 - pooling_weight) * B
        # atol for float32, eps 1.2e-7
        assert jnp.allclose(pooled, jnp.eye(2), atol=1e-6)


@pytest.mark.parametrize("pooling_weight,bound", [(ODDITY_W, 1.5), (TWOAFC_W, 2.0)])
def test_eigenvalues_of_A_are_bounded_by_one_over_w(pooling_weight, bound):
    """B must stay positive definite, which caps eig(A) at 1/pooling_weight."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        Sr, Sc, mr, mc = _trial(rng)
        A, _ = normalize_by_noise(Sr, Sc, mr, mc, pooling_weight=pooling_weight)
        ev = jnp.linalg.eigvalsh(A)
        assert float(ev.min()) > 0.0
        assert float(ev.max()) < bound


def test_normalized_distance_equals_mahalanobis_distance():
    """After normalizing by noise, the S_bar^-1 in the decision rule is gone."""
    rng = np.random.default_rng(2)
    Sr, Sc, mr, mc = _trial(rng)
    Sbar = ODDITY_W * Sr + (1 - ODDITY_W) * Sc
    x, y = jnp.asarray(rng.normal(size=2)), jnp.asarray(rng.normal(size=2))

    maha = (x - y) @ jnp.linalg.inv(Sbar) @ (x - y)
    M = jnp.linalg.inv(jnp.linalg.cholesky(Sbar))
    eucl = jnp.sum((M @ x - M @ y) ** 2)
    assert float(abs(maha - eucl)) < 1e-6  # float32; see note above


# --------------------------------------------------------------------------
# the invariances that leave 4 features
# --------------------------------------------------------------------------


def test_invariant_under_translation():
    rng = np.random.default_rng(3)
    Sr, Sc, mr, mc = _trial(rng)
    shift = jnp.asarray(rng.normal(size=2))
    base = oddity_features(Sr, Sc, mr, mc)
    moved = oddity_features(Sr, Sc, mr + shift, mc + shift)
    # float32 gap measured at 5e-7 over 200 seeds. A feature that leaks the shift
    # moves by about |shift| (here ~1), which is 1e5 times atol.
    assert jnp.allclose(base, moved, rtol=0, atol=1e-5)


def test_invariant_under_rotation():
    rng = np.random.default_rng(4)
    Sr, Sc, mr, mc = _trial(rng)
    th = 0.7
    R = jnp.asarray([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    base = oddity_features(Sr, Sc, mr, mc)
    turned = oddity_features(R @ Sr @ R.T, R @ Sc @ R.T, R @ mr, R @ mc)
    assert jnp.allclose(base, turned, atol=1e-6)


def test_invariant_under_a_general_affine_map():
    """A rotation-only test would pass a wrong normalization."""
    rng = np.random.default_rng(5)
    Sr, Sc, mr, mc = _trial(rng)
    L = jnp.asarray([[2.0, 0.7], [-0.4, 0.5]])  # shear + anisotropic scale
    assert abs(float(jnp.linalg.det(L))) > 1e-6
    base = oddity_features(Sr, Sc, mr, mc)
    mapped = oddity_features(L @ Sr @ L.T, L @ Sc @ L.T, L @ mr, L @ mc)
    assert jnp.allclose(base, mapped, atol=1e-5)


# --------------------------------------------------------------------------
# shape, coincident stimuli, and pooling_weight
# --------------------------------------------------------------------------


def test_feature_vector_is_2d_long():
    """2*input_dim, which for the 2-D oddity task is 4."""
    rng = np.random.default_rng(6)
    Sr, Sc, mr, mc = _trial(rng)
    assert oddity_features(Sr, Sc, mr, mc).shape == (4,)


def test_coincident_stimuli_are_finite():
    """ref == comparison makes the rotation undefined; it must not produce NaN."""
    f = oddity_features(jnp.eye(2), jnp.eye(2), jnp.zeros(2), jnp.zeros(2))
    assert bool(jnp.all(jnp.isfinite(f)))
    assert float(f[0]) == pytest.approx(0.0, abs=1e-12)


def test_coincident_stimuli_have_a_finite_gradient():
    """The gradient of ||d|| at d = 0 is 0/0 unless the norm is guarded."""

    def feature_sum(Sc, mc):
        return jnp.sum(oddity_features(jnp.eye(2), Sc, jnp.zeros(2), mc))

    g_cov, g_mu = jax.grad(feature_sum, argnums=(0, 1))(jnp.eye(2), jnp.zeros(2))
    assert bool(jnp.all(jnp.isfinite(g_cov)))
    assert bool(jnp.all(jnp.isfinite(g_mu)))


def test_normalize_by_noise_with_another_pooling_weight_changes_the_features():
    """normalize_by_noise keeps pooling_weight, so a 2AFC reduction differs from the oddity features."""
    rng = np.random.default_rng(7)
    Sr, Sc, mr, mc = _trial(rng)
    odd = oddity_features(Sr, Sc, mr, mc)
    afc = align_displacement_2d(
        *normalize_by_noise(Sr, Sc, mr, mc, pooling_weight=TWOAFC_W)
    )
    assert not jnp.allclose(odd, afc, atol=1e-6)


def test_oddity_features_refuse_3d_stimuli():
    """A 3-D trial would otherwise return 4 numbers, mixing a 3-D norm with a 2-D rotation."""
    Sr = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    Sc = jnp.diag(jnp.array([0.5, 1.0, 4.0]))
    mr, mc = jnp.zeros(3), jnp.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="input_dim=2, got input_dim=3"):
        oddity_features(Sr, Sc, mr, mc)


def test_align_displacement_2d_composes_with_normalize_by_noise():
    """oddity_features must be exactly normalize_by_noise + align_displacement_2d."""
    rng = np.random.default_rng(8)
    Sr, Sc, mr, mc = _trial(rng)
    A, d = normalize_by_noise(Sr, Sc, mr, mc, pooling_weight=ODDITY_W)
    assert jnp.allclose(
        align_displacement_2d(A, d), oddity_features(Sr, Sc, mr, mc), atol=1e-7
    )


# --------------------------------------------------------------------------
# agreement with the reference implementation
# --------------------------------------------------------------------------

# Printed at full precision from the reference `whiten_inputs`, which runs in
# float64. To regenerate:
#
#     from nn_inputs import whiten_inputs
#     whiten_inputs(jnp.array([Sr]), jnp.array([Sc]), jnp.array([mr]), jnp.array([mc]))
#
_REF_CASES = [
    (
        [[1.0, 0.2], [0.2, 0.5]],
        [[0.6, -0.1], [-0.1, 0.4]],
        [0.0, 0.0],
        [0.8, 0.4],
        [
            0.9728714577165046,
            0.9428571428571427,
            -0.02274620967575469,
            1.237424547283702,
        ],
    ),
    (
        [[1.0, 0.8], [0.8, 1.0]],
        [[0.4, 0.0], [0.0, 0.9]],
        [0.0, 0.0],
        [1.0, 0.5],
        [
            1.1381803659589922,
            0.710526315789474,
            -0.29356611300456403,
            1.157655502392345,
        ],
    ),
    (
        # Covariances at psyphy's own scale, where the normalization is worse conditioned.
        [[0.05, 0.01], [0.01, 0.02]],
        [[0.03, 0.0], [0.0, 0.04]],
        [0.2, 0.1],
        [0.35, 0.18],
        [
            0.8034923770640265,
            0.7388475836431225,
            0.13550185873605952,
            1.1211524163568771,
        ],
    ),
]


@pytest.mark.parametrize("Sr,Sc,mr,mc,expected", _REF_CASES)
def test_matches_reference_implementation(Sr, Sc, mr, mc, expected):
    """Agreement with the reference implementation the weights were trained on.

    atol 1e-5: the reference ran in float64 and psyphy runs float32. Measured end
    to end, the gap in p(correct) is about 2e-7.
    """
    got = oddity_features(
        jnp.asarray(Sr), jnp.asarray(Sc), jnp.asarray(mr), jnp.asarray(mc)
    )
    assert jnp.allclose(got, jnp.asarray(expected), atol=1e-5), (
        f"got {np.asarray(got)}, reference {expected}"
    )
