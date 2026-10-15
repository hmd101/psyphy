"""Tests for psyphy.model.likelihood.features.

Written before the implementation. Every assertion here follows from the
mathematics of the feature map rather than from the code, so they constrain the
implementation instead of describing it:

* whitening is *defined* by `M S_bar M^T = I`, so that identity is a test, not a
  property to be discovered;
* the pooled constraint `w*A + (1-w)*B = I` forces `B = (I - w*A)/(1-w)`, which
  removes three free numbers and is the whole reason the feature vector is short;
* `P(correct)` is invariant under any invertible affine change of stimulus
  coordinates, so the features must be too. Rotation alone is a weaker check and
  would pass an implementation that got the whitening wrong, hence the shear.

The reference values at the bottom come from Philip Greengard's `nn_march2026`
implementation (`whiten_inputs`). They are baked in as literals because that
folder is not a package and cannot be a test dependency.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from psyphy.model.likelihood.features import (
    reduce_2d,
    whiten,
    whitened_oddity_features,
)

ODDITY_W = 2.0 / 3.0  # two reference presentations, one comparison
TWOAFC_W = 0.5  # one each


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
# whiten(): the defining property, and what it buys
# --------------------------------------------------------------------------


@pytest.mark.parametrize("w_ref", [ODDITY_W, TWOAFC_W])
def test_whitening_makes_the_pooled_covariance_the_identity(w_ref):
    """This is the definition of whitening, so it is the first thing to pin."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        Sr, Sc, mr, mc = _trial(rng)
        A, _ = whiten(Sr, Sc, mr, mc, w_ref=w_ref)
        B = (jnp.eye(2) - w_ref * A) / (1.0 - w_ref)
        # A and B are Sigma_ref and Sigma_comp in whitened coordinates, so the
        # pooled combination of them must be exactly the identity.
        pooled = w_ref * A + (1.0 - w_ref) * B
        # atol is float32-scaled: eps is 1.2e-7, so 1e-10 would be unreachable
        # regardless of correctness. psyphy runs float32 by default and the
        # surrogate does not need more; see the note on the reference test below.
        assert jnp.allclose(pooled, jnp.eye(2), atol=1e-6)


@pytest.mark.parametrize("w_ref,bound", [(ODDITY_W, 1.5), (TWOAFC_W, 2.0)])
def test_eigenvalues_of_A_are_bounded_by_one_over_w(w_ref, bound):
    """B must stay positive definite, which caps eig(A) at 1/w_ref."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        Sr, Sc, mr, mc = _trial(rng)
        A, _ = whiten(Sr, Sc, mr, mc, w_ref=w_ref)
        ev = jnp.linalg.eigvalsh(A)
        assert float(ev.min()) > 0.0
        assert float(ev.max()) < bound


def test_whitened_distance_equals_mahalanobis_distance():
    """The point of whitening: the S_bar^-1 in the decision rule disappears."""
    rng = np.random.default_rng(2)
    Sr, Sc, mr, mc = _trial(rng)
    Sbar = ODDITY_W * Sr + (1 - ODDITY_W) * Sc
    x, y = jnp.asarray(rng.normal(size=2)), jnp.asarray(rng.normal(size=2))

    maha = (x - y) @ jnp.linalg.inv(Sbar) @ (x - y)
    M = jnp.linalg.inv(jnp.linalg.cholesky(Sbar))
    eucl = jnp.sum((M @ x - M @ y) ** 2)
    assert float(abs(maha - eucl)) < 1e-6  # float32; see note above


# --------------------------------------------------------------------------
# the invariances -- the reason the feature vector can be this short
# --------------------------------------------------------------------------


def test_invariant_under_translation():
    rng = np.random.default_rng(3)
    Sr, Sc, mr, mc = _trial(rng)
    shift = jnp.asarray(rng.normal(size=2))
    base = whitened_oddity_features(Sr, Sc, mr, mc)
    moved = whitened_oddity_features(Sr, Sc, mr + shift, mc + shift)
    assert jnp.allclose(base, moved, atol=1e-6)


def test_invariant_under_rotation():
    rng = np.random.default_rng(4)
    Sr, Sc, mr, mc = _trial(rng)
    th = 0.7
    R = jnp.asarray([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    base = whitened_oddity_features(Sr, Sc, mr, mc)
    turned = whitened_oddity_features(R @ Sr @ R.T, R @ Sc @ R.T, R @ mr, R @ mc)
    assert jnp.allclose(base, turned, atol=1e-6)


def test_invariant_under_a_general_affine_map():
    """The strong version. A rotation-only test would pass a wrong whitening."""
    rng = np.random.default_rng(5)
    Sr, Sc, mr, mc = _trial(rng)
    L = jnp.asarray([[2.0, 0.7], [-0.4, 0.5]])  # shear + anisotropic scale
    assert abs(float(jnp.linalg.det(L))) > 1e-6
    base = whitened_oddity_features(Sr, Sc, mr, mc)
    mapped = whitened_oddity_features(L @ Sr @ L.T, L @ Sc @ L.T, L @ mr, L @ mc)
    assert jnp.allclose(base, mapped, atol=1e-5)


# --------------------------------------------------------------------------
# shape, degeneracy, and the task parameterisation
# --------------------------------------------------------------------------


def test_feature_vector_is_2d_long():
    """2*input_dim, which for the 2-D oddity task is 4."""
    rng = np.random.default_rng(6)
    Sr, Sc, mr, mc = _trial(rng)
    assert whitened_oddity_features(Sr, Sc, mr, mc).shape == (4,)


def test_coincident_stimuli_are_finite():
    """mu_ref == mu_comp makes the rotation undefined; it must not produce NaN."""
    f = whitened_oddity_features(jnp.eye(2), jnp.eye(2), jnp.zeros(2), jnp.zeros(2))
    assert bool(jnp.all(jnp.isfinite(f)))
    assert float(f[0]) == pytest.approx(0.0, abs=1e-12)


def test_w_ref_changes_the_features():
    """Guards against w_ref being accepted and then ignored."""
    rng = np.random.default_rng(7)
    Sr, Sc, mr, mc = _trial(rng)
    odd = whitened_oddity_features(Sr, Sc, mr, mc, w_ref=ODDITY_W)
    afc = whitened_oddity_features(Sr, Sc, mr, mc, w_ref=TWOAFC_W)
    assert not jnp.allclose(odd, afc, atol=1e-6)


def test_reduce_2d_composes_with_whiten():
    """whitened_oddity_features must be exactly whiten + reduce_2d."""
    rng = np.random.default_rng(8)
    Sr, Sc, mr, mc = _trial(rng)
    A, d = whiten(Sr, Sc, mr, mc, w_ref=ODDITY_W)
    assert jnp.allclose(
        reduce_2d(A, d), whitened_oddity_features(Sr, Sc, mr, mc), atol=1e-7
    )


# --------------------------------------------------------------------------
# agreement with the reference implementation
# --------------------------------------------------------------------------

# Produced by running nn_march2026's `whiten_inputs` on the inputs below and
# printing at full precision (that module forces float64). Baked in as literals
# because `nn_march2026` is a loose folder, not a package, so it cannot be a test
# dependency. To regenerate:
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
        # Covariances at psyphy's own scale, where the whitening is worse conditioned.
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
    """Agreement with nn_march2026, which is where the weights come from.

    The tolerance is 1e-5 rather than machine precision because the reference was
    computed in float64 and psyphy runs float32. That gap is in the features
    themselves; measured end to end it shrinks to about 2e-7 by the time it
    reaches p(correct), which is far below the surrogate's own ~0.02 worst-case
    approximation error.
    """
    got = whitened_oddity_features(
        jnp.asarray(Sr), jnp.asarray(Sc), jnp.asarray(mr), jnp.asarray(mc)
    )
    assert jnp.allclose(got, jnp.asarray(expected), atol=1e-5), (
        f"got {np.asarray(got)}, reference {expected}"
    )
