"""Tests for psyphy.model.likelihood.neural: the surrogate container and its file format.

Written before the implementation, against the schema specified in

The point of nearly every test here is a *silent* failure. A surrogate that is
handed the wrong weights does not crash and does not produce NaN: it returns a
plausible probability in (0, 1) that is simply wrong. Measured, loading oddity
weights but building features with the 2AFC pooling weight gives 0.3469 where the
right answer is 0.3627. Nothing in the arrays distinguishes the two cases, because
a 2-D oddity and a 2-D 2AFC network have identical shapes. Only declared metadata,
checked at load, can tell them apart -- which is what these tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from psyphy.model.likelihood.neural import (  # noqa: E402
    SCHEMA_VERSION,
    NeuralSurrogate,
    mlp_forward,
    save_surrogate_npz,
)

ODDITY_REQUIRE = {
    "task": "oddity",
    "input_dim": 2,
    "feature_map": "whitened_rotation_reduced",
    "feature_map_version": 1,
    "w_ref": 2.0 / 3.0,
}


# --------------------------------------------------------------------------
# fixtures: a small, well-formed surrogate written to a temp file
# --------------------------------------------------------------------------


def _toy_params(rng, widths=(4, 8, 8, 1)):
    out = []
    for i in range(len(widths) - 1):
        out.append(
            (
                jnp.asarray(rng.normal(size=(widths[i], widths[i + 1])) * 0.3),
                jnp.asarray(rng.normal(size=(widths[i + 1],)) * 0.1),
            )
        )
    return out


@pytest.fixture
def toy_file(tmp_path):
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    path = tmp_path / "oddity_2d_toy.npz"
    save_surrogate_npz(
        path,
        params=params,
        input_mean=jnp.zeros(4),
        input_std=jnp.ones(4),
        meta=dict(ODDITY_REQUIRE, n_features=4),
    )
    return path


# --------------------------------------------------------------------------
# the forward pass
# --------------------------------------------------------------------------


def test_mlp_forward_shape_and_range():
    rng = np.random.default_rng(1)
    params = _toy_params(rng)
    x = jnp.asarray(rng.normal(size=(7, 4)))
    p = mlp_forward(params, x)
    assert p.shape == (7,), "one probability per row, with the trailing axis squeezed"
    assert bool(jnp.all((p > 0) & (p < 1))), "sigmoid output must be a probability"


def test_mlp_forward_is_batch_polymorphic():
    """A single feature vector and a batch must both work; loglik vmaps this."""
    rng = np.random.default_rng(2)
    params = _toy_params(rng)
    x = jnp.asarray(rng.normal(size=4))
    assert mlp_forward(params, x).shape == ()
    assert mlp_forward(params, x[None, :]).shape == (1,)


def test_mlp_forward_reads_depth_from_the_params_list():
    """Depth is whatever the file holds; 3 layers is not assumed."""
    rng = np.random.default_rng(3)
    for widths in [(4, 5, 1), (4, 5, 6, 1), (4, 5, 6, 7, 1)]:
        p = mlp_forward(_toy_params(rng, widths), jnp.zeros(4))
        assert jnp.isfinite(p)


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_round_trip(toy_file):
    s = NeuralSurrogate.from_npz(toy_file, require=ODDITY_REQUIRE)
    assert s.meta["task"] == "oddity"
    assert s.meta["input_dim"] == 2
    assert s.meta["psyphy_surrogate_version"] == SCHEMA_VERSION
    out = s(jnp.zeros(4))
    assert out.shape == () and 0.0 < float(out) < 1.0


def test_standardization_is_applied(toy_file, tmp_path):
    """The surrogate must standardize before the forward pass.

    Weights and standardization statistics are stored together precisely because
    using one without the other is silent; this pins that they are used together.
    """
    rng = np.random.default_rng(4)
    params = _toy_params(rng)
    shifted = tmp_path / "shifted.npz"
    save_surrogate_npz(
        shifted,
        params=params,
        input_mean=jnp.full(4, 3.0),
        input_std=jnp.full(4, 2.0),
        meta=dict(ODDITY_REQUIRE, n_features=4),
    )
    plain = tmp_path / "plain.npz"
    save_surrogate_npz(
        plain,
        params=params,
        input_mean=jnp.zeros(4),
        input_std=jnp.ones(4),
        meta=dict(ODDITY_REQUIRE, n_features=4),
    )
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    a = NeuralSurrogate.from_npz(shifted, require=ODDITY_REQUIRE)(x)
    b = NeuralSurrogate.from_npz(plain, require=ODDITY_REQUIRE)(x)
    assert abs(float(a) - float(b)) > 1e-6, "input_mean/std were ignored"


def test_weights_are_cast_to_the_ambient_dtype(toy_file):
    """Shipped weights are float64; psyphy runs float32 and must not be forced up."""
    s = NeuralSurrogate.from_npz(toy_file, require=ODDITY_REQUIRE)
    want = jnp.zeros(()).dtype
    assert all(W.dtype == want and b.dtype == want for W, b in s.params)
    assert s.input_mean.dtype == want


# --------------------------------------------------------------------------
# verification: every one of these is a silent failure if unchecked
# --------------------------------------------------------------------------


def test_refuses_a_file_with_no_metadata(tmp_path):
    """Pre-schema files, e.g. the reference drop's originals, must not load."""
    path = tmp_path / "bare.npz"
    rng = np.random.default_rng(5)
    params = _toy_params(rng)
    np.savez(
        path,
        **{f"W_{i}": np.asarray(W) for i, (W, _) in enumerate(params)},
        **{f"b_{i}": np.asarray(b) for i, (_, b) in enumerate(params)},
        n_layers=np.array(len(params)),
        input_mean=np.zeros(4),
        input_std=np.ones(4),
    )
    with pytest.raises(ValueError, match="task"):
        NeuralSurrogate.from_npz(path, require=ODDITY_REQUIRE)


def test_refuses_the_wrong_task_and_names_both_sides(toy_file):
    """The headline case: identical shapes, different task."""
    with pytest.raises(ValueError) as e:
        NeuralSurrogate.from_npz(
            toy_file, require=dict(ODDITY_REQUIRE, task="2afc", w_ref=0.5)
        )
    msg = str(e.value)
    assert "oddity" in msg and "2afc" in msg, (
        f"the error must name what the file is AND what was wanted; got: {msg}"
    )


def test_refuses_the_wrong_dimension(toy_file):
    with pytest.raises(ValueError, match="input_dim"):
        NeuralSurrogate.from_npz(toy_file, require=dict(ODDITY_REQUIRE, input_dim=3))


def test_refuses_a_stale_feature_map_version(toy_file):
    """The nastiest case: same task, same shapes, feature definition changed."""
    with pytest.raises(ValueError, match="feature_map_version"):
        NeuralSurrogate.from_npz(
            toy_file, require=dict(ODDITY_REQUIRE, feature_map_version=2)
        )


def test_w_ref_is_compared_with_a_tolerance(toy_file):
    """2/3 is not exact in binary; a naive == would reject a correct file."""
    NeuralSurrogate.from_npz(
        toy_file, require=dict(ODDITY_REQUIRE, w_ref=float(np.float32(2.0 / 3.0)))
    )


def test_refuses_when_n_features_disagrees_with_the_weights(tmp_path):
    """Declared width vs the width the network actually accepts.

    This is checked against the weights rather than against any rule relating
    n_features to input_dim: how many features a task needs is a property of its
    feature map, and a different map may legitimately use a different count.
    """
    rng = np.random.default_rng(6)
    path = tmp_path / "inconsistent.npz"
    save_surrogate_npz(
        path,
        params=_toy_params(rng),  # W_0 is (4, 8), so the real width is 4
        input_mean=jnp.zeros(4),
        input_std=jnp.ones(4),
        meta=dict(ODDITY_REQUIRE, n_features=6),  # claims 6
    )
    with pytest.raises(ValueError, match="n_features"):
        NeuralSurrogate.from_npz(path, require=ODDITY_REQUIRE)


def test_refuses_a_future_schema_version(tmp_path):
    rng = np.random.default_rng(7)
    path = tmp_path / "future.npz"
    save_surrogate_npz(
        path,
        params=_toy_params(rng),
        input_mean=jnp.zeros(4),
        input_std=jnp.ones(4),
        meta=dict(ODDITY_REQUIRE, n_features=4),
        schema_version=SCHEMA_VERSION + 1,
    )
    with pytest.raises(ValueError, match="version"):
        NeuralSurrogate.from_npz(path, require=ODDITY_REQUIRE)


# --------------------------------------------------------------------------
# optional fields: absent means unknown, not unbounded
# --------------------------------------------------------------------------


def test_optional_fields_round_trip(tmp_path):
    rng = np.random.default_rng(8)
    path = tmp_path / "with_optional.npz"
    save_surrogate_npz(
        path,
        params=_toy_params(rng),
        input_mean=jnp.zeros(4),
        input_std=jnp.ones(4),
        meta=dict(
            ODDITY_REQUIRE,
            n_features=4,
            domain_lo=np.array([0.0, 0.2, -0.5, 0.2]),
            domain_hi=np.array([20.0, 1.5, 0.5, 1.5]),
            label_mc_samples=200_000,
            val_max_err=0.019,
        ),
    )
    s = NeuralSurrogate.from_npz(path, require=ODDITY_REQUIRE)
    assert s.meta["label_mc_samples"] == 200_000
    assert np.allclose(s.meta["domain_hi"], [20.0, 1.5, 0.5, 1.5])


def test_absent_optional_fields_are_absent_not_defaulted(toy_file):
    """'Unknown' must stay distinguishable from a guessed value.

    A surrogate extrapolates silently outside its training range, so a caller may
    want to warn about that -- but only for files that carry a measured range.
    Defaulting an absent box to something plausible would make the two cases
    indistinguishable.
    """
    s = NeuralSurrogate.from_npz(toy_file, require=ODDITY_REQUIRE)
    assert "domain_lo" not in s.meta
    assert s.domain is None, (
        "an unknown training domain must not become a fabricated box"
    )
