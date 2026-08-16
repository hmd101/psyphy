"""Tests for :mod:`psyphy.data.published.hong2025`.

Two clearly separated halves:

1. **Parsing tests** run everywhere. They use small synthetic CSVs written to
   ``tmp_path`` that mimic the published layout, so CI needs no network and no
   third-party data.

2. **The external-validity test** is skipped unless the real OSF data has been
   downloaded. It is the reason this module exists: it asserts that psyphy's
   covariance construction reproduces the field Hong et al. published, given
   their weights. Run ``hong2025.fetch(1, noise_ellipses=True)`` to enable it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from psyphy.data.dataset import TrialData
from psyphy.data.published import hong2025
from psyphy.model.covariance_field import WPPMCovarianceField


@pytest.fixture
def x64():
    """Enable float64 for the duration of a single test, then restore.

    Comparing against the published covariances needs float64 -- in float32 the
    agreement floors around 1e-7 rather than resolving the ~1e-9 that is
    actually there.

    This is deliberately a scoped context manager and *not* a module-level
    ``jax.config.update``. pytest imports every test module during collection,
    before running any test, so a module-level flag flip changes precision for
    the entire session. Other modules here assert behaviour that only holds in
    float32 (e.g. ``test_mc_likelihood.py::test_gradients_are_finite_normal_case``),
    and would fail depending on collection order.
    """
    with enable_x64():
        yield

# ----------------------------------------------------------------------
# Synthetic fixtures mimicking the published CSV layout
# ----------------------------------------------------------------------

_TRIALS_CSV = """TrialType,xref,x1,y
AEPsych_0_Sobol_0,"0.10,-0.20","0.15,-0.25",1
AEPsych_1_adaptive_0,"-0.30,0.40","-0.35,0.45",0
AEPsych_2_adaptive_1,"0.50,0.60","0.55,0.65",1
MOCS_0_cond_0_level_0_trial_0,"0.00,0.00","0.05,0.05",1
MOCS_0_cond_0_level_0_trial_1,"0.00,0.00","0.05,0.05",0
"""

_SIGMA_CSV = """grid_ref,Sigmas_thres_grid_org,Sigmas_thres_grid_btst1_rank0
"-0.70000000,-0.70000000","[[0.008,0.003],[0.003,0.005]]","[[0.009,0.004],[0.004,0.006]]"
"0.00000000,0.00000000","[[0.001,0.000],[0.000,0.002]]","[[0.002,0.001],[0.001,0.003]]"
"""


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


def _weights_csv(shape=(2, 2, 2, 3), *, column: str = "W_org") -> str:
    """A dense weights table over `shape`, values enumerated so gaps are visible."""
    lines = [f'"i,j,k,l",{column},W_btst0_rank0']
    for n, idx in enumerate(np.ndindex(*shape)):
        key = ",".join(str(i) for i in idx)
        lines.append(f'"{key}",{n * 0.001:.6f},{n * 0.002:.6f}')
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# load_trials
# ----------------------------------------------------------------------


class TestLoadTrials:
    def test_returns_trial_data_with_expected_shapes(self, tmp_path):
        data = hong2025.load_trials(_write(tmp_path, "t.csv", _TRIALS_CSV))
        assert isinstance(data, TrialData)
        # 3 AEPsych rows kept by default; K=2 (ref, comp) and d=2.
        assert data.stimuli.shape == (3, 2, 2)
        assert data.responses.shape == (3, 1)
        assert data.stimulus_names == ("ref", "comp")

    def test_default_keeps_only_aepsych_trials(self, tmp_path):
        """The published fit excluded MOCS rows; the default must too."""
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        assert hong2025.load_trials(path).num_trials == 3
        assert hong2025.load_trials(path, trial_types=None).num_trials == 5
        assert hong2025.load_trials(path, trial_types=("MOCS",)).num_trials == 2

    def test_named_stimulus_slots_map_to_ref_and_comparison(self, tmp_path):
        data = hong2025.load_trials(_write(tmp_path, "t.csv", _TRIALS_CSV))
        np.testing.assert_allclose(data.stimulus("ref")[0], [0.10, -0.20])
        np.testing.assert_allclose(data.stimulus("comp")[0], [0.15, -0.25])

    def test_responses_preserve_row_order(self, tmp_path):
        data = hong2025.load_trials(_write(tmp_path, "t.csv", _TRIALS_CSV))
        np.testing.assert_array_equal(np.asarray(data.responses)[:, 0], [1, 0, 1])

    def test_subsampling_is_reproducible_given_seed(self, tmp_path):
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        a = hong2025.load_trials(path, max_trials=2, seed=7)
        b = hong2025.load_trials(path, max_trials=2, seed=7)
        assert a.num_trials == 2
        np.testing.assert_array_equal(np.asarray(a.stimuli), np.asarray(b.stimuli))

    def test_max_trials_above_available_is_a_noop(self, tmp_path):
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        assert hong2025.load_trials(path, max_trials=99).num_trials == 3

    def test_missing_column_raises_naming_the_column(self, tmp_path):
        path = _write(tmp_path, "bad.csv", 'TrialType,xref,x1\nA,"0,0","0,0"\n')
        with pytest.raises(ValueError, match="missing expected column"):
            hong2025.load_trials(path)

    def test_no_matching_trial_type_raises(self, tmp_path):
        path = _write(tmp_path, "t.csv", _TRIALS_CSV)
        with pytest.raises(ValueError, match="No trials matched"):
            hong2025.load_trials(path, trial_types=("Nonexistent",))


# ----------------------------------------------------------------------
# load_reference_W
# ----------------------------------------------------------------------


class TestLoadReferenceW:
    def test_shape_is_inferred_from_index_keys(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        assert hong2025.load_reference_W(path).shape == (2, 2, 2, 3)

    def test_values_land_at_their_index(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        W = np.asarray(hong2025.load_reference_W(path))
        # Values were enumerated in np.ndindex order.
        for n, idx in enumerate(np.ndindex(2, 2, 2, 3)):
            assert W[idx] == pytest.approx(n * 0.001)

    def test_alternate_column_can_be_selected(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        main = np.asarray(hong2025.load_reference_W(path))
        btst = np.asarray(hong2025.load_reference_W(path, column="W_btst0_rank0"))
        np.testing.assert_allclose(btst, 2.0 * main)

    def test_duplicate_index_keys_are_rejected(self, tmp_path):
        """A duplicated key plus a dropped one preserves both the row count and
        the per-axis maxima, so the density check alone cannot catch it."""
        rows = _weights_csv((2, 2, 2, 3)).splitlines()
        rows[5] = rows[1]  # duplicate the first index key onto another row
        path = _write(tmp_path, "w.csv", "\n".join(rows) + "\n")
        with pytest.raises(ValueError, match="duplicate index keys"):
            hong2025.load_reference_W(path)

    def test_sparse_table_is_rejected(self, tmp_path):
        rows = _weights_csv((2, 2, 2, 3)).splitlines()
        del rows[3]  # drop a row; maxima unchanged, count now short
        path = _write(tmp_path, "w.csv", "\n".join(rows) + "\n")
        with pytest.raises(ValueError, match="not dense"):
            hong2025.load_reference_W(path)

    def test_unknown_column_raises(self, tmp_path):
        path = _write(tmp_path, "w.csv", _weights_csv((2, 2, 2, 3)))
        with pytest.raises(ValueError, match="not found"):
            hong2025.load_reference_W(path, column="W_nope")

    def test_wrong_index_column_raises(self, tmp_path):
        path = _write(tmp_path, "w.csv", 'idx,W_org\n"0,0,0,0",1.0\n')
        with pytest.raises(ValueError, match="expected first column"):
            hong2025.load_reference_W(path)


# ----------------------------------------------------------------------
# load_sigma_table
# ----------------------------------------------------------------------


class TestLoadSigmaTable:
    def test_parses_coordinates_and_matrices(self, tmp_path):
        coords, sigmas = hong2025.load_sigma_table(
            _write(tmp_path, "s.csv", _SIGMA_CSV)
        )
        assert coords.shape == (2, 2)
        assert sigmas.shape == (2, 2, 2)
        np.testing.assert_allclose(coords[0], [-0.7, -0.7])
        np.testing.assert_allclose(sigmas[0], [[0.008, 0.003], [0.003, 0.005]])

    def test_org_column_is_auto_detected(self, tmp_path):
        _, sigmas = hong2025.load_sigma_table(_write(tmp_path, "s.csv", _SIGMA_CSV))
        # Must pick *_org, not merely the first value column.
        np.testing.assert_allclose(sigmas[1], [[0.001, 0.0], [0.0, 0.002]])

    def test_explicit_column_overrides_auto_detection(self, tmp_path):
        _, sigmas = hong2025.load_sigma_table(
            _write(tmp_path, "s.csv", _SIGMA_CSV),
            value_column="Sigmas_thres_grid_btst1_rank0",
        )
        np.testing.assert_allclose(sigmas[0], [[0.009, 0.004], [0.004, 0.006]])

    def test_fine_grid_coordinate_column_is_accepted(self, tmp_path):
        text = _SIGMA_CSV.replace("grid_ref,", "grid_ref_fine,", 1).replace(
            "Sigmas_thres", "Sigmas_noise"
        )
        coords, _ = hong2025.load_sigma_table(_write(tmp_path, "n.csv", text))
        assert coords.shape == (2, 2)

    def test_unexpected_coordinate_column_raises(self, tmp_path):
        text = _SIGMA_CSV.replace("grid_ref,", "location,", 1)
        with pytest.raises(ValueError, match="expected first column"):
            hong2025.load_sigma_table(_write(tmp_path, "s.csv", text))


# ----------------------------------------------------------------------
# build_paper_model
# ----------------------------------------------------------------------


class TestBuildPaperModel:
    def test_matches_the_published_configuration(self):
        model = hong2025.build_paper_model(mc_samples=4)
        assert model.input_dim == 2
        assert model.basis_degree == 4  # paper's degree=5 counts T0..T4
        assert model.embedding_dim == 3  # input_dim + extra_dims
        assert model.diag_term == 0.0
        assert model.prior.variance_scale == pytest.approx(3e-4)
        assert model.prior.decay_rate == pytest.approx(0.4)

    def test_mc_controls_are_overridable_but_default_to_the_paper(self):
        assert hong2025.build_paper_model().likelihood.config.num_samples == 2000
        assert hong2025.build_paper_model().likelihood.config.bandwidth == 5e-3
        fast = hong2025.build_paper_model(mc_samples=8, bandwidth=1e-2)
        assert fast.likelihood.config.num_samples == 8
        assert fast.likelihood.config.bandwidth == 1e-2

    def test_prior_sample_has_the_published_weight_shape(self):
        model = hong2025.build_paper_model(mc_samples=4)
        params = model.init_params(jax.random.PRNGKey(0))
        # Same shape as the published Bestfit_W table, so the two are swappable.
        assert params["W"].shape == (5, 5, 2, 3)


# ----------------------------------------------------------------------
# External validity — requires the OSF download
# ----------------------------------------------------------------------

_SUB1 = hong2025.default_data_dir() / "sub1"
_WEIGHTS = _SUB1 / "Bestfit_W_sub1.csv"
_NOISE = _SUB1 / "Noise_ellipses_sub1.csv"

requires_osf_data = pytest.mark.skipif(
    not (_WEIGHTS.exists() and _NOISE.exists()),
    reason=(
        "Published data not downloaded. Run: python -c "
        "'from psyphy.data.published import hong2025; "
        "hong2025.fetch(1, noise_ellipses=True)'"
    ),
)


@requires_osf_data
def test_covariance_field_matches_published_sigma_noise(x64):
    """psyphy must reproduce the paper's covariance field from the paper's weights.

    This is the external-validity check for the whole WPPM covariance path:
    Chebyshev basis, weight layout, the einsum contraction, and
    ``Sigma = U U^T + diag_term I``. It is deterministic -- no optimizer, no
    Monte Carlo -- so any deviation is a real regression, not noise.

    The tolerance is set by the published CSV, which is rounded to 8 decimals;
    agreement is otherwise tighter than this.
    """
    W_org = hong2025.load_reference_W(_WEIGHTS)
    assert W_org.dtype == jnp.float64, "x64 must be enabled for this comparison"

    coords, sigma_published = hong2025.load_sigma_table(_NOISE)
    model = hong2025.build_paper_model(mc_samples=1)
    sigma_psyphy = np.asarray(WPPMCovarianceField(model, {"W": W_org})(coords))

    assert sigma_psyphy.shape == sigma_published.shape
    max_abs = float(np.abs(sigma_psyphy - sigma_published).max())
    assert max_abs < 1e-8, f"max |diff| = {max_abs:.3e} over {len(coords)} points"


@requires_osf_data
def test_published_trials_load_with_expected_counts():
    """Guards the trial-type split the published fit depends on."""
    trials_csv = _SUB1 / "trial_data_pooled_by_type_sub1.csv"
    if not trials_csv.exists():
        pytest.skip("trial data not downloaded")

    fitted = hong2025.load_trials(trials_csv)
    everything = hong2025.load_trials(trials_csv, trial_types=None)
    assert fitted.num_trials == 6000  # AEPsych only -- what the paper fitted
    assert everything.num_trials == 12000  # + held-out MOCS validation trials
    assert float(fitted.responses.mean()) == pytest.approx(0.7077, abs=1e-3)
