"""
Reproducing Hong et al. (2025) with psyphy
------------------------------------------

This  script fits *published human
data from Hong et al. (2025)* and compares against the authors' own published fit.

It runs in three stages, deliberately separated. The first two are given the
paper's own fitted weights and check what psyphy *computes* from them; only the
third asks psyphy to *fit* anything.

  Stage 1 -- covariance field, exact. Feed the paper's published weights into
      psyphy's covariance field and compare to their published covariances.
      Deterministic: no optimizer, no Monte Carlo. Seconds.

  Stage 2 -- threshold contours, reproducing the paper's Figure 2B. Invert the
      oddity task to turn that noise field into 66.7%-correct discrimination
      thresholds, and plot them in the paper's own monitor-calibrated colors.
      Monte Carlo but no optimizer. ~20 s on CPU at the default settings.

  Stage 3 -- refit. Start from a prior sample and fit psyphy's WPPM to the
      paper's trials, then compare the resulting field to theirs. At the
      paper's settings this is a GPU/cluster job; `--mode quick` is a
      seconds-long smoke test that does NOT reproduce anything.

The ordering is the point: stages 1-2 hold the model to account with the
optimizer removed from the picture, so if stage 3 disagrees you already know
the disagreement is the optimizer's and not the model's.

Usage
-----
    python hong2025_reproduction.py    # ~1 min CPU; stage 3 is only a smoke test
    python hong2025_reproduction.py --skip-refit    # stages 1-2 only, no fitting
    python hong2025_reproduction.py --mode full     # paper settings, GPU/cluster
    python hong2025_reproduction.py --skip-thresholds   # stages 1 and 3 only

Measured runtimes
-----------------
Keep these current when settings change -- they are quoted in the accompanying
markdown page. CPU figures are from an Apple Silicon laptop with JAX using
~12 cores; the GPU figure is a single CUDA device.

    Stage 1  covariance check, 10 609 pts, deterministic  CPU   seconds
    Stage 2  Figure 2B, 49 refs, n_theta=16,
             n_length=300, mc=500                         CPU   20-23 s
    Stage 2  at the paper's settings (n_length=1000,
             mc=2000): 13.4 s per reference point         CPU   ~11 min
    Stages 1+2 together                                   CPU   24-36 s
             (spread is CPU scheduling, not workload)
    Stage 3  quick smoke test, 500 trials, 20 steps,
             mc=50 -- the fit itself                      CPU   0.8 s
    Stage 3  full: 6 000 trials, 1 500 steps, mc=2000,
             3 restarts (3 x ~317 s)                      GPU   ~16 min

For reference, the paper's own SLURM header requests an H100 for 14 h -- but
that covers the main fit *plus* 120 bootstrap refits, not a single fit.

Data is downloaded on first run into ~/.cache/psyphy/ (override with
$PSYPHY_DATA_HOME). psyphy ships no data; see the accompanying markdown page.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# --8<-- [start:x64]
import jax

# float64 must be enabled before the first JAX array exists. The paper's code
# requires it throughout; without it the stage-1 comparison floors at ~1e-7
# instead of resolving the ~1e-9 agreement that is actually there.
jax.config.update("jax_enable_x64", True)
# --8<-- [end:x64]

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

# --8<-- [start:imports]
from psyphy.data.published import hong2025
from psyphy.inference import MAPOptimizer
from psyphy.model import WPPMCovarianceField
from psyphy.posterior import MAPPosterior, ThresholdConfig, WPPMPredictivePosterior

# --8<-- [end:imports]

PLOTS_DIR = Path(__file__).parent / "plots"

# --8<-- [start:modes]
# Stage-3 compute settings. "full" is the paper's own configuration; "quick"
# exists only to prove the pipeline runs -- it will NOT reproduce the paper.
MODES = {
    "quick": {"max_trials": 500, "mc_samples": 50, "steps": 20, "restarts": 1},
    "full": {"max_trials": None, "mc_samples": 2000, "steps": 1500, "restarts": 3},
}
# --8<-- [end:modes]

# --8<-- [start:threshold_settings]
# Stage-2 (threshold inversion) settings.
#
# The paper uses n_theta=16, n_length=1000, mc_samples=2000, which costs ~13 s
# per reference point -- about 11 CPU minutes for the whole 7x7 grid. The
# reduced settings below reproduce the published semi-axes to a median 2.18 %
# (max 10.78 %) in 20-23 s of CPU wall clock, measured over the full 49-point
# grid. That is the right trade for a tutorial; raise them toward the paper's
# for publication figures, where the residual shrinks and the cost grows.
THRESHOLD_MC_SAMPLES = 500
THRESHOLD_CONFIG = ThresholdConfig(n_theta=16, n_length=300)
# --8<-- [end:threshold_settings]


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
# --8<-- [start:metrics]
def _sqrtm_psd(M: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix."""
    vals, vecs = np.linalg.eigh(M)
    return vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def normalized_bures_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Normalized Bures Similarity between two PD matrices (1.0 == identical).

    The similarity measure Hong et al. use to rank bootstrap fits, reproduced
    here so the numbers are directly comparable to the paper's.
    """
    sa = _sqrtm_psd(A)
    inner = _sqrtm_psd(sa @ B @ sa)
    return float(np.trace(inner) / np.sqrt(np.trace(A) * np.trace(B)))


def compare_fields(Sigma_fit: np.ndarray, Sigma_ref: np.ndarray) -> dict[str, float]:
    """Compare two stacks of 2x2 covariances, shape (M, 2, 2).

    Compares Sigma  (never W ). U -> U Q for orthogonal Q leaves Sigma = U U^T
    unchanged and the prior is isotropic in the embedding axis, so the weights
    are not identifiable while the covariance field is.
    """
    rel_frob = np.linalg.norm(Sigma_fit - Sigma_ref, axis=(1, 2)) / np.linalg.norm(
        Sigma_ref, axis=(1, 2)
    )
    # Ellipse area is proportional to sqrt(det Sigma).
    area_ratio = np.sqrt(np.linalg.det(Sigma_fit) / np.linalg.det(Sigma_ref))

    def major_axis_angle(S: np.ndarray) -> np.ndarray:
        vecs = np.linalg.eigh(S)[1][..., -1]  # eigh returns ascending eigenvalues
        return np.arctan2(vecs[..., 1], vecs[..., 0])

    # Ellipse axes are undirected, so angle error lives on [0, 90] degrees.
    d_ang = np.degrees(major_axis_angle(Sigma_fit) - major_axis_angle(Sigma_ref))
    d_ang = np.abs((d_ang + 90.0) % 180.0 - 90.0)

    nbs = np.array(
        [
            normalized_bures_similarity(a, b)
            for a, b in zip(Sigma_ref, Sigma_fit, strict=True)
        ]
    )
    return {
        "rel_frobenius_median": float(np.median(rel_frob)),
        "rel_frobenius_max": float(rel_frob.max()),
        "area_ratio_median": float(np.median(area_ratio)),
        "angle_err_deg_median": float(np.median(d_ang)),
        "nbs_median": float(np.median(nbs)),
        "nbs_min": float(nbs.min()),
    }


# --8<-- [end:metrics]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
_THETA = np.linspace(0, 2 * np.pi, 100)
_UNIT_CIRCLE = np.vstack([np.cos(_THETA), np.sin(_THETA)])


def _ellipse_segments(centers, covs, scale):
    """Batched covariances -> polyline segments, plus a positive-definite mask.

    One LineCollection instead of hundreds of Line2D artists, as in
    full_wppm_fit_example.py.
    """
    valid = np.all(np.linalg.eigvalsh(covs) > 0, axis=-1)
    segs = [
        (c[:, None] + scale * (np.linalg.cholesky(S) @ _UNIT_CIRCLE)).T
        for c, S, ok in zip(centers, covs, valid, strict=True)
        if ok
    ]
    return segs, valid


def ellipse_plot_scale(coords: np.ndarray, Sigma_ref: np.ndarray) -> float:
    """Magnification that makes ellipses visible without colliding.

    Thresholds are ~0.05 in W units against a grid spacing of ~0.23, so drawn at
    true size they are legible but tiny, while any fixed magnification would be
    wrong for a different grid. Size the median ellipse to a set fraction of the
    nearest-neighbour spacing instead. Purely cosmetic -- the same scale applies
    to both fields, so the comparison is unaffected.

    Uses a KD-tree rather than a full pairwise distance matrix: the matrix is
    Theta(M^2) in time and space, which is nothing at M=49 but ~1.8 GB at the
    M=10609 of the fine noise grid.
    """
    dists, _ = cKDTree(coords).query(coords, k=2)  # k=2: [0] is the point itself
    spacing = float(np.median(dists[:, 1]))
    typical_radius = float(np.median(np.sqrt(np.linalg.eigvalsh(Sigma_ref).mean(-1))))
    return 0.35 * spacing / typical_radius


def plot_comparison(coords, Sigma_fit, Sigma_ref, out_path, title, scale):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for covs, color, label in [
        (Sigma_ref, "black", "published Σ_noise (Hong et al. 2025)"),
        (Sigma_fit, "crimson", "psyphy Σ_noise (MAP fit)"),
    ]:
        segs, valid = _ellipse_segments(coords, covs, scale)
        ax.add_collection(LineCollection(segs, colors=color, linewidths=1.2, alpha=0.8))
        ax.plot([], [], color=color, lw=1.2, label=f"{label}")

    ax.scatter(coords[:, 0], coords[:, 1], c="gray", s=4, zorder=5)
    ticks = np.linspace(-0.7, 0.7, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Model Dimension 1")
    ax.set_ylabel("Model Dimension 2")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# --8<-- [start:plot_thresholds]
def plot_threshold_figure(coords, Sigma_psyphy, Sigma_published, out_path, scale, M):
    """Figure 2B: threshold contours, colored by reference stimulus.

    Each ellipse sits at its reference location and takes that location's own
    color, which is what makes the paper's version readable as a *color*
    figure rather than an abstract field of ellipses. ``M`` is the monitor
    calibration matrix; when it is None the plot falls back to neutral grey and
    says so in the legend, so the figure is never silently mislabelled.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)

    if M is not None:
        colors = hong2025.w2d_to_rgb(coords, M)
        color_note = "color = reference stimulus (monitor-calibrated)"
    else:
        colors = np.full((len(coords), 3), 0.45)
        color_note = "neutral grey — calibration matrix not downloaded"

    # Published contours first, as a dashed dark outline underneath, so the
    # comparison is visible where the two nearly coincide.
    segs_pub, _ = _ellipse_segments(coords, Sigma_published, scale)
    ax.add_collection(
        LineCollection(
            segs_pub, colors="black", linewidths=2.2, alpha=0.35, linestyles="--"
        )
    )

    segs_psy, valid = _ellipse_segments(coords, Sigma_psyphy, scale)
    ax.add_collection(LineCollection(segs_psy, colors=colors[valid], linewidths=1.6))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=14, zorder=5, edgecolors="none")

    ax.plot(
        [],
        [],
        color="black",
        lw=2.2,
        ls="--",
        alpha=0.5,
        label="published (Hong et al. 2025)",
    )
    ax.plot([], [], color="0.2", lw=1.6, label="psyphy (oddity inversion)")

    ticks = np.linspace(-0.7, 0.7, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-0.95, 0.95)
    ax.set_ylim(-0.95, 0.95)
    ax.set_aspect("equal")
    ax.set_xlabel("Model Dimension 1")
    ax.set_ylabel("Model Dimension 2")
    ax.set_title(
        " 66.7%-correct discrimination thresholds\n"
        f"Figure 2B in Hong et al. 2025 reproduced, subject 1 (CH)",#; {color_note}",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# --8<-- [end:plot_thresholds]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage1_exact_check(paths: dict[str, Path]) -> None:
    """Compare psyphy's covariance field to the published one, given the paper's W."""
    print("\n=== Stage 1: psyphy's Sigma(W_org) vs the published field ===")
    if "noise_ellipses" not in paths:
        print("  skipped (re-run with --noise-ellipses to download the 68 MB table)")
        return

    # --8<-- [start:stage1]
    W_org = hong2025.load_reference_W(paths["weights"])  # (5, 5, 2, 3)
    coords, sigma_published = hong2025.load_sigma_table(paths["noise_ellipses"])

    model = hong2025.build_paper_model(mc_samples=1)  # MC unused: no likelihood here
    field = WPPMCovarianceField(model, {"W": W_org})
    sigma_psyphy = np.asarray(field(jnp.asarray(coords)))

    max_abs = float(np.abs(sigma_psyphy - sigma_published).max())
    # --8<-- [end:stage1]

    print(f"  grid points  : {len(coords)}")
    print(f"  max |diff|   : {max_abs:.3e}")
    print(
        f"  mean |diff|  : {float(np.abs(sigma_psyphy - sigma_published).mean()):.3e}"
    )
    verdict = "PASS" if max_abs < 1e-8 else "FAIL"
    print(f"  {verdict} — the published CSV is rounded to 8 decimals, so this is")
    print("  agreement to the precision the published file can express.")


def stage2_thresholds(paths: dict[str, Path]) -> None:
    """Reproduce Figure 2B: threshold contours from the paper's own weights."""
    print("\n=== Stage 2: threshold contours (Figure 2B) from W_org ===")

    # --8<-- [start:thresholds]
    W_org = hong2025.load_reference_W(paths["weights"])
    coords, thres_published = hong2025.load_sigma_table(paths["thres_ellipses"])

    # No fitting here -- the paper's weights go straight in, so this isolates
    # the oddity inversion from the optimizer entirely.
    model = hong2025.build_paper_model(mc_samples=THRESHOLD_MC_SAMPLES)
    posterior = MAPPosterior({"W": W_org}, model)

    predictive = WPPMPredictivePosterior(
        posterior,
        jnp.asarray(coords),  # bare reference points, not (ref, comparison) pairs
        n_samples=1,  # MAPPosterior is a point estimate: 1 draw is all there is
        threshold_pred=True,
        threshold_config=THRESHOLD_CONFIG,
    )
    thres_psyphy = np.asarray(predictive.mean)  # (49, 2, 2)
    # --8<-- [end:thresholds]

    # --8<-- [start:threshold_error]
    # Compare semi-axis lengths: sqrt of the covariance eigenvalues.
    got = np.sqrt(np.linalg.eigvalsh(thres_psyphy))
    want = np.sqrt(np.linalg.eigvalsh(thres_published))
    rel_err = np.abs(got - want) / want
    # --8<-- [end:threshold_error]

    print(f"  reference points : {len(coords)}")
    print(
        f"  semi-axis error  : median {np.median(rel_err) * 100:.2f} %, "
        f"max {rel_err.max() * 100:.2f} %"
    )
    print(
        f"  settings         : n_theta={THRESHOLD_CONFIG.n_theta}, "
        f"n_length={THRESHOLD_CONFIG.n_length}, mc={THRESHOLD_MC_SAMPLES}"
    )

    # --8<-- [start:colors]
    # The paper colors each ellipse by its reference stimulus, via a monitor
    # calibration matrix published alongside the data. Optional: the figure
    # falls back to neutral grey when it has not been downloaded.
    try:
        M = hong2025.load_calibration_matrix(hong2025.fetch_calibration_matrix())
    except Exception as exc:  # network, or OSF layout change
        print(f"  color calibration unavailable ({exc}); plotting in grey")
        M = None
    # --8<-- [end:colors]

    plot_threshold_figure(
        coords,
        thres_psyphy,
        thres_published,
        PLOTS_DIR / "hong2025_thresholds.png",
        scale=ellipse_plot_scale(coords, thres_published),
        M=M,
    )


def stage3_refit(paths: dict[str, Path], cfg: dict, mode: str, seed: int) -> None:
    """Fit psyphy's WPPM to the published trials and compare fields."""
    print(f"\n=== Stage 3: refit from a prior sample (mode={mode}) ===")

    # --8<-- [start:load]
    # Only the AEPsych trials were used for the published fit; the MOCS trials
    # in the same file are held-out validation. This is the default.
    data = hong2025.load_trials(
        paths["trials"], max_trials=cfg["max_trials"], seed=seed
    )
    # --8<-- [end:load]
    print(f"  trials: {data.num_trials} (p_correct={float(data.responses.mean()):.4f})")

    # --8<-- [start:fit]
    model = hong2025.build_paper_model(mc_samples=cfg["mc_samples"])
    optimizer = MAPOptimizer(
        steps=cfg["steps"],
        learning_rate=hong2025.PAPER_HYPERPARAMS["learning_rate"],
        momentum=hong2025.PAPER_HYPERPARAMS["momentum"],
        # The paper's optimizer scales the objective per trial and does no
        # gradient clipping; both are needed for its learning rate to transfer.
        reduction="mean",
        max_grad_norm=None,
    )

    # The paper fits from 3 random initializations and keeps the lowest final
    # objective, guarding against a bad local optimum.
    best = None
    for r in range(cfg["restarts"]):
        t0 = time.time()
        init = model.init_params(jax.random.PRNGKey(seed + 1000 * r))
        posterior = optimizer.fit(model, data, init_params=init)
        _, losses = optimizer.get_history()
        print(
            f"  restart {r}: loss {losses[0]:.5f} -> {losses[-1]:.5f}"
            f"  ({time.time() - t0:.1f}s)"
        )
        if best is None or losses[-1] < best[1]:
            best = (posterior.params, losses[-1], list(losses))
    params, _, loss_hist = best
    # --8<-- [end:fit]

    # --8<-- [start:compare]
    # Grid coordinates come from the published table, so there is no meshgrid
    # ordering convention to get wrong.
    coords, _ = hong2025.load_sigma_table(paths["thres_ellipses"])
    W_org = hong2025.load_reference_W(paths["weights"])  # original best fit Weights

    Sigma_ref = np.asarray(
        WPPMCovarianceField(model, {"W": W_org})(jnp.asarray(coords))
    )
    Sigma_fit = np.asarray(WPPMCovarianceField(model, params)(jnp.asarray(coords)))
    metrics = compare_fields(Sigma_fit, Sigma_ref)
    # --8<-- [end:compare]

    print(f"\n  --- fitted vs published field, {len(coords)} grid points ---")
    for key, value in metrics.items():
        print(f"    {key:24s} {value: .4f}")

    scale = ellipse_plot_scale(coords, Sigma_ref)
    plot_comparison(
        coords,
        Sigma_fit,
        Sigma_ref,
        PLOTS_DIR / f"hong2025_{mode}_ellipses.png",
        f"Σ_noise(x) — psyphy MAP fit vs Hong et al. 2025 (subj 1 CH)\n"
        f" N={data.num_trials}, mc={cfg['mc_samples']}, steps={cfg['steps']}",
        # f"  (not the published threshold contours)"
        # f"— ellipses magnified {scale:.1f}x"
        scale=scale,
    )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(loss_hist, color="#4444aa")
    ax.set_xlabel("Step")
    ax.set_ylabel("Neg log posterior (per trial)")
    ax.set_title(f"Learning curve — mode={mode}, N={data.num_trials}", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"hong2025_{mode}_learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved hong2025_{mode}_learning_curve.png")

    if mode == "quick":
        print(
            "\n  Quick mode does NOT reproduce the paper: after a few steps the fit\n"
            "  is still close to its prior. Use --mode full on a GPU."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), default="quick")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-ellipses",
        action="store_true",
        default=True,
        help="download the 68 MB noise-covariance table needed by stage 1",
    )
    parser.add_argument(
        "--no-noise-ellipses",
        dest="noise_ellipses",
        action="store_false",
        help="skip stage 1 and the 68 MB download",
    )
    parser.add_argument(
        "--skip-thresholds",
        action="store_true",
        help="skip stage 2 (the Figure 2B threshold inversion, ~20 s on CPU)",
    )
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help=(
            "skip stage 3. Stages 1-2 reproduce published results on CPU; the "
            "stage-3 refit at paper settings is a GPU/cluster job."
        ),
    )
    args = parser.parse_args()

    print(f"device: {jax.devices()[0]}   x64: {jax.config.read('jax_enable_x64')}")

    # --8<-- [start:fetch]
    paths = hong2025.fetch(subject=args.subject, noise_ellipses=args.noise_ellipses)
    # --8<-- [end:fetch]

    stage1_exact_check(paths)
    if args.skip_thresholds:
        print("\n=== Stage 2: skipped (--skip-thresholds) ===")
    else:
        stage2_thresholds(paths)
    if args.skip_refit:
        print("\n=== Stage 3: skipped (--skip-refit) ===")
    else:
        if args.mode == "full" and jax.devices()[0].platform == "cpu":
            print(
                "\n  WARNING: --mode full on CPU. The paper's settings are a "
                "GPU/cluster job\n  (~16 min on one GPU; far longer here). "
                "Ctrl-C and pass --mode quick to smoke-test."
            )
        stage3_refit(paths, MODES[args.mode], args.mode, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
