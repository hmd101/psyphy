"""Figures: posterior mean (Σ(E[W]) vs E[Σ]) and R-hat on W vs on Σ.

Uses psyphy.viz for ellipses, ArviZ for traces and R-hat distributions, and
matplotlib's default colours (C0, C1, ... and named "k"). One set of panels per
mode found in results/{quick,balanced}.npz (written by collect.py), plus a
stitched overview. No sampling happens here; this runs in seconds.

usage (from the repo root):
    python research/rhat_on_sigma/figures.py
"""

import os
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)
from psyphy.viz import plot_ellipses  # noqa: E402

SCALE = 1.2  # one magnification for every ellipse in every panel
Q = ["log_var1", "log_var2", "corr"]
W_DIMS = ["deg_x", "deg_y", "U_row", "U_col"]
modes = [m for m in ("quick", "balanced") if os.path.exists(f"{RESULTS}/{m}.npz")]
panels = {}

for m in modes:
    d = np.load(f"{RESULTS}/{m}.npz")
    C, S = d["W"].shape[:2]
    head = f"{m}: {C} chains × {S} draws, {d['refs'].shape[0]} ref points"

    # ArviZ DataTree with the raw W and the identifiable Σ quantities per ref.
    dt = az.from_dict(
        {"posterior": {"W": d["W"], "Sigma": d["feats"]}},
        dims={"W": W_DIMS, "Sigma": ["ref", "quantity"]},
        coords={"quantity": Q},
    )
    rhat = az.rhat(dt)

    # --- (a) posterior mean: psyphy.viz ---------------------------------------
    Sig = d["Sig"].reshape(C * S, *d["Sig"].shape[2:])
    size = lambda A: np.sqrt(np.linalg.det(A))  # noqa: E731  raw Σ, no plot scale
    med = np.median(size(Sig), axis=0)
    old, new = np.mean(size(d["Sig_EW"]) / med), np.mean(size(d["E_Sig"]) / med)
    fig, ax = plt.subplots(figsize=(6, 6.4))
    pick = np.random.default_rng(0).choice(C * S, size=min(40, C * S), replace=False)
    plot_ellipses(
        d["refs"], Sig[pick], ax=ax, scale=SCALE, colors="C7", linewidths=0.4, alpha=0.2
    )
    plot_ellipses(
        d["refs"],
        [d["truth"], d["E_Sig"], d["Sig_EW"]],
        ax=ax,
        scale=SCALE,
        colors=["k", "C0", "C7"],
        linestyles=["-", "-", "--"],
        linewidths=[1.2, 2.0, 1.5],
        labels=["ground truth", "E[Σ] (fix)", "Σ(E[W]) (old)"],
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    ax.set_title(
        f"{head}\nposterior mean ÷ median draw: old {old:.2f}, fix {new:.2f}\n"
        f"(all ellipses ×{SCALE}; light grey: 40 posterior draws)",
        fontsize=10,
    )
    # JPEG: many thin, semi-transparent outlines compress badly as PNG (same
    # reasoning as save_ellipse_figure in the NUTS example).
    panels[m, "mean"] = f"{FIGURES}/mean_{m}.jpg"
    fig.savefig(
        panels[m, "mean"], dpi=130, bbox_inches="tight", pil_kwargs={"quality": 88}
    )
    plt.close(fig)

    # --- (b) traces of the worst W coefficient and worst Σ quantity: ArviZ -----
    iw = np.unravel_index(int(np.argmax(rhat["W"].values)), rhat["W"].shape)
    iS = np.unravel_index(int(np.argmax(rhat["Sigma"].values)), rhat["Sigma"].shape)
    coords = dict(zip(W_DIMS, map(int, iw)))
    coords.update(ref=int(iS[0]), quantity=Q[iS[1]])
    pc = az.plot_trace(dt, var_names=["W", "Sigma"], coords=coords)
    fig = pc.viz["figure"].item()
    ax_w, ax_s = fig.axes[:2]
    ax_w.set_title(
        f"worst W coefficient W{tuple(map(int, iw))}\n"
        f"R-hat {float(rhat['W'].max()):.2f}",
        fontsize=10,
    )
    ax_s.set_title(
        f"worst Σ quantity: {Q[iS[1]]} at ref {int(iS[0])}\n"
        f"R-hat {float(rhat['Sigma'].max()):.2f}",
        fontsize=10,
    )
    for a in (ax_w, ax_s):
        a.set_xlabel("draw (post-warmup); one colour per chain")
    panels[m, "trace"] = f"{FIGURES}/trace_{m}.png"
    fig.savefig(panels[m, "trace"], dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- (c) distribution of R-hat over all quantities: ArviZ ------------------
    pc = az.plot_convergence_dist(
        dt, var_names=["W", "Sigma"], diagnostics=["rhat_rank"], grouped=False
    )
    fig = pc.viz["figure"].item()
    hi = max(float(rhat["W"].max()), float(rhat["Sigma"].max())) * 1.02
    for a, (name, var) in zip(
        fig.axes[:2],
        (
            ("W (54 raw coefficients)", "W"),
            ("Σ (log-variances, correlation per ref)", "Sigma"),
        ),
    ):
        v = rhat[var].values.ravel()
        a.set_xlim(0.99, hi)  # same x-range for both, so they can be compared
        a.set_title(
            f"{name}: median {np.median(v):.2f}, {np.mean(v < 1.01):.0%} below 1.01",
            fontsize=10,
        )
        a.set_ylabel("fraction of quantities")
    fig.axes[1].set_xlabel("R-hat (rank-normalized); dashed line = 1.01")
    fig.suptitle(head, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    panels[m, "rhat"] = f"{FIGURES}/rhat_{m}.png"
    fig.savefig(panels[m, "rhat"], dpi=130, bbox_inches="tight")
    plt.close(fig)

# --- stitched overview: rows = modes, columns = mean | traces | R-hat ---------
fig, axes = plt.subplots(
    len(modes),
    3,
    figsize=(22, 6.2 * len(modes)),
    squeeze=False,
    gridspec_kw={"width_ratios": [0.8, 1.6, 1.0], "wspace": 0.02, "hspace": 0.04},
)
for r, m in enumerate(modes):
    for c, k in enumerate(("mean", "trace", "rhat")):
        axes[r, c].imshow(plt.imread(panels[m, k]))
        axes[r, c].axis("off")
out = f"{FIGURES}/overview.jpg"
fig.savefig(out, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
print(
    "wrote",
    out,
    "and per-panel figures:",
    sorted(os.path.basename(p) for p in panels.values()),
)
