"""
Full WPPM Example: Fitting a Spatially-Varying Covariance Field
---------------------------------------------------------------

This example demonstrates how to use the full Wishart Process
Psychophysical Model (WPPM)to fit a spatially-varying covariance
field to synthetic 2D data. It visualizes the ground-truth
covariance field, the initial prior sample, and the fitted
field as ellipsoid contours.

"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Ensure local src is importable when running directly
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
import jax.random as jr

from psyphy.data.dataset import ResponseData
from psyphy.inference.map_optimizer import MAPOptimizer
from psyphy.model.noise import GaussianNoise
from psyphy.model.prior import Prior
from psyphy.model.task import OddityTask
from psyphy.model.wppm import WPPM

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# Helper: invert criterion to d* for Oddity task


# Robust ellipse plotting utilities (like in covariance_field_demo.py)
def matrix_sqrt(Sigma: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0))
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


def plot_ellipse_at_point(
    ax,
    center: np.ndarray,
    Sigma: np.ndarray,
    scale: float = 1.0,
    color: str = "blue",
    alpha: float = 1.0,
    linewidth: float = 0.5,
    label: str | None = None,
):
    sqrt_Sigma = matrix_sqrt(Sigma)
    ellipse_points = scale * (sqrt_Sigma @ _UNIT_CIRCLE)
    x_coords = center[0] + ellipse_points[0]
    y_coords = center[1] + ellipse_points[1]
    ax.plot(
        x_coords, y_coords, color=color, alpha=alpha, linewidth=linewidth, label=label
    )


_THETAS = np.linspace(0, 2 * np.pi, 100)
_UNIT_CIRCLE = np.vstack([np.cos(_THETAS), np.sin(_THETAS)])


# 1) Ground truth: Wishart process field

# Original constants describing simulation:
# NUM_GRID_PTS = jnp.float32(10)      # Number of reference points over stimulus space.
# MC_SAMPLES = jnp.float32(50)        # Number of simulated trials to compute likelihood.
# NUM_TRIALS = jnp.float32(4000)      # Number of trials in simulated dataset.
# # MIN_LR = jnp.float32(-7)
# # MAX_LR = jnp.float32(-3)

NUM_GRID_PTS = 10  # Number of reference points over stimulus space.
MC_SAMPLES = 60  # Number of simulated trials to compute likelihood.
NUM_TRIALS = 4000  # Number of trials in simulated dataset.
# 4000 trials does not work on cpu


print("[1/5] Setting up ground-truth WPPM and simulating data...")
input_dim = 2
basis_degree = 4  # controls smoothness/complexity
extra_dims = 1  # embedding dim for Wishart process
decay_rate = 0.4  # decay rate for basis functions
variance_scale = 4e-3
diag_term = 1e-9

task = OddityTask()
noise = GaussianNoise(sigma=0.1)
# Set all Wishart process arguments in Prior
truth_prior = Prior(
    input_dim=input_dim,
    basis_degree=basis_degree,
    extra_embedding_dims=extra_dims,
    decay_rate=decay_rate,
    variance_scale=variance_scale,
)
truth_model = WPPM(
    input_dim=input_dim,
    extra_dims=1,
    prior=truth_prior,
    task=task,
    noise=noise,
    diag_term=diag_term,  # ensure positive-definite covariances
)

# Sample ground-truth Wishart process weights
truth_params = truth_model.init_params(jax.random.PRNGKey(123))


# 2) Simulate synthetic data from the ground-truth field
#
# We *must* generate synthetic responses using the SAME observer model that the
# likelihood assumes.
#
# `OddityTask.loglik`
#   - samples internal reps around the stimulus means (ref / comparison)
#   - uses the 3-stimulus oddity decision rule via Mahalanobis distances under
#     an averaged covariance
#   - uses a logistic CDF smoothing with a configurable bandwidth
#
# Here we generate y ~ Bernoulli(p_correct) where p_correct is computed by the
# same MC simulation used by the task.


data = ResponseData()
num_trials_per_ref = NUM_TRIALS  # 4000
n_ref_grid = 5  # NUM_GRID_PTS
ref_grid = np.linspace(-1, 1, n_ref_grid)  # [-1,1] space
ref_points = np.stack(np.meshgrid(ref_grid, ref_grid), axis=-1).reshape(-1, 2)
max_radius = 0.15
mc_samples = MC_SAMPLES
seed = 3
key = jr.PRNGKey(seed)
trial_idx = 0
for ref_np in ref_points:
    for _ in range(num_trials_per_ref):
        # Sample comparison point.
        # Fold in the trial index so each trial gets independent randomness.
        trial_key = jr.fold_in(key, trial_idx)
        k_angle, k_radius, k_pred, k_y = jr.split(trial_key, 4)

        angle = float(jax.random.uniform(k_angle, (), minval=0.0, maxval=2.0 * np.pi))
        radius = float(jax.random.uniform(k_radius, (), minval=0.0, maxval=max_radius))
        delta = np.array([np.cos(angle), np.sin(angle)]) * radius
        comparison_np = ref_np + delta
        comparison_np = np.clip(comparison_np, -1.0, 1.0)

        # Use the MC oddity observer model to compute p(correct).
        # This matches the decision rule used by `OddityTask.loglik`.
        ref = jnp.array(ref_np)
        comparison = jnp.array(comparison_np)
        p_correct = task.predict_with_kwargs(
            truth_params,
            (ref, comparison),
            truth_model,
            truth_model.noise,
            num_samples=mc_samples,
            bandwidth=1e-2,
            key=k_pred,
        )

        # Sample an observed response y ~ Bernoulli(p_correct)
        y = int(jr.bernoulli(k_y, p_correct))
        data.add_trial(ref=np.array(ref), comparison=np.array(comparison), resp=y)
        trial_idx += 1

# 3) Model to fit and MAPOptimizer
print("[2/5] Building model and optimizer...")
prior = Prior(
    input_dim=input_dim,
    basis_degree=basis_degree,
    extra_embedding_dims=extra_dims,
    decay_rate=decay_rate,
    variance_scale=variance_scale,
)
model = WPPM(
    input_dim=input_dim,
    prior=prior,
    task=task,
    noise=noise,
    diag_term=1e-4,  # ensure positive-definite covariances
)


# 4) Fit using optimizer
print("[3/5] Fitting via MAPOptimizer ...")
steps = 3000
lr = 1e-5
momentum = 0.9
map_optimizer = MAPOptimizer(
    steps=steps, learning_rate=lr, momentum=momentum, track_history=True, log_every=1
)
init_params = model.init_params(jax.random.PRNGKey(42))

print(
    "MAPOptimizer settings:",
    f"steps={map_optimizer.steps}",
    f"track_history={map_optimizer.track_history}",
    f"log_every={map_optimizer.log_every}",
)

map_posterior = map_optimizer.fit(model, data, init_params=init_params)

# NOTE: MAPOptimizer.fit(...) optimizes a *point estimate* (MAP), not a sampled
# posterior. The returned object is a MAPPosterior (delta distribution at theta_MAP).
#
# MC-only likelihood note:
# - During optimization, the log-likelihood is evaluated via MC simulation inside
#   OddityTask.loglik.
# - MC samples are *created on the fly* each time loglik is called; they are not
#   stored in the posterior.


# 5) Visualize ellipsoid field: overlay ground truth, prior, and fit at each grid point
print("[4/5] Plotting covariance field ellipses ...")

# Visualization-only stabilization: if a covariance is numerically slightly
# indefinite, we add a small diagonal jitter for plotting.
_PLOT_JITTER_DEFAULT = 1e-6
_PLOT_JITTER_FIT = 1e-5

# Grid for field visualization in [-1,1] space
n_grid = 12
grid_x = np.linspace(-1, 1, n_grid)
grid_y = np.linspace(-1, 1, n_grid)
centers = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)


# Helper to get 2x2 covariance at a point
def get_cov(params, x):
    return np.array(model.local_covariance(params, jnp.array(x)))


# Compute average scale from ground truth covariances for ellipse scaling
gt_covs = np.stack([get_cov(truth_params, c) for c in ref_points])
gt_scales = np.sqrt([np.mean(np.linalg.eigvalsh(cov)) for cov in gt_covs])
avg_scale = float(np.mean(gt_scales))
ellipse_scale = 0.3  # 0.3 * avg_scale  # match simulate_3d.py style, scale is .3 in

fig, ax = plt.subplots(figsize=(7, 7))
non_pd_counts = [0, 0, 0]
labels = ["Ground Truth", "Prior Sample", "Fitted (MAP)", "Reference Points"]
colors = ["k", "b", "r", "g"]
params_list = [truth_params, init_params, map_posterior.params]
legend_handles = []
for i, (params, color, label) in enumerate(zip(params_list, colors, labels)):
    first = True
    plot_jitter = _PLOT_JITTER_DEFAULT
    if label.startswith("Fitted"):
        plot_jitter = _PLOT_JITTER_FIT
    for center in ref_points:  # centers:
        cov = get_cov(params, center)
        cov = cov + plot_jitter * np.eye(cov.shape[0])
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            non_pd_counts[i] += 1
            continue
        (h,) = ax.plot(
            [],
            [],
            color=color,
            alpha=0.5,
            linewidth=0.8,
            label=label if first else None,
        )
        plot_ellipse_at_point(
            ax,
            center=center,
            Sigma=cov,
            scale=ellipse_scale,
            color=color,
            alpha=0.5,
            linewidth=0.8,
            label=None,
        )
        if first:
            legend_handles.append(h)
            first = False

print(
    f"Ellipse plot skips (Fit/MAP): {non_pd_counts[2]} / {len(ref_points)} "
    f"(plot jitter={_PLOT_JITTER_FIT:g})"
)
# Plot reference points
ref_scatter = ax.scatter(
    ref_points[:, 0], ref_points[:, 1], c="g", s=3, zorder=5, label="Reference Points"
)
legend_handles.append(ref_scatter)
ax.set_title(
    f"Covariance field ellipses (overlay)\nSkipped non-PD: GT={non_pd_counts[0]}, Prior={non_pd_counts[1]}, Fit={non_pd_counts[2]}"
)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_handles, loc="upper right")
plt.tight_layout()

os.makedirs(PLOTS_DIR, exist_ok=True)
fig.savefig(
    os.path.join(PLOTS_DIR, "ellipses_overlay.png"),
    dpi=200,
    bbox_inches="tight",
)

# Learning curve
print("[5/5] Plotting learning curve...")
steps_hist, loss_hist = map_optimizer.get_history()
print(f"num steps: {len(steps_hist)}, num losses: {len(loss_hist)}")
if steps_hist:
    print(f"history step range: [{steps_hist[0]}, {steps_hist[-1]}]")
if steps_hist and loss_hist:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.set_xlim(steps_hist[0], steps_hist[-1])
    ax2.plot(steps_hist, loss_hist, color="#4444aa")
    ax2.set_title(f"Learning curve (neg log posterior) — lr={lr}, steps={steps}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(
        os.path.join(PLOTS_DIR, "learning_curve.png"),
        dpi=200,
        bbox_inches="tight",
    )
else:
    print("No history recorded — set track_history=True in MAPOptimizer to enable.")
