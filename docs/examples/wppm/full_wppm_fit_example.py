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


def invert_oddity_criterion_to_d(criterion: float, slope: float = 1.5) -> float:
    chance = 1.0 / 3.0
    perf_range = 1.0 - chance
    g = (criterion - chance) / perf_range
    val = np.clip(2.0 * float(g) - 1.0, -0.999999, 0.999999)
    return float(np.arctanh(val) / slope)


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
print("[1/5] Setting up ground-truth WPPM and simulating data...")
input_dim = 2
basis_degree = 4  # controls smoothness/complexity
extra_dims = 1  # embedding dim for Wishart process
lengthscale = 0.5
variance_scale = 0.2

task = OddityTask(slope=1.5)
noise = GaussianNoise(sigma=0.1)
# Set all Wishart process arguments in Prior
truth_prior = Prior(
    input_dim=input_dim,
    basis_degree=basis_degree,
    extra_embedding_dims=extra_dims,
    lengthscale=lengthscale,
    variance_scale=variance_scale,
)
truth_model = WPPM(
    input_dim=input_dim,
    extra_dims=1,
    prior=truth_prior,
    task=task,
    noise=noise,
    diag_term=1e-4,  # ensure positive-definite covariances
)

# Sample ground-truth Wishart process weights
truth_params = truth_model.init_params(jax.random.PRNGKey(123))


# 2) Simulate synthetic data from the ground-truth field (MC-based oddity simulation)


data = ResponseData()
num_trials_per_ref = 50
n_ref_grid = 5
ref_grid = np.linspace(0.1, 0.9, n_ref_grid)  # [0,1] space, avoid edges
ref_points = np.stack(np.meshgrid(ref_grid, ref_grid), axis=-1).reshape(-1, 2)
max_radius = 0.15
mc_samples = 25
seed = 3
key = jr.PRNGKey(seed)
trial_idx = 0
for ref_np in ref_points:
    for _ in range(num_trials_per_ref):
        # Sample comparison point
        angle = float(jax.random.uniform(key, (), minval=0.0, maxval=2.0 * np.pi))
        radius = float(jax.random.uniform(key, (), minval=0.0, maxval=max_radius))
        delta = np.array([np.cos(angle), np.sin(angle)]) * radius
        comparison_np = ref_np + delta
        comparison_np = np.clip(comparison_np, 0.0, 1.0)
        # MC simulate oddity response
        # For 2D: oddity task with 3 stimuli (ref, comparison, ref)
        # Use a single key per trial, split for each stimulus
        trial_key = jr.fold_in(key, trial_idx)
        stim_keys = jr.split(trial_key, 3)
        # For each stimulus, sample mc_samples noisy reps from N(0, Sigma)
        means = [np.zeros(input_dim)] * 3
        covs = [
            np.array(truth_model.local_covariance(truth_params, jnp.array(stim)))
            for stim in [ref_np, comparison_np, ref_np]
        ]
        reps = [
            jax.random.multivariate_normal(
                stim_keys[i], means[i], covs[i], shape=(mc_samples,)
            )
            for i in range(3)
        ]
        # Compute oddity responses for each MC sample
        oddity_responses = []
        for i in range(mc_samples):
            samples = [reps[0][i], reps[1][i], reps[2][i]]
            avg_dists = [
                np.mean(
                    [
                        np.linalg.norm(samples[j] - samples[k])
                        for k in range(3)
                        if k != j
                    ]
                )
                for j in range(3)
            ]
            odd_idx = np.argmax(avg_dists)
            oddity_responses.append(
                1 if odd_idx == 1 else 0
            )  # 1 if comparison chosen as odd
        # Majority vote over MC samples
        y = int(np.mean(oddity_responses) > 0.5)
        data.add_trial(ref=ref_np.copy(), comparison=comparison_np.copy(), resp=y)
        trial_idx += 1

# 3) Model to fit and MAPOptimizer
print("[2/5] Building model and optimizer...")
prior = Prior(
    input_dim=input_dim,
    # scale=0.5,
    basis_degree=basis_degree,
    extra_embedding_dims=extra_dims,
    lengthscale=lengthscale,
    variance_scale=variance_scale,
)
model = WPPM(
    input_dim=input_dim,
    prior=prior,
    task=task,
    noise=noise,
    diag_term=1e-4,  # ensure positive-definite covariances
)


# 4) Fit using optimizer (MC-based likelihood)
print("[3/5] Fitting via MAPOptimizer (MC-based likelihood)...")
steps = 1200
lr = 1e-2
momentum = 0.9
optimizer = MAPOptimizer(
    steps=steps, learning_rate=lr, momentum=momentum, track_history=True, log_every=10
)
init_params = model.init_params(jax.random.PRNGKey(42))

posterior = optimizer.fit(model, data, init_params=init_params)


# 5) Visualize ellipsoid field: overlay ground truth, prior, and fit at each grid point
print("[4/5] Plotting covariance field ellipses (overlay style)...")

# Grid for field visualization in [0,1] space
n_grid = 11
grid_x = np.linspace(0.1, 0.9, n_grid)
grid_y = np.linspace(0.1, 0.9, n_grid)
centers = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)


# Helper to get 2x2 covariance at a point
def get_cov(params, x):
    return np.array(model.local_covariance(params, jnp.array(x)))


# Compute average scale from ground truth covariances for ellipse scaling
gt_covs = np.stack([get_cov(truth_params, c) for c in centers])
gt_scales = np.sqrt([np.mean(np.linalg.eigvalsh(cov)) for cov in gt_covs])
avg_scale = float(np.mean(gt_scales))
ellipse_scale = 0.3 * avg_scale  # match simulate_3d.py style

fig, ax = plt.subplots(figsize=(7, 7))
non_pd_counts = [0, 0, 0]
labels = ["Ground Truth", "Prior Sample", "Fitted (MAP)", "Reference Points"]
colors = ["k", "b", "r", "g"]
params_list = [truth_params, init_params, posterior.params]
legend_handles = []
for i, (params, color, label) in enumerate(zip(params_list, colors, labels)):
    first = True
    for center in centers:
        cov = get_cov(params, center)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 0):
            non_pd_counts[i] += 1
            continue
        (h,) = ax.plot(
            [],
            [],
            color=color,
            alpha=0.5,
            linewidth=2.0,
            label=label if first else None,
        )
        plot_ellipse_at_point(
            ax,
            center=center,
            Sigma=cov,
            scale=ellipse_scale,
            color=color,
            alpha=0.5,
            linewidth=2.0,
            label=None,
        )
        if first:
            legend_handles.append(h)
            first = False
# Plot reference points
ref_scatter = ax.scatter(
    ref_points[:, 0], ref_points[:, 1], c="g", s=30, zorder=5, label="Reference Points"
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
    os.path.join(PLOTS_DIR, "full_wppm_field_ellipses_overlay.png"),
    dpi=200,
    bbox_inches="tight",
)

# Learning curve
print("[5/5] Plotting learning curve...")
steps_hist, loss_hist = optimizer.get_history()
if steps_hist and loss_hist:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(steps_hist, loss_hist, color="#4444aa")
    ax2.set_title(f"Learning curve (neg log posterior) — lr={lr}, steps={steps}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(
        os.path.join(PLOTS_DIR, "full_wppm_learning_curve.png"),
        dpi=200,
        bbox_inches="tight",
    )
else:
    print("No history recorded — set track_history=True in MAPOptimizer to enable.")
