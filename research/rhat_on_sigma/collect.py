"""Run the NUTS example for one mode and save what the figures need.

Sampling is the slow part (balanced: ~36 min on an A100), so it is done once
here and the figures are drawn from the saved .npz by ``figures.py``.

The example script is copied to a temporary directory and run from there, so
its own plots never overwrite the committed tutorial figures in
docs/examples/wppm/plots/. Imports resolve to this repo's src/.

usage (from the repo root):
    python research/rhat_on_sigma/collect.py quick
    python research/rhat_on_sigma/collect.py balanced

If another JAX process is using the GPU, set XLA_PYTHON_CLIENT_PREALLOCATE=false
(JAX otherwise reserves 75% of GPU memory up front and the second process fails).
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXAMPLE = os.path.join(REPO, "docs", "examples", "wppm", "nuts_inference_example.py")
sys.path.insert(0, os.path.join(REPO, "src"))

import arviz as az  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

mode = sys.argv[1]
out = os.path.join(HERE, "results", f"{mode}.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    example = shutil.copy(EXAMPLE, tmp)
    sys.argv = [example, "--mode", mode]
    t0 = time.time()
    g = runpy.run_path(example, run_name="__main__")
    runtime = time.time() - t0

model, post, refs = g["model"], g["nuts_posterior"], g["ref_points"]
Field = g["WPPMCovarianceField"]
W = post._samples["W"]  # (C, S, *W_shape)


def sigma_at_refs(Wi):
    """Σ at every reference point for ONE parameter draw."""
    return Field(model, {"W": Wi})(refs)


Sig = jax.vmap(jax.vmap(sigma_at_refs))(W)  # (C, S, n_ref, 2, 2)
Sig_EW = sigma_at_refs(W.mean(axis=(0, 1)))  # Σ(E[W]): average W, then Σ
E_Sig = Sig.mean(axis=(0, 1))  # E[Σ]: Σ per draw, then average
truth = g["truth_field"](refs)

# Identifiable per-ref quantities for diagnostics. Unlike W, these do not
# change under the embedding rotation W -> W R. Variances on log scale
# (positive, skewed); the correlation carries the orientation.
feats = np.stack(
    [
        np.asarray(jnp.log(Sig[..., 0, 0])),
        np.asarray(jnp.log(Sig[..., 1, 1])),
        np.asarray(Sig[..., 0, 1] / jnp.sqrt(Sig[..., 0, 0] * Sig[..., 1, 1])),
    ],
    axis=-1,
)  # (C, S, n_ref, 3): log_var1, log_var2, corr

dt = az.from_dict({"posterior": {"W": np.asarray(W), "Sigma": feats}})
rhat = az.rhat(dt)

np.savez(
    out,
    mode=mode,
    runtime_s=runtime,
    W=np.asarray(W),
    Sig=np.asarray(Sig),
    Sig_EW=np.asarray(Sig_EW),
    E_Sig=np.asarray(E_Sig),
    truth=np.asarray(truth),
    refs=np.asarray(refs),
    feats=feats,
    rhat_W=np.asarray(rhat["W"]).ravel(),
    rhat_S=np.asarray(rhat["Sigma"]).ravel(),
    n_steps=np.asarray(post.sampler_stats["n_steps"]),
    max_num_doublings=g["MAX_NUM_DOUBLINGS"],
)
print(f"saved {out}: {mode}, {W.shape[0]}x{W.shape[1]} draws, {runtime / 60:.1f} min")
