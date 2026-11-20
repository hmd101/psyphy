#!/usr/bin/env python3
"""Train the 2-D oddity network that ships with psyphy.

The output, ``oddity_2d_16x2.npz``, is the default weight file of
:class:`psyphy.model.likelihood.FastOddityTask`. It is ours and ships under psyphy's MIT
license, like the code.

What we do, in order:

1. Draw points (lambda, eta) in whitened coordinates. lambda holds the 2 eigenvalues of the
   reference covariance (0.001 to 1.499, largest first). eta is the offset between the
   stimuli, with length below 12 and a direction over the whole circle.
2. Turn each point into 4 network inputs with ``oddity_features`` from a trial that has it
   as its whitened form.
3. Label each point with a Monte Carlo estimate of P(correct), using a hard indicator.
4. Fit a 4 -> 16 -> 16 -> 1 GELU network with Adam on the cross-entropy, 3 times from
   different initial weights. The one with the lowest error on a selection set is kept.
5. Measure the kept network on a separate test set and on 300 trials like real stimuli.
   The file is written only if every check below passes.

Checks, on points with P(correct) between 1/3 and 1:

- RMS error at most 1.5e-3 and maximum error at most 1.5e-2, for a12 > 0 and for a12 < 0
  separately

Below 1/3, RMS error at most 3e-3. On the 300 trials, maximum error at most 0.01.

A miss means training again with other seeds. The bounds stay.

Usage
-----
    python scripts/train_oddity_fast_likelihood.py

takes about 20 minutes on a laptop CPU. The seeds are constants, so a rerun on the same JAX
version gives the same file. A short run that checks the script still works:

    python scripts/train_oddity_fast_likelihood.py --quick --force-write \
        --out "$TMPDIR/oddity_quick.npz"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from psyphy.model.likelihood import (
    TrainedNetwork,
    forward_pass,
    normalize_by_noise,
    oddity_features,
    save_network,
)
from psyphy.model.likelihood.features import ODDITY_POOLING_WEIGHT
from psyphy.model.likelihood.neural import ODDITY_2D_REQUIRE

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "src/psyphy/model/likelihood/weights/oddity_2d_16x2.npz"

LAM_LO, LAM_HI = 1e-3, 1.5 - 1e-3  # eigenvalues of the whitened reference covariance
ETA_MAX = 12.0  # length of the whitened offset
WIDTHS = (4, 16, 16, 1)
BATCH = 1024
LR_START, LR_END = 1e-3, 1e-5

# seeds: training points, selection set, test set, probe trials, network restarts
TRAIN_SEED, SELECT_SEED, TEST_SEED, TRIALS_SEED = 1, 2000, 1000, 7
RESTART_SEEDS = (0, 1, 2)

# acceptance bounds
H_LO = 1 / 3
RMS_ABOVE, MAX_ABOVE = 1.5e-3, 1.5e-2
RMS_BELOW = 3e-3
MAX_TRIALS = 0.01

CHUNK = 10_000  # Monte Carlo samples per point per scan step
BLOCK = 512  # points per jitted call


def sample_points(n, rng):
    """lam (n, 2), largest first, and eta (n, 2), length uniform on (0, ETA_MAX)."""
    lam = np.sort(rng.uniform(LAM_LO, LAM_HI, size=(n, 2)), axis=1)[:, ::-1].copy()
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    length = rng.uniform(0.0, ETA_MAX, size=n)
    eta = np.stack([np.cos(angle), np.sin(angle)], axis=1) * length[:, None]
    return lam, eta


@jax.jit
def _features(lam, eta):
    """(n, 4) network inputs for (n, 2) lam and eta."""

    def one(lam_i, eta_i):
        # a trial whose whitened form is (lam_i, eta_i): the pooled covariance is the identity
        cov_ref = jnp.diag(lam_i)
        cov_comparison = 3.0 * jnp.eye(2) - 2.0 * cov_ref
        return oddity_features(cov_ref, cov_comparison, jnp.zeros(2), eta_i)

    return jax.vmap(one)(lam, eta)


def features(lam, eta):
    return np.asarray(
        _features(jnp.asarray(lam, jnp.float32), jnp.asarray(eta, jnp.float32))
    )


@partial(jax.jit, static_argnames="n_chunks")
def _hits(key, lam, eta, n_chunks):
    """(BLOCK,) number of correct trials out of n_chunks * CHUNK, per point."""
    sx = jnp.sqrt(lam)[:, None, :]  # (BLOCK, 1, 2), reference noise
    sy = jnp.sqrt(3.0 - 2.0 * lam)[:, None, :]  # comparison noise
    mu = eta[:, None, :]

    def body(count, k):
        z = jax.random.normal(k, (3, lam.shape[0], CHUNK, 2), jnp.float32)
        x1, x2, y = z[0] * sx, z[1] * sx, mu + z[2] * sy  # (BLOCK, CHUNK, 2)
        d12 = jnp.sum((x1 - x2) ** 2, axis=-1)
        d1y = jnp.sum((x1 - y) ** 2, axis=-1)
        d2y = jnp.sum((x2 - y) ** 2, axis=-1)
        correct = d12 < jnp.minimum(
            d1y, d2y
        )  # the pair shown twice is the closest pair
        return count + correct.sum(axis=-1), None

    keys = jax.random.split(key, n_chunks)
    count, _ = jax.lax.scan(body, jnp.zeros(lam.shape[0], jnp.int32), keys)
    return count


def label(lam, eta, n_samples, seed):
    """P(correct) for each of n points, with n_samples rounded up to a multiple of CHUNK."""
    n_chunks = int(-(-int(n_samples) // CHUNK))
    n = lam.shape[0]
    pad = -n % BLOCK
    lam_p = np.concatenate([lam, np.ones((pad, 2))]).astype(np.float32)
    eta_p = np.concatenate([eta, np.zeros((pad, 2))]).astype(np.float32)
    keys = jax.random.split(jax.random.PRNGKey(seed), (n + pad) // BLOCK)
    out = []
    for i, k in enumerate(keys):
        sl = slice(i * BLOCK, (i + 1) * BLOCK)
        out.append(
            np.asarray(
                _hits(k, jnp.asarray(lam_p[sl]), jnp.asarray(eta_p[sl]), n_chunks)
            )
        )
    return np.concatenate(out)[:n] / (n_chunks * CHUNK)


def labeled_set(seed, per_bin, n_samples, n_candidates):
    """Points spread over P(correct): per_bin in each of 24 bins on [1/3, 1], 2 * per_bin below."""
    rng = np.random.default_rng(seed)
    lam, eta = sample_points(n_candidates, rng)
    rough = label(lam, eta, 10_000, seed)  # only to place each candidate in a bin
    edges = np.linspace(H_LO, 1.0, 25)
    pick = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((rough >= lo) & (rough < hi))
        pick.append(rng.choice(idx, size=min(per_bin, idx.size), replace=False))
    below = np.flatnonzero(rough < H_LO)
    pick.append(rng.choice(below, size=min(2 * per_bin, below.size), replace=False))
    pick = np.concatenate(pick)
    return lam[pick], eta[pick], label(lam[pick], eta[pick], n_samples, seed + 1)


def init(key):
    """Glorot-normal weights, zero biases."""
    params = []
    for n_in, n_out in zip(WIDTHS[:-1], WIDTHS[1:]):
        key, sub = jax.random.split(key)
        scale = jnp.sqrt(2.0 / (n_in + n_out))
        params.append((scale * jax.random.normal(sub, (n_in, n_out)), jnp.zeros(n_out)))
    return params


def train(x, y, seed, n_steps):
    """Fit one network. x (n, 4) standardized, y (n,) labels. Returns its parameters."""
    schedule = optax.cosine_decay_schedule(LR_START, n_steps, alpha=LR_END / LR_START)
    opt = optax.adam(schedule)
    params = init(jax.random.PRNGKey(seed))
    state = opt.init(params)

    def loss(p, xb, yb):
        return optax.sigmoid_binary_cross_entropy(forward_pass(p, xb), yb).mean()

    @jax.jit
    def step(p, s, k):
        idx = jax.random.randint(k, (BATCH,), 0, x.shape[0])
        grads = jax.grad(loss)(p, x[idx], y[idx])
        updates, s = opt.update(grads, s)
        return optax.apply_updates(p, updates), s

    for k in jax.random.split(jax.random.PRNGKey(seed + 100), n_steps):
        params, state = step(params, state, k)
    return params


def predict(params, mean, std, f):
    """P(correct) for (n, 4) features f."""
    x = jnp.asarray((f - mean) / std, jnp.float32)
    return np.asarray(jax.nn.sigmoid(forward_pass(params, x)))


def errors(p, y):
    """RMS and maximum absolute error."""
    e = p - y
    return float(np.sqrt(np.mean(e**2))), float(np.abs(e).max())


def with_a12_sign(eta, sign):
    """Same label, a12 of the features positive (sign=1) or negative (sign=-1)."""
    return np.abs(eta) * np.array([1.0, sign])


def realistic_trials(n=300, seed=0):
    """lam, eta of n trials like the ones a fit sees: small covariances, mixed orientations."""
    rng = np.random.RandomState(seed)

    def root(lams, theta):
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return rot @ np.diag(np.sqrt(lams))

    covs, trials = [], []
    for _ in range(n):
        scale = 10 ** rng.uniform(-3, -1)
        lam_ref = scale * 10 ** rng.uniform(-1, 0, size=2)
        lam_comp = scale * 10 ** rng.uniform(-1, 0, size=2)
        dist = np.sqrt(scale) * 10 ** rng.uniform(-1.5, 0.7)
        th_ref, th_comp, phi = (
            rng.uniform(0, np.pi),
            rng.uniform(0, np.pi),
            rng.uniform(0, 2 * np.pi),
        )
        cov_ref = root(lam_ref, th_ref) @ root(lam_ref, th_ref).T
        cov_comp = root(lam_comp, th_comp) @ root(lam_comp, th_comp).T
        ref = np.array([0.3, -0.2])
        comp = ref + dist * np.array([np.cos(phi), np.sin(phi)])
        covs.append((cov_ref, cov_comp))
        trials.append((ref, comp))
    return covs, trials


def trial_inputs(covs, trials):
    """Network inputs f (n, 4), and the whitened lam (n, 2), eta (n, 2) that fix the label."""
    f, lam, eta = [], [], []
    for (cov_ref, cov_comp), (ref, comp) in zip(covs, trials):
        args = [jnp.asarray(a, jnp.float32) for a in (cov_ref, cov_comp, ref, comp)]
        f.append(np.asarray(oddity_features(*args)))
        a_mat, disp = normalize_by_noise(*args)
        w, v = jnp.linalg.eigh(a_mat)  # ascending
        lam.append(np.asarray(w[::-1]))
        eta.append(np.abs(np.asarray(v[:, ::-1].T @ disp)))
    return np.array(f), np.array(lam), np.array(eta)


def git_rev():
    """Short commit of the checkout this script runs from, with a mark if the script changed."""
    cmd = ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"]
    rev = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    status = [
        "git",
        "-C",
        str(HERE),
        "status",
        "--porcelain",
        "--",
        str(Path(__file__).resolve()),
    ]
    changed = bool(
        subprocess.run(status, capture_output=True, text=True).stdout.strip()
    )
    return rev + ("+uncommitted" if changed else "")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="weight file to write"
    )
    ap.add_argument(
        "--quick", action="store_true", help="1000 points, 1e3 samples, 100 steps"
    )
    ap.add_argument(
        "--force-write", action="store_true", help="write the file even on a miss"
    )
    ap.add_argument(
        "--generator-rev",
        default=None,
        help="commit to record, default: this checkout's HEAD",
    )
    args = ap.parse_args()

    if args.quick:
        n_points, label_samples, n_steps = 1000, 1_000, 100
        select = {"per_bin": 2, "n_samples": 1_000, "n_candidates": 5_000}
        test = {"per_bin": 4, "n_samples": 1_000, "n_candidates": 5_000}
    else:
        n_points, label_samples, n_steps = 300_000, 100_000, 112_000
        select = {"per_bin": 104, "n_samples": 1_000_000, "n_candidates": 200_000}
        test = {"per_bin": 208, "n_samples": 10_000_000, "n_candidates": 200_000}

    t0 = time.perf_counter()

    def clock(what):
        print(f"[{time.perf_counter() - t0:6.0f} s] {what}", flush=True)

    # training set
    lam, eta = sample_points(n_points, np.random.default_rng(TRAIN_SEED))
    f_train = features(lam, eta)
    y_train = label(lam, eta, label_samples, TRAIN_SEED)
    clock(f"training labels: {n_points} points at {label_samples:.0e} samples")
    mean, std = f_train.mean(axis=0), f_train.std(axis=0)
    x_train = jnp.asarray((f_train - mean) / std, jnp.float32)
    y_train = jnp.asarray(y_train, jnp.float32)

    # selection and test sets
    lam_s, eta_s, y_s = labeled_set(SELECT_SEED, **select)
    f_s = features(lam_s, eta_s)
    clock(f"selection set: {len(y_s)} points")
    lam_t, eta_t, y_t = labeled_set(TEST_SEED, **test)
    clock(f"test set: {len(y_t)} points")

    # restarts, the lowest selection RMS wins
    best = None
    for seed in RESTART_SEEDS:
        params = train(x_train, y_train, seed, n_steps)
        rms, _ = errors(predict(params, mean, std, f_s), y_s)
        clock(f"restart {seed}: selection RMS {rms:.2e}")
        if best is None or rms < best[0]:
            best = (rms, seed, params)
    _, seed, params = best
    print(f"kept restart {seed}")

    # the checks
    above = y_t >= H_LO
    p_plus = predict(params, mean, std, features(lam_t, with_a12_sign(eta_t, 1.0)))
    p_minus = predict(params, mean, std, features(lam_t, with_a12_sign(eta_t, -1.0)))
    p_test = predict(params, mean, std, features(lam_t, eta_t))
    rms_plus, max_plus = errors(p_plus[above], y_t[above])
    rms_minus, max_minus = errors(p_minus[above], y_t[above])
    rms_above, max_above = errors(p_test[above], y_t[above])
    rms_below, max_below = errors(p_test[~above], y_t[~above])

    covs, trials = realistic_trials()
    f_trials, lam_trials, eta_trials = trial_inputs(covs, trials)
    y_trials = label(
        lam_trials, eta_trials, 1_000_000 if not args.quick else 1_000, TRIALS_SEED
    )
    p_trials = predict(params, mean, std, f_trials)
    trial_err = np.abs(p_trials - y_trials)

    checks = [
        ("h in [1/3, 1], a12 > 0: RMS", rms_plus, RMS_ABOVE),
        ("h in [1/3, 1], a12 > 0: maximum", max_plus, MAX_ABOVE),
        ("h in [1/3, 1], a12 < 0: RMS", rms_minus, RMS_ABOVE),
        ("h in [1/3, 1], a12 < 0: maximum", max_minus, MAX_ABOVE),
        ("h below 1/3: RMS", rms_below, RMS_BELOW),
        ("300 trials: maximum", trial_err.max(), MAX_TRIALS),
    ]
    print(
        f"\ntest set: {above.sum()} points with h in [1/3, 1], {(~above).sum()} below"
    )
    print(f"h in [1/3, 1], as sampled: RMS {rms_above:.2e}, maximum {max_above:.2e}")
    print(f"h below 1/3, as sampled: RMS {rms_below:.2e}, maximum {max_below:.2e}")
    print(
        f"300 trials ({(y_trials < H_LO).sum()} below 1/3, {(f_trials[:, 2] < 0).sum()} with a12 < 0): "
        f"median {np.median(trial_err):.4f}, p95 {np.percentile(trial_err, 95):.4f}, "
        f"maximum {trial_err.max():.4f}, RMS {np.sqrt(np.mean(trial_err**2)):.4f}"
    )
    print("\nacceptance:")
    ok = True
    for name, value, bound in checks:
        passed = value <= bound
        ok &= passed
        print(f"  {name:<36} {value:.2e}  <= {bound:.1e}  {'ok' if passed else 'MISS'}")

    if not ok and not args.force_write:
        print(
            "\nacceptance missed, no file written. Train again with other restart seeds."
        )
        sys.exit(1)

    metadata = {
        "task": ODDITY_2D_REQUIRE["task"],
        "input_dim": 2,
        "n_features": 4,
        "feature_map": ODDITY_2D_REQUIRE["feature_map"],
        "feature_map_version": ODDITY_2D_REQUIRE["feature_map_version"],
        "pooling_weight": float(ODDITY_POOLING_WEIGHT),
        "label_mc_samples": int(label_samples),
        "label_bandwidth": 0.0,
        "domain_lo": f_train.min(axis=0),
        "domain_hi": f_train.max(axis=0),
        "val_rmse": rms_above,
        "val_max_err": max_above,
        "trained_utc": _dt.datetime.now(_dt.timezone.utc).date().isoformat(),
        "generator_rev": f"scripts/train_oddity_fast_likelihood.py @ {args.generator_rev or git_rev()}",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_network(
        args.out, params=params, input_mean=mean, input_std=std, metadata=metadata
    )

    # read it back through the loader psyphy uses
    net = TrainedNetwork.from_npz(args.out, require=ODDITY_2D_REQUIRE)
    size = args.out.stat().st_size
    print(
        f"\nwrote {args.out} ({size:,} bytes), loads as {net.metadata['task']} "
        f"{net.metadata['input_dim']}-D, domain {'recorded' if net.domain else 'unknown'}"
    )
    print(
        f"training features: min {np.round(f_train.min(axis=0), 4)}, max {np.round(f_train.max(axis=0), 4)}"
    )
    clock("done")


if __name__ == "__main__":
    main()
