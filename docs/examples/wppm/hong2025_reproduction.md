# Reproducing Hong et al. (2025)

We reproduce **Figure 2B** of Hong et al. (2025) — human color discrimination
thresholds — using psyphy and the authors' own data. Two things happen here:
we recover their published threshold contours from their published model, and
we refit the model from scratch to check that we land where they landed.

If you know the paper, this shows how its pipeline maps onto psyphy. If you
don't, it is a worked example of psyphy on real data, with an external ground
truth to check against.

> Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P.,
> Williams, A. H., & Brainard, D. H. (2026). *Comprehensive characterization of
> human color discrimination thresholds.* eLife 14:RP108943.
> <https://doi.org/10.7554/eLife.108943.2>

Runnable script:
[`hong2025_reproduction.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/hong2025_reproduction.py).

---

## The result

<div align="center">
    <img src="../plots/hong2025_thresholds.png"
         alt="Figure 2B reproduced: 66.7%-correct discrimination threshold contours"
         width="620"/>
    <p><em>Colored ellipses are the contours we recover with psyphy; dashed gray
    are the published ones. Each ellipse takes the color of its own reference
    stimulus, via the monitor calibration matrix published with the data.</em></p>
</div>

The paper's own caption for this panel:

> Discrimination threshold contours (66.7% correct) read out from the WPPM on a
> grid of reference stimuli for a representative participant, based [on] fits to
> the 6,000 AEPsych trials.

Each ellipse says how far a comparison color must move from its reference
before this observer distinguishes the two 66.7% of the time. We match the
published contours to a **median 2.2% semi-axis error in ~20 s on a laptop**.

## The whole recipe

```python title="Published data to threshold contours"
import jax
jax.config.update("jax_enable_x64", True)   # the authors used float64
import jax.numpy as jnp

from psyphy.data.published import hong2025
from psyphy.posterior import MAPPosterior, ThresholdConfig, WPPMPredictivePosterior

paths = hong2025.fetch(subject=1)                          # download from OSF
W_org = hong2025.load_reference_W(paths["weights"])        # the paper's fitted weights
coords, published = hong2025.load_sigma_table(paths["thres_ellipses"])

model = hong2025.build_paper_model(mc_samples=500)         # the paper's model
thresholds = WPPMPredictivePosterior(
    MAPPosterior({"W": W_org}, model),
    jnp.asarray(coords),
    n_samples=1,
    threshold_pred=True,
    threshold_config=ThresholdConfig(n_theta=16, n_length=300),
).mean                                                     # -> (49, 2, 2)
```

The rest of this page explains those calls, then refits the model from the raw
trials.

```bash
python hong2025_reproduction.py --skip-refit   # everything except the refit, <1 min CPU
python hong2025_reproduction.py --mode full    # add the refit; wants a GPU
```

---

## Data

psyphy ships no data. The OSF node carries no explicit license, so we download
on request into `~/.cache/psyphy/` 

```python title="Download one observer's files"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fetch"
```

| File | Size | Used for |
|---|---|---|
| `trial_data_pooled_by_type_sub1.csv` | 1 MB | trials, for the refit |
| `Bestfit_W_sub1.csv` | 212 KB | fitted weights, plus 120 bootstraps |
| `Thres_ellipses_sub1.csv` | 320 KB | the 7×7 grid and published thresholds |
| `Noise_ellipses_sub1.csv` | 68 MB | published Σ_noise on a 103×103 grid |

All three result files also carry all 120 bootstrap fits, which we use to
calibrate what "close enough" means.

The figure's colors need one more file: a 3×3 calibration matrix in a
different OSF folder, not per-observer, so we need one more  call:

```python title="Color calibration"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:colors"
```

`load_trials` returns psyphy's ordinary `TrialData`, so nothing downstream
needs an adapter:

```python title="Load the trials"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:load"
```

!!! warning "Only 6,000 of the 12,000 trials were fitted"
    The file holds 5,100 adaptive + 900 Sobol (`AEPsych_*`) trials and 6,000
    `MOCS_*` trials. The paper fits the `AEPsych_*` rows; MOCS is held-out
    validation. Fitting all 12,000 gives a plausible result that is not the
    published one. `load_trials` defaults to `trial_types=("AEPsych",)`.
    For more information on how the authors did adaptive trial placement using the library AEPsych, we refer the reader to the paper.

Two conventions psyphy handles for us: stimulus coordinates are already in the
Chebyshev domain `[-1, 1]`, so no normalization is needed; and oddity trials
are stored with `K=2`, not 3 — the observer sees three items but only two
distinct means, and the duplication lives in the likelihood.

---

## Model

`build_paper_model()` assembles a WPPM from `PAPER_HYPERPARAMS`, transcribed
from the paper's fitting script. Most settings map one-to-one. The ones worth
knowing:

| Paper | psyphy | Note |
|---|---|---|
| `degree=5` | `basis_degree=4` | **Off-by-one.** Theirs counts basis *functions* (T₀…T₄); ours is the *maximum degree*. Same 5×5 grid. |
| `variance_scale=3e-4` | same | psyphy's default is `4e-3` |
| `diag_term=0` | same | psyphy's default is `1e-6`; theirs leaves Σ unregularized |
| `mc_samples=2000`, `bandwidth=5e-3` | `OddityTaskConfig` | |
| `learning_rate=1e-4`, `momentum=0.2`, `total_steps=1500`, 3 restarts | `MAPOptimizer` + a loop | refit only |

The published weight tensor is `(5, 5, 2, 3)` — exactly psyphy's `params["W"]`
layout, so it drops straight in with no reshaping.

---

## Thresholds (Figure 2B)

The model is parameterized in `Σ_noise(x)`, the covariance of the observer's
_internal representation_. The paper reports **thresholds**. Those are different
objects! The map between them is as follows:

```
forward  (psyphy's OddityTask):  Σ_noise(x_ref), Σ_noise(x_1)  ->  P(correct)
inverse  (what Figure 2B plots): P(correct) = 2/3             ->  x_1
```

There is no closed form for the inverse. `P(correct)` for the 3-alternative
oddity task is the probability that `min(d_02, d_12) > d_01` over three correlated
quadratic forms, which is why the paper estimates it by Monte Carlo in the
first place. So we invert numerically, the same way they do:

1. Probe `n_theta` directions around each reference point.
2. Along each, evaluate `P(correct)` at `n_length` distances and keep the one
   closest to 2/3. One boundary point per direction.
3. Fit an ellipse to those points.

Step 3 is closed-form: a point at radius `r` in direction `u` satisfies
`uᵀΣ⁻¹u = 1/r²`, which is **linear** in the three free entries of `Σ⁻¹`. Least
squares, then one inverse — no optimizer. Step 2 uses a dense sweep rather than
bisection because `P(correct)` is monotone only up to Monte Carlo noise, and a
root-finder can walk off a noisy plateau.

```python title="Threshold inversion at every published reference point"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:thresholds"
```

Note what is absent: no optimizer, no trial data. We feed the paper's own
weights in via `MAPPosterior`, so the covariance field is identical by
construction and the only thing that can differ is the inversion itself.

Two API details specific to threshold mode:

- **`X` is bare reference points**, `(n_test, input_dim)` — not the paired
  `(n_test, k_stimuli, input_dim)` shape the class takes otherwise. Threshold
  mode generates its own comparisons. Passing the paired shape raises
  `ValueError`.
- **`mean` and `variance` are matrix-valued**, `(n_test, input_dim, input_dim)`
  — one threshold covariance per reference point.

```python title="Compute settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:threshold_settings"
```

```
reference points : 49
semi-axis error  : median 2.18 %, max 10.78 %
settings         : n_theta=16, n_length=300, mc=500
```

The ~2% residual is the 16-direction fan plus Monte Carlo noise, not anything
structural; raising `n_theta` and `mc_samples` toward the paper's settings
shrinks it, at ~30× the runtime.


---

## Exact check
### does psyphy build the same covariance field they published?

A narrower question with a sharper answer: given the paper's weights, does
psyphy build the same covariance field they published? This is fully
deterministic. 

```python title="Published weights through psyphy's covariance field"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:stage1"
```

```
grid points  : 10609
max |diff|   : 6.778e-09
mean |diff|  : 2.538e-09
```

Plain elementwise subtraction over all 42,436 entries. The
published CSV stores 8 decimals; our values round to theirs exactly in 96% of
cases and agree to within one unit in the last printed digit in 100%. **This is
agreement to the precision the file can express.**

This runs as a test (`test_covariance_field_matches_published_sigma_noise`),
skipped automatically when the data has not been downloaded, so CI stays
network-free.

!!! note "Why float64 — and where it actually matters"
    ```python
    --8<-- "docs/examples/wppm/hong2025_reproduction.py:x64"
    ```
    We use float64 because the paper does, and because it is the safe default
    for the refit, where gradients accumulate over 6,000 × 2,000 × 1,500 and
    `diag_term=0` leaves Σ unregularized.

    It is **not required for this check or for the thresholds**. Measured:
    float32 gives max |diff| 6.86e-9 and median threshold error 1.77%, against
    float64's 6.78e-9 and 2.18% — both pass, and the threshold gap is Monte
    Carlo noise. float32's epsilon is *relative* (~1.2e-7) and Σ entries are
    ~1e-3, so float32 resolves them to ~5e-10 absolute, finer than the file's
    own 1e-8 rounding. Separately, at this point,  `jax-metal` has no float64, so Apple GPUs
    are out for the refit regardless.

---

## Refit
### Does psyphy's fit find the paper's covariance field?

Everything above started from the paper's weights. The stronger question: given
only their **trials**, does psyphy's fit find their field?

```python title="MAP fit with the paper's optimizer settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fit"
```

Two settings make the paper's `learning_rate=1e-4` mean the same thing here:
`reduction="mean"` (they minimize a per-trial objective) and
`max_grad_norm=None` (they do no clipping, psyphy clips at 1.0 by default,
which would silently rescale the effective learning rate).

We compare **Σ, never W**: `U -> UQ` for orthogonal `Q` leaves `Σ = UUᵀ`
unchanged and the prior is isotropic in the embedding axis, so the weights are
not identifiable while the field is.

```python title="Comparison"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:compare"
```

Four metrics, each blind to something different, so together they say *how* a
fit is wrong rather than only *that* it is:

| Metric | Sensitive to | Blind to |
|---|---|---|
| Relative Frobenius | everything at once | — |
| Area ratio `√(det Σ_fit / det Σ_ref)` | size | shape, orientation |
| Major-axis angle error, folded to [0°, 90°] | orientation | size, shape |
| Normalized Bures Similarity (the paper's own) | shape + orientation | **size, exactly** |



**What counts as good.** Rather than invent a tolerance, we use the authors'
own run-to-run spread: each of their 120 bootstraps against their main fit, on
the same 49 points.

The authors refit their own model 120 times on resampled data. This table shows how much those refits disagree with their main fit. That's the disagreement you get from finite data alone. If our refit disagrees less than that, it's as good as the data allows. Ours disagrees less on every metric, which makes sense because we didn't resample


| Metric | p5 | median | p95 | Our full refit |
|---|---|---|---|---|
| `rel_frobenius_median` | 0.144 | 0.197 | 0.263 | **0.096** |
| `rel_frobenius_max` | 0.347 | 0.581 | 1.110 | **0.277** |
| `area_ratio_median` | 0.928 | 0.988 | 1.068 | **0.987** |
| `angle_err_deg_median` | 3.47° | 5.30° | 7.75° | **1.41°** |
| `nbs_median` | 0.9945 | 0.9964 | 0.9982 | **0.9995** |
| `nbs_min` | 0.912 | 0.968 | 0.986 | **0.9911** |



<div align="center">
    <img src="../plots/hong2025_full_ellipses.png"
         alt="Sigma_noise: published field vs a full-settings psyphy fit"
         width="560"/>
    <p><em>Σ_noise(x): published (black) vs our full-settings MAP fit (red),
    subject 1 (CH). This is the internal noise field — the paper's
    supplementary Figure S3 — not the threshold contours above.</em></p>
</div>

Three restarts from independent prior draws ended at losses 0.550 / 0.512 /
0.505, with no sign of a multimodal landscape.

!!! warning "Scope"
    One subject (CH, 1 of 8), one run, one GPU. Not repeated for seed stability,
    not run for the other seven. Read it as "the fitting pipeline reproduces the
    paper for this subject," not as a general guarantee.

`--mode quick` exists only to prove the code path runs on a laptop: 500 trials
and 20 steps leave the fit essentially at its prior (`nbs_median` 0.950, which
looks like a pass in isolation and is a decisive failure against the envelope
above). Never quote a similarity score without a calibrated reference.

---

## Runtimes

CPU figures are an Apple Silicon laptop (~12 cores); GPU is one CUDA device.

| Step | Hardware | Wall clock | Settings |
|---|---|---|---|
| Thresholds (Figure 2B) | CPU | **20–23 s** | 49 refs, `n_theta=16`, `n_length=300`, `mc=500` |
| Thresholds at paper settings | CPU | ~11 min | `n_length=1000`, `mc=2000` (13.4 s per ref) |
| Exact covariance check | CPU | seconds | 10,609 points, deterministic |
| Refit — quick smoke test | CPU | 0.8 s | 500 trials, 20 steps |
| **Refit — full** | 1 GPU | **~16 min** | 6,000 trials, 1,500 steps, `mc=2000`, 3 restarts |
| Paper's SLURM request | H100 | 14 h | main fit **+ 120 bootstraps** |

**The published figure is a ~20-second laptop computation.** "GPU job" applies
to the refit alone, and the paper's 14-hour budget is dominated by bootstraps,
not by the single fit.

If the refit runs out of memory: `OddityTask.loglik` vmaps over all trials at
once with no chunking, so budget several GB. Lower `mc_samples` first — it
divides memory linearly and only adds gradient noise, whereas cutting trials
discards data.

---

## Watch out for

- **`Σ_noise` and `Σ_thres` are different objects.** The thresholds above are
  Figure 2B; the exact check and the refit compare the noise field, which is
  supplementary Figure S3. Both arrive as `(49, 2, 2)` stacks on the same grid,
  which makes them easy to conflate.
- **Monte Carlo results are not bit-reproducible across platforms.** The exact
  check is exact anywhere; thresholds and refits reproduce to a neighborhood.
  We have seen `rel_frobenius_median` of 2.39 and 2.65 for the same quick-mode
  configuration on different machines.
- **Loss values are not comparable to the paper's.** psyphy's `Prior.log_prob`
  drops a constant the paper keeps — identical gradients, different printed
  numbers.
- **Figure 2C is not reproduced here.** It aggregates the same readout across
  all 8 subjects; we do one. That is a matter of repeating the steps per
  subject, not new machinery.

---

## See also

- [Full WPPM fit (simulated data)](full_wppm_fit_example.md) — same machinery with ground truth available.
- [Quick start](quick_start.md) — the minimal version.
- `psyphy.data.published.hong2025` in [Data](../../reference/data.md); `WPPMPredictivePosterior` and `ThresholdConfig` in [Posterior](../../reference/posterior.md).
