# Reproducing Hong et al. (2025)
This tutorial is accompanied by av
** runnable script: **
[`hong2025_reproduction.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/hong2025_reproduction.py).

```bash
python hong2025_reproduction.py --skip-refit   # everything except the refit, <1 min CPU
python hong2025_reproduction.py --mode full    # add the refit; wants a GPU
```

---

### For who this is

You might find this tutorial of interest
- to see  a worked example of `psyphy` on real data, with an external ground
truth to check against 
- or, if you know the Hong et al paper, this shows how `psyphy` can be used to reproduce its results.

---

This Tutorial shows how to reproduce the key finding shown by Hong et al 2025. They introduce the Wishart Pyschophysical Process Model 

We reproduce **Figure 2B** of Hong et al. (2025) — human color discrimination
thresholds — using psyphy and the authors' own data. Two things happen here:

- we recover their published threshold contours from their published model, 

- and we refit the model from scratch to check that we land where they landed.


> Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P.,
> Williams, A. H., & Brainard, D. H. (2026). *Comprehensive characterization of
> human color discrimination thresholds.* eLife 14:RP108943.
> <https://doi.org/10.7554/eLife.108943.2>

---

## Background — what the Whishart Psychophysical Process Model (WPPM) is

Measuring a discrimination threshold the usual way means fixing one color and
asking, over many trials, how far a second color has to move before someone
notices the difference. That tells you about one color. Repeating it across a
whole plane of colors is impractical — too many locations, far too many trials.

The WPPM takes a different route. It assumes the observer's internal noise
changes *smoothly* across color space: nearby colors are confusable in similar
ways. That lets us fit one smooth field over the entire space instead of many
separate measurements, so every trial informs the whole picture. Once fit, we
can ask the model about any pair of colors — including pairs nobody was ever
shown.

The picture that comes out: discrimination is finest near gray and gets coarser
for more saturated colors, and the threshold ellipses point outward from gray,
so sensitivity depends on direction as well as position.

The authors checked this by holding back trials the model never saw and
measuring thresholds at those points. The two agreed, which is
good evidence that assuming smoothness didn't smooth away real structure.

**What psyphy adds.** psyphy implements the WPPM in general form: any number of
stimulus dimensions, any task you can write a likelihood for. The color setup
here is only one configuration of it, which is why this page doubles as an external
check on psyphy and a worked example of the general machinery. As the authors
already mention in the paper, the approach carries beyond color to any domain where the noise
limiting performance varies smoothly across stimulus space.

---

## The result

<div align="center">
    <img src="../plots/hong2025_thresholds.png"
         alt="Paper Figure 2B reproduced: 66.7%-correct discrimination threshold contours"
         width="620"/>
    <p><em>Colored ellipses are the contours we recover with psyphy; dashed gray
    are the published ones. Each ellipse takes the color of its own reference
    stimulus, via the monitor calibration matrix published with the data.</em></p>
</div>

The paper's own caption for this panel:

> Discrimination threshold contours (66.7% correct) read out from the [Wishart Psychophysical Process Model] WPPM on a
> grid of reference stimuli for a representative participant, based  on fits to
> the 6,000 AEPsych trials.

Each ellipse says how far a comparison color must move from its reference
before this observer distinguishes the two 66.7% of the time. 

## The whole recipe


```python title="Published data to threshold contours"
import jax
jax.config.update("jax_enable_x64", True)   # the authors used float64
import jax.numpy as jnp

from psyphy.data.published import hong2025
from psyphy.posterior import MAPPosterior, ThresholdConfig, WPPMPredictivePosterior

paths = hong2025.fetch(subject=1)                       # download from OSF
W_org = hong2025.load_reference_W(paths["weights"])     # the paper's fitted weights
coords, published = hong2025.load_sigma_table(paths["thres_ellipses"])

# Model: given weights W, how noisy is perception at each color?
model = hong2025.build_paper_model(mc_samples=500)

# Parameter posterior: which W do we believe? 
posterior = MAPPosterior({"W": W_org}, model)
# Search settings: how carefully to look for each threshold

config = ThresholdConfig(n_theta=16, n_length=300)
# Predictive posterior: given what we believe about W, what do we predict here?
thresholds = WPPMPredictivePosterior(
    posterior,
    jnp.asarray(coords),                                # reference points only
    n_samples=1,
    threshold_pred=True,                                # ask for thresholds
    threshold_config=config,
).mean                                                  # -> (49, 2, 2)
```


The sections below will dive deeper into details, such as how to load the data or how to compute the thresholds. 


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

We match the paper's hyper parameters. 


`build_paper_model()` assembles a WPPM from `PAPER_HYPERPARAMS`. 


Toggle down for hyperparameter details:

TODO: Wrap in toggle

| Paper | psyphy | Note |
|---|---|---|
| `degree=5` | `basis_degree=4` | **Off-by-one.** Theirs counts basis *functions* (T₀…T₄); ours is the *maximum degree*. Same 5×5 grid. |
| `variance_scale=3e-4` | same | psyphy's default is `4e-3` |
| `diag_term=0` | same | psyphy's default is `1e-6`; theirs leaves Σ unregularized |
| `mc_samples=2000`, `bandwidth=5e-3` | `OddityTaskConfig` | |
| `learning_rate=1e-4`, `momentum=0.2`, `total_steps=1500`, 3 restarts | `MAPOptimizer`  | refit only |

The published weight tensor is `(5, 5, 2, 3)` — exactly psyphy's `params["W"]`
layout, so it drops straight in with no reshaping.

---

## Thresholds (as in Paper Figure 2B)

The model is parameterized in `Σ_noise(x)`, the covariance of the observer's
_internal representation_. The paper reports **thresholds**, i.e. how much do we have to move in stimulus space, until the observer will notice a difference in 66% of the cases. Those are different
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
squares, then one inverse. 

```python title="Threshold inversion at every published reference point"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:thresholds"
```

Note we only feed the paper's own
weights in via `MAPPosterior` (no data), so the covariance field is identical by
construction and the only thing that can differ is the inversion itself.

Two API details specific to threshold mode:

- **`X` is bare reference points**, `(n_test, input_dim)` — not the paired
  `(n_test, k_stimuli, input_dim)` shape the class takes otherwise. Threshold
  mode generates its own comparisons. Passing the paired shape raises
  `ValueError`.
- **`mean` and `variance` are matrix-valued**, `(n_test, input_dim, input_dim)`
  — one threshold covariance per reference point.

```python title="Compute settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:threshold_settings
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
### does psyphy build the same covariance field Hong et al published?

This is fully
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

---

## Refit
### Does psyphy's fit find the paper's covariance field?

Everything above started from the paper's weights. The stronger question is: given
only their **trials**, does psyphy's fit find their covariance field?

```python title="MAP fit with the paper's optimizer settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fit"
```

Two settings make the paper's `learning_rate=1e-4` mean the same thing here:
`reduction="mean"` (they minimize a per-trial objective) and
`max_grad_norm=None` (they do no clipping, psyphy clips at 1.0 by default,
which would silently rescale the effective learning rate).



```python title="Comparison"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:compare"
```



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
    One subject (CH, 1 of 8), 1 run, 1 GPU. Not repeated for seed stability and 
    not run for the other seven. Read this purely as "the fitting pipeline reproduces the
    paper for this subject,".

`--mode quick` exists only to prove the code path runs on a laptop: 500 trials
and 20 steps leave the fit essentially at its prior 

---

## Runtimes

CPU figures are an Apple Silicon laptop (~12 cores); GPU is one CUDA device.

| Step | Hardware | Wall clock | Settings |
|---|---|---|---|
| Thresholds (Figure 2B) | CPU | **20–23 s** | 49 refs, `n_theta=16`, `n_length=300`, `mc=500` |
| Thresholds at paper settings | CPU | ~11 min | `n_length=1000`, `mc=2000` (13.4 s per ref) |
| Exact covariance check | CPU | seconds | 10,609 points, deterministic |
| **Refit — full** | 1 GPU | **~16 min** | 6,000 trials, 1,500 steps, `mc=2000`, 3 restarts |
| Paper's SLURM request | H100 | 14 h | main fit **+ 120 bootstraps** |


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
  drops a constant, which the paper keeps — still  identical gradients but different numbers


---

## See also

- [Full WPPM fit (simulated data)](full_wppm_fit_example.md) — same machinery with ground truth available.
- [Quick start](quick_start.md) — the minimal version.
- `psyphy.data.published.hong2025` in [Data](../../reference/data.md); `WPPMPredictivePosterior` and `ThresholdConfig` in [Posterior](../../reference/posterior.md).
