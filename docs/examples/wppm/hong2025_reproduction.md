# Reproducing Hong et al. (2025)
This tutorial is accompanied by a
**runnable script: **
[`hong2025_reproduction.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/hong2025_reproduction.py).

```bash
python hong2025_reproduction.py --skip-refit   # everything except the refit, <1 min CPU
python hong2025_reproduction.py --mode full    # add the refit; wants a GPU
```

---

This tutorial shows how to reproduce the key finding shown by Hong et al 2025. They introduce the Wishart Pyschophysical Process Model, which allows for a comprehensive characterization of human color discrimination thresholds.

More specifically, we reproduce **Figure 2B** of Hong et al. (2025) — the human color discrimination
thresholds — using psyphy and the authors' own data. Two things happen here:

- we recover their published threshold contours

- and we refit the model from scratch to check that we land where they landed.

To that end, you might find this tutorial of interest
- to see  a worked example of `psyphy` on real data, with an external ground
truth to check against
- or, if you know the Hong et al paper, this shows how `psyphy` can be used to reproduce its results.


> Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P.,
> Williams, A. H., & Brainard, D. H. (2026). *Comprehensive characterization of
> human color discrimination thresholds.* eLife 14:RP108943.
> <https://doi.org/10.7554/eLife.108943.2>

---

## Background — what the Whishart Psychophysical Process Model (WPPM) is

Measuring a discrimination threshold the usual way means fixing one color and
asking, over many trials, how far a second color has to move before someone
notices the difference. That tells you about one color. Repeating it across a
whole plane of colors is impractical: too many locations and far too many trials, so we run into the curse of dimensionality.

The WPPM takes a different approach. It assumes the observer's internal noise
changes *smoothly* across color space, or more generally, the input space: nearby colors are confusable in similar
ways. That lets us fit one smooth field over the entire space instead of many
separate measurements, so every trial informs the whole picture. Once fit, we
can evaluate the model at any point in stimulus space.

**What psyphy adds.** psyphy implements the Wishart Psychophysical Process Model (WPPM) in general form: any number of
stimulus dimensions (doesn't have to be color), any task you can write a likelihood for. The color setup
here is only one configuration of it, which is why this page doubles as an external
check on psyphy and a worked example of the general machinery. The WPPM approach carries beyond color to any domain where the noise
limiting performance varies smoothly across input space.

---

## The result

Each ellipse can be thought of as a *Just-Noticable Distance (JND)* around a reference color (center): the smallest difference in a color a person can detect. Here, it's operationalized as
 how far a comparison color must move from its reference
before this observer distinguishes the two 66.7% of the time, determining the size of the ellipse.

<div align="center">
    <img src="../plots/hong2025_thresholds.png"
         alt="Paper Figure 2B reproduced: 66.7%-correct discrimination threshold contours"
         width="620"/>
    <p><em>Colored ellipses are the contours we recover with psyphy; dashed gray
    are the published ones. Each ellipse takes the color of its own reference
    stimulus (center dot), via the monitor calibration matrix published with the data. The dimensions of the figure here are called model dimensions and are arbitrary in that they can result from any affine transformation of the input space, here RGB from the isoluminant plane. </em></p>
</div>



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

??? note "Paper -> psyphy parameter mapping"

    Most settings map one to one. The ones worth knowing:

    | Paper | psyphy | Note |
    |---|---|---|
    | `degree=5` | `basis_degree=4` | **Off-by-one.** Theirs counts basis *functions* (T₀…T₄); ours is the *maximum degree*. Same 5×5 grid. |
    | `variance_scale=3e-4` | same | psyphy's default is `4e-3` |
    | `diag_term=0` | same | psyphy's default is `1e-6`; theirs leaves Σ unregularized |
    | `mc_samples=2000`, `bandwidth=5e-3` | `OddityTaskConfig` | |
    | `learning_rate=1e-4`, `momentum=0.2`, `total_steps=1500`, 3 restarts | `MAPOptimizer` | refit only |

    The published weight tensor is `(5, 5, 2, 3)` , which is exactly psyphy's `params["W"]` layout.



---

## Thresholds (as in Paper Figure 2B)

The model is parameterized in `Σ_noise(x)`, the covariance of the observer's
_internal representation_. The paper reports **thresholds**, i.e. how much do we have to move in stimulus space, until the observer will notice a difference in 66% of the cases. Those are different
objects! The map between them is as follows:


$$
\begin{aligned}
\text{forward (psyphy's OddityTask)}:\qquad
  & \Sigma_{\text{noise}}(x_{\text{ref}}),\ \Sigma_{\text{noise}}(x_{1})
  && \longrightarrow\ P(\text{correct}) \\[4pt]
\text{inverse (what Figure 2B plots)}:\qquad
  & P(\text{correct}) = \tfrac{2}{3}
  && \longrightarrow\ x_{1}
\end{aligned}
$$


There is no closed form for the inverse. `P(correct)` for the 3-alternative
oddity task is the probability that `min(d_02, d_12) > d_01` over three correlated
quadratic forms, which is why the paper estimates it by
[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) in the
first place. So we invert numerically:

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

### Plotting it

Both contour fields go on one axes in a single
[`plot_ellipses`](../../reference/viz.md) call: published dashed underneath, ours on
top colored by reference stimulus:

```python title="The plotting call"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:plot_call"
```

`scale` comes from `auto_scale(coords, thres_published)` and `colors` from
`hong2025.w2d_to_rgb(coords, M)`. Passing **one** `scale` for both fields is the
point. Independently scaled fields cannot be compared by eye. `plot_ellipses` draws into an axes
and returns it. It never saves or shows *for* you, so you style the figure
first and then `fig.savefig(...)` when you're ready.


---

Now that we've seen how to reproduce we the key findings from the paper, we


## Exact check
### does psyphy build the same covariance field Hong et al published?

This is fully
deterministic.

```python title="Published weights through psyphy's covariance field"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:stage1"
```


Plain elementwise subtraction over all 42,436 entries. The
published CSV stores 8 decimals (the differences are tiny!):

```
max |diff|   : 6.778e-09
mean |diff|  : 2.538e-09
```

 Our values round to theirs exactly in 96% of
cases and agree to within one unit in the last printed digit in 100%. **This is
agreement to the precision the file can express.**

This runs as a test (`test_covariance_field_matches_published_sigma_noise`),
skipped automatically when the data has not been downloaded, so CI stays
network-free.

---

## Refit
### Does psyphy's fit find the paper's covariance field?

Everything above started from the paper's weights. The stronger question is: given
only the paper's **trials**, does psyphy's fit find the paper's covariance field?

Looking at the alignment of the ellipses in the figure below, the answer to that question is yes.

```python title="MAP fit with the paper's optimizer settings"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fit"
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

The full refit refit needs a GPU,
and that is **~16 min** but there's quick mode available to check the whether the script runs.

??? note "Measured runtimes, step by step"

    CPU figures are an Apple Silicon laptop (M5); GPU is one A100 unless otherwise noted.

    | Step | Hardware | Wall clock | Settings |
    |---|---|---|---|
    | Thresholds (Figure 2B) | CPU | **20–23 s** | 49 refs, `n_theta=16`, `n_length=300`, `mc=500` |
    | Thresholds at paper settings | CPU | ~11 min | `n_length=1000`, `mc=2000` (13.4 s per ref) |
    | Exact covariance check | CPU | seconds | 10,609 points, deterministic |
    | **Refit — full** | 1 GPU | **~16 min** | 6,000 trials, 1,500 steps, `mc=2000`, 3 restarts |
    | Paper's SLURM request | H100 | 14 h | main fit **+ 120 bootstraps** |

    The paper's 14-hour budget covers the main fit *plus* 120 bootstrap refits,
    not a single fit.


---

## Watch out for

- **`Σ_noise` and `Σ_thres` are different things.** The thresholds above are
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
- [Plotting ellipse fields](../viz/ellipse_plots.md) — `plot_ellipses` on its own, with synthetic data.
- `psyphy.data.published.hong2025` in [Data](../../reference/data.md); `WPPMPredictivePosterior` and `ThresholdConfig` in [Posterior](../../reference/posterior.md).
