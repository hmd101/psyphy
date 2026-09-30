# Reproducing Hong et al. (2025)
This tutorial is accompanied by a
**runnable script:**
[`hong2025_reproduction.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/hong2025_reproduction.py).


??? note "How to run the script"

    Everything on this page comes from one script. Pick a mode by how much
    compute you want to spend:

    ```bash
    # stages 1 and 2 only: the exact check and Figure 2B. <1 min on CPU.
    python hong2025_reproduction.py --skip-refit

    # add stage 3, the refit at the paper's settings. Wants a GPU.
    python hong2025_reproduction.py --mode full
    ```

    To check the code path runs on your laptop before committing to any of
    that:

    ```bash
    # a smoke test, not a reproduction: 500 trials and 20 steps
    # leave the fit essentially at its prior.
    python hong2025_reproduction.py --mode quick
    ```



---

# Reproducing Hong et al. (2025)

Hong et al. measured how finely people can tell colors apart, across a whole
plane of colors rather than at a handful of points. This page reproduces their
central figure from their own published data, in three stages: an exact
check of the model's arithmetic, the threshold contours of Figure 2B, and a
refit from their raw trials to see whether we land where they landed.

**Who this is for**

- You want a worked example of psyphy on real data, with an external ground
  truth to check against.
- You know the paper and want to see how psyphy reproduces it.

No familiarity with the model is needed to start. The next section introduces it
at a high level, [the simulated-data tutorial](full_wppm_fit_example.md) goes
further, and the paper itself is the full reference:


> Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P.,
> Williams, A. H., & Brainard, D. H. (2025). *Comprehensive characterization of
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
check on psyphy and a worked example of the general pipeline. The WPPM approach carries beyond color to any domain where the noise
limiting performance varies smoothly across input space.

**The task.** Hong et al. collect each judgement from the human subjects with an **oddity task**: on
every trial the observer sees three stimuli — two identical, one different —
and picks the odd one out. Chance is therefore 1/3, and the threshold is placed
at the usual midpoint between chance and perfect performance,
`P(correct) = 2/3`. That is the 66.7% contour this page reproduces.

---

## The result

Each ellipse is a *Just-Noticeable Difference (JND)* threshold contour around a reference
color at its center: the smallest color difference this observer can reliably
detect. Operationally, it is how far a comparison color must move from the
reference before they pick it out as the odd one 66.7% of the time. It is an
ellipse rather than a circle because sensitivity depends on *direction* — some
color changes are easier to see than others of the same physical size. The
orientation and elongation of each ellipse are exactly what the WPPM estimates.

That sensitivity also scales with the baseline stimulus, which is the
[Weber–Fechner law](https://en.wikipedia.org/wiki/Weber%E2%80%93Fechner_law).
psyphy can recover it from simulated data — see
[Recovering Weber's Law](weber_law.md) for a worked example on a
one-dimensional stimulus.


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

`psyphy` ships no data. The OSF node carries no explicit license, so we download
on request into `~/.cache/psyphy/`

```python title="Download one observer's files"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:fetch"
```

??? note "What each data file is, and how big"

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

Two conventions psyphy handles for us:

- **Coordinates are already in the Chebyshev domain** `[-1, 1]`, so no
  normalization is needed.
- **Oddity trials are stored with `K=2`, not 3.** The task presents three
  stimuli, reference, reference, comparison, but only **two distinct** ones,
  and `K` counts the distinct stimuli. The duplication lives in the likelihood,
  not in the stored data.

---

## Model

We match the paper's hyperparameters.


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
_internal representation_. The paper reports **thresholds**, i.e., how much do we have to move in stimulus space, until the observer picks it out as the odd one 66.7% of the time. Those are different
objects! The map between them runs in two directions, and only the forward pass is easy:

- **Forward**: given the noise at two points, how often does the observer get
  the trial right? That is what the model computes directly.
- **Inverse**: given that they get it right two-thirds of the time, how far
  apart were the stimuli? That is what Figure 2B plots and it is the
  direction with no closed form.

Written out:

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
oddity task is the probability that `min(d_02, d_12) > d_01`, where d_ij refers to the Mahalanobis distance between any two stimuli representations, which is why the paper estimates it by
[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) in the
first place. So we invert numerically:

1. Probe `n_theta` directions around each reference point.
2. Along each, evaluate `P(correct)` at `n_length` distances and keep the one
   closest to 2/3. One boundary point per direction.
3. Fit an ellipse to those points.



```python title="Threshold inversion at every published reference point"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:thresholds"
```


??? note "Why threshold mode takes and returns different shapes"

    `threshold_pred` selects which direction of the map above you are asking
    for, so both the input and the output change shape with it.

    | | `threshold_pred=False` | `threshold_pred=True` (used here) |
    |---|---|---|
    | **Direction** | forward | inverse |
    | **`X` you pass** | assembled trials, `(n_test, k_stimuli, input_dim)` | bare reference points, `(n_test, input_dim)` |
    | **`mean`/`variance` you get** | one probability per trial, `(n_test,)` | one covariance per point, `(n_test, input_dim, input_dim)` |

    **Why bare points go in.** Normally you supply the comparison stimulus and
    the model scores that pair. In threshold mode, *finding* the comparison is
    what we want: the threshold is the distance at which `P(correct)` reaches
    2/3. So, supplying one would be handing over the answer. Instead, it generates its own by sweeping `n_theta` directions by `n_length` distances around
    each reference (that is what `ThresholdConfig` controls).




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
shrinks it, at ca. 30× the runtime.

### Plotting it

Both contour fields go on one axes in a single
[`plot_ellipses`](../../reference/viz.md) call: published dashed underneath, ours on
top colored by reference stimulus:

```python title="The plotting call"
--8<-- "docs/examples/wppm/hong2025_reproduction.py:plot_call"
```

`scale` comes from `auto_scale(coords, thres_published)` and `colors` from
`hong2025.w2d_to_rgb(coords, M)`. We recommend only passing  **one** `scale` for both fields because otherwise the comparison independently scaled fields cannot be
compared by eye.

The rest of the API, such as  `scale="auto"`, per-ellipse colors, posterior draws,
non-positive-definite covariances, and why nothing is saved or shown for you,
is covered in [Plotting ellipse fields](../viz/ellipse_plots.md).


---

That reproduces the published figure, but agreement by eye is the weakest
evidence on this page. Getting there involved a numerical inversion,
[Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) sampling and a
shared plotting scale, so a mismatch could have come from any of them.

The next two sections take those away in order. First a fully deterministic
check: published weights straight through psyphy's covariance field, with no
optimizer and no sampling anywhere. Then the refit, with both back in; so that
if *that* disagrees, we already know the disagreement is the optimizer's and
not the model's.


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
