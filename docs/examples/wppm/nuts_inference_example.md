# Sampling the posterior with NUTS

The [MAP tutorial](full_wppm_fit_example.md) gives one threshold ellipse per reference
point. Here we get a distribution of ellipses instead. Tight clusters mean the data
constrain the threshold ellipse and wide spread means they don't. A single MAP ellipse hides
that difference. Crucially, adaptive trial placement depends on this uncertainty, because
the next trial should go where the model is least certain.

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty_full.jpg" width="560"/>
  <p><em>What we build in this tutorial. The axes are the two model dimensions, arbitrary up to
  an affine transformation of the input space (e.g., RGB). Each green dot is one of the 25 reference
  points at which the model is compared against the ground truth. Around each of them, black
  is the true threshold contour, red is the posterior mean \(\mathbb{E}[\Sigma(x)]\), and the
  200 faint blue contours are individual draws from the posterior. Built in <a href="#step-6-threshold-uncertainty">Step 6</a>.</em></p>
</div>

The companion script is
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).

| step | what we do | what we get |
|---|---|---|
| 1–2 | simulate data, build the WPPM | a [`TrialData`](../../reference/data.md#psyphy.data.TrialData) and a model, as in the [MAP tutorial](full_wppm_fit_example.md) |
| 3 | run a short MAP fit | a starting point near the posterior peak |
| 4 | run the sampler, here NUTS, via [BlackJAX](https://blackjax-devs.github.io/blackjax/) | an `MCMCPosterior`: thousands of plausible parameter arrays $W$ |
| 5 | check the chains with [ArviZ](https://python.arviz.org/) | evidence that the draws are representitive of the posterior, or are not|
| 6 | evaluate the covariance field for every draw | a distribution of threshold contours, and a credible band |

??? note "How to run the script"

    The following 3 dependencies are required, none of which psyphy requires by default:

    ```bash
    pip install 'psyphy[sampling,diagnostics,viz]'
    ```

    | extra | brings in | needed for |
    |---|---|---|
    | `sampling` | `blackjax>=1.4,<2` | `NUTSSampler` |
    | `diagnostics` | [`arviz>=1.0`](https://python.arviz.org/en/latest/user_guide/getting_started.html) | `covariance_to_arviz`, R-hat, ESS, the diagnostic figures |
    | `viz` | `matplotlib>=3.5` | the figures |

    If you want to run the companion script
    [`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py)
    yourself to reproduce the results shown here, you have three options: `--mode` selects a
    compute preset.

    ```bash
    python nuts_inference_example.py --mode quick      # smoke test, ~1 min on an A100
    python nuts_inference_example.py --mode balanced   # a real fit, ~35 min on an A100
    python nuts_inference_example.py --mode full       # 11 h on a GPU, and what we show here
    ```

    Every figure and number here comes from `--mode full`. `quick` is a smoke test for the
    code path. Its chains are far too short to converge, so its diagnostics are not a result.

    Only MAP and NUTS are expensive, and the script caches them. The diagnostics and all
    figures are post-processing of the stored draws and need no GPU:

    ```bash
    python nuts_inference_example.py --mode full --from-fit
    ```

    The cache (`fits/nuts_full_fit.npz`) is committed and records the settings it was
    produced with.

---

## Why sample the posterior

MAP estimation returns the single parameter array that maximizes the posterior:

$$\theta_\text{MAP} = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

That locates the peak but doesn't tell us anything  about its width. What we want is the posterior
$p(\theta \mid \mathcal{D})$ itself, and Bayes' rule gives it in one line containing an
integral over all of parameter space that we cannot evaluate here.

[Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) gets
around this by constructing a random walk whose long-run visiting frequency is the
posterior, so the places it has been form a sample from it. A plain random walk mixes too
slowly to be usable in the roughly 100 dimensions of our parameteres $W$, so
[Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) uses the
gradient of the log posterior to move along the distribution's geometry instead. HMC needs
two settings: the step size, and how many steps to take before stopping. NUTS, the No-U-Turn
Sampler of [Hoffman and Gelman (2014)](https://jmlr.org/papers/v15/hoffman14a.html), removes
the second by extending each trajectory until it starts to double back on itself. The step
size is still tuned, during warmup ([Step 4](#step-4-nuts-sampling)). Betancourt's
[conceptual introduction](https://arxiv.org/abs/1701.02434) covers why this works and where
it fails.

Two things follow for psyphy:

1. The sampler needs gradients, which is why the log posterior is written in JAX and
   `NUTSSampler` hands it straight to BlackJAX.
2. The sampler needs the log density to be a fixed function of the parameters $\theta$. Since psyphy's  oddity
   likelihood is estimated by simulation introducing slight randomness, psyphy freezes one PRNG key for the whole run
   (`logdensity_key_seed`) to make it deterministic.

What we get in exchange for the extra compute:

- **Uncertainty quantification**: how spread out is our belief over model parameters?
- **Posterior predictive distributions**: how uncertain are our predictions at new test points?

??? note "What the frozen key costs"

    [`OddityTask`](../../reference/model.md#psyphy.model.OddityTask) has no closed-form likelihood. It estimates $P(\text{correct})$ by
    simulating `mc_samples` noisy internal representations per trial, which makes the log
    density expensive and slightly different on every call. Freezing one key removes the
    variation, at the cost of sampling the posterior implied by that one noise realization
    rather than the exact posterior. More `mc_samples` reduces the discrepancy and costs
    proportionally more. A deterministic surrogate likelihood would remove the compromise.
    See future `NeuralSurrogateTask`.

---

## Step-by-step walkthrough

### Step 1: Simulate data

Same setup as the [MAP tutorial](full_wppm_fit_example.md): oddity-task trials from a known
ground-truth WPPM, so there is something to compare the fit against.

```python title="Data container"
--8<-- "docs/examples/wppm/nuts_inference_example.py:data"
```

### Step 2: Build the model

```python title="Model definition"
--8<-- "docs/examples/wppm/nuts_inference_example.py:build_model"
```

??? note "Recap: the model and its parameters"

    The model is a [`Prior`](../../reference/model.md#psyphy.model.Prior) over the parameters, plus a task likelihood and a noise model.
    Every diagnostic and figure below refers to the parameters, so here is what they are.

    **Parameters.** The WPPM fits a covariance *field* $\Sigma(x)$: a $2\times2$ matrix at
    each point $x$ in stimulus space. It is defined as

    $$\Sigma(x) = U(x)\,U(x)^\top + \lambda I, \qquad U(x) = \sum_{j} W_j \,\phi_j(x)$$

    The $\phi_j$ are fixed Chebyshev basis functions, up to degree 2 per input dimension.
    Only the coefficients $W_j$ are fitted. With `input_dim=2`, `basis_degree=2` and
    `extra_embedding_dims=1` they form an array of shape `(3, 3, 2, 3)`. $\lambda$
    (`diag_term`) is a small floor that keeps $\Sigma$ invertible.

    For the basis, the prior and `diag_term` in more depth, see the
    [full WPPM fit](full_wppm_fit_example.md).

### Step 3: MAP warm-start

A sampler started far from where the posterior mass lives spends its first stretch walking
there, and those draws are discarded. Starting at the MAP estimate skips that walk, so
warmup can be spent on tuning instead.

```python title="MAP warm-start"
--8<-- "docs/examples/wppm/nuts_inference_example.py:map_warmstart"
```

`map_posterior.params` then goes to `NUTSSampler.fit()` as `init_params`.

### Step 4: NUTS sampling

```python title="NUTS sampling with NUTSSampler"
--8<-- "docs/examples/wppm/nuts_inference_example.py:nuts_fit"
```

`NUTSSampler` has the same `fit(model, data)` interface as [`MAPOptimizer`](../../reference/inference.md#psyphy.inference.MAPOptimizer), and the BlackJAX
backend stays hidden. Each chain runs a warmup phase, where the step size and mass matrix
are tuned and the draws discarded, then a sampling phase where the tuning is frozen and
every draw is kept. What comes back is an `MCMCPosterior`. Three things on it are used
below:

| call | returns | used in |
|---|---|---|
| `nuts_posterior.sample(n, key=...)` | `n` draws, `{"W": (n, *W_shape)}` | Step 6, the contour ensemble |
| `PosteriorMeanCovarianceField(nuts_posterior)` | the posterior mean field $\mathbb{E}[\Sigma(x)]$ | Step 6, the red contours |
| `covariance_to_arviz(nuts_posterior, points)` | $\Sigma$ at chosen points for every draw, in ArviZ's format | Step 5 |

One draw is one complete covariance field, so one threshold contour at every reference
point. `MCMCPosterior` and [`MAPPosterior`](../../reference/posterior.md#psyphy.posterior.MAPPosterior) share this interface, so code written for a MAP
fit, [`WPPMPredictivePosterior`](../../reference/posterior.md#psyphy.posterior.WPPMPredictivePosterior) for instance, takes a NUTS fit unchanged.

!!! warning "Do not build a covariance field from `nuts_posterior.params`"
    `nuts_posterior.params` is the mean *parameter* $\mathbb{E}[W]$. It is a valid parameter
    set and works as `init_params`, but
    [`WPPMCovarianceField`](../../reference/model.md#psyphy.model.WPPMCovarianceField)`(model, nuts_posterior.params)`
    gives contours that are far too small, because $\Sigma$ is quadratic in $W$ and $W$ is
    only identified up to a rotation. Use `PosteriorMeanCovarianceField`
    ([Step 6](#step-6-threshold-uncertainty)), whose docstring has the full argument.
    `WPPMCovarianceField.from_posterior` warns when handed an MCMC posterior.

### Step 5: Check the chains with ArviZ

Sampling produces draws whether or not they represent the posterior, and MCMC fails quietly:
for example, a chain can look like a result while having explored only part of the distribution.
[ArviZ](https://python.arviz.org/) is the standard Python library for the check, and psyphy
provides the bridge to it. Four quantities, with the upstream reference for each:

| check | what it detects | suspect when |
|---|---|---|
| divergences, `sampler_stats["diverging"]` | the integrator failed, usually in a region of high curvature | any divergence at all |
| [R-hat](https://python.arviz.org/projects/stats/en/latest/) | chains have not mixed to the same distribution | $\hat{R} > 1.01$ |
| [ESS](https://python.arviz.org/projects/stats/en/latest/) | autocorrelation: how many independent draws the chain is worth | bulk or tail ESS $< 100$ per chain |
| `sampler_stats["n_steps"]` | trajectories truncated at the tree-depth cap | at the cap for most draws |

Diagnostics run on $\Sigma$, not on the raw parameters, for the reason in
[Step 4](#step-4-nuts-sampling). `covariance_to_arviz` evaluates $\Sigma$ at points we
choose, here the reference points, and returns it in ArviZ's format. It takes the posterior
itself rather than draws from it, because R-hat compares chains against each other and
`posterior.sample()` would pool them.

```python title="R-hat and ESS on the covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

```python title="Trace plot at the worst reference point"
--8<-- "docs/examples/wppm/nuts_inference_example.py:trace_plot"
```

<div align="center">
  <img src="../plots/nuts_trace_sigma_full.png" width="760"/>
  <p><em>One panel per quantity at the reference point with the worst R-hat, one colored
  line per chain. Horizontal axis is the draw index, vertical axis the value of that
  quantity.</em></p>
</div>

`covariance_to_arviz` summarizes each $2\times2$ $\Sigma$ by the three numbers that
determine it: `log_var_0` and `log_var_1`, the log variances along each axis, and
`corr_0_1`, their correlation. A trace plot shows how each of those moved over the run. We
are looking for the four chains to overlap, to stay level, and to vary
fast from draw to draw. A chain sitting at its own offset, or still sloping at the end of
the run, has not mixed.

```python title="ESS and R-hat over all quantities"
--8<-- "docs/examples/wppm/nuts_inference_example.py:convergence_plot"
```

<div align="center">
  <img src="../plots/nuts_convergence_sigma_full.png" width="760"/>
  <p><em>Bulk ESS, tail ESS and R-hat, one panel each. The curve is the cumulative fraction
  of quantities at or below a value, so the vertical axis runs from 0 to 1 and the
  horizontal axis carries the diagnostic. Dashed lines are ArviZ's reference
  thresholds.</em></p>
</div>

The trace plot covers one point, this covers all of them at once. Each curve is the
distribution of a diagnostic over every $\Sigma$ quantity at every reference point, so a
point on the curve reads as "this fraction of quantities scored at most this value". More is
better for ESS and less is better for R-hat, which means we want both ESS curves to sit
entirely to the right of their dashed line and the R-hat curve entirely to its left. A curve
crossing its threshold says what fraction of quantities fall short.

This run has no divergences, and the printed `R-hat max` and `ESS min` above are the two
numbers to quote.

A failing check is fixed by the sampler arguments, which
[`blackjax.nuts`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html)
and
[`window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html)
document. psyphy's `NUTSSampler` passes them straight through under the same names.

### Step 6: Threshold uncertainty

Every draw is a complete covariance field, so the ensemble of draws is an ensemble of
threshold contours.

```python title="Draw posterior parameter samples"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_samples"
```

For one summary field, average the covariance matrices over draws.
`PosteriorMeanCovarianceField` evaluates $\Sigma$ per draw and averages afterwards, which is
the reason it exists rather than `WPPMCovarianceField(model, posterior.params)`:

```python title="Posterior mean covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_mean"
```

```python title="Plot the ensemble"
--8<-- "docs/examples/wppm/nuts_inference_example.py:plot_uncertainty"
```

This produces the figure at the top of the page.

A cloud of contours is the full answer, but a paper reports an interval. psyphy computes
one with [`threshold_band`](../../reference/model.md#psyphy.model.threshold_band), which
takes the posterior and the points to evaluate at.

??? note "What to pass, and what comes back"

    | argument | what to pass |
    |---|---|
    | `posterior` | the fitted posterior, here the `MCMCPosterior` from Step 4 |
    | `points` | the reference points, `(n_points, 2)` |
    | `ci_prob` | how much posterior mass the interval holds. `0.89` by default, matching ArviZ |
    | `ci_kind` | `"eti"` for equal-tailed, `"hdi"` for the narrowest interval of that mass |
    | `n_samples`, `key` | how many draws to use, and the key to draw them with |

    What comes back is a
    [`ThresholdBand`](../../reference/model.md#psyphy.model.ThresholdBand): `lower`,
    `median` and `upper` radii, each `(n_points, n_directions)`, plus the `directions` they
    were measured along and the `ci_prob` and `ci_kind` that produced them, so a figure
    caption can state which interval it draws.

    Passing a sequence, `ci_prob=[0.5, 0.8, 0.95]`, returns nested bands for one figure. The
    three arrays then gain a leading level axis and keep the same shape as each other, so a
    loop over levels can index all of them.

It works by writing each contour as a radius per direction: along the unit direction
$u(\varphi) = (\cos\varphi, \sin\varphi)$, draw $i$'s contour at reference point $x$ sits at

$$r_i(\varphi) \;=\; \frac{1}{\sqrt{u(\varphi)^\top \Sigma_i(x)^{-1} u(\varphi)}}$$

which is the discrimination threshold, the just noticeable difference, in direction
$\varphi$. That is a scalar, one number per draw, so quantiles over draws give an interval
at each direction. Filling between them gives us the credible band.

??? note "Where the radius formula comes from"

    The threshold contour of draw $i$ is the set $\{d : d^\top \Sigma_i(x)^{-1} d = 1\}$.
    Along a fixed unit direction $u$, writing $d = r\,u$ gives
    $r^2 u^\top \Sigma_i^{-1} u = 1$, hence the expression above. Along an eigenvector of
    $\Sigma_i$ it reduces to $\sqrt{\lambda}$, the semi-axis the ellipse plots already draw,
    so this is the same contour in polar form.

    For a $1-\alpha$ interval, at each direction separately, the band runs between
    $q_{\alpha/2}\{r_i(\varphi)\}$ and $q_{1-\alpha/2}\{r_i(\varphi)\}$ over the $N$ draws.
    [`threshold_radii`](../../reference/model.md#psyphy.model.threshold_radii) computes
    $r_i(\varphi)$ on its own, in any dimension, for anyone who wants a quantile psyphy does
    not offer.

```python title="Pointwise credible band on the threshold contour"
--8<-- "docs/examples/wppm/nuts_inference_example.py:credible_band"
```

<div align="center">
  <img src="../plots/nuts_threshold_band_full.jpg" width="620"/>
  <p><em>Pointwise 89% credible band on each threshold contour (blue fill), the posterior
  median contour (blue line) and ground truth (black).</em></p>
</div>

The band is narrow where the data constrain the threshold and wide where they do not. A MAP
fit reports the same contours with no way to tell those two cases apart.

!!! warning "This band is pointwise, not simultaneous"
    The interval covers the true radius in each direction separately with probability
    $1-\alpha$. It does not say that the whole true contour lies inside the band with
    probability $1-\alpha$, which is a stronger statement needing a different construction.
    An 89% pointwise band is expected to miss some directions, so the ground truth leaving
    it in places is not a failure of the fit.

---

## Reference: tuning and runtime

Three collapsed blocks, none of which the walkthrough depends on.

??? note "Sampler arguments"

    | argument | what it controls | raise it when |
    |---|---|---|
    | `num_warmup` | tuning steps per chain, discarded | trace plots still drift at the start |
    | `num_samples` | kept draws per chain | ESS is too low |
    | `num_chains` | independent chains, R-hat needs at least 2 | a stronger convergence check is wanted |
    | `max_num_doublings` | tree-depth cap, at most $2^k - 1$ gradients per draw | most draws hit the cap, which the script prints |
    | `target_acceptance_rate` | how cautious the tuned step size is | divergences appear |

    [`blackjax.nuts`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html)
    and
    [`window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html)
    document them upstream. `NUTSSampler` passes them through under the same names.

??? note "Compute presets"

    ```python title="Compute settings"
    --8<-- "docs/examples/wppm/nuts_inference_example.py:compute_settings"
    ```

    The presets scale three separate things: how much data there is, how accurately the
    likelihood is estimated (`mc_samples`), and how long the sampler runs. The first two
    change the posterior being targeted, so a `quick` fit is a smaller problem rather than a
    rough version of a `full` one.

    ??? note "Why a run costs what it costs"

        - **The Monte Carlo likelihood dominates.** Each gradient evaluation simulates
          `mc_samples` draws for every trial, so cost scales with `mc_samples × n_trials`.
          `quick` to `full` raises that product by about 440 times.
        - **Cost per draw is not fixed.** NUTS needs $O(2^{\text{tree depth}})$ gradient
          evaluations and the depth is data dependent, commonly 16 to 64. The script times
          one gradient and prints a bracketed projection before sampling starts.
        - **Tree depth is capped.** The default `max_num_doublings=8` allows at most
          $2^8 - 1$ gradient evaluations per draw. If `sampler_stats["n_steps"]` sits at the
          cap for most draws, trajectories are being truncated: still valid, but mixing more
          slowly.
        - **Warmup starts from a prior-scaled metric.** By default the sampler works in
          $z = W / \sigma_\text{prior}$ (`prior_preconditioning=True`), so adaptation does
          not begin from an identity mass matrix mismatched to the coefficient scales.
        - **Chains are vmapped.** All chains advance in one compiled call per phase. Under
          vmap each step costs as much as the deepest tree across chains, which is another
          reason for the depth cap. On a GPU four chains cost about as much as one. On a
          CPU, many chains can be slower than running them one at a time.

??? note "What `step_size` does"

    `step_size` picks between two code paths. The paths are BlackJAX's, but the choice is
    psyphy's: BlackJAX has no notion of modes, and `NUTSSampler` dispatches on whether this
    argument is `None`.

    | `step_size` | what happens | call |
    |---|---|---|
    | `None` (default) | `window_adaptation` tunes the step size and mass matrix per chain, starting from a prior-scaled metric | `NUTSSampler(num_warmup=200, num_samples=500)` |
    | a float | no adaptation and an identity mass matrix. Warmup becomes plain burn-in, so it is faster, and the step size has to be chosen by hand | `NUTSSampler(..., step_size=0.01)` |

    Two things in the adaptive path are psyphy's own rather than BlackJAX's.
    `prior_preconditioning=True` samples $z = W / \sigma_\text{prior}$ and maps the draws
    back to $W$, so adaptation starts from a metric matched to the prior scales instead of
    the identity. And `max_num_doublings` defaults to 8 here against BlackJAX's own 10.

---

## Next steps

- **Predictions at new stimuli.** `WPPMPredictivePosterior` takes this posterior and gives
  the posterior predictive distribution over responses, averaging $P(\text{correct})$ over
  draws instead of over one point estimate.
- **Adaptive trial placement.** The spread shown here is what an acquisition function needs
  in order to choose where the next trial goes.
- **Other samplers.** Nothing downstream of the sampler is NUTS-specific: any
  [`InferenceEngine`](../../reference/inference.md#psyphy.inference.InferenceEngine)
  returning an `MCMCPosterior` with draws shaped `(n_chains, n_draws, *W_shape)` works with
  `covariance_to_arviz`, `PosteriorMeanCovarianceField` and `WPPMPredictivePosterior`
  unchanged. Note that not all the diagnostics carry over, since a sampler without trajectories has
  no divergences, for example.

## Further reading

- API reference: [`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler),
  [`InferenceEngine`](../../reference/inference.md#psyphy.inference.InferenceEngine),
  [`MCMCPosterior`](../../reference/posterior.md#psyphy.posterior.MCMCPosterior),
  [`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz),
  [`PosteriorMeanCovarianceField`](../../reference/model.md#psyphy.model.PosteriorMeanCovarianceField),
  [`threshold_band`](../../reference/model.md#psyphy.model.threshold_band),
  [`threshold_radii`](../../reference/model.md#psyphy.model.threshold_radii)
- MAP fitting and the model in more depth:
  [full WPPM fit](full_wppm_fit_example.md).
- [BlackJAX](https://blackjax-devs.github.io/blackjax/) for the sampler and its arguments.
- [ArviZ](https://python.arviz.org/) for the diagnostics, with its
  [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html)
  and the [arviz-stats API](https://python.arviz.org/projects/stats/en/latest/).

> Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: adaptively setting path
> lengths in Hamiltonian Monte Carlo.* Journal of Machine Learning Research, 15, 1593-1623.
> [jmlr.org](https://jmlr.org/papers/v15/hoffman14a.html)
>
> Betancourt, M. (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo.*
> [arXiv:1701.02434](https://arxiv.org/abs/1701.02434)
