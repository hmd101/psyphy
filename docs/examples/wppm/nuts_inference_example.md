# NUTS Posterior Inference for WPPM

This tutorial demonstrates **full posterior inference** using the No-U-Turn Sampler (NUTS) for the Wishart Process Psychophysical Model (WPPM).

The [MAP tutorial](full_wppm_fit_example.md) ends with *one* threshold ellipse per reference
point: the model's single best guess. This page ends with a *distribution* of ellipses per reference
point. Where the distribution is
tight, the data pin the threshold down. Where it fans out, they do not, and a single MAP
ellipse would have hidden that.

The companion script can be found at
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).

**The route through this page:**

| step | what you do | what you get |
|---|---|---|
| 1–2 | simulate data, build the WPPM | `TrialData` and a model, same as the MAP tutorial |
| 3 | run a short MAP fit | a starting point near the posterior peak |
| 4 | run NUTS ([BlackJAX](https://blackjax-devs.github.io/blackjax/)) | an `MCMCPosterior`: hundreds to thousands of plausible sets of parameter `W`s |
| 5 | check the chains with [ArviZ](https://python.arviz.org/) | evidence that those draws can be trusted (or that they cannot) |
| 6 | push the draws through the covariance field | a distribution of threshold ellipses at every reference point, overlaid on ground truth and MAP |

??? note "How to run the script"

    An install of psyphy on its own is not enough. This page needs three optional
    dependencies, none of which psyphy requires by default.

    ```bash
    pip install 'psyphy[sampling,diagnostics,viz]'
    ```

    | extra | brings in | what it is for |
    |---|---|---|
    | `sampling` | `blackjax>=1.4,<2` | `NUTSSampler` itself |
    | `diagnostics` | [`arviz>=1.0`](https://python.arviz.org/en/latest/user_guide/getting_started.html) | `MCMCPosterior.to_arviz()`, R-hat, ESS, the trace plot |
    | `viz` | `matplotlib>=3.5` | every figure on this page |


    Everything on this page comes from one script. Pick a preset by how much compute
    you want to spend:

    ```bash
    # smoke test: 9 refs, 50 MC samples, 2 chains x (100 warmup + 150 draws)
    python nuts_inference_example.py --mode quick

    # 25 refs, 500 MC samples, 4 chains x (500 + 500). A real fit.
    python nuts_inference_example.py --mode balanced

    # 25 refs, 2000 MC samples, 4 chains x (1000 + 1000). Paper scale; a batch job.
    python nuts_inference_example.py --mode full
    ```

    `quick` exists so the page can be regenerated quickly (about a minute on an A100). Its
    diagnostics are deliberately not publication grade; see Step 5.

---

## Background: why Markov Chain Monte Carlo (MCMC), and what it is

MAP estimation returns a single point estimate $\theta_\text{MAP}$ for our parameters $W$:

$$\theta_\text{MAP} = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

This tells you *where the posterior peaks*, but not *how wide* it is. If the data are
sparse or ambiguous the posterior may be broad, and the MAP estimate alone cannot tell you
that. For a threshold contour the difference is the whole point: a point estimate gives one
curve, where what you want to report is a curve with a band around it.

What we actually want instead is the posterior $p(\theta \mid \mathcal{D})$, the full distribution
over parameters given the data. Bayes' rule writes it down in one line, but the line
contains an integral over all of parameter space that nobody can evaluate here: `W` has 54
entries in the smallest setting on this page. We cannot plot the posterior, and we cannot
integrate against it directly.

[Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) sidesteps
that. Instead of computing the distribution, you construct a random walk whose long-run
visiting frequency *is* that distribution. Run it long enough and the places it has been
form a sample from the posterior, which is all you need: averages over the sample
approximate averages over the posterior, and the spread of the sample is the uncertainty.
Nothing is ever normalized, which is the trick, because the intractable integral is exactly
the normalising constant.

The challenge is building a walk that explores efficiently. A naive random walk works in two or
three dimensions and fails badly in fifty, where almost every direction you could step in
leads somewhere less probable, so nearly all proposals are rejected and the chain crawls.
[Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) fixes this
by using the *gradient* of the log posterior: it gives the parameter vector a random
momentum and simulates it rolling along the posterior's own geometry, so a single proposal
can travel a long way and still be accepted. **NUTS**, the No-U-Turn Sampler of
[Hoffman and Gelman (2014)](https://jmlr.org/papers/v15/hoffman14a.html), removes the one
awkward tuning knob by extending each trajectory until it starts to double back on itself.
Betancourt's [*Conceptual Introduction to Hamiltonian Monte Carlo*](https://arxiv.org/abs/1701.02434)
is the readable account of why this works and where it fails; it is worth an hour if you
intend to trust these numbers.

Two things follow for psyphy:

1. **The sampler needs gradients**, which is why the log posterior is written in JAX and
   why psyphy's `NUTSSampler` can hand it straight to BlackJAX.
2. **The sampler needs the log density to be a fixed function.** psyphy's oddity likelihood
   is estimated by Monte Carlo simulation, so it is *not* fixed; it returns a slightly
   different number each call. That tension is real, and the determinism note below explains
   what is done about it.

What you get in exchange for the extra compute:

- **Uncertainty quantification**: how spread out is our belief over model parameters?
- **Posterior predictive distributions**: how uncertain are our predictions at new test points?

---

## The posterior predictive distribution

Given posterior samples $\theta_i \sim p(\theta \mid \mathcal{D})$, predictions at new test points $X^*$ are:

$$p(\text{correct} \mid X^*, \mathcal{D}) = \int p(\text{correct} \mid X^*, \theta)\; p(\theta \mid \mathcal{D})\; d\theta \approx \frac{1}{N} \sum_{i=1}^{N} \underbrace{p(\text{correct} \mid X^*, \theta_i)}_{\texttt{OddityTask.predict}(\theta_i,\, X^*)}$$

In psyphy this is computed by `WPPMPredictivePosterior`, which calls `nuts_posterior.sample(N)` to get $N$ parameter draws and vmaps the task likelihood over them. The same class works identically with a `MAPPosterior` (which just returns $N$ copies of $\theta_\text{MAP}$).

---

## A note on the likelihood

The oddity-task likelihood has no closed form. `OddityTask.loglik` *estimates* the
probability of a correct response by simulating the observer: for every trial it draws
`mc_samples` noisy internal representations of the three stimuli and counts how often
the odd one out is picked. That has two consequences you will feel on this page:

- **It is expensive.** Every gradient NUTS asks for repeats that simulation for every
  trial, and NUTS asks for many gradients per draw. This is why the run-time notes in
  Step 4 exist.
- **It is noisy.** Two calls with different random keys give slightly different answers.
  NUTS needs the *same* answer every time, so psyphy freezes one key for the whole run
  (`logdensity_key_seed` in `NUTSSampler`). The sampler then explores the posterior implied
  by that *one* noise realization, not the exact posterior, and at low settings the
  difference is not small: at `mc_samples=200` the key-to-key spread in the gradient is
  about 1.5 times the gradient itself. The bias shrinks as `mc_samples` grows, which is the
  main reason `--mode balanced` and `--mode full` raise it. **Treat results from small
  `mc_samples` as approximate.**

**Future alternative:** a *neural surrogate likelihood* (see `NeuralSurrogateTask`)
replaces the simulation with a trained network. It is deterministic and cheap to
differentiate, which removes both problems.

---

## Step-by-step walkthrough

### Step 1: Simulate data

Same setup as the full MAP example. Data is a `TrialData` object:

```python title="Simulated data"
--8<-- "docs/examples/wppm/nuts_inference_example.py:data"
```

### Step 2: Build the model

```python title="Model definition"
--8<-- "docs/examples/wppm/nuts_inference_example.py:build_model"
```

### Step 3: MAP warm-start (recommended)

A sampler started far from where the posterior mass lives spends its first stretch just
*walking there*, and those draws are useless. Starting at the MAP estimate, which sits
at the posterior peak, skips that walk, so warmup can be spent on tuning instead.
Run a short MAP fit first:

```python title="MAP warm-start"
--8<-- "docs/examples/wppm/nuts_inference_example.py:map_warmstart"
```

Then pass `map_posterior.params` as `init_params` to `NUTSSampler.fit()`.

### Step 4: NUTS sampling

```python title="NUTS sampling with NUTSSampler"
--8<-- "docs/examples/wppm/nuts_inference_example.py:nuts_fit"
```

`NUTSSampler` has the same `fit(model, data)` interface as `MAPOptimizer`; the
[BlackJAX](https://blackjax-devs.github.io/blackjax/) backend is entirely hidden. What comes
back is an `MCMCPosterior`, and these are the three things you will use from it:

| call | returns | used in |
|---|---|---|
| `nuts_posterior.sample(n, key=...)` | `n` draws `{"W": (n, *W_shape)}` | Step 6, the ellipse ensemble |
| `PosteriorMeanCovarianceField(nuts_posterior)` | the posterior mean covariance field E[Σ(x)] | Step 6, the red "posterior mean" ellipses |
| `nuts_posterior.to_arviz()` | all chains, for diagnostics | Step 5 |

`MCMCPosterior` and `MAPPosterior` share this interface, so code written for a MAP fit,
for example `WPPMPredictivePosterior`, accepts a NUTS fit unchanged.

!!! warning "Don't draw ellipses from `nuts_posterior.params`"
    `nuts_posterior.params` is the average **parameter** E[W]. It is a valid parameter set
    (you can pass it as `init_params`), but the covariance field built from it,
    `WPPMCovarianceField(model, nuts_posterior.params)`, comes out far too small. Step 6
    shows why and what to use instead. `WPPMCovarianceField.from_posterior` builds exactly
    this field, so it warns when given an MCMC posterior.

**What happens inside `fit`.** Each chain runs in two phases:

1. **Warmup** (`num_warmup` steps). The sampler tunes itself and the draws are thrown away.
   It learns a *step size* (how far one simulated move goes) and a *mass matrix* (how far
   to move along each parameter, since a degree-0 coefficient varies on a several-times
   larger scale than a higher-degree one). This is BlackJAX's
   [`window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html).
2. **Sampling** (`num_samples` steps). The tuning is frozen, and every draw is kept.

The arguments you are most likely to touch:

| argument | what it controls | turn it up when |
|---|---|---|
| `num_warmup` | tuning steps per chain, discarded | trace plots still drift at the start |
| `num_samples` | kept draws per chain | ESS is too low |
| `num_chains` | independent chains; R-hat needs ≥ 2 | you want a stronger convergence check |
| `max_num_doublings` | tree-depth cap, ≤ 2^k − 1 gradients per draw | most draws hit the cap (the script prints this) |
| `target_acceptance_rate` | how cautious the tuned step size is | you see divergences (try 0.9) |

**Compute settings.** The script takes `--mode`, which selects a preset (see
*How to run the script* at the top). The *model* is identical in every mode, so only the
thoroughness of the fit changes, never what is being fit:

```python
--8<-- "docs/examples/wppm/nuts_inference_example.py:compute_settings"
```

??? note "Runtime notes"

    - **The MC likelihood dominates everything.** Each gradient evaluation simulates
      `mc_samples` Monte Carlo draws for every trial, so cost scales with
      `mc_samples × n_trials`. Going from `quick` to `full` raises that product by ~440×,
      which is why `full` is a batch job and not something to run interactively.
    - **NUTS cost per draw is not fixed.** It is `O(2^tree_depth)` gradient evaluations, and
      the depth is data-dependent, commonly 16 to 64 evaluations per draw. The script therefore
      **times one gradient and prints a bracketed projection** before sampling starts, so you
      can see whether a run fits inside a Slurm time limit before committing to it.
    - **Tree depth is capped.** `NUTSSampler(max_num_doublings=8)` (the default) limits a draw
      to at most 2^8 − 1 = 255 gradient evaluations; BlackJAX's own default of 10 allows 1023.
      If `sampler_stats["n_steps"]` sits at the cap for most draws, trajectories are being
      truncated: still valid MCMC, but slower mixing. Raise the cap or reparameterize.
    - **Warmup starts from a prior-scaled metric.** By default the sampler works in
      `z = W / prior_std` (`prior_preconditioning=True`), so adaptation does not begin from an
      identity mass matrix that is badly mismatched to coefficients whose scales span
      0.06 to 0.01. Without this, early warmup runs every draw to the maximum tree depth.
    - **Chains are vmapped, in both warmup modes.** All chains advance in one compiled call per
      phase, so XLA compiles once rather than once per chain. Under vmap each step costs as much
      as the *deepest* tree across chains, which is one more reason for the depth cap. On a
      GPU, four chains cost about as much as one; on a CPU, many chains can be slower than
      running them one by one.
      Note: the progress bar moves only when each phase (warmup, sampling) finishes.



#### Two warmup modes

| Mode | When | How |
|------|------|-----|
| Adaptive (default, `step_size=None`) | Best quality; adapts step size and mass matrix per chain via `blackjax.window_adaptation`, starting from a prior-scaled metric. Chains vmapped; each adapts independently. | `NUTSSampler(num_warmup=200, num_samples=500)` |
| Fixed (`step_size=<float>`) | Faster with many chains; no adaptation. All chains vmapped in one compiled call. User must choose a good step size. | `NUTSSampler(..., step_size=0.01)` |

### Step 5: Diagnostics with ArviZ

Sampling gives you draws; it does not tell you whether to trust them. MCMC fails
quietly, so a chain can look like a result while having explored only part of the
posterior. [**ArviZ**](https://python.arviz.org/) is the standard Python library for that
check. It is backend-agnostic, so the same diagnostics apply whether the draws came from
BlackJAX, NumPyro, Stan or PyMC. Since version 1.0 it is split into three packages, and the
function references below link into them:
[arviz-stats](https://python.arviz.org/projects/stats/en/latest/) (numbers such as R-hat
and ESS) and [arviz-plots](https://python.arviz.org/projects/plots/en/latest/) (figures),
both on top of arviz-base (the data format). `import arviz as az` gives you all three.

psyphy hands your chains over in one call:

```python
idata = nuts_posterior.to_arviz()
```

`to_arviz()` converts the `(n_chains, n_draws, *W_shape)` arrays, plus the sampler
statistics, into ArviZ's format, an
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html)
with a `posterior` group (the draws of `W`) and a `sample_stats` group (acceptance rate,
divergences, energy, tree size). No reshaping is needed: ArviZ handles arbitrary-rank
variables natively and computes diagnostics per scalar element of `W`. Everything ArviZ
offers is then available on `idata`.

```python title="ArviZ diagnostics"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

[`az.summary`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.summary.html)
returns one row per scalar element of `W`, 54 rows here. The script prints only the
spread of each column, because 54 rows are not readable. If you print `summary` itself,
the columns mean:

| column | meaning |
|---|---|
| `mean`, `sd` | posterior mean and standard deviation of that coefficient |
| `eti89_lb`, `eti89_ub` | lower and upper end of the 89% credible interval: the coefficient lies in this range with 89% posterior probability (equal-tailed by default; `ci_kind="hdi"` and `ci_prob=` change it) |
| `ess_bulk`, `ess_tail` | effective sample size in the middle and in the tails of the distribution ([`az.ess`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html)) |
| `r_hat` | between-chain agreement ([`az.rhat`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html)) |
| `mcse_mean`, `mcse_sd` | Monte Carlo error of `mean` and `sd`: how much they would move if you re-ran the sampler |

The single row to look at first is the one with the **largest** `r_hat` and the
**smallest** `ess_bulk`. If the worst coefficient passes, all of them do.

```python title="Trace plot"
--8<-- "docs/examples/wppm/nuts_inference_example.py:trace_plot"
```

<div align="center">
  <img src="../plots/nuts_trace_plot.png" width="700"/>
  <p><em>Posterior traces for the slice <code>W[:, :, 0, 0]</code>.</em></p>
</div>

**How to read it.** `W` has shape `(3, 3, 2, 3)` = `(deg_x, deg_y, U_row, U_col)`: the
first two axes index the Chebyshev basis, giving degree along stimulus dimension 1 and
along dimension 2. The last two index the entry of `U(x)`, where `Σ = U Uᵀ + diag_term·I`.
That is 54 scalars, past ArviZ's `plot.max_subplots` limit and illegible anyway, so the
figure fixes `U[0, 0]` and plots the remaining 3×3 coefficient grid, laid out *as* that
grid by `col_wrap=3`.

| element of the figure | what it is |
|---|---|
| one panel | one Chebyshev coefficient; the title `i, j` is degree `i` × degree `j` |
| x axis | draw index **within a chain**, post-warmup (warmup draws are discarded, not plotted) |
| y axis | the value of that coefficient |
| one coloured line per panel | one chain |
| black ticks along the bottom | divergences; they appear because `sampler_stats` carries `diverging` under ArviZ's standard name |

**What good looks like:** chains overlapping each other, flat, and wiggling fast, the
"fuzzy caterpillar". A chain that drifts steadily across the panel has not converged:
warmup was too short and the sampler is still finding the typical set. That is the single
most common thing this plot catches, and it is a reason to raise `--mode`, not a reason to
distrust the model.

It is generated by **ArviZ**, from `MCMCPosterior.to_arviz()`; no psyphy plotting code is
involved. On ArviZ ≥ 1.0
[`plot_trace`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_trace.html)
returns a
[`PlotCollection`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.PlotCollection.html)
(not an array of matplotlib axes), so it is saved with `pc.savefig()`; the 0.x kwargs
`compact=` and `figsize=` no longer exist and raise if passed.

??? tip "Three more ArviZ plots worth a look on your own run"

    Each takes the same `idata`, and the same `var_names` / `coords` selection as the trace
    plot above.

    | plot | what it shows | what bad looks like |
    |---|---|---|
    | [`az.plot_trace_dist`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_trace_dist.html) | the classic pairing: each coefficient's marginal density next to its trace | one chain's density sitting apart from the others |
    | [`az.plot_rank`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_rank.html) | for each chain, how its draws rank among *all* chains' draws | one chain's curve leaving the band: it is exploring somewhere else |
    | [`az.plot_energy`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_energy.html) | NUTS-specific: whether the momentum resampling can move the chain between energy levels (BFMI) | the two energy distributions differ clearly in width, or BFMI < 0.3 |

    The [arviz-plots gallery](https://arviz-plots.readthedocs.io/en/latest/gallery/index.html)
    shows every plot type with example output.

#### Interpreting diagnostics

The most important diagnostics to check in order:

| Diagnostic | What it measures | Good value |
|---|---|---|
| **Divergences** | Transitions where the integrator blew up, usually in a region of high curvature the step size cannot resolve. | **0.** Any divergence makes the run suspect, whatever R-hat says. |
| [**R-hat**](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html) | Between-chain variance over within-chain variance; near 1 means the chains agree on the same distribution. | < 1.01 |
| [**ESS (bulk)**](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html) | Effective sample size: how many independent draws the chains are worth once autocorrelation is accounted for. | > 100 per chain |
| **Acceptance rate** | Fraction of NUTS proposals accepted. Far below target means the step size is too large; far above means it is inefficiently small. | 0.6 to 0.9 |


To make ESS concrete: 1000 draws with an ESS of 50 carry about as much information as
50 independent draws. Neighbouring draws in a chain are correlated, so they partly
repeat each other.

**Divergences come first**, because the other three can all look healthy while the sampler
is systematically missing part of the posterior. A divergence means the simulated trajectory
blew up rather than following the posterior's geometry, which usually points at a region of
curvature the step size cannot resolve, and the draws you keep are then biased in a way no
amount of extra sampling fixes. Betancourt's
[*Conceptual Introduction*](https://arxiv.org/abs/1701.02434) is the place to read what they
mean geometrically. psyphy records them for you: `sampler_stats` carries `diverging`,
`energy` and `n_steps` under ArviZ's standard names, so
[`az.plot_energy(idata)`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_energy.html)
and the divergence ticks on the trace plot work with no extra wiring.

**R-hat** compares chains against each other, so it needs at least two of them; a single
chain cannot tell you it is stuck. The idea goes back to
[Gelman and Rubin (1992)](https://doi.org/10.1214/ss/1177011136), but the version ArviZ
computes is the rank-normalized, split variant of
[Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221), which is also where the
`< 1.01` and `> 100` thresholds above come from. They are recommendations from that paper.

With `--mode quick` (2 chains and 150 draws) R-hat and ESS will not reach these
thresholds, and they are not meant to. Use `--mode balanced` or `--mode full`, which exist
for exactly this reason, before drawing any conclusion from the numbers.

### Step 6: Posterior predictive uncertainty

```python title="Draw posterior parameter samples"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_samples"
```

Each draw gives one complete covariance field, so every blue ellipse below is a field the
data cannot rule out. For a single summary field, the *posterior mean* E[Σ(x)], average the
covariance matrices over all draws:

```python title="Posterior mean covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_mean"
```

??? note "Why the mean is computed from Σ, not from W"

    The tempting shortcut is to average the parameters and build one field from the
    result, `WPPMCovarianceField(model, nuts_posterior.params)`. That computes Σ(E[W]),
    and for the WPPM it gives ellipses that are far too small.

    The reason is that Σ(x) = U(x) U(x)ᵀ + λI, with U linear in W, so Σ is roughly
    *W squared*. Squaring and then averaging is not the same as averaging and then
    squaring. A one-number example with two draws, W = +1 and W = −1:

    | | draw 1 | draw 2 | average |
    |---|---|---|---|
    | W | +1 | −1 | **0** |
    | Σ = W² | 1 | 1 | **1** |

    Both draws describe the same ellipse, Σ = 1. Averaging Σ gives 1, the right answer.
    Averaging W first gives 0, and 0² = 0: the ellipse disappears.

    The WPPM has this situation built in. Rotating `W` along its embedding axis (its
    last axis) leaves Σ unchanged, so many different `W`s produce the same ellipses, and
    different draws, especially from different chains, settle on different ones. On their
    own each draw is fine. Averaged, they partly cancel. Even without that symmetry,
    E[Σ] = Σ(E[W]) + Cov[U] holds exactly, so averaging W first always loses the spread
    term and never gives larger ellipses.

    `PosteriorMeanCovarianceField` computes Σ for every draw first and averages afterwards,
    so neither effect applies. One thing to keep in mind: an average of covariance matrices
    is somewhat larger than a typical single draw, because draws differ in size. That is
    what a mean does, not an error.

```python title="Plot ensemble of ellipses"
--8<-- "docs/examples/wppm/nuts_inference_example.py:plot_uncertainty"
```

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty.jpg" width="600"/>
  <p><em>Faint blue: 30 covariance fields sampled from the posterior. Red: posterior mean E[Σ(x)], averaged over all draws. Black: ground truth. The spread of blue ellipses shows where the model is uncertain.</em></p>
</div>




!!! note "Every ellipse here is drawn by `psyphy.viz`"
    Both covariance figures on this page go through `plot_ellipses`, which accepts the
    covariances in whichever shape you already have:

    | shape | meaning |
    |---|---|
    | `(n_points, 2, 2)` | one field, for instance the posterior mean |
    | list of `(n_points, 2, 2)` | several fields overlaid, each with its own colour, width, linestyle and legend entry |
    | `(n_samples, n_points, 2, 2)` | a stack of fields, which is exactly what `sample()` returns |



<div align="center">
  <img src="../plots/nuts_overlay.jpg" width="650"/>
  <p><em>Faint blue: posterior draws. Black: ground truth. Dashed orange: MAP. Red:
  posterior mean. Drawn at the same <code>ELLIPSE_SCALE</code> as the figure above, so the
  two are comparable.</em></p>
</div>

!!! note "`scale` and honest ellipse sizes"
    Every figure on this page multiplies the ellipses by `ELLIPSE_SCALE` so they are
    visible against the grid. `plot_ellipses` also
    accepts `scale="auto"`, which sizes the median ellipse to a fraction of the grid
    spacing and can return the factor it used via `return_scale=True`, so a figure can
    report its own magnification.

---

## Further reading

- API reference: [`NUTSSampler`](../../reference/inference.md), [`MCMCPosterior`](../../reference/posterior.md)
- MAP fitting tutorial: [`full_wppm_fit_example`](full_wppm_fit_example.md)
- Model details: [`wppm_tutorial`](wppm_tutorial.md)
- BlackJAX documentation: [blackjax-devs.github.io/blackjax](https://blackjax-devs.github.io/blackjax/)
- ArviZ documentation: [python.arviz.org](https://python.arviz.org/), with its
  [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html),
  the [arviz-stats API](https://python.arviz.org/projects/stats/en/latest/) (R-hat, ESS,
  `summary`) and the [arviz-plots gallery](https://arviz-plots.readthedocs.io/en/latest/gallery/index.html)

The papers behind the method and the diagnostics, cited where they are used above and
collected here:

> Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: adaptively setting path
> lengths in Hamiltonian Monte Carlo.* Journal of Machine Learning Research, 15, 1593-1623.
> [jmlr.org](https://jmlr.org/papers/v15/hoffman14a.html)
>
> Betancourt, M. (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo.*
> [arXiv:1701.02434](https://arxiv.org/abs/1701.02434)
>
> Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
> *Rank-normalization, folding, and localization: an improved $\widehat{R}$ for assessing
> convergence of MCMC.* Bayesian Analysis, 16(2), 667-718.
> [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
>
> Gelman, A., & Rubin, D. B. (1992). *Inference from iterative simulation using multiple
> sequences.* Statistical Science, 7(4), 457-472.
> [doi:10.1214/ss/1177011136](https://doi.org/10.1214/ss/1177011136)
