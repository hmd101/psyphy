# NUTS Posterior Inference for WPPM

This tutorial demonstrates **full posterior inference** using the No-U-Turn Sampler (NUTS) for the Wishart Process Psychophysical Model (WPPM).

The [Maximum A Posteriori (MAP) tutorial](full_wppm_fit_example.md) ends with *one* threshold ellipse per reference
point: the model's single best guess. This page ends with a *distribution* of ellipses per reference
point (green dots in figure below). Where the distribution is
concentrated, the data pin the threshold down. Where  the ellipses fan out, the data do not, and a single MAP
ellipse would have hidden that.
This then allows for adaptive trial placement where we care about
placing new trials where the model is currently maximally uncertain.
This allows for efficient ressource management: maximizing the information gained from each trial.

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty_full.jpg" width="520"/>
  <p><em>What this page builds toward: a whole distribution of threshold contours at every
  reference point, where a point estimate gives just one. The axes are the two model
  dimensions, arbitrary up to an affine transformation of the input space. Each green dot is
  one of the 25 reference points at which the model is compared against the ground truth.
  Around each of them, black is the true threshold contour and red is the posterior mean
  \(\mathbb{E}[\Sigma(x)]\); the 200 faint blue contours are individual posterior draws,
  each one a threshold the data cannot rule out. Where they sit on top of one another the
  data pin the threshold down, and where they fan out they do not. Built in
  <a href="#step-6-posterior-predictive-uncertainty">Step 6</a>.</em></p>
</div>

The companion script can be found at
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).

**Outline:**

| step | what we do | what we get |
|---|---|---|
| 1–2 | simulate data, build the WPPM | `TrialData` and a model, same as the [MAP tutorial](full_wppm_fit_example.md) |
| 3 | run a short Maximum a Posteriori (MAP) fit | a starting point near the posterior peak |
| 4 | run a sampler, here NUTS ([BlackJAX](https://blackjax-devs.github.io/blackjax/)) | an `MCMCPosterior`: hundreds to thousands of plausible parameter arrays $W$ |
| 5 | check the chains with [ArviZ](https://python.arviz.org/) | evidence that those draws can be trusted (or that they cannot) |
| 6 | push the draws through the covariance field | a distribution of threshold ellipses at every reference point, overlaid on ground truth and MAP |

??? note "How to run the script"

    This page needs three optional
    dependencies, none of which psyphy requires by default.

    ```bash
    pip install 'psyphy[sampling,diagnostics,viz]'
    ```

    | extra | brings in | what it is for |
    |---|---|---|
    | `sampling` | `blackjax>=1.4,<2` | `NUTSSampler` itself |
    | `diagnostics` | [`arviz>=1.0`](https://python.arviz.org/en/latest/user_guide/getting_started.html) | `covariance_to_arviz`, `MCMCPosterior.to_arviz()`, R-hat, ESS, the diagnostic plots |
    | `viz` | `matplotlib>=3.5` | every figure on this page |


    Everything on this page comes from one script. Pick a preset by how much compute
    you want to spend:

    ```bash
    # smoke test: 9 refs, 50 MC samples, 2 chains x (100 warmup + 150 draws)
    python nuts_inference_example.py --mode quick

    # 25 refs, 500 MC samples, 4 chains x (500 + 500). A real fit; ~35 min on an A100.
    python nuts_inference_example.py --mode balanced

    # 25 refs, 2000 MC samples, 4 chains x (1000 + 1000). 11h on GPU for only slightly better fits.
    python nuts_inference_example.py --mode full
    ```

    Only the MAP and NUTS steps are expensive, and the script caches their result.
    Everything after them is post-processing of the stored draws: the diagnostics, and
    every figure on this page. All of it can be redone without a GPU:

    ```bash
    # rebuild every figure and every printed diagnostic from the cached fit
    python nuts_inference_example.py --mode full --from-fit
    ```

    The cache (`fits/nuts_full_fit.npz`) is committed, and it records the settings it
    was produced with: a run whose settings disagree with it refuses to load rather than
    quietly drawing a `full` figure from `quick` draws.

    `quick` is a smoke test for the code path: about a minute on an A100 GPU, about 30
    minutes on a CPU. Its chains are far too short to converge, so don't read its
    diagnostics as a result.

    **Every figure and every number on this page comes from `--mode full`.** It is the
    only one of the three that is fully converged but the results from balanced look already very good; the table in
    [Step 5](#what-to-check-and-in-what-order) shows more details.


---

## Background: why Markov Chain Monte Carlo (MCMC), and what it is

Throughout this page $\theta$ and $W$ are the same object: the WPPM's single parameter
array, the basis coefficients that define the covariance field $\Sigma(x)$ (Step 2 says what
is in it). $\theta$ is used where a statement is about Bayesian inference in general and
$W$ where it is about this model; in code the same array is the entry `"W"` of the
parameter dict. MAP estimation returns one value of it:

$$\theta_\text{MAP} = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

This tells you *where the posterior peaks*, but not *how wide* it is. If the data are
sparse or ambiguous the posterior may be broad, and the MAP estimate alone cannot tell you
that. For a threshold contour the difference is the whole point: a point estimate gives one
curve, where what you want to report is that curve together with the cloud of curves the
data cannot rule out.

What we actually want instead is the posterior $p(\theta \mid \mathcal{D})$, the full distribution
over parameters given the data. Bayes' rule gives this to us in one line, but the line
contains an integral over all of parameter space that is intractable to  evaluate here. We cannot plot the posterior, and we cannot
integrate against it directly.

[Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) sidesteps
that. Instead of computing the distribution, we construct a random walk whose long-run
visiting frequency *is* that distribution. Run it long enough and the places it has been
form a sample from the posterior, which is all we need: averages over the sample
approximate averages over the posterior, and the spread of the sample is the uncertainty.


The challenge is building a walk that explores efficiently. A naive random walk works maybe in two or
three dimensions and fails badly in fifty, where almost every direction you could step in
leads somewhere less probable, so nearly all proposals are rejected.
[Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) fixes this
by using the *gradient* of the log posterior: it gives the parameter vector a random
momentum and simulates it rolling along the posterior's own geometry, so a single proposal
can travel a long way and still be accepted. **NUTS**, the No-U-Turn Sampler of
[Hoffman and Gelman (2014)](https://jmlr.org/papers/v15/hoffman14a.html), removes the one
awkward tuning knob by extending each trajectory until it starts to double back on itself.
Betancourt's [*Conceptual Introduction to Hamiltonian Monte Carlo*](https://arxiv.org/abs/1701.02434)
is a good source of why this works and where it fails; it is worth skimming.

Two things follow for psyphy:

1. **The sampler needs gradients**, which is why the log posterior is written in JAX and
   why psyphy's `NUTSSampler` can hand it straight to BlackJAX.
2. **The sampler needs the log density to be a fixed function.** psyphy's oddity likelihood
   is estimated by Monte Carlo simulation, so it is *not* fixed; it returns a slightly
   different number each call. To fix this  non-determinism, the note below explains
   what is done about it.

What we get in exchange for the extra compute:

- **Uncertainty quantification**: how spread out is our belief over model parameters?
- **Posterior predictive distributions**: how uncertain are our predictions at new test points?

---

## The posterior predictive distribution

Given posterior samples $\theta_i \sim p(\theta \mid \mathcal{D})$, predictions at new test points $X^*$ are:

$$p(\text{correct} \mid X^*, \mathcal{D}) = \int p(\text{correct} \mid X^*, \theta)\; p(\theta \mid \mathcal{D})\; d\theta \approx \frac{1}{N} \sum_{i=1}^{N} \underbrace{p(\text{correct} \mid X^*, \theta_i)}_{\texttt{OddityTask.predict}(\theta_i,\, X^*)}$$

In psyphy this is computed by `WPPMPredictivePosterior`, which calls `nuts_posterior.sample(N)` to get $N$ parameter draws and vmaps the task likelihood over them. The same class works identically with a `MAPPosterior` (which just returns $N$ copies of $\theta_\text{MAP}$).

---

## A note on the likelihood and its non-determinism

The oddity-task likelihood has no closed form. `OddityTask.loglik` *estimates* the
probability of a correct response by simulating the observer: for every trial it draws
`mc_samples` noisy internal representations of the three stimuli and counts how often
the odd one out is picked. That has two consequences:

- **It is expensive.** Every gradient NUTS asks for repeats that simulation for every
  trial, and NUTS asks for many gradients per draw. This is what the
  [runtime notes](#runtime) are about.
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

## A note on the parameters: $W$ is not identified

$W$ is not uniquely determined by the data, and not because the data are too few. Look again
at how the field is built, $\Sigma(x) = U(x)U(x)^\top + \lambda I$ with $U(x) = \sum_j W_j \phi_j(x)$:
the last axis of $W$ is the *embedding* axis, and it is summed over in $UU^\top$. Rotate $W$
along that axis, $W \mapsto WR$ with $R$ orthogonal, and every $U(x)R$ gives
$U(x)RR^\top U(x)^\top = U(x)U(x)^\top$. So $\Sigma$ is unchanged, the likelihood is unchanged,
and the prior, being isotropic in that axis, is unchanged too. The posterior over $W$ is
therefore a continuum of orthogonal copies of itself, and different chains, started from
different points, happily settle on different ones.

This is a property of the model, not of NUTS or of any sampler, and it has exactly two
consequences on this page. Both are about treating $W$ as if it meant something on its own:

- **Diagnostics on $W$ report failures that do not exist.** Chains that disagree about $W$
  while agreeing about every ellipse look non-convergent. So psyphy diagnoses $\Sigma$
  instead ([Step 5](#what-to-diagnose-sigma-not-w)).
- **Averaging $W$ destroys the answer.** Draws at different rotations cancel, so the mean
  parameter $\mathbb{E}[W]$ gives ellipses that are far too small. So psyphy averages
  $\Sigma$ instead ([Step 6](#step-6-posterior-predictive-uncertainty)).

Everything the model predicts is invariant under the rotation: every covariance,
threshold, ellipse and response probability. Nothing you would report is affected.

---

## Step-by-step walkthrough

### Step 1: Simulate data

Same setup as the [MAP tutorial](full_wppm_fit_example.md). Data is a `TrialData`
object:

```python title="Simulated data"
--8<-- "docs/examples/wppm/nuts_inference_example.py:data"
```

### Step 2: Build the model

The model is a `Prior` over the parameters plus a task likelihood and a noise model. Two
things about it are worth having in mind for the rest of the page, because every diagnostic
and every figure below refers to them.

**What the parameters are.** The WPPM does not have one covariance matrix; it has a
covariance *field* $\Sigma(x)$, a $2\times2$ matrix at every point $x$ of the stimulus space.
The field is built as

$$\Sigma(x) = U(x)\,U(x)^\top + \lambda I, \qquad U(x) = \sum_{j} W_j \,\phi_j(x)$$

where the $\phi_j$ are fixed polynomial basis functions (Chebyshev, here up to degree 2 per
input dimension) and the coefficients $W_j$ are the only thing being fitted. With
`input_dim=2`, `basis_degree=2` and `extra_embedding_dims=1` that is an array of shape
`(3, 3, 2, 3)`.  $\lambda$ is
`diag_term`, a small floor that keeps $\Sigma$ invertible.

Two consequences come straight out of this form:

- $\Sigma$ is *quadratic* in $W$. Averaging draws of $W$ and squaring is not the same as
  squaring and averaging, which is why Step 6 never averages $W$.
- The prior shrinks the coefficients as their total degree rises (`decay_rate`), so a
  degree-0 coefficient and a degree-4 one live on genuinely different scales. That is what
  the sampler's mass matrix has to discover, and why Step 4 preconditions by the prior
  standard deviation.

For the basis, the prior and `diag_term` in more depth, see the
[WPPM tutorial](wppm_tutorial.md).

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
back is an `MCMCPosterior`, and these are the three things we will use from it:

| call | returns | used in |
|---|---|---|
| `nuts_posterior.sample(n, key=...)` | `n` draws `{"W": (n, *W_shape)}` | Step 6, the ellipse ensemble |
| `PosteriorMeanCovarianceField(nuts_posterior)` | the posterior mean covariance field $\mathbb{E}[\Sigma(x)]$ | Step 6, the red "posterior mean" ellipses |
| `covariance_to_arviz(nuts_posterior, points)` | $\Sigma$ at chosen points for every draw, in ArviZ's format, for diagnostics | Step 5 |

`MCMCPosterior` and `MAPPosterior` share this interface, so code written for a MAP fit,
for example `WPPMPredictivePosterior`, accepts a NUTS fit unchanged.

Concretely, what a draw means: **one draw is one complete covariance field**, hence one
ellipse at every reference point. 2000 draws are 2000 ellipses per point, and their spread
*is* the posterior uncertainty about that threshold. That is the figure at the top of this
page, and Step 6 is where it gets built.

!!! warning "Don't draw ellipses from `nuts_posterior.params`"
    `nuts_posterior.params` is the average **parameter** $\mathbb{E}[W]$. It is a valid parameter
    set; you can pass it as `init_params`. But the covariance field built from it,
    `WPPMCovarianceField(model, nuts_posterior.params)`, comes out far too small, for the
    reason given in [$W$ is not identified](#a-note-on-the-parameters-w-is-not-identified).
    Use `PosteriorMeanCovarianceField` instead ([Step 6](#step-6-posterior-predictive-uncertainty)).
    `WPPMCovarianceField.from_posterior` builds the wrong one, so it warns when handed an
    MCMC posterior.

**What happens inside `fit`.** Each chain runs in two phases:

1. **Warmup** (`num_warmup` steps). The sampler tunes itself and the draws are thrown away.
   It learns a *step size* (how far one simulated move goes) and a *mass matrix* (how far
   to move along each parameter, since a degree-0 coefficient varies on a several-times
   larger scale than a higher-degree one). This is BlackJAX's
   [`window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html).
2. **Sampling** (`num_samples` steps). The tuning is frozen, and every draw is kept.

The knobs behind all of this are gathered in
[Reference: tuning and runtime](#reference-tuning-and-runtime) at the end of the page: the
arguments, the `--mode` presets, the runtime behaviour and the two warmup modes. That lets
the walkthrough go straight on to the question that matters more, which is whether the
draws can be trusted at all.

### Step 5: Diagnostics with ArviZ

Sampling gives us draws; it does not tell us whether to trust them. MCMC fails
quietly, so a chain can look like a result while having explored only part of the
posterior. [**ArviZ**](https://python.arviz.org/) is the standard Python library for that
check. It is backend-agnostic, so the same diagnostics apply whether the draws came from
BlackJAX, NumPyro, Stan or PyMC. Since version 1.0 it is split into three packages, and the
function references below link into them:
[arviz-stats](https://python.arviz.org/projects/stats/en/latest/) (numbers such as R-hat
and ESS) and [arviz-plots](https://python.arviz.org/projects/plots/en/latest/) (figures),
both on top of arviz-base (the data format). `import arviz as az` gives you all three.

#### What to check, and in what order

Four numbers decide whether the draws are usable. Check them in this order:

| Diagnostic | What it measures | Good value |
|---|---|---|
| **Divergences** | Transitions where the integrator blew up, usually in a region of high curvature the step size cannot resolve. | **0.** Any divergence makes the run suspect, whatever R-hat says. |
| [**R-hat**](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html) on $\Sigma$ | Whether the chains agree on the same distribution ([below](#what-r-hat-measures-and-what-rank-normalized-means)). | < 1.01 |
| [**ESS (bulk)**](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html) on $\Sigma$ | How many independent draws the chains are worth. | > 100 per chain |
| **Acceptance rate** | Fraction of NUTS proposals accepted. Far below target means the step size is too large; far above means it is inefficiently small. | 0.6 to 0.9 |

**Divergences come first**, because the other three can all look healthy while the sampler
is systematically missing part of the posterior. A divergence means the simulated trajectory
blew up rather than following the posterior's geometry, which usually points at a region of
curvature the step size cannot resolve, and the draws you keep are then biased in a way no
amount of extra sampling fixes. Betancourt's
[*Conceptual Introduction*](https://arxiv.org/abs/1701.02434) is the place to read what they
mean geometrically. psyphy records them for you: the sampler statistics carry `diverging`,
`energy` and `n_steps` under ArviZ's standard names, so `az.plot_energy` and the
divergence ticks on the trace plot work with no extra wiring, and the script prints the
count as soon as sampling finishes, before any of the figures; a divergent run is not
worth plotting.

With `--mode quick` (2 chains and 150 draws) R-hat and ESS will not reach these
thresholds, even on $\Sigma$, and they are not meant to. Use `--mode full`, which is the
run this page is written from, before drawing any conclusion from the numbers.

The rest of this step is how to compute these four for the WPPM: *on what* (next), *where*
(after that), and then how to read the two figures and what R-hat is actually doing.

#### What to diagnose: $\Sigma$, not $W$

The obvious thing to check would be the parameters themselves:
`nuts_posterior.to_arviz()` hands the raw draws of $W$ to ArviZ. For the WPPM that is
misleading, because $W$ is [not identified](#a-note-on-the-parameters-w-is-not-identified).
Chains settle at different rotations, disagree about $W$, and agree about every ellipse.

The size of the effect is worth seeing once. On the `full` run behind this page, R-hat on
$W$ reaches **1.34** (median 1.07), while the worst of the 75 $\Sigma$ quantities is
**1.017**. Same draws, same fit: one number says the chains disagree, the other says they
agree to within 2%, and only the second is about something the model predicts.

So psyphy diagnoses $\Sigma$ itself.
[`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz)
evaluates $\Sigma$ for every posterior draw at points you choose and returns the result in
ArviZ's format, an
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html).
For each draw and point, the 2×2 matrix is summarised by three numbers that together
determine it:

| quantity | meaning |
|---|---|
| `log_var_0` | log of the variance along stimulus dimension 1, $\log \Sigma_{11}$ |
| `log_var_1` | log of the variance along stimulus dimension 2, $\log \Sigma_{22}$ |
| `corr_0_1` | correlation $\Sigma_{12}/\sqrt{\Sigma_{11}\Sigma_{22}}$, which carries the ellipse's orientation |

The result holds one variable, `Sigma`, with dimensions `(chain, draw, point, quantity)`.
The evaluation points are stored with it as the coordinates `location_0` and
`location_1`, and the sampler statistics (divergences, energy, tree size) are carried
over, so everything below works on it directly.

#### Choosing where to evaluate $\Sigma$

$W$ is one fixed set of numbers, but $\Sigma$ is a function of the stimulus location $x$. A
diagnostic of $\Sigma$ is therefore always a diagnostic *at some $x$*, and with the current
implementation **you choose those locations**: `covariance_to_arviz` deliberately has no
default for `points`.

- **Start with the locations you report.** If you plot or interpret ellipses at certain
  points, check convergence there. This script uses the reference points, the same
  ellipses Step 6 plots.
- **Add coverage where it matters.** Between and beyond the data points the posterior is
  shaped more by the prior and can mix differently. A modest grid over the whole stimulus
  space catches that, for example 7×7 points:
  `g = jnp.linspace(-1, 1, 7); grid = jnp.stack(jnp.meshgrid(g, g), -1).reshape(-1, 2)`.
- **Report the worst value** over all points and quantities.
- **A few points are enough.** $\Sigma(x)$ is a low-degree polynomial in $x$ (degree 2 here), so it
  varies smoothly, and a dense grid mostly adds near-duplicates.

Choosing points is a property of this approach, not a law of nature. Quantities of $W$
that are unchanged by the rotation, such as $W W^\top$ summed over the embedding axis, would
allow a point-free check; psyphy does not implement that.

#### Running the diagnostics

```python title="ArviZ diagnostics"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

This is what the `full` run behind this page prints: the four checks above, then the
spread of `az.summary` over all 75 quantities:

```text
  Posterior: 4 chains × 1000 draws
  Mean acceptance rate: 0.786
  Divergences: 0 / 4000 draws (0.0%)
  Leapfrog steps/draw: mean 255, 100% at the cap (255)
[5/7] Computing ArviZ diagnostics...
            mean         sd      r_hat     ess_bulk
count  75.000000  75.000000  75.000000    75.000000
mean   -2.796666   0.148029   1.004584  1012.223180
std     1.892634   0.053201   0.003398   314.698960
min    -6.242783   0.077725   0.999981   398.686765
25%    -4.171345   0.108828   1.002054   806.841445
50%    -3.697674   0.142442   1.003868  1034.452500
75%    -0.410639   0.163172   1.006311  1207.651958
max     0.333164   0.337189   1.017289  1837.560237
  R-hat max on Σ: 1.017  (target < 1.01)
  ESS min on Σ:   399    (target > 100 per chain)
```

Read in the order of the table above: **no divergences**, so the geometry was never
pathological; acceptance 0.79, in range; R-hat at worst 1.017, with 93% of quantities under
1.01; bulk ESS at worst 399 against a target of 400, which is a miss by one effective draw
out of 4000. This is what a converged fit of this model looks like. The one blemish is
unchanged from the cheaper presets: every draw still ran to the tree-depth cap of 255
integration steps, so trajectories are truncated and consecutive draws stay correlated. The
cap, not the number of draws, is what is now limiting ESS.

??? note "What the cheaper presets give, and why we  use `full`"

    The same four checks, measured on all three presets. Only the thoroughness of the fit
    changes between the rows that matter; `quick` also uses less data, so it is not simply
    a shorter version of the other two.

    | | `quick` | `balanced` | `full` | target |
    |---|---|---|---|---|
    | chains × draws | 2 × 150 | 4 × 500 | 4 × 1000 | n/a |
    | `mc_samples`, trials | 50, 450 | 500, 2500 | 2000, 5000 | n/a |
    | divergences | 0 / 300 | 0 / 2000 | 0 / 4000 | 0 |
    | acceptance | 0.802 | 0.829 | 0.786 | 0.6–0.9 |
    | R-hat max on $\Sigma$ | 1.569 | 1.033 | **1.017** | < 1.01 |
    | share R-hat < 1.01 | not measured | 44% | **93%** | 100% |
    | bulk ESS min | 4 | 92 | **399** | 400 |
    | draws at the depth cap | 100% | 100% | 100% | n/a |

    Two things are worth reading off this table. Divergences are zero at every preset, so
    the sampler never had trouble with the geometry; what the cheap presets lack is simply
    draws. And the cap saturates at 100% everywhere, which is why ESS remains the binding
    constraint even at `full`: raising `max_num_doublings` is the untried lever.

[`az.summary`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.summary.html)
returns one row per point and quantity: 27 rows in `quick` (9 points × 3) and 75 in
`balanced` and `full` (25 × 3). The script prints only the spread of each column, because that many
rows are not readable.

??? note "What the columns of `az.summary` mean"

    | column | meaning |
    |---|---|
    | `mean`, `sd` | posterior mean and standard deviation of that quantity |
    | `eti89_lb`, `eti89_ub` | lower and upper end of the 89% credible interval: the quantity lies in this range with 89% posterior probability (equal-tailed by default; `ci_kind="hdi"` and `ci_prob=` change it). This is the pointwise interval that [Step 6](#from-a-cloud-of-curves-to-an-interval) turns into a band on the threshold contour |
    | `ess_bulk`, `ess_tail` | effective sample size in the middle and in the tails of the distribution ([`az.ess`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html)) |
    | `r_hat` | between-chain agreement ([`az.rhat`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html)); explained below |
    | `mcse_mean`, `mcse_sd` | Monte Carlo error of `mean` and `sd`: how much they would move if you re-ran the sampler |

The single row to look at first is the one with the **largest** `r_hat` and the
**smallest** `ess_bulk`. If the worst quantity passes, all of them do.

#### Trace and convergence plots

Both figures below are plain ArviZ output, with no extra styling. They come from
`--mode full` (4 chains × 1000 draws).

```python title="Trace plot of Σ at the worst reference point"
--8<-- "docs/examples/wppm/nuts_inference_example.py:trace_plot"
```

<div align="center">
  <img src="../plots/nuts_trace_sigma_full.png" width="800"/>
  <p><em>The three \(\Sigma\) quantities at the reference point with the highest R-hat, one
  colour per chain, from <code>--mode full</code>.</em></p>
</div>

The first line picks the reference point whose worst quantity has the highest R-hat, so
the plot shows the worst case, chosen by the numbers rather than by hand. ArviZ then draws
one panel per quantity:

| element of the figure | what it is |
|---|---|
| panel title, e.g. `Sigma` / `log_var_0` | the variable and the quantity shown. The selected point is not part of the title; it is the worst one, as described above |
| x axis, `Draw` | draw index **within a chain**, post-warmup (warmup draws are discarded, not plotted) |
| y axis | the value of that quantity: a log variance, or the correlation |
| one coloured line | one chain |
| black ticks along the bottom | divergences, if any; they appear because the sampler statistics carry `diverging` under ArviZ's standard name |

**What good looks like:** chains overlapping each other, flat, and wiggling fast; the
"fuzzy caterpillar". That is what this run shows: in all three panels the four chains cover
the same range, none drifts, and none sits at its own level. There are no black ticks along
the bottom either, because this run had no divergences; on a run that did, that is where
they would appear.

A chain that drifts steadily across a panel, or sits at its own level, has not converged:
warmup was too short and the sampler is still finding the typical set. That is the most
common thing this plot catches, and it is a reason to raise `--mode`, not to distrust the
model.

```python title="ESS and R-hat over all Σ quantities"
--8<-- "docs/examples/wppm/nuts_inference_example.py:convergence_plot"
```

<div align="center">
  <img src="../plots/nuts_convergence_sigma_full.png" width="800"/>
  <p><em>Bulk ESS, tail ESS and R-hat for all 75 \(\Sigma\) quantities (25 reference points × 3),
  from <code>--mode full</code>, with ArviZ's default settings.</em></p>
</div>

**How to read it.** There is one panel per diagnostic, titled with ArviZ's name for it
(`ess_bulk`, `ess_tail`, `rhat`). Each panel is a cumulative curve over all 75 quantities:
its height at a value $x$ is the **fraction of quantities** whose diagnostic is at or below $x$.
The dashed lines are ArviZ's reference values: 1.01 for R-hat, and 100 effective draws
per chain for ESS, which is 400 here with 4 chains (200 in `quick` with 2).

- **R-hat:** you want the curve to reach 1 by the dashed line, meaning every quantity is
  below 1.01.
- **ESS:** you want the opposite, the curve still at 0 at the dashed line, meaning every
  quantity has more effective draws than the threshold.

In this run both curves sit where you want them. **93%** of the quantities have an R-hat
below 1.01 and the largest is **1.017**; that is ArviZ's default, the rank-normalized split
R-hat, and the comparison below shows what the other variants give. The ESS curves barely
start before their dashed line: the smallest bulk ESS is **399** against a target of 400,
so exactly one quantity out of 75 falls short, by one effective draw, and **no** quantity
falls short on tail ESS, where the smallest is 686.

That is a converged fit, with one cosmetic miss. The figures in Step 6 can be read as the
posterior rather than as an approximation to it. What remains is the tree-depth cap at
100%: it is why ESS is the diagnostic that comes closest to failing, and why raising
`max_num_doublings` would buy more than another thousand draws.

??? note "The same figure before the fit converged (`--mode balanced`)"

    <div align="center">
      <img src="../plots/nuts_convergence_sigma_balanced.png" width="760"/>
      <p><em>Bulk ESS, tail ESS and R-hat from <code>--mode balanced</code>: 4 chains ×
      500 draws, <code>mc_samples=500</code>, half the trials.</em></p>
    </div>

    Same model, same diagnostics, a quarter of the sampling effort. Every curve has moved
    to the wrong side of its dashed line: 57% of quantities have a bulk ESS below 400
    (smallest 92), 11% fall short on tail ESS, and only 44% reach R-hat < 1.01, with a
    worst case of 1.033. Nothing about it is *wrong*; there were no divergences here
    either, and the ellipses it produces are close to the converged ones. It is simply not
    finished, and this is the shape to expect from your own first run. The fix was more
    draws and more Monte Carlo samples, nothing cleverer.

#### What R-hat measures, and what "rank-normalized" means

The classic R-hat ([Gelman and Rubin, 1992](https://doi.org/10.1214/ss/1177011136))
compares the spread *between* chains with the spread *within* each chain. If every chain
explores the same distribution, the two agree and R-hat is about 1. If the chains sit in
different places, the between-chain spread is larger and R-hat rises above 1. It needs at
least two chains: a single chain cannot tell you it is stuck.

ArviZ does not compute that classic version. By default it computes the improved R-hat of
[Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221) (`az.rhat` uses `method="rank"`;
`az.summary` and `az.plot_convergence_dist` above use the same), which is stricter in three
ways. That is also where the `< 1.01` threshold comes from, together with `> 100`
effective draws per chain. Both are recommendations from that paper, which is why the older
`< 1.1` rule is now considered too lenient.

??? note "The three changes, and why each one exists"

    1. **Split chains.** Each chain is cut in half and the halves are compared as separate
       chains. A single chain that drifts slowly then shows up as its first half
       disagreeing with its second half, which no between-chain comparison of whole
       chains can see.
    2. **Rank-normalize.** All draws from all chains are pooled and replaced by their ranks
       (1 for the smallest), and the ranks are mapped onto a standard normal scale. The
       classic formula is then applied to these scores rather than to the raw values. This
       makes the result robust to heavy tails and outliers, and unchanged by any
       order-preserving transform: taking the log of a variance does not change its R-hat.
    3. **Check the tails as well.** The computation is repeated on the distance of each
       draw from the median, which catches chains that agree on location but differ in
       spread. The larger of the two values is reported.

The extra steps matter in practice. On the `full` run behind this page, the share of the
75 $\Sigma$ quantities that pass `< 1.01` still depends on which version you compute:

| R-hat variant | share below 1.01, `full` | same, `balanced` |
|---|---|---|
| classic (Gelman & Rubin, 1992) | 100% | 79% |
| + split chains (step 1) | 93% | 45% |
| + rank-normalization (step 2); ArviZ's default, and the figure above | 93% | 44% |

Splitting is what does the work. The classic formula calls this fit perfect, with every
quantity under 1.01 and a worst case of 1.009; splitting each chain in half finds drift
within chains that no comparison of whole chains can see, and drops the pass rate to 93%.
Rank-normalization changes nothing here, because these quantities have no heavy tails; on
a posterior that did, it would matter more.

The `balanced` column is the same comparison on a fit that had not converged, and it makes
the point sharper: there, the classic R-hat still passed 79% of quantities while the
version ArviZ actually computes passed 44%. "R-hat looked fine" is not a claim until you
say which R-hat.

**ESS** (effective sample size) is computed with the same rank-normalization. It answers
how many *independent* draws your correlated chain is worth. To make it concrete: 1000
draws with an ESS of 50 carry about as much information as 50 independent draws, because
neighbouring draws in a chain partly repeat each other.

#### Figures ArviZ gives you

The two figures above are `az.plot_trace` and `az.plot_convergence_dist`. Three more are
worth knowing about when those two leave you unsure.

??? note "Three further ArviZ figures, and what bad looks like in each"

    All of these work on the result of `covariance_to_arviz` directly; for $\Sigma$, pick a
    point or quantity with `coords`, e.g. `coords={"point": 0}`.

    | function | what it shows | what bad looks like |
    |---|---|---|
    | [`az.plot_rank`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_rank.html) | for each chain, how its draws rank among *all* chains' draws | one chain's curve leaving the band: it is exploring somewhere else |
    | [`az.plot_trace_dist`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_trace_dist.html) | each quantity's distribution per chain, next to its trace | one chain's distribution sitting apart from the others |
    | [`az.plot_energy`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_energy.html) | NUTS-specific: whether the sampler moves freely between energy levels (BFMI) | the two energy distributions differ clearly in width, or BFMI < 0.3 |

On ArviZ ≥ 1.0 these return a
[`PlotCollection`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.PlotCollection.html)
rather than an array of matplotlib axes, so they are saved with `pc.savefig()`. The
[arviz-plots gallery](https://arviz-plots.readthedocs.io/en/latest/gallery/index.html)
shows every plot type with example output.

### Step 6: Posterior predictive uncertainty

```python title="Draw posterior parameter samples"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_samples"
```

Each draw gives one complete covariance field, so every blue ellipse below is a field the
data cannot rule out. For a single summary field, the *posterior mean* $\mathbb{E}[\Sigma(x)]$, average the
covariance matrices over all draws:

```python title="Posterior mean covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_mean"
```

??? note "Why the mean is computed from \(\Sigma\), not from \(W\)"

    The tempting shortcut is to average the parameters and build one field from the
    result, `WPPMCovarianceField(model, nuts_posterior.params)`. That computes $\Sigma(\mathbb{E}[W])$,
    and for the WPPM it gives ellipses that are far too small.

    The reason is that $\Sigma(x) = U(x)\,U(x)^\top + \lambda I$, with $U$ linear in $W$, so $\Sigma$ is roughly
    *$W$ squared*. Squaring and then averaging is not the same as averaging and then
    squaring. A one-number example with two draws, $W = +1$ and $W = -1$:

    | | draw 1 | draw 2 | average |
    |---|---|---|---|
    | $W$ | $+1$ | $-1$ | $\mathbf{0}$ |
    | $\Sigma = W^2$ | $1$ | $1$ | $\mathbf{1}$ |

    Both draws describe the same ellipse, $\Sigma = 1$. Averaging $\Sigma$ gives $1$, the right answer.
    Averaging $W$ first gives $0$, and $0^2 = 0$: the ellipse disappears.

    The $W = \pm 1$ example is not a contrivance: it is exactly the rotation symmetry of
    [$W$ is not identified](#a-note-on-the-parameters-w-is-not-identified) in one dimension,
    where the only orthogonal transforms are $\pm 1$. Draws that each describe the same
    ellipse cancel when their parameters are averaged. And even without any symmetry,
    $\mathbb{E}[\Sigma] = \Sigma(\mathbb{E}[W]) + \operatorname{Cov}[U]$ holds exactly, so averaging $W$
    first always drops the spread term and can never give larger ellipses.

    `PosteriorMeanCovarianceField` computes $\Sigma$ for every draw first and averages afterwards,
    so neither effect applies. One thing to keep in mind: an average of covariance matrices
    is somewhat larger than a typical single draw, because draws differ in size. That is
    what a mean does, not an error.

```python title="Plot ensemble of ellipses"
--8<-- "docs/examples/wppm/nuts_inference_example.py:plot_uncertainty"
```

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty_full.jpg" width="600"/>
  <p><em>Faint blue: 200 covariance fields sampled from the posterior. Red: posterior mean \(\mathbb{E}[\Sigma(x)]\), averaged over all draws. Black: ground truth. The spread of blue ellipses shows where the model is uncertain.</em></p>
</div>

#### From a cloud of curves to an interval

The figure above is a *cloud*: a hundred contours the data cannot rule out, drawn on top of
one another. That is the honest picture, but it is not what goes in a paper caption. A
paper reports an interval: "the threshold in this direction is 0.08, 89% CI [0.06, 0.11]".
The draws already contain it.

The step that makes it possible is writing a contour as a **radius per direction**. The
threshold contour of draw $i$ at reference point $x$ is the set
$\{d : d^\top \Sigma_i(x)^{-1} d = 1\}$, and along the unit direction
$u(\varphi) = (\cos\varphi, \sin\varphi)$ that set is a single point, at distance

$$r_i(\varphi) \;=\; \frac{1}{\sqrt{u(\varphi)^\top \Sigma_i(x)^{-1} u(\varphi)}}$$

from the reference point. Along an eigenvector of $\Sigma_i$ this is $\sqrt{\lambda}$, the
semi-axis the ellipse plots are already drawing, so nothing new is being shown; it is the
same contour in polar form. The gain is that $r_i(\varphi)$ is a *scalar*, one number per
draw, so the ordinary recipe for a credible interval applies: sort the draws and take
quantiles. For a $1-\alpha$ interval, at each direction separately,

$$\big[\,q_{\alpha/2}\{r_i(\varphi)\}_{i=1}^{N},\;\; q_{1-\alpha/2}\{r_i(\varphi)\}_{i=1}^{N}\,\big]$$

and filling between the two curves over all $\varphi$ gives a band. $r_i(\varphi)$ also has a
meaning the entries of $\Sigma$ do not. It is the discrimination threshold, the just
noticeable difference, in direction $\varphi$, which is the quantity an experiment set out
to measure.

```python title="Pointwise credible band on the threshold contour"
--8<-- "docs/examples/wppm/nuts_inference_example.py:credible_band"
```

<div align="center">
  <img src="../plots/nuts_threshold_band_full.jpg" width="620"/>
  <p><em>Pointwise 89% credible band on each threshold contour (blue fill), the posterior
  median contour (blue line) and ground truth (black), from <code>--mode full</code>
  over 1000 draws.</em></p>
</div>

The band is narrow where the data pin the threshold down and wide where they do not: its
width runs from **14%** of the contour's own radius at the best-determined reference point
to **64%** at the worst, with a median of **24%**. A MAP fit reports the same ellipses with
no way to tell those two cases apart.

The black contour grazes the band, and leaves it in places. That is worth looking at rather
than away from, and it is where a pointwise band earns its caveats. An 89% pointwise band
is *supposed* to miss about 11% of directions. Counted over all 25 × 181 (point, direction)
pairs, the truth falls inside this band **82%** of the time, with **11 of the 25** points
covered in every direction.



!!! warning "This band is pointwise, not simultaneous"
    The interval above covers the true radius **in each direction separately** with
    probability $1-\alpha$. It does *not* say that the whole true contour lies inside the
    band with probability $1-\alpha$. That is a stronger, *simultaneous* statement, and it
    needs a different construction (calibrate one global factor on the worst standardized
    deviation over $\varphi$, rather than taking a quantile at each $\varphi$ independently).
    A simultaneous band is wider.

    The same distinction applies to the `eti89_lb`/`eti89_ub` columns of `az.summary` in
    Step 5: those are pointwise intervals on single $\Sigma$ quantities, not a joint region
    for the matrix.

The default here is $1-\alpha = 0.89$, which is ArviZ's default `ci_prob`, so the band and
the `az.summary` columns are the same kind of interval at the same level. There is nothing
special about 89%; it is a round number chosen to be visibly *not* 95%, so that nobody
reads it as a significance test.




!!! note "Every ellipse here is drawn by `psyphy.viz`"
    Both covariance figures on this page go through `plot_ellipses`, which accepts the
    covariances in whichever shape you already have:

    | shape | meaning |
    |---|---|
    | `(n_points, 2, 2)` | one field, for instance the posterior mean |
    | list of `(n_points, 2, 2)` | several fields overlaid, each with its own colour, width, linestyle and legend entry |
    | `(n_samples, n_points, 2, 2)` | a stack of fields, which is exactly what `sample()` returns |



<div align="center">
  <img src="../plots/nuts_overlay_full.jpg" width="650"/>
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

## Reference: tuning and runtime

Everything in this section is a knob, not a step. It is here rather than inside
[Step 4](#step-4-nuts-sampling) so that the walkthrough stays a walkthrough.

### Sampler arguments

The ones you are most likely to touch:

| argument | what it controls | turn it up when |
|---|---|---|
| `num_warmup` | tuning steps per chain, discarded | trace plots still drift at the start |
| `num_samples` | kept draws per chain | ESS is too low |
| `num_chains` | independent chains; R-hat needs ≥ 2 | you want a stronger convergence check |
| `max_num_doublings` | tree-depth cap, $\le 2^k - 1$ gradients per draw | most draws hit the cap (the script prints this) |
| `target_acceptance_rate` | how cautious the tuned step size is | you see divergences (try 0.9) |

### Compute presets

The script takes `--mode`, which selects a preset (see *How to run
the script* at the top).
What the presets scale is three different things, and only the third is "how hard the
sampler works":

- **how much data** there is (9 reference points × 50 trials in `quick`, 25 × 200 in `full`),
- **how accurately the likelihood is estimated** (`mc_samples` 50 → 2000),
- **how long the sampler runs** (chains, warmup, draws).

The first two change the posterior *being targeted*, not just how well it is explored, so a
`quick` fit is not a rough version of a `full` fit; it is a different, smaller problem.

```python
--8<-- "docs/examples/wppm/nuts_inference_example.py:compute_settings"
```

### Runtime

??? note "Why a run costs what it costs"

    - **The MC likelihood dominates everything.** Each gradient evaluation simulates
      `mc_samples` Monte Carlo draws for every trial, so cost scales with
      `mc_samples × n_trials`. Going from `quick` to `full` raises that product by ~440×,
      which is why `full` is a batch job and not something to run interactively.
    - **NUTS cost per draw is not fixed.** It is `O(2^tree_depth)` gradient evaluations, and
      the depth is data-dependent, commonly 16 to 64 evaluations per draw. The script therefore
      **times one gradient and prints a bracketed projection** before sampling starts, so you
      can see whether a run fits inside a Slurm time limit before committing to it.
    - **Tree depth is capped.** `NUTSSampler(max_num_doublings=8)` (the default) limits a draw
      to at most $2^8 - 1 = 255$ gradient evaluations; BlackJAX's own default of 10 allows 1023.
      If `sampler_stats["n_steps"]` sits at the cap for most draws, trajectories are being
      truncated: still valid MCMC, but slower mixing. Raise the cap or reparameterize.
    - **Warmup starts from a prior-scaled metric.** By default the sampler works in
      `z = W / prior_std` (`prior_preconditioning=True`), so adaptation does not begin from an
      identity mass matrix that is badly mismatched to coefficients. Without this, early warmup runs every draw to the maximum tree depth.
    - **Chains are vmapped, in both warmup modes.** All chains advance in one compiled call per
      phase, so XLA compiles once rather than once per chain. Under vmap each step costs as much
      as the *deepest* tree across chains, which is one more reason for the depth cap. On a
      GPU, four chains cost about as much as one; on a CPU, many chains can be slower than
      running them one by one.
      Note: the progress bar moves only when each phase (warmup, sampling) finishes.



### Two warmup modes

| Mode | When | How |
|------|------|-----|
| Adaptive (default, `step_size=None`) | Best quality; adapts step size and mass matrix per chain via `blackjax.window_adaptation`, starting from a prior-scaled metric. Chains vmapped; each adapts independently. | `NUTSSampler(num_warmup=200, num_samples=500)` |
| Fixed (`step_size=<float>`) | Faster with many chains; no adaptation. All chains vmapped in one compiled call. User must choose a good step size. | `NUTSSampler(..., step_size=0.01)` |

!!! info "Samplers other than NUTS"
    psyphy ships NUTS only. Nothing *downstream* of the sampler is NUTS-specific, though:
    any engine that returns an `MCMCPosterior` whose draws are shaped
    `(n_chains, n_draws, *W_shape)` works unchanged with `to_arviz`, `covariance_to_arviz`,
    `PosteriorMeanCovarianceField`, `WPPMPredictivePosterior` and every figure on this page.
    BlackJAX itself offers MALA, plain HMC and random-walk Metropolis.

    What is missing is the *upstream* half: the pieces that make a sampler a psyphy
    sampler. The log density with the frozen Monte Carlo key, the prior preconditioning and
    the vmapped chains all live inside `NUTSSampler.fit` rather than in a shared base class,
    so adding a second engine today means copying them rather than setting an option.
    Factoring them out is planned.

    Two caveats would carry over to any gradient-based sampler, because they belong to the
    model and the likelihood rather than to NUTS: the
    [frozen-key likelihood](#a-note-on-the-likelihood) and the
    [rotation symmetry of $W$](#a-note-on-the-parameters-w-is-not-identified). What does not
    carry over is the diagnostics. A sampler without trajectories has no divergences, no
    energies and no tree sizes, so parts of Step 5 would simply not apply.

---

## Further reading

- API reference: [`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler),
  [`MCMCPosterior`](../../reference/posterior.md#psyphy.posterior.MCMCPosterior),
  [`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz),
  [`PosteriorMeanCovarianceField`](../../reference/model.md#psyphy.model.PosteriorMeanCovarianceField),
  [`InferenceEngine`](../../reference/inference.md#psyphy.inference.base.InferenceEngine)
- MAP fitting tutorial: [`full_wppm_fit_example`](full_wppm_fit_example.md)
- Model details: [`wppm_tutorial`](wppm_tutorial.md)
- BlackJAX documentation: [blackjax-devs.github.io/blackjax](https://blackjax-devs.github.io/blackjax/)
- ArviZ documentation: [python.arviz.org](https://python.arviz.org/), with its
  [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html),
  the [arviz-stats API](https://python.arviz.org/projects/stats/en/latest/) (R-hat, ESS,
  `summary`) and the [arviz-plots gallery](https://arviz-plots.readthedocs.io/en/latest/gallery/index.html)

The papers behind the method and the diagnostics cited above:

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
