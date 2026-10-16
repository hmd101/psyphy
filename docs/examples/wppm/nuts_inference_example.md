# NUTS Posterior Inference for WPPM

This tutorial demonstrates **full posterior inference** using the No-U-Turn Sampler (NUTS) for the Wishart Process Psychophysical Model (WPPM).

The [Maximum A Posteriori (MAP) tutorial](full_wppm_fit_example.md) ends with *one* threshold ellipse per reference
point: the model's single best guess. This page ends with a *distribution* of ellipses per reference
point. Where the distribution is
tight, the data pin the threshold down. Where it fans out, they do not, and a single MAP
ellipse would have hidden that.

The companion script can be found at
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).

**Outline:**

| step | what you do | what you get |
|---|---|---|
| 1–2 | simulate data, build the WPPM | `TrialData` and a model, same as the MAP tutorial |
| 3 | run a short MAP fit | a starting point near the posterior peak |
| 4 | run NUTS ([BlackJAX](https://blackjax-devs.github.io/blackjax/)) | an `MCMCPosterior`: hundreds to thousands of plausible sets of parameter `W`s |
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

    # 25 refs, 2000 MC samples, 4 chains x (1000 + 1000).
    python nuts_inference_example.py --mode full
    ```

    `quick` exists so the page can be regenerated quickly: about a minute on an A100 GPU,
    about 20 minutes on a CPU. Don't expect good diagnostics from these settings yet.


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
the normalizing constant.

The challenge is building a walk that explores efficiently. A naive random walk works maybe in two or
three dimensions and fails badly in fifty, where almost every direction you could step in
leads somewhere less probable, so nearly all proposals are rejected and the chain crawls.
[Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) fixes this
by using the *gradient* of the log posterior: it gives the parameter vector a random
momentum and simulates it rolling along the posterior's own geometry, so a single proposal
can travel a long way and still be accepted. **NUTS**, the No-U-Turn Sampler of
[Hoffman and Gelman (2014)](https://jmlr.org/papers/v15/hoffman14a.html), removes the one
awkward tuning knob by extending each trajectory until it starts to double back on itself.
Betancourt's [*Conceptual Introduction to Hamiltonian Monte Carlo*](https://arxiv.org/abs/1701.02434)
is the readable account of why this works and where it fails; it is worth skimming.

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
| `PosteriorMeanCovarianceField(nuts_posterior)` | the posterior mean covariance field $\mathbb{E}[\Sigma(x)]$ | Step 6, the red "posterior mean" ellipses |
| `covariance_to_arviz(nuts_posterior, points)` | $\Sigma$ at chosen points for every draw, in ArviZ's format, for diagnostics | Step 5 |

`MCMCPosterior` and `MAPPosterior` share this interface, so code written for a MAP fit,
for example `WPPMPredictivePosterior`, accepts a NUTS fit unchanged.

!!! warning "Don't draw ellipses from `nuts_posterior.params`"
    `nuts_posterior.params` is the average **parameter** $\mathbb{E}[W]$. It is a valid parameter set
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
| `max_num_doublings` | tree-depth cap, $\le 2^k - 1$ gradients per draw | most draws hit the cap (the script prints this) |
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



#### Two warmup modes

| Mode | When | How |
|------|------|-----|
| Adaptive (default, `step_size=None`) | Best quality; adapts step size and mass matrix per chain via `blackjax.window_adaptation`, starting from a prior-scaled metric. Chains vmapped; each adapts independently. | `NUTSSampler(num_warmup=200, num_samples=500)` |
| Fixed (`step_size=<float>`) | Faster with many chains; no adaptation. All chains vmapped in one compiled call. User must choose a good step size. | `NUTSSampler(..., step_size=0.01)` |

??? note "Using a different BlackJAX sampler, e.g. MALA"

    psyphy ships NUTS only, but BlackJAX offers other samplers, such as MALA
    (`blackjax.mala`), plain HMC or random-walk Metropolis. Here is what using one would take.

    **What psyphy needs from a sampler.** A subclass of
    [`InferenceEngine`](../../reference/inference.md#psyphy.inference.base.InferenceEngine)
    whose `fit(model, data)` returns an `MCMCPosterior`, with draws shaped
    `(n_chains, n_draws, *W_shape)`. Once it does, everything downstream works unchanged:
    `to_arviz`, `covariance_to_arviz`, `PosteriorMeanCovarianceField`,
    `WPPMPredictivePosterior` and every figure on this page.

    **What BlackJAX needs from psyphy.** Only a log-density function. In
    [`nuts.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/inference/nuts.py)
    the block marked "BlackJAX API boundary" builds it from
    `model.log_posterior_from_data`, with the frozen Monte Carlo key from
    [A note on the likelihood](#a-note-on-the-likelihood).

    **For MALA specifically**, the change at the centre is replacing the kernel,
    `blackjax.nuts(logdensity_fn, step_size=..., inverse_mass_matrix=...)`, with
    `blackjax.mala(logdensity_fn, step_size)`. Three things differ:

    - **No automatic tuning.** `window_adaptation` (Mode A above) tunes a mass matrix, which
      MALA does not have, so its step size must be chosen some other way.
    - **Fewer diagnostics.** MALA reports only an acceptance rate and whether each proposal
      was accepted, so there are no divergences, energies or tree sizes in `sampler_stats`.
    - **Same caveats.** The frozen-key likelihood applies to any gradient-based sampler,
      and the rotation symmetry of `W` (Step 5) is a property of the model, not of NUTS.

    **How much work that is today.** It is copy-and-adapt rather than configuration. The
    psyphy-specific parts live inside
    [`NUTSSampler.fit`](../../reference/inference.md#psyphy.inference.NUTSSampler), not in a
    shared base class: the log density with the frozen key, the prior preconditioning,
    running chains in parallel and wrapping the result in `MCMCPosterior`. A MALA sampler
    would copy those (about a hundred lines, mostly the fixed-step-size path) and swap the
    kernel. The unimplemented
    [`LangevinSampler`](../../reference/inference.md#psyphy.inference.langevin.LangevinSampler)
    stub is the natural place for it.

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

#### What to diagnose: $\Sigma$, not $W$

The obvious thing to check would be the parameters themselves:
`nuts_posterior.to_arviz()` hands the raw draws of `W` to ArviZ. For the WPPM that is
misleading. `W` is only defined up to a rotation: rotating `W` along its embedding axis
(its last axis) leaves every covariance matrix $\Sigma(x)$ unchanged, and the prior too. So
different chains can settle at different rotations. They then disagree about `W` while
agreeing about every ellipse, and diagnostics computed on `W` report a failure that no
ellipse, threshold or prediction is affected by. (It is the same symmetry that makes the
naive posterior mean too small; see the note in Step 6.)

If you run diagnostics on `W` yourself, for example
`az.summary(nuts_posterior.to_arviz())`, expect them to look bad: in our `balanced` runs,
R-hat on `W` reaches about 2, while on $\Sigma$ it stays around 1.07. That is the symmetry, not a
failed fit.

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

`W` is one fixed set of numbers, but $\Sigma$ is a function of the stimulus location $x$. A
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

Choosing points is a property of this approach, not a law of nature. Quantities of `W`
that are unchanged by the rotation, such as $W W^\top$ summed over the embedding axis, would
allow a point-free check; psyphy does not implement that.

#### Running the diagnostics

```python title="ArviZ diagnostics"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

[`az.summary`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.summary.html)
returns one row per point and quantity: 27 rows in `quick` (9 points × 3) and 75 in
`balanced` (25 × 3). The script prints only the spread of each column, because that many
rows are not readable. If you print `summary` itself, the columns mean:

| column | meaning |
|---|---|
| `mean`, `sd` | posterior mean and standard deviation of that quantity |
| `eti89_lb`, `eti89_ub` | lower and upper end of the 89% credible interval: the quantity lies in this range with 89% posterior probability (equal-tailed by default; `ci_kind="hdi"` and `ci_prob=` change it) |
| `ess_bulk`, `ess_tail` | effective sample size in the middle and in the tails of the distribution ([`az.ess`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html)) |
| `r_hat` | between-chain agreement ([`az.rhat`](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html)); explained below |
| `mcse_mean`, `mcse_sd` | Monte Carlo error of `mean` and `sd`: how much they would move if you re-ran the sampler |

The single row to look at first is the one with the **largest** `r_hat` and the
**smallest** `ess_bulk`. If the worst quantity passes, all of them do.

#### Trace and convergence plots

Both figures below are plain ArviZ output, with no extra styling. They come from
`--mode balanced` (4 chains × 500 draws): `quick` is too short for the chains to settle,
so its plots show little beyond "not converged yet". Running the script with
`--mode balanced` writes them under these names.

```python title="Trace plot of Σ at the worst reference point"
--8<-- "docs/examples/wppm/nuts_inference_example.py:trace_plot"
```

<div align="center">
  <img src="../plots/nuts_trace_sigma_balanced.png" width="800"/>
  <p><em>The three \(\Sigma\) quantities at the reference point with the highest R-hat, one
  colour per chain, from <code>--mode balanced</code>.</em></p>
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

**What good looks like:** chains overlapping each other, flat, and wiggling fast, the
"fuzzy caterpillar". That is what this run shows: the four chains cover the same range in
all three panels, and only one chain sits a little high in `log_var_0` for its first
50 or so draws.

A chain that drifts steadily across a panel, or sits at its own level, has not converged:
warmup was too short and the sampler is still finding the typical set. That is the most
common thing this plot catches, and it is a reason to raise `--mode`, not to distrust the
model.

```python title="ESS and R-hat over all Σ quantities"
--8<-- "docs/examples/wppm/nuts_inference_example.py:convergence_plot"
```

<div align="center">
  <img src="../plots/nuts_convergence_sigma_balanced.png" width="800"/>
  <p><em>Bulk ESS, tail ESS and R-hat for all 75 \(\Sigma\) quantities (25 reference points × 3),
  from <code>--mode balanced</code>, with ArviZ's default settings.</em></p>
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

In this run, about 40% of the quantities have R-hat below 1.01, and the largest is 1.066.
Roughly three quarters have a bulk ESS below 400 (the smallest is 50), and about one in six
has a tail ESS below 400. So the fit is close to converged on $\Sigma$ but does not meet the strict
thresholds, mainly because it collected too few effective draws. That fits another number
the script prints: every draw ran to the tree-depth cap of 255 integration steps, so the
trajectories are cut short and consecutive draws stay correlated. The remedy is more
draws, from `--mode full` (4 × 1000) or a larger `num_samples`.

#### What R-hat measures, and what "rank-normalized" means

The classic R-hat ([Gelman and Rubin, 1992](https://doi.org/10.1214/ss/1177011136))
compares the spread *between* chains with the spread *within* each chain. If every chain
explores the same distribution, the two agree and R-hat is about 1. If the chains sit in
different places, the between-chain spread is larger and R-hat rises above 1. It needs at
least two chains: a single chain cannot tell you it is stuck.

ArviZ computes the improved version of
[Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221) by default (`az.rhat` uses
`method="rank"`; `az.summary` and `az.plot_convergence_dist` above use the same). Three
changes make it stricter and more robust:

1. **Split chains.** Each chain is cut in half and the halves are compared as separate
   chains. A single chain that drifts slowly then shows up as its first half disagreeing
   with its second half.
2. **Rank-normalize.** All draws from all chains are pooled and replaced by their ranks
   (1 for the smallest), and the ranks are mapped onto a standard normal scale. The classic
   formula is then applied to these scores rather than to the raw values. This makes the
   result robust to heavy tails and outliers, and unchanged by any order-preserving
   transform: taking the log of a variance does not change its R-hat.
3. **Check the tails as well.** The computation is repeated on the distance of each draw
   from the median, which catches chains that agree on location but differ in spread. The
   larger of the two values is reported.

That is also where the `< 1.01` threshold comes from, together with `> 100` effective
draws per chain: they are recommendations from the same paper, which explains why the
older `< 1.1` rule turned out to be too lenient.

The extra steps matter in practice. On a `balanced` run, the classic R-hat puts 61% of
the $\Sigma$ quantities below 1.01. Splitting each chain in half (step 1) catches slow drift
within a chain, and drops that to 35%. Rank-normalization adds almost nothing on top
here, because the $\Sigma$ quantities have no heavy tails.

**ESS** (effective sample size) is computed with the same rank-normalization. It answers
how many *independent* draws your correlated chain is worth. To make it concrete: 1000
draws with an ESS of 50 carry about as much information as 50 independent draws, because
neighbouring draws in a chain partly repeat each other.

#### Figures ArviZ gives you

All of these work on the result of `covariance_to_arviz` directly; for $\Sigma$, pick a point or
quantity with `coords`, e.g. `coords={"point": 0}`. The first two produce the figures above.

| function | what it shows | what bad looks like |
|---|---|---|
| [`az.plot_trace`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_trace.html) | value at every draw, one line per chain | chains at separate levels, or drifting |
| [`az.plot_convergence_dist`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_convergence_dist.html) | distribution of ESS (bulk and tail) and R-hat over all quantities | the R-hat curve reaching 1 far to the right of 1.01, or the ESS curves rising before their dashed line |
| [`az.plot_rank`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_rank.html) | for each chain, how its draws rank among *all* chains' draws | one chain's curve leaving the band: it is exploring somewhere else |
| [`az.plot_trace_dist`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_trace_dist.html) | each quantity's distribution per chain, next to its trace | one chain's distribution sitting apart from the others |
| [`az.plot_energy`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.plot_energy.html) | NUTS-specific: whether the sampler moves freely between energy levels (BFMI) | the two energy distributions differ clearly in width, or BFMI < 0.3 |

On ArviZ ≥ 1.0 these return a
[`PlotCollection`](https://python.arviz.org/projects/plots/en/latest/api/generated/arviz_plots.PlotCollection.html)
rather than an array of matplotlib axes, so they are saved with `pc.savefig()`; the 0.x
keyword arguments `compact=` and `figsize=` no longer exist and raise if passed. The
[arviz-plots gallery](https://arviz-plots.readthedocs.io/en/latest/gallery/index.html)
shows every plot type with example output.

#### Interpreting diagnostics

The most important diagnostics, in the order to check them:

| Diagnostic | What it measures | Good value |
|---|---|---|
| **Divergences** | Transitions where the integrator blew up, usually in a region of high curvature the step size cannot resolve. | **0.** Any divergence makes the run suspect, whatever R-hat says. |
| [**R-hat**](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html) on $\Sigma$ | Whether the chains agree on the same distribution; see above. | < 1.01 |
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
divergence ticks on the trace plot work with no extra wiring.

With `--mode quick` (2 chains and 150 draws) R-hat and ESS will not reach these
thresholds, even on $\Sigma$, and they are not meant to. Use `--mode balanced` or `--mode full`,
which exist for exactly this reason, before drawing any conclusion from the numbers.

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

    The WPPM has this situation built in. Rotating `W` along its embedding axis (its
    last axis) leaves $\Sigma$ unchanged, so many different `W`s produce the same ellipses, and
    different draws, especially from different chains, settle on different ones. On their
    own each draw is fine. Averaged, they partly cancel. Even without that symmetry,
    $\mathbb{E}[\Sigma] = \Sigma(\mathbb{E}[W]) + \operatorname{Cov}[U]$ holds exactly, so averaging $W$ first always loses the spread
    term and never gives larger ellipses.

    `PosteriorMeanCovarianceField` computes $\Sigma$ for every draw first and averages afterwards,
    so neither effect applies. One thing to keep in mind: an average of covariance matrices
    is somewhat larger than a typical single draw, because draws differ in size. That is
    what a mean does, not an error.

```python title="Plot ensemble of ellipses"
--8<-- "docs/examples/wppm/nuts_inference_example.py:plot_uncertainty"
```

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty.jpg" width="600"/>
  <p><em>Faint blue: 30 covariance fields sampled from the posterior. Red: posterior mean \(\mathbb{E}[\Sigma(x)]\), averaged over all draws. Black: ground truth. The spread of blue ellipses shows where the model is uncertain.</em></p>
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
