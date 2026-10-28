# Checking MCMC draws with ArviZ

[Sampling the posterior with the No-U-Turn Sampler (NUTS)](nuts_inference_example.md) ends
with a distribution of threshold contours, samples we drew from the posterior. Here we show
you how to check whether the samples represent the posterior, and which sampler argument to
change when they do not.

[ArviZ](https://python.arviz.org/projects/stats/en/latest/index.html) is an established
library for diagnosing Bayesian models.
[`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz) hands a
psyphy fit to it in the shape it expects, so every ArviZ function works on our draws with no
reshaping. The figure below carries two of the core diagnostics and is ArviZ's own output.
It is what this tutorial builds towards.

<div align="center">
  <img src="../plots/nuts_convergence_sigma_full.png" width="760"/>
  <p><em>The summary check for posterior samples with ArviZ: Bulk effective sample size (ESS), tail ESS and R-hat, one panel
  each, over every quantity at every reference point. Dashed lines are ArviZ's reference
  thresholds, in total ESS across all chains. We want both ESS curves entirely right of
  their line and the R-hat curve entirely left of its.</em></p>
</div>

Everything here runs on the fit from the
[sampling tutorial](nuts_inference_example.md), from the same script,
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).
You can rebuild every diagnostic and figure from the cached draws without a GPU (a laptop
CPU is fine), like this:

```bash
python nuts_inference_example.py --mode full --from-fit
```

---

## What to check
4 quantities to check. The R-hat and ESS thresholds are ArviZ's defaults, which follow
[Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221). The divergence and tree-depth
guidance comes from Stan, a probabilistic programming language for Bayesian models whose
MCMC advice is the usual reference, in its
[reference manual](https://mc-stan.org/docs/reference-manual/mcmc.html): a divergence is a
validity problem (e.g. step size too large), tree-depth saturation only an efficiency one (e.g., whether we need to increase tree-depth).

| check | what it detects | suspect when |
|---|---|---|
| divergences, `sampler_stats["diverging"]` | the integrator failed, usually in a region of high curvature | any divergence at all |
| [R-hat](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html) | chains have not mixed to the same distribution | $\hat{R} > 1.01$ |
| [Effective Sample Size (ESS)](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html) | autocorrelation: how many independent draws the chain is worth | bulk or tail ESS $< 100$ per chain, so $< 400$ in total across 4 chains |
| `sampler_stats["n_steps"]` | trajectories truncated at the tree-depth cap | at the cap for most draws |

BlackJAX flags a divergence on every draw it takes. `NUTSSampler` keeps those flags in
`sampler_stats`, a plain dict of JAX arrays:

```python title="Divergences and tree-depth saturation"
--8<-- "docs/examples/wppm/nuts_inference_example.py:divergence_check"
```

A divergent transition means the integrator, the thing approximating the trajectory in small
steps ([sampling tutorial](nuts_inference_example.md#why-sample-the-posterior)), left the
posterior's geometry. More sampling doesn't fix the draws it produced.

!!! warning "Any divergence makes the run suspect"
    One is enough, regardless of what R-hat says, which is why we check divergences first.

The second print is the fraction of draws whose trajectory hit
`max_num_doublings`: a large fraction means the cap is limiting mixing.

## Diagnostics on $\Sigma$, with `covariance_to_arviz`

Diagnostics run on $\Sigma$, not on the raw parameters, for the reason we discussed in the
[sampling tutorial](nuts_inference_example.md#step-4-nuts-sampling). One call hands the fit
to ArviZ:

```python title="R-hat and ESS on the covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

`az.summary` gives one row per quantity: two log variances and one correlation at each of
the 25 reference points we chose, (75 rows total). We see the print output from above for the  `--mode full`
run below:

```text
            mean         sd      r_hat     ess_bulk
count  75.000000  75.000000  75.000000    75.000000
mean   -2.796666   0.148029   1.004584  1012.223390
std     1.892634   0.053201   0.003398   314.699325
min    -6.242783   0.077725   0.999981   398.680579
25%    -4.171345   0.108828   1.002054   806.839604
50%    -3.697674   0.142442   1.003868  1034.450751
75%    -0.410639   0.163172   1.006310  1207.652176
max     0.333164   0.337189   1.017289  1837.560237
  R-hat max on Σ: 1.017  (target < 1.01)
  ESS min on Σ:   399    (target > 100 per chain)
```

Drop the `.describe()` to see all 75 rows instead:

```python title="Every quantity, one row each"
print(summary[["mean", "sd", "r_hat", "ess_bulk"]])
```

[`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz) takes the
posterior and the points to evaluate $\Sigma$ at, here the reference points. From there
`az.summary`, `az.rhat`, `az.ess` and every ArviZ plotting function work on a psyphy fit
directly, which is where the two figures below come from.

??? note "Careful about ArviZ the spelling of  keywords"

    `covariance_to_arviz` returns an `xarray.DataTree` with dimensions
    `(chain, draw, point, quantity)`, and carries the sampler statistics into ArviZ's
    `sample_stats` group under the keys ArviZ expects:

    - `diverging`
    - `energy`
    - `n_steps`

    Nothing needs renaming, so `az.plot_energy` and the divergence ticks on a trace plot
    work directly. ArviZ's
    [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html)
    covers the format in more detail.

??? note "Watch out what to compute the ESS on"

    `posterior.sample(n)` pools the chains: `(n_chains, n_draws, ...)` flattens to
    `(n_chains * n_draws, ...)`, so a draw no longer carries the chain it came from. R-hat
    and ESS both need that axis. On pooled draws R-hat returns about 1.00 however badly the
    real chains disagreed, so the check passes when it should fail.

## Reading the figures

Both figures below are one ArviZ call each, `az.plot_trace` and
`az.plot_convergence_dist`, on the object `covariance_to_arviz` returned. (We write no additional
plotting code here.)

`covariance_to_arviz` summarizes each $2\times2$ $\Sigma$ by the three numbers that
determine it: `log_var_0` and `log_var_1`, the log variances along each axis, and
`corr_0_1`, their correlation. A trace plot shows how each of those moved over the run. If
the chains converged, they should overlap, stay level, and vary fast from draw to draw.

```python title="Trace plot at the worst reference point"
--8<-- "docs/examples/wppm/nuts_inference_example.py:trace_plot"
```

<div align="center">
  <img src="../plots/nuts_trace_sigma_full.png" width="760"/>
  <p><em>One panel per quantity at the reference point with the worst R-hat, one colored
  line per chain. Horizontal axis is the draw index, vertical axis the value of that
  quantity.</em></p>
</div>

Look along each panel for a chain holding its own level, or a slope at either end. Either
means the chains have not mixed.

<div align="center">
  <img src="../plots/nuts_trace_sigma_quick.png" width="760"/>
  <p><em>The same plot from `--mode quick`: 2 chains, 150 draws. Not converged.</em></p>
</div>

Compare the two. Here each chain wanders in slow swings rather than varying fast from draw
to draw, and in `corr_0_1` the two chains hold different levels for stretches of 40 draws or
more. Both are what a trace plot catches. The slower failure, chains that look fine but
carry too few independent draws, is what effective sample size (ESS) and R-hat catch
instead.

We see one reference point here. The figure at the top covers all 25 at once.

```python title="ESS and R-hat over all quantities"
--8<-- "docs/examples/wppm/nuts_inference_example.py:convergence_plot"
```

Each curve there is the distribution of one diagnostic over every $\Sigma$ quantity at
every reference point, so a point on it reads as "this fraction of quantities scored at most
this value". More is better for ESS and less is better for R-hat. For R-hat: where its
curve crosses the dashed line, that fraction of quantities sits above 1.01.

---

## What to change when a check fails

A failing check is fixed by the sampler arguments, which
[`blackjax.nuts`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html)
and
[`window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html)
document.

| argument | what it controls | psyphy default | raise it when |
|---|---|---|---|
| `num_warmup` | tuning steps per chain, discarded | 500 | trace plots still drift at the start |
| `num_samples` | kept draws per chain | 1000 | ESS is too low |
| `num_chains` | independent chains, R-hat needs at least 2 | 4 | a stronger convergence check is wanted |
| `max_num_doublings` | tree-depth cap, at most $2^k - 1$ gradients per draw | 8, against BlackJAX's 10 | most draws hit the cap, which the script prints |
| `target_acceptance_rate` | how cautious the tuned step size is | 0.8 | divergences appear |

??? note "Compute presets"

    ```python title="Compute settings"
    --8<-- "docs/examples/wppm/nuts_inference_example.py:compute_settings"
    ```

    The presets scale three separate things: how much data there is, how accurately the
    likelihood is estimated (`mc_samples`), and how long the sampler runs. The first two
    change the posterior being targeted, so a `quick` fit is a smaller problem rather than a
    rough version of a `full` one.

    ??? note "Compute costs of a run"

        - The Monte Carlo likelihood dominates. Each gradient evaluation simulates
          `mc_samples` draws for every trial, so cost scales with `mc_samples * n_trials`.
          `quick` to `full` raises that product by about 440 times.
        - Cost per draw varies. NUTS needs $O(2^{\text{tree depth}})$ gradient
          evaluations and the depth is data dependent, commonly 16 to 64.
        - Tree depth is capped. The default `max_num_doublings=8` allows at most
          $2^8 - 1$ gradient evaluations per draw. If `sampler_stats["n_steps"]` sits at the
          cap for most draws, trajectories are being truncated: still valid, but mixing more
          slowly.
        - Warmup starts from a prior-scaled metric. By default the sampler works in
          $z = W / \sigma_\text{prior}$ (`prior_preconditioning=True`), so adaptation does
          not begin from an identity mass matrix mismatched to the coefficient scales.
        - Chains are vmapped. All chains advance in one compiled call per phase. Under
          vmap each step costs as much as the deepest tree across chains (which is another
          reason for the depth cap). On a GPU four chains cost about as much as one. On a
          CPU, many chains can be slower than running them one at a time.

??? note "What `step_size` does"

    `step_size` picks between two code paths. The paths are BlackJAX's, but the choice is
    psyphy's: BlackJAX has no notion of modes, and `NUTSSampler` dispatches on whether this
    argument is `None`.

    | `step_size` | what happens | call |
    |---|---|---|
    | `None` (default) | `window_adaptation` tunes the step size and mass matrix per chain, starting from a prior-scaled metric | `NUTSSampler(num_warmup=200, num_samples=500)` |
    | a float | no adaptation and an identity mass matrix. Warmup becomes plain burn-in, so it is faster, and the step size has to be chosen by hand | `NUTSSampler(..., step_size=0.01)` |

    psyphy deviates in two places from the BlackJax defaults:
    `prior_preconditioning=True` samples $z = W / \sigma_\text{prior}$ and maps the draws
    back to $W$, so adaptation starts from a metric matched to the prior scales. And
    `max_num_doublings` defaults to 8 here (against BlackJAX's own 10).

---

## Further reading

- API reference:
  [`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz),
  [`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler),
  [`MCMCPosterior`](../../reference/posterior.md#psyphy.posterior.MCMCPosterior)
- The fit these diagnostics run on:
  [Sampling the posterior with NUTS](nuts_inference_example.md).
- [ArviZ](https://python.arviz.org/) for the diagnostics, with its
  [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html)
  and the [arviz-stats API](https://python.arviz.org/projects/stats/en/latest/)
- [BlackJAX](https://blackjax-devs.github.io/blackjax/) for the sampler and its arguments
- [Stan's reference manual](https://mc-stan.org/docs/reference-manual/mcmc.html) for
  divergences and tree depth, including the reparameterization route when raising
  `target_acceptance_rate` is not enough

The paper behind the ArviZ thresholds:

> Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
> *Rank-normalization, folding, and localization: an improved $\widehat{R}$ for assessing
> convergence of MCMC.* Bayesian Analysis, 16(2).
> [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
