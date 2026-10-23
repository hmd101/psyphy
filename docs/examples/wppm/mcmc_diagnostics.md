# Checking MCMC draws with ArviZ

[Sampling the posterior with the No-U-Turn Sampler (NUTS)](nuts_inference_example.md) ends
with a distribution of threshold contours, samples we drew from the posterior. Here we
check whether the draws represent the posterior, and which sampler argument to change when
they do not.

<div align="center">
  <img src="../plots/nuts_convergence_sigma_full.png" width="760"/>
  <p><em>The summary check for posterior samples with Arviz: Bulk effective sample size (ESS), tail ESS and R-hat, one panel
  each, over every quantity at every reference point. Dashed lines are ArviZ's reference
  thresholds, in total ESS across all chains. We want both ESS curves entirely right of
  their line and the R-hat curve entirely left of its.</em></p>
</div>

Everything here runs on the fit from the
[sampling tutorial](nuts_inference_example.md), from the same script,
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).
Rebuilding every diagnostic and figure from the cached draws needs no GPU and can quickly be run like this:

```bash
python nuts_inference_example.py --mode full --from-fit
```

---

## What to check

4 quantities to check, in this order, at ArviZ's default thresholds. Those defaults
follow [Vehtari et al. (2021)](https://doi.org/10.1214/20-BA1221) for R-hat and ESS.

| check | what it detects | suspect when |
|---|---|---|
| divergences, `sampler_stats["diverging"]` | the integrator failed, usually in a region of high curvature | any divergence at all |
| [R-hat](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html) | chains have not mixed to the same distribution | $\hat{R} > 1.01$ |
| [ESS](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html) | autocorrelation: how many independent draws the chain is worth | bulk or tail ESS $< 100$ per chain, so $< 400$ in total across 4 chains |
| `sampler_stats["n_steps"]` | trajectories truncated at the tree-depth cap | at the cap for most draws |

Divergences and the tree-depth cap come straight off `sampler_stats`, with no ArviZ needed:

```python title="Divergences and tree-depth saturation"
--8<-- "docs/examples/wppm/nuts_inference_example.py:divergence_check"
```

A divergent transition means the integrator left the posterior's geometry, so more sampling
does not fix the draws it produced. Any non-zero count makes the run suspect regardless of
what R-hat says, which is why we check divergences first. The second print is the fraction
of draws whose trajectory hit `max_num_doublings`: a large fraction means the cap is
limiting mixing.

## Diagnostics on $\Sigma$, with `covariance_to_arviz`

Diagnostics run on $\Sigma$, not on the raw parameters, for the reason we discussed in the
[sampling tutorial](nuts_inference_example.md#step-4-nuts-sampling). One call hands the fit
to ArviZ:

```python title="R-hat and ESS on the covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

[`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz) takes the
posterior and the points to evaluate $\Sigma$ at, here the reference points. From there
`az.summary`, `az.rhat`, `az.ess` and every ArviZ plotting function work on a psyphy fit
directly, which is where the two figures below come from.

??? note "What ArviZ wants, and what it gets"

    `covariance_to_arviz` returns an `xarray.DataTree` with dimensions
    `(chain, draw, point, quantity)`, and carries the sampler statistics across as ArviZ's
    `sample_stats` group. ArviZ's
    [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html)
    covers that format if you want the detail.

    Note it takes the posterior itself, not draws from it. `posterior.sample(n)` *pools* the
    chains: it flattens `(n_chains, n_draws, ...)` into `(n_chains * n_draws, ...)` and picks
    `n` at random, so a draw no longer carries the chain it came from.

    R-hat asks whether the chains agree, by comparing the spread between them to the spread
    within each one. Pooling blends the chains evenly into one collection, so any way of
    splitting it afterwards gives pieces that look alike. ArviZ returns `nan` for a single
    chain, and if pooled draws are reshaped into chains it returns about 1.00 no matter how
    badly the real chains disagreed. ESS behaves the same way, since shuffling removes the
    autocorrelation it measures.

## Reading the figures

Both figures below are one ArviZ call each, `az.plot_trace` and
`az.plot_convergence_dist`, on the object `covariance_to_arviz` returned. We write no
plotting code.

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

We see one reference point here. The figure at the top covers all 25 at once.

```python title="ESS and R-hat over all quantities"
--8<-- "docs/examples/wppm/nuts_inference_example.py:convergence_plot"
```

Each curve there is the distribution of one diagnostic over every $\Sigma$ quantity at
every reference point, so a point on it reads as "this fraction of quantities scored at most
this value". More is better for ESS and less is better for R-hat. For R-hat: where its
curve crosses the dashed line, that fraction of quantities sits above 1.01.

Useful: `az.summary` prints `R-hat max` and `ESS min`.

---

## What to change when a check fails

A failing check is fixed by the sampler arguments, which
[`blackjax.nuts`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html)
and
[`window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html)
document. psyphy's
[`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler) passes them
straight through under the same names.

| argument | what it controls | raise it when |
|---|---|---|
| `num_warmup` | tuning steps per chain, discarded | trace plots still drift at the start |
| `num_samples` | kept draws per chain | ESS is too low |
| `num_chains` | independent chains, R-hat needs at least 2 | a stronger convergence check is wanted |
| `max_num_doublings` | tree-depth cap, at most $2^k - 1$ gradients per draw | most draws hit the cap, which the script prints |
| `target_acceptance_rate` | how cautious the tuned step size is | divergences appear |

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

    Two things in the adaptive path are psyphy's own.
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

The paper behind the thresholds:

> Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).
> *Rank-normalization, folding, and localization: an improved $\widehat{R}$ for assessing
> convergence of MCMC.* Bayesian Analysis, 16(2).
> [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
