# Checking MCMC draws with ArviZ

[Sampling the posterior with the No-U-Turn Sampler (NUTS)](nuts_inference_example.md) ends
with a distribution of threshold contours. Those contours are only as good as the chains
behind them, and Markov chain Monte Carlo (MCMC) fails quietly: a chain can look like a
result while having explored one part of the posterior. Here we check whether the draws
represent the posterior, and which sampler argument to change when they do not.

<div align="center">
  <img src="../plots/nuts_convergence_sigma_full.png" width="760"/>
  <p><em>The summary check. Bulk effective sample size (ESS), tail ESS and R-hat, one panel
  each, over every quantity at every reference point. Each curve is a cumulative fraction, so
  the vertical axis runs from 0 to 1 and the horizontal axis carries the diagnostic. Dashed
  lines are ArviZ's reference thresholds. We want both ESS curves entirely right of their
  line and the R-hat curve entirely left of its.</em></p>
</div>

Everything here runs on the fit from the sampling tutorial, from the same script,
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).
Rebuilding every diagnostic and figure from the cached draws needs no GPU:

```bash
python nuts_inference_example.py --mode full --from-fit
```

??? note "Recap: what $\Sigma$ is here"

    The Wishart Process Psychophysical Model (WPPM) fits a covariance *field* $\Sigma(x)$:
    a $2\times2$ covariance matrix at every point $x$ of the stimulus space, whose contour
    is the threshold contour. Every $\Sigma$ below means this, evaluated at one reference
    point and one posterior draw unless said otherwise. The
    [sampling tutorial](nuts_inference_example.md#step-2-build-the-model) has the full
    parameterization.

---

## What to check

4 quantities to check, in this order, at ArviZ's default thresholds:

| check | what it detects | suspect when |
|---|---|---|
| divergences, `sampler_stats["diverging"]` | the integrator failed, usually in a region of high curvature | any divergence at all |
| [R-hat](https://python.arviz.org/projects/stats/en/latest/) | chains have not mixed to the same distribution | $\hat{R} > 1.01$ |
| [ESS](https://python.arviz.org/projects/stats/en/latest/) | autocorrelation: how many independent draws the chain is worth | bulk or tail ESS $< 100$ per chain |
| `sampler_stats["n_steps"]` | trajectories truncated at the tree-depth cap | at the cap for most draws |

## The bridge to ArviZ

Diagnostics run on $\Sigma$, not on the raw parameters, for the reason in the
[sampling tutorial](nuts_inference_example.md#step-4-nuts-sampling).
[`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz) is the
bridge. It evaluates $\Sigma$ at points we choose, here the reference points, and returns an
`xarray.DataTree` with dimensions `(chain, draw, point, quantity)`, carrying the sampler
statistics across as ArviZ's `sample_stats` group. That is the shape ArviZ expects, so
`az.summary`, `az.rhat`, `az.ess` and the plotting functions all take a psyphy fit directly,
with no reshaping.

It takes the posterior itself, not draws from it. `posterior.sample(n)` *pools* the chains:
it flattens `(n_chains, n_draws, ...)` into `(n_chains * n_draws, ...)` and picks `n` at
random, so a draw no longer carries the chain it came from.

That label is what R-hat works on. R-hat asks whether the chains agree, by comparing the
spread between them to the spread within each one. Pooling blends the chains evenly into one
collection, so any way of splitting it afterwards gives pieces that look alike. ArviZ returns
`nan` for a single chain, and if pooled draws are reshaped into chains it returns about 1.00
no matter how badly the real chains disagreed. ESS goes the same way, since shuffling removes
the autocorrelation it measures.

```python title="R-hat and ESS on the covariance field"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

## Reading the figures

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
are looking for the four chains to overlap, to stay level, and to vary fast from draw to
draw. A chain sitting at its own offset, or still sloping at the end of the run, has not
mixed.

```python title="ESS and R-hat over all quantities"
--8<-- "docs/examples/wppm/nuts_inference_example.py:convergence_plot"
```

The trace plot covers one point, the figure at the top of this page covers all of them at
once. Each curve is the distribution of a diagnostic over every $\Sigma$ quantity at every
reference point, so a point on the curve reads as "this fraction of quantities scored at most
this value". More is better for ESS and less is better for R-hat. A curve crossing its
threshold says what fraction of quantities fall short.

This run has no divergences, and the printed `R-hat max` and `ESS min` are the two numbers
to quote.

---

## What to change when a check fails

A failing check is fixed by the sampler arguments, which
[`blackjax.nuts`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html)
and
[`window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html)
document. psyphy's
[`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler) passes them
straight through under the same names.

??? note "Sampler arguments"

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

## Further reading

- API reference:
  [`covariance_to_arviz`](../../reference/model.md#psyphy.model.covariance_to_arviz),
  [`NUTSSampler`](../../reference/inference.md#psyphy.inference.NUTSSampler),
  [`MCMCPosterior`](../../reference/posterior.md#psyphy.posterior.MCMCPosterior)
- The fit these diagnostics run on:
  [Sampling the posterior with NUTS](nuts_inference_example.md).
- [ArviZ](https://python.arviz.org/) for the diagnostics, with its
  [getting started guide](https://python.arviz.org/en/latest/user_guide/getting_started.html)
  and the [arviz-stats API](https://python.arviz.org/projects/stats/en/latest/).
- [BlackJAX](https://blackjax-devs.github.io/blackjax/) for the sampler and its arguments.
