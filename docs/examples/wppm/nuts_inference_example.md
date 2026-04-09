# NUTS Posterior Inference for WPPM

This tutorial demonstrates **full posterior inference** using the No-U-Turn Sampler (NUTS) for the Wishart Process Psychophysical Model (WPPM).

The companion script can be found at
[`nuts_inference_example.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/wppm/nuts_inference_example.py).

---

## Why MCMC instead of MAP?

MAP estimation returns a single point estimate $\theta_\text{MAP}$:

$$\theta_\text{MAP} = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

This tells you *where the posterior peaks*, but not *how wide* it is. If the data are sparse or ambiguous, the posterior may be broad — and the MAP estimate alone cannot tell you that.

MCMC (and specifically NUTS) draws samples from the **full posterior** $p(\theta \mid \mathcal{D})$. This enables:

- **Uncertainty quantification**: how spread out is our belief over model parameters?
- **Posterior predictive distributions**: how uncertain are our predictions at new stimuli?
- **Convergence diagnostics**: R-hat and ESS tell you whether the chains have mixed.

---

## The posterior predictive distribution

Given posterior samples $\theta_i \sim p(\theta \mid \mathcal{D})$, predictions at new test stimuli $X^*$ are:

$$p(\text{correct} \mid X^*, \mathcal{D}) = \int p(\text{correct} \mid X^*, \theta)\; p(\theta \mid \mathcal{D})\; d\theta \approx \frac{1}{N} \sum_{i=1}^{N} \underbrace{p(\text{correct} \mid X^*, \theta_i)}_{\texttt{OddityTask.predict}(\theta_i,\, X^*)}$$

In psyphy this is computed by `WPPMPredictivePosterior`, which calls `nuts_posterior.sample(N)` to get $N$ parameter draws and vmaps the task likelihood over them. The same class works identically with a `MAPPosterior` (which just returns $N$ copies of $\theta_\text{MAP}$).

---

## A note on the stochastic likelihood

WPPM's likelihood is estimated via Monte Carlo inside `OddityTask.loglik`. NUTS requires a (nearly) deterministic log-density to compute consistent Hamiltonian dynamics.

**Chosen approach:** a single fixed JAX key is baked into the `logdensity_fn` closure at construction time, making the MC noise realization constant throughout sampling:

```python
fixed_mc_key = jax.random.PRNGKey(logdensity_key_seed)
logdensity_fn = lambda params: model.log_posterior_from_data(params, data, key=fixed_mc_key)
```

This means NUTS is sampling from a slightly perturbed posterior. The perturbation decreases as `MC_SAMPLES → ∞`. We recommend `MC_SAMPLES ≥ 200`.

**Alternatives** (not implemented here but worth knowing):
- *Pseudo-marginal MCMC*: fresh key per HMC *trajectory* (not per leapfrog step) — theoretically correct noisy HMC.
- *Neural surrogate likelihood*: replace the MC estimator with a trained neural network for exact gradients (see `NeuralSurrogateTask`).

---

## Step-by-step walkthrough

### Step 0 — Install requirements

```bash
pip install 'psyphy[nuts]'
# installs: blackjax>=1.0, arviz>=0.16, matplotlib>=3.5
```

### Step 1 — Simulate data

Same setup as the full MAP example. Data is a `TrialData` object:

```python title="Simulated data"
--8<-- "docs/examples/wppm/nuts_inference_example.py:data"
```

### Step 2 — Build the model

```python title="Model definition"
--8<-- "docs/examples/wppm/nuts_inference_example.py:build_model"
```

### Step 3 — MAP warm-start (recommended)

Starting NUTS near the posterior mode dramatically reduces the warmup needed. Run a short MAP fit first:

```python title="MAP warm-start"
--8<-- "docs/examples/wppm/nuts_inference_example.py:map_warmstart"
```

Then pass `map_posterior.params` as `init_params` to `NUTSSampler.fit()`.

### Step 4 — NUTS sampling

```python title="NUTS sampling with NUTSSampler"
--8<-- "docs/examples/wppm/nuts_inference_example.py:nuts_fit"
```

`NUTSSampler` has the same `fit(model, data)` interface as `MAPOptimizer` — the BlackJAX backend is entirely hidden. It returns an `MCMCPosterior` satisfying the same `ParameterPosterior` protocol as `MAPPosterior`.

**Compute settings used in this example:**

```python
--8<-- "docs/examples/wppm/nuts_inference_example.py:compute_settings"
```

**Runtime notes:**
- The first chain triggers JIT compilation (~30 s on CPU).
- Subsequent chains reuse the compiled kernel.
- GPU (A100): ~2 min total. CPU (M-series): ~15–30 min.

#### Two warmup modes

| Mode | When | How |
|------|------|-----|
| Adaptive (default, `step_size=None`) | Best quality; adapts step size and mass matrix per chain via `blackjax.window_adaptation`. Chains run sequentially. | `NUTSSampler(num_warmup=200, num_samples=500)` |
| Fixed (`step_size=<float>`) | Faster with many chains; no adaptation. All chains vmapped in one compiled call. User must choose a good step size. | `NUTSSampler(..., step_size=0.01)` |

### Step 5 — ArviZ diagnostics

```python title="ArviZ diagnostics"
--8<-- "docs/examples/wppm/nuts_inference_example.py:arviz_diagnostics"
```

```python title="Trace plot"
--8<-- "docs/examples/wppm/nuts_inference_example.py:trace_plot"
```

<div align="center">
  <img src="../plots/nuts_trace_plot.png" width="700"/>
  <p><em>Trace plots for NUTS chains. Well-mixed chains produce caterpillar-shaped traces.</em></p>
</div>

#### Interpreting diagnostics

| Diagnostic | What it measures | Good value |
|---|---|---|
| **R-hat** | Ratio of between-chain to within-chain variance. Values near 1 indicate convergence. | < 1.01 |
| **ESS (bulk)** | Effective sample size — how many independent draws the chains are equivalent to. | > 100 per chain |
| **Acceptance rate** | Fraction of NUTS proposals accepted. Too low → too large step_size. Too high → too small. | 0.6–0.9 |

With only `NUM_CHAINS=2` and `NUM_SAMPLES=300` (the demo settings), R-hat and ESS may not reach textbook thresholds. Increase both for production use.

### Step 6 — Posterior predictive uncertainty

```python title="Draw posterior parameter samples"
--8<-- "docs/examples/wppm/nuts_inference_example.py:posterior_samples"
```

```python title="Plot ensemble of ellipses"
--8<-- "docs/examples/wppm/nuts_inference_example.py:plot_uncertainty"
```

<div align="center">
  <img src="../plots/nuts_ellipses_uncertainty.png" width="600"/>
  <p><em>Faint blue: 30 covariance fields sampled from the posterior. Red: posterior mean. Black: ground truth. The spread of blue ellipses shows where the model is uncertain.</em></p>
</div>

### Step 7 — MAP vs NUTS comparison

<div align="center">
  <img src="../plots/nuts_vs_map_comparison.png" width="900"/>
  <p><em>Ground truth (left), MAP point estimate (center), NUTS posterior mean (right). The NUTS mean is the average over many samples — it may differ slightly from MAP due to posterior asymmetry.</em></p>
</div>

---

## Architecture recap

```
NUTSSampler.fit(model, data)        ← same interface as MAPOptimizer
        ↓  returns
MCMCPosterior                        ← satisfies ParameterPosterior protocol
  .params        → posterior mean W
  .sample(n,key) → n draws {"W": (n, *W_shape)}
  .to_arviz()    → az.InferenceData (n_chains, n_draws, *W_shape)
        ↓
WPPMPredictivePosterior(nuts_posterior, X_test, ...)
  .mean     → E[p(correct | X*, θ) | D]
  .variance → Var[p(correct | X*, θ) | D]
```

`WPPMPredictivePosterior` is permanently agnostic about whether it wraps a `MAPPosterior` or `MCMCPosterior` — the `ParameterPosterior` protocol is the firewall.

---

## Limitations and future extensions

- **Fixed MC key bias**: as noted above, using a fixed key introduces a slight bias. Mitigate with larger `MC_SAMPLES` or switch to the neural surrogate likelihood.
- **Sequential warmup**: adaptive mode runs one chain at a time. For >8 chains, fixed `step_size` mode + vmap is faster.
- **Other samplers**: any sampler that produces an `MCMCPosterior` plugs into the same downstream pipeline. Future additions: `MALASampler`, `SGLDSampler`, NumPyro integration.

---

## Further reading

- API reference: [`NUTSSampler`](../../reference/inference.md), [`MCMCPosterior`](../../reference/posterior.md)
- MAP fitting tutorial: [`full_wppm_fit_example`](full_wppm_fit_example.md)
- Model details: [`wppm_tutorial`](wppm_tutorial.md)
- BlackJAX documentation: [blackjax-devs.github.io/blackjax](https://blackjax-devs.github.io/blackjax/)
- ArviZ documentation: [python.arviz.org](https://python.arviz.org/)
