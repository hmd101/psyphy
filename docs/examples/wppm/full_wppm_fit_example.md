# Full WPPM fit (end-to-end) — simulated 2D data

This note explains what the example script [`full_wppm_fit_example.py`](full_wppm_fit_example.py) is doing, and where the key functions live in the `psyphy` codebase.

- **Goal:** Fit a *spatially varying covariance field* \(\Sigma(x)\) over a 2D stimulus space \(x \in [-1,1]^2\) using the **Wishart Process Psychophysical Model (WPPM)**.
- **Data:** synthetic oddity-task responses simulated from a “ground-truth” WPPM.
- **Inference:** MAP (maximum a posteriori) optimization of the WPPM parameters.

> You can treat this as a “recipe” for using WPPM in your own project: build a model, initialize parameters, get predicted response probabilities, and fit parameters.

---

## What the WPPM is in a nutshell

WPPM defines a *covariance matrix field* \(\Sigma(x)\) over stimulus space. Intuitively, \(\Sigma(x)\) describes the local noise/uncertainty ellipse around stimulus \(x\). The model represents \(\Sigma(x)\) as

\[
\Sigma(x) = U(x)U(x)^\top + \varepsilon I,
\]

where \(U(x)\) is a smooth, basis-expanded matrix-valued function and \(\varepsilon\) is a small diagonal “jitter” (`diag_term`) to avoid numerical issues.

A psychophysical task model (here: `OddityTask`) uses \(\Sigma\) to compute probability of a correct response on each trial, and `MAPOptimizer` fits WPPM parameters by maximizing

\[
\log p(\theta \mid \mathcal{D}) = \log p(\mathcal{D} \mid \theta) + \log p(\theta).
\]

For more details on how the Wishart Psychophysical Model (WPPM) works, please checkout this [tutorial](`docs/examples/wppm/wppm_tutorial.md`).
---

## Files to know (where to look in the repo)

<!--
Note on links:
MkDocs warns when a Markdown link points outside the `docs/` tree.
For source files under `src/`, we therefore link to the repository (GitHub)
instead of using relative filesystem paths.
-->

- Example script: [`docs/examples/wppm/full_wppm_fit_example.py`](full_wppm_fit_example.py)
- Prior (how weights are initialized / regularized): [`src/psyphy/model/prior.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/prior.py)
- Model definition: [`src/psyphy/model/wppm.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/wppm.py) (see `WPPM`)
- Covariance field wrapper: [`src/psyphy/model/covariance_field.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/covariance_field.py) (see `WPPMCovarianceField`)
- Task / likelihood: [`src/psyphy/model/task.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/task.py) (see `OddityTask`)
- Noise model: [`src/psyphy/model/noise.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/noise.py) (see `GaussianNoise`)
- MAP fitting: [`src/psyphy/inference/map_optimizer.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/inference/map_optimizer.py) (see `MAPOptimizer`)
- Data container: [`src/psyphy/data/dataset.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/data/dataset.py) (see `ResponseData`)

If you want to “follow the call graph”:

1. `WPPM.init_params(...)` (defined in [`src/psyphy/model/wppm.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/wppm.py)) → delegates to the prior’s `Prior.sample_params(...)` (defined in [`src/psyphy/model/prior.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/prior.py)).
2. `OddityTask.predict_with_kwargs(...)` / `OddityTask.loglik(...)` (defined in [`src/psyphy/model/task.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/task.py)) → calls into the model to get \(\Sigma(x)\) and then runs the task’s decision rule (Monte Carlo in the full model).
3. `WPPMCovarianceField(model, params)` (defined in [`src/psyphy/model/covariance_field.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/model/covariance_field.py)) → provides a callable `field(x)` that returns \(\Sigma(x)\) for single points or batches.
4. `MAPOptimizer.fit(...)` (defined in [`src/psyphy/inference/map_optimizer.py`](https://github.com/flatironinstitute/psyphy/blob/main/src/psyphy/inference/map_optimizer.py)) → runs gradient-based optimization of the negative log posterior.

---

## Step 0 — Imports and setup

In `full_wppm_fit_example.py`, the important imports are:

- `ResponseData` (trial container)
- `MAPOptimizer` (fitter)
- `WPPMCovarianceField` (fast batched \(\Sigma\) evaluation)
- `GaussianNoise`, `Prior`, `OddityTask`, `WPPM`




---

## Step 1 — Define the prior (how weights are distributed initially)

The WPPM parameters are basis weights stored as a dict, typically

- `params = {"W": W}`

where `W` is a tensor of Chebyshev-basis coefficients.

### Prior distribution over weights

See `src/psyphy/model/prior.py`:

- `Prior.sample_params(key)` samples weights `W` from a **zero-mean Gaussian** with a *degree-dependent variance*.

For 2D, the weight tensor shape is

\[
W \in \mathbb{R}^{(d+1) \times (d+1) \times D \times E},
\]

where:

- \(d\) = `basis_degree`
- \(D\) = `input_dim` (here 2)
- \(E\) = `embedding_dim = input_dim + extra_embedding_dims`

The prior variance decays with basis “total degree”. In code:

- `Prior._compute_basis_degree_grid()` constructs degrees \(i+j\) (2D) or \(i+j+k\) (3D).
- `Prior._compute_W_prior_variances()` returns

\[
\sigma^2_{ij} = \texttt{variance_scale} \cdot (\texttt{decay_rate})^{(i+j)}.
\]

- `Prior.sample_params(...)` then samples

\[
W_{ijde} \sim \mathcal{N}(0, \sigma^2_{ij}).
\]

This is the “mathematical start” of WPPM: **before any data**, WPPM draws smooth random fields because high-frequency coefficients are shrunk by the decay.

### Corresponding code block in the example

Look for the section where the script constructs a prior:

- Ground-truth prior:
  - `truth_prior = Prior(...)`
- Fit model prior:
  - `prior = Prior(...)`

and where it draws initial parameters:

- `truth_params = truth_model.init_params(jax.random.PRNGKey(...))`
- `init_params = model.init_params(jax.random.PRNGKey(...))`

```python title="Ground-truth model + prior sample"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:truth_model"
```

---

## Step 2 — Build the model (WPPM + task + noise)

In the example script, the model is created like:

- `task = OddityTask()`
- `noise = GaussianNoise(sigma=0.1)`
- `model = WPPM(input_dim=..., prior=prior, task=task, noise=noise, diag_term=...)`

### What the constructor arguments mean

- **`input_dim`**: dimensionality of the stimulus space.
- **`prior`**: controls initialization *and* the regularization term \(\log p(\theta)\).
- **`task`**: defines likelihood \(p(y\mid x,\theta)\). Here it is an oddity decision rule.
- **`noise`**: defines additional noise assumptions used by the task.
- **`diag_term`**: adds \(\varepsilon I\) to keep \(\Sigma\) positive definite.

---

## Step 3 — Evaluate the covariance field $\Sigma(x)$

The example uses a convenience wrapper:

```python title="Covariance field evaluation (Σ(x))"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:simulate_data"
```

### What `field(x)` does

At a high level:

- Input: `x` with shape `(D,)` or `(..., D)`.
- Output: covariance matrix/matrices \(\Sigma(x)\) with shape `(..., D, D)`.

Mathematically:

1. Compute a basis feature vector \(\phi(x)\) (Chebyshev basis products).
2. Form a matrix

\[
U(x) = \sum_{i,j} W_{ij}\, \phi_{ij}(x)
\]

(where indices suppressed; the actual tensor contraction is done via `einsum`).

3. Produce

\[
\Sigma(x) = U(x)U(x)^\top + \varepsilon I.
\]

In the code, the name “sqrt” is often used for \(U(x)\): it is a *square-root factor* of the covariance (up to the diagonal term).

> If you’re looking for the implementation details of the “sqrt” computation, search in `src/psyphy/model/wppm.py` for a helper named like `_compute_sqrt` (or similarly named). That’s where you’ll find the `einsum` contraction turning `W` and basis features into `U(x)`.

### Corresponding code block in the example

- Field wrapper construction:
  - `truth_field = WPPMCovarianceField(truth_model, truth_params)`
  - `init_field = WPPMCovarianceField(model, init_params)`
  - `map_field = WPPMCovarianceField(model, map_posterior.params)`

- Batched evaluation:
  - `gt_covs = truth_field(ref_points)`

---

## Step 4 — Simulate oddity-task data from a ground-truth WPPM

The script creates synthetic trials \((x_\text{ref}, x_\text{comp}, y)\):

1. Choose reference points `refs` on a grid.
2. Compute ground-truth covariances \(\Sigma_\text{truth}(x_\text{ref})\).
3. Sample probe directions on the unit circle.
4. Create a displacement using a Cholesky factor \(L\) so that probes have roughly constant **Mahalanobis radius**.

The key part is:

\[
\Delta = r\, L(x_\text{ref})\, u, \quad u \sim \text{Uniform on the unit circle},
\]

and

\[
x_\text{comp} = x_\text{ref} + \Delta.
\]

Then the oddity task uses Monte Carlo simulation to compute \(p(y=1)\), and responses are drawn

\[
y \sim \text{Bernoulli}(p_\theta(\text{correct} \mid x_\text{ref}, x_\text{comp})).
\]

### Corresponding code block in the example

Look for:

- `Sigmas_ref = truth_field(refs)`
- `L = jnp.linalg.cholesky(Sigmas_ref)`
- `deltas = MAHAL_RADIUS * einsum(...)`
- `p_correct = jax.vmap(_p_correct_one)(...)`
- `ys = jr.bernoulli(..., p_correct, ...)`
- `data.add_trial(...)`

```python title="Simulating trials (refs, comparisons, responses)"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:simulate_data"
```

---

## Step 5 — Fit with MAP optimization

The example fits parameters with:

```python title="Model definition"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:build_model"
```

```python title="Fitting with psyphy (MAPOptimizer)"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:fit_map"
```

### What is being optimized

MAP fitting finds

\[
\theta_\text{MAP} = \arg\max_{\theta} \big[\log p(\mathcal{D}\mid\theta) + \log p(\theta)\big].
\]

- \(\log p(\theta)\) is from `Prior.log_prob(params)` (see `prior.py`).
- \(\log p(\mathcal{D}\mid\theta)\) is computed by the task’s log-likelihood (here via Monte Carlo inside `OddityTask.loglik`).

The result in this example is a `MAPPosterior` object that contains a point estimate `map_posterior.params`.

### Corresponding code block in the example

- `map_optimizer = MAPOptimizer(...)`
- `map_posterior = map_optimizer.fit(...)`

---

## Step 6 — Visualize fit vs. truth vs. prior sample

The final part overlays ellipses from three covariance fields evaluated at reference points:

- “Ground Truth” (truth model + truth params)
- “Prior Sample (init)” (fit model + init prior draw)
- “Fitted (MAP)” (fit model + learned params)

The core idea is:

1. Evaluate \(\Sigma(x)\) in batch.
2. Convert each covariance to an ellipse by applying a matrix square root (for plotting), and transforming unit circle points.

For plotting performance, the script aims to build a `matplotlib.collections.LineCollection` of many ellipses at once instead of calling `ax.plot` 100s of times.

```python title="Plot: ellipses overlay (truth vs init vs MAP)"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:plot_ellipses"
```

```python title="Plot: learning curve"
--8<-- "docs/examples/wppm/full_wppm_fit_example.py:plot_learning_curve"
```

---

## Minimal recipe (copy/paste mental model)

To use WPPM on your own data, these are the essential calls:

1. **Create** task + noise + prior:
   - `task = OddityTask()`
   - `noise = GaussianNoise(sigma=...)`
   - `prior = Prior(input_dim=..., basis_degree=..., extra_embedding_dims=..., decay_rate=..., variance_scale=...)`

2. **Create** WPPM:
   - `model = WPPM(input_dim=..., prior=prior, task=task, noise=noise, diag_term=...)`

3. **Initialize** parameters:
   - `params0 = model.init_params(jax.random.PRNGKey(...))`  (draws from `Prior.sample_params`)

4. **Load/build** a dataset:
   - `data = ResponseData(); data.add_trial(ref=..., comparison=..., resp=...)`

5. **Fit**:
   - `map = MAPOptimizer(...).fit(model, data, init_params=params0, ...)`

6. **Inspect** \(\Sigma(x)\):
   - `field = WPPMCovarianceField(model, map.params)`
   - `Sigmas = field(xs)`

---

## Notes and pitfalls

- **CPU vs GPU:** this example can be heavy because the oddity likelihood uses Monte Carlo. A GPU can help a lot.
- **Positive definiteness:** `diag_term` is important. If you ever see a non-PD covariance, increase `diag_term` slightly.
- **MC variance:** optimization stability depends on `MC_SAMPLES`. Too small means noisy gradients.

---

## Next places to explore

- Read the API docs in `docs/reference/` (especially model + inference sections).
- Inspect `src/psyphy/model/prior.py` if you want to change smoothness/regularization.
- Inspect `src/psyphy/model/covariance_field.py` if you want faster / vmapped field evaluation patterns.
