# Changelog

All notable changes to `psyphy` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `psyphy` is
pre-1.0: while the version is `0.0.x`, **any release may change the public API**, so
breaking changes are called out below but do not force a minor-version bump.

## [Unreleased]

### Added

- **`psyphy.model.likelihood` is now a package** rather than a single module, with
  `base.py` (the `TaskLikelihood` hierarchy), `oddity.py` (`OddityTask`), `features.py`
  and `neural.py`. Every public name is re-exported, so imports are unchanged.
- **`NeuralOddityTask`**, a neural-network surrogate for the 3-AFC oddity likelihood. It
  describes the same observer and the same decision rule as `OddityTask`, but evaluates a
  small network fitted to that simulation instead of simulating trials. Two-dimensional
  models with Gaussian noise only; both limits raise rather than mispredicting.

  Besides being much cheaper per evaluation, it is **deterministic**, which the Monte
  Carlo likelihood is not. That matters for gradient-based samplers, which assume the log
  density is a fixed function of the parameters, and it is a prerequisite for any method
  built on second derivatives. Accuracy is not traded away: at the sample counts normally
  used, the surrogate is closer to a high-accuracy simulation than the simulation is at
  its own defaults.

      from psyphy.model.likelihood import NeuralOddityTask
      model = WPPM(input_dim=2, prior=..., likelihood=NeuralOddityTask(), noise=GaussianNoise())

  Weights ship with psyphy, so no download or configuration is needed.
- **`psyphy.model.likelihood.features`**, the input transform a surrogate consumes:
  `whiten`, `reduce_2d` and `whitened_oddity_features`. Public because deriving a feature
  map for a new task is something users should be able to do. The pooling weight `w_ref`
  is an argument rather than a constant, so the same whitening serves other tasks.
- **`NeuralSurrogate` and a self-describing weight-file format.** A surrogate file records
  which task, stimulus dimension and feature map it was trained for, and the loader
  refuses a file that does not match what the task requires.

  This exists because the failure it prevents is silent. Networks for different tasks can
  have identical shapes, so loading the wrong weights produces a plausible probability
  rather than an error. Files written before this format cannot be verified and are
  refused; `scripts/convert_reference_surrogate.py` re-saves them with the metadata
  attached.
- `TaskLikelihood.is_stochastic`, declaring whether a likelihood consumes randomness.
  `False` for `NeuralOddityTask`, so callers can stop inferring it.

## [0.0.5] — 2026-10-09

 Two new subpackages
(`psyphy.viz`, `psyphy.data.published`), a reproduction of a published figure, and two
breaking API changes in the model layer.

### Breaking

- **`TaskLikelihood.predict` signature and return type changed** (#139). Stimuli are now
  passed as one packed, slot-indexed array instead of separate arguments, and the return
  is a tuple of sufficient statistics instead of a bare scalar:

  ```python
  # before (0.0.4)
  def predict(self, params, ref, comparison, model, *, key=None) -> jnp.ndarray: ...
  #   returns p_correct

  # now (0.0.5)
  def predict(self, params, stimuli, model, *, key=None) -> tuple[jnp.ndarray, ...]: ...
  #   stimuli has shape (K, input_dim);  OddityTask reads stimuli[0], stimuli[1]
  #   returns (p_correct,) for Bernoulli tasks, (mu, sigma) for Gaussian task likelihoods
  ```

  **If you wrote a custom `TaskLikelihood`, it must be updated.** Unpack `stimuli`
  yourself and wrap the return in a tuple. Returning a bare array no longer works
  correctly: `BernoulliTaskLikelihood.loglik` indexes element 0 of the returned tuple, so
  a bare scalar silently collapses every trial onto the first trial's probability.

- **`TrialData` now stores stimuli as one array** (#136). `refs=` / `comparisons=` are
  replaced by a single `stimuli` array of shape `(N, K, d)`, with `responses` of shape
  `(N, R)`; `K` is the number of stimuli per trial, so tasks with more than two slots are
  now representable.

  ```python
  # before
  TrialData(refs=refs, comparisons=comparisons, responses=responses)
  # now
  TrialData(stimuli=jnp.stack([refs, comparisons], axis=1), responses=responses)
  ```

  A 1-D `responses` array of shape `(N,)` is still accepted and normalized to `(N, 1)`.
  Slots may optionally be named, e.g.,  `stimulus_names=("ref", "comp")` enables
  `data.stimulus("ref")` alongside positional `data.stimuli[:, 0, :]`.

  **For the oddity task, `K` is 2, not 3.** The observer is shown three stimuli, but only
  the two distinct means are stored; presenting the reference twice is encoded in
  `OddityTask`, which draws two samples from the reference distribution and one from the
  comparison. `K` counts stored stimuli, not presentations.

### Added

- **`psyphy.data.published`** loaders for published datasets, starting with
  `hong2025`: `fetch`, `fetch_calibration_matrix`, `load_calibration_matrix`,
  `load_trials`, `load_reference_W`, `load_sigma_table`, `build_paper_model`,
  `w2d_to_rgb`, `default_data_dir`. Datasets are downloaded on demand (resumable, cached
  outside the repo) rather than shipped with the package.
- **`psyphy.viz`** : `plot_ellipses`, plus `auto_scale` and `ellipse_segments`. The
  geometry is separated from the drawing layer so it is testable without a plotting
  backend, and `matplotlib` is imported lazily.
- **Threshold prediction as first-class API** ,  `WPPMPredictivePosterior(...,
  threshold_pred=True)` with a `ThresholdConfig`, recovering threshold contours by
  numerically inverting `P(correct)`. In this mode `X` is bare reference points of shape
  `(n_test, input_dim)` rather than paired stimuli.
- **1-D WPPM support** `input_dim=1` is now handled by the Chebyshev basis, the
  prior, and the covariance-field computation, alongside the existing 2-D and 3-D cases.
- **A distributional layer in the likelihood hierarchy**
  `BernoulliTaskLikelihood` and `GaussianTaskLikelihood` sit between `TaskLikelihood` and
  concrete tasks, each providing `loglik` and `simulate` so a new task only implements
  `predict`. `OddityTask` is now a `BernoulliTaskLikelihood`.
- **`reduction` on `MAPOptimizer`**  `"mean"` (new default) or `"sum"`, controlling how
  the per-trial objective is aggregated. This interacts with gradient clipping: under
  `"sum"` the gradient scales with the number of trials, so a learning rate tuned on one
  dataset size does not transfer. `"mean"` makes learning rates comparable to those
  reported by Hong et al 2025.
- **Tutorials** reproduction of Hong et al. (2025) Figure 2B (threshold contours) from the published data,
  recovery of Weber's law with a flexible WPPM, and ellipse-field plotting.
- **Tests**: `test_data_published_hong2025.py`, `test_viz.py`, `test_docs_recipe.py`
  (parses the code quoted in the docs and checks every call against the live API),
  `test_quick_start_recovery.py`, `test_map_optimizer_clipping.py`,
  `test_likelihood_logic.py`, `test_data_format.py`.

### Fixed

- **Response-shape broadcasting in the Bernoulli log-likelihood** (#140). With
  `responses` of shape `(N, 1)` and probabilities of shape `(N,)`, `jnp.where` broadcast
  to `(N, N)`, so the summed objective mixed every trial's response with every trial's
  probability and the gradient was wrong. Responses are now reduced to `(N,)` first.
- **Deprecated `jax.tree_map` replaced** with `jax.tree.map` (#71 — thanks
  @rohansood10).

### Changed

- `requires-python` is `>=3.10`; CI runs lint, type checks, and tests on 3.10, 3.11, and
  3.12.
- Documentation pages now quote code from the runnable scripts beside them via snippet
  includes, so a page and its script cannot drift apart.

## [0.0.4] — 2026-03-31

Likelihood refactor. `loglik` and `simulate` became concrete methods on
`TaskLikelihood`, so a concrete task implements only `predict`; `OddityTask` lost its own
`loglik` and the duplicated vectorised Monte Carlo path (`likelihood.py` net −200 lines).
Documentation restructuring.

## [0.0.2] — 2026-03-27

First release published to PyPI. Full initial module set: `psyphy.model` (WPPM, priors,
noise models, the oddity task), `psyphy.inference` (MAP optimizer), `psyphy.posterior`,
 `psyphy.data`,  `psyphy.utils`.

[0.0.5]: https://github.com/flatironinstitute/psyphy/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/flatironinstitute/psyphy/compare/v0.0.2...v0.0.4
[0.0.2]: https://github.com/flatironinstitute/psyphy/releases/tag/v0.0.2
