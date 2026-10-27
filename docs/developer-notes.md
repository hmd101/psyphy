# Developer notes

Decisions in psyphy that are not obvious from the signature: what we were trying to achieve,
what we chose, and potentially at what it costs. Written for someone reading or extending the source (not
for someone using the library). The tutorials under
[Examples](examples/index.md) show how to *use* these pieces.


---

## The log density uses one frozen PRNG key

**Symbols:** [`NUTSSampler`](reference/inference.md#psyphy.inference.NUTSSampler)
(`logdensity_key_seed`), [`OddityTask`](reference/model.md#psyphy.model.OddityTask).
**Checked updated:** 2026-10-27

**What we wanted.** Gradient-based samplers assume the log density is a fixed function of
the parameters. BlackJAX's NUTS integrates a trajectory and then accepts or rejects it by
comparing energies along the way, which only means anything if the same parameters give the
same density every time.

**What psyphy does** `OddityTask` has no closed-form likelihood. It estimates
$P(\text{correct})$ by simulating `mc_samples` noisy internal representations per trial, so
two calls with different keys give slightly different answers. `NUTSSampler` freezes one key
(`logdensity_key_seed`) for the whole run, so the sampler sees a deterministic function.

**What it costs.** The sampler explores the posterior implied by that one noise realization,
not the exact posterior. Raising `mc_samples` shrinks the difference and costs proportionally
more compute. A deterministic surrogate likelihood would remove the compromise entirely, and
that is the main argument for building one.

---

## The posterior mean field is a class (as opposed to a flag)

**Symbols:**
[`PosteriorMeanCovarianceField`](reference/model.md#psyphy.model.PosteriorMeanCovarianceField),
[`WPPMCovarianceField`](reference/model.md#psyphy.model.WPPMCovarianceField).
**Checked updated:**  2026-10-27

**What we wanted.** One covariance field summarizing a sample-based posterior, without
inviting the wrong summary.

**What psyphy does.** `PosteriorMeanCovarianceField` takes a posterior, builds $\Sigma$ from
each draw, and averages the matrices. It is a separate class rather than a `mean=True`
argument on `WPPMCovarianceField`, because the two need different inputs: one takes a
parameter set, the other takes a posterior and samples from it. A flag would make `params`
meaningful in one mode and ignored in the other.

**What it costs.** Two names to learn instead of one.
`WPPMCovarianceField.from_posterior` warns rather than raises when handed a sample-based
posterior, since $\Sigma(\mathbb{E}[W])$ is a legitimate object that somebody may want
deliberately. The class docstring has the identity showing why it is the wrong posterior
mean.

---

## Diagnostics run on $\Sigma$, and take the posterior rather than samples

**Symbols:** [`covariance_to_arviz`](reference/model.md#psyphy.model.covariance_to_arviz).
**Last updated:** 2026-10-27

**What we wanted.** Convergence diagnostics that report real failures and not artifacts of
the parameterization.

**What psyphy does.** $W$ is identified only up to an orthogonal rotation of its embedding
axis, so two chains can describe the same covariance field from different orientations.
R-hat on $W$ calls that non-convergence. `covariance_to_arviz` evaluates $\Sigma$ at points
the caller chooses and hands ArviZ that instead, in an `xarray.DataTree` with dimensions
`(chain, draw, point, quantity)`. It takes the posterior object rather than draws from it,
because `posterior.sample(n)` pools the chains and R-hat needs to know which chain a draw
came from.

**What it costs.** The caller has to choose evaluation points, which is why `points` has no
default. The diagnostics then describe $\Sigma$ at those points rather than the parameters
themselves.
