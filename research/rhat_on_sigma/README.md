# Posterior mean and R-hat for the WPPM: W versus Σ

Exploratory analysis, not part of the psyphy package or its documentation.

## Question

The WPPM covariance field is Σ(x) = U(x) U(x)ᵀ + λI, with U linear in the
parameters W. Two consequences for summarising an MCMC posterior over W:

1. **Posterior mean.** Σ is quadratic in W, so the field built from the mean
   parameter, Σ(E[W]), is not the posterior mean of Σ. In general
   E[Σ] = Σ(E[W]) + Cov[U], so Σ(E[W]) is never larger than E[Σ] in any direction.
2. **Identifiability and R-hat.** W is only identified up to an orthogonal
   rotation R of its embedding axis: W → W R leaves Σ and the prior unchanged.
   Chains can settle at different orientations that describe the same Σ. Two
   things follow: averaging W partly cancels, which shrinks Σ(E[W]) further, and
   R-hat computed on W reports chain disagreement that has no effect on any Σ.

Here we measure both effects on the NUTS example
(`docs/examples/wppm/nuts_inference_example.py`). R-hat is computed on W and on
identifiable per-reference-point quantities of Σ: log Σ₁₁, log Σ₂₂, and the
correlation Σ₁₂/√(Σ₁₁Σ₂₂).

## Results

One run per mode on an NVIDIA A100.
| | quick | balanced |
|---|---|---|
| chains × draws | 2 × 150 | 4 × 500 |
| sampling time | 1 min | 36 min |
| size of Σ(E[W]) ÷ median posterior draw | 0.24 | 0.23 |
| size of E[Σ] ÷ median posterior draw | 1.20 | 1.05 |
| size of median posterior draw ÷ ground truth | 1.43 | 1.11 |
| R-hat on W: median / max / share < 1.01 | 1.43 / 2.40 / 0% | 1.12 / 2.03 / 0% |
| R-hat on Σ quantities: median / max / share < 1.01 | 1.14 / 1.43 / 0% | 1.01 / 1.08 / 35% |
| draws at the tree-depth cap (255 leapfrog steps) | 100% | 100% |

"Size" is √det Σ computed on the raw matrices, one value per reference point,
and the ratio is averaged over reference points. The plotting magnification
plays no part in it.

Reading the table:

- **Σ(E[W]) is about a quarter of a typical posterior draw in both modes.** More
  data and longer chains do not fix it. E[Σ] sits among the draws, slightly above
  the median, as expected for the mean of covariance matrices.
- **R-hat on W stays high even where Σ has nearly converged.** In balanced, the
  worst W coefficient has R-hat 2.03, with four chains holding separate levels for
  500 draws, while the worst Σ quantity has R-hat 1.08 with the chains fully
  overlapping (`figures/trace_balanced.png`).
- **Σ is close to converged in balanced but not strictly.** Its median is 1.01,
  and about two thirds of the quantities are above 1.01.
- **Every draw hits the tree-depth cap in both modes.** This fits trajectories
  that glide along the ridge of rotation-equivalent Ws without turning back.

## Figures

All figures are drawn from the saved data in `results/`.

| file | content | tool |
|---|---|---|
| `figures/mean_{mode}.jpg` | ground truth, E[Σ], Σ(E[W]) and 40 posterior draws; every ellipse at the same ×1.2 | `psyphy.viz.plot_ellipses` |
| `figures/trace_{mode}.png` | traces of the W coefficient and the Σ quantity with the highest R-hat, one colour per chain | `arviz.plot_trace` |
| `figures/rhat_{mode}.png` | cumulative distribution of R-hat over all W coefficients and over all Σ quantities, same x-range | `arviz.plot_convergence_dist` |
| `figures/overview.jpg` | the three panels side by side, one row per mode | stitched |

## Reproducing

From the repo root:

```bash
python research/rhat_on_sigma/collect.py quick      # ~1 min on an A100
python research/rhat_on_sigma/collect.py balanced   # ~36 min on an A100
python research/rhat_on_sigma/figures.py            # seconds, no sampling
```

- `collect.py` runs the NUTS example from a temporary copy, so the committed
  tutorial figures are not overwritten. It saves per-draw Σ, both means and the
  R-hat values to `results/{mode}.npz`.
- If another JAX process is using the GPU, prefix the commands with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- `results/` (about 2 MB) is not in git: `research/**/results/` is listed in the
  repo's `.gitignore`. Run `collect.py` first to recreate it; `figures.py`
  needs it.

## Open questions

- Does `--mode full` bring all Σ quantities below 1.01, and at what cost given
  that every draw already hits the depth cap?
- Would removing the rotation symmetry from the parameterization, for example
  by fixing the embedding orientation through a triangular constraint on part
  of U, cut the trajectory length and make W itself identifiable?
