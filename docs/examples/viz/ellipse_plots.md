# Plotting covariance ellipse fields

A 2×2 covariance drawn as an ellipse is how psyphy shows most of what it
computes: a model's internal noise field, a threshold contour, a posterior
draw. `psyphy.viz.plot_ellipses` draws any of those, one field or several.

```bash
pip install 'psyphy[viz]'     # matplotlib is an optional dependency
```

```python title="Import"
--8<-- "docs/examples/viz/ellipse_plots.py:imports"
```

Runnable script:
[`ellipse_plots.py`](https://github.com/flatironinstitute/psyphy/blob/main/docs/examples/viz/ellipse_plots.py).
Everything below is synthetic and runs in about a second.

```python title="The example field"
--8<-- "docs/examples/viz/ellipse_plots.py:field"
```

---

## One field

Pass centers and one `(n_points, 2, 2)` stack. The default `scale=1.0` draws
**true size**: the ellipse semi-axes are the square roots of the covariance
eigenvalues, in the units of your stimulus space.

```python
--8<-- "docs/examples/viz/ellipse_plots.py:single"
```

<div align="center">
    <img src="../plots/viz_single_field.png" alt="One covariance field at true size" width="420"/>
</div>

Nothing is saved or shown for you — the function draws into an `Axes` and
returns it, so the figure stays yours to title, style, and save.

---

## Several fields overlaid

Pass a **list** of fields to compare them. This is the "published vs. fitted"
figure that shows up throughout the WPPM examples.

```python
--8<-- "docs/examples/viz/ellipse_plots.py:overlay"
```

<div align="center">
    <img src="../plots/viz_overlay.png" alt="Two covariance fields overlaid" width="420"/>
</div>

`colors`, `linestyles`, `linewidths`, `alpha`, and `labels` each accept either
one value for all fields or a list with one entry per field. A legend appears
only if you pass `labels`.

!!! note "`scale='auto'` sizes ellipses to the grid — and shares one factor"
    True size is honest but can be hard to read: on a dense grid the ellipses
    shrink to specks, and on a coarse one they float in white space.
    `scale="auto"` sets the *median* ellipse to a fixed fraction of the median
    spacing between centers, so the same call works on a 7×7 grid and a
    103×103 one.

    Two things follow. **It is not always a magnification** — on a dense grid
    the factor drops below 1, shrinking ellipses so they do not overlap. And
    when several fields are drawn, the factor is computed from the **first**
    one and applied to all of them; independently scaled fields could not be
    compared by eye. Pass `return_scale=True` to get the number back and put it
    in your title, since a rescaled figure cannot be read for absolute size.

---

## Posterior draws

A `(n_samples, n_points, 2, 2)` array draws one translucent field per sample —
the natural way to show parameter uncertainty.

```python
--8<-- "docs/examples/viz/ellipse_plots.py:samples"
```

<div align="center">
    <img src="../plots/viz_samples.png" alt="Twelve posterior draws of one field" width="420"/>
</div>

This is the shape `WPPMPredictivePosterior.rsample()` returns. With a
`MAPPosterior` every draw is identical (a point estimate has no spread), so the
band collapses to a single field — but the same call gives real error bars once
a sampling posterior is available.

---

## Per-ellipse colors

`colors` also accepts an `(n_points, 3)` or `(n_points, 4)` array, giving each
ellipse its own color. That is how the
[Hong et al. reproduction](../wppm/hong2025_reproduction.md) colors each
contour by its own reference stimulus:

```python
colors = hong2025.w2d_to_rgb(coords, M)          # (n_points, 3)
plot_ellipses(coords, thresholds, colors=colors)
```

---

## Non-positive-definite covariances

A covariance that is not positive-definite has no ellipse. By default those are
skipped and a warning names how many were dropped:

```
UserWarning: 3 non-positive-definite covariance(s) were not drawn;
the plotted field is incomplete.
```

Silence is the wrong default here — a field quietly missing a few ellipses is
hard to spot by eye, and it usually means something upstream went wrong, such
as an ill-conditioned threshold inversion. Pass `skip_non_pd=False` to raise
instead.

---

## Without matplotlib

The geometry is separate from the drawing, in `psyphy.viz.geometry`, and needs
only numpy and scipy. Use it to feed another plotting backend, or to test
without a display:

```python
from psyphy.viz import ellipse_segments

segments, valid = ellipse_segments(centers, covs, scale=1.0)
# segments: list of (n_points, 2) arrays, one per positive-definite covariance
# valid:    bool mask of which covariances were drawable
```

---

## Limits

- **2-D only.** Higher-dimensional covariances raise `NotImplementedError`;
  drawing them would mean ellipsoid rendering.
- **`scale="auto"` needs at least two centers**, since it measures spacing
  between them. With one point, pass an explicit scale.

---

## See also

- API reference: `psyphy.viz` in [Visualization](../../reference/viz.md).
- [Reproducing Hong et al. (2025)](../wppm/hong2025_reproduction.md) — these
  functions on real published data.
