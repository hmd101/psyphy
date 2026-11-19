r"""psyphy.model.likelihood.features

Feature maps for the fast likelihood: the few numbers per trial that a trained
network (also called a surrogate) takes as input.

For a task whose decision rule compares Mahalanobis distances, ``P(correct)``
depends on the two covariances and the two stimulus locations only through a few
numbers. A 2-D trial has 8: 3 per symmetric covariance and 2 for the difference
between the locations. ``P(correct)`` does not change under an invertible affine
change of stimulus coordinates, which removes 4 of them, so a 2-D network takes
4 features. In general the count is ``2 * input_dim``.

We reduce a trial in 2 steps:

``normalize_by_noise``
    Change coordinates so the pooled covariance becomes the identity, which
    expresses the trial in noise units. Works in any dimension and for any task.
    The task enters only through ``pooling_weight``. Removes ``d(d+1)/2`` numbers.
``align_displacement_2d``
    Rotate so the mean difference lies along an axis, which leaves only its
    length. 2-D only. A 3-D version would give 6 features.

These are free functions over arrays, like :mod:`psyphy.viz.geometry`, and need
no model or trained network. To build a feature map for a new task, start from
:func:`normalize_by_noise` with that task's ``pooling_weight``. A weight file
names the feature map it expects in its ``feature_map`` metadata.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

__all__ = ["normalize_by_noise", "align_displacement_2d", "oddity_features"]

#: Pooling weight for the 3-AFC oddity task: the reference is presented twice
#: and the comparison once, so the pooled covariance is weighted 2:1.
ODDITY_POOLING_WEIGHT = 2.0 / 3.0


def normalize_by_noise(
    cov_ref: jnp.ndarray,
    cov_comparison: jnp.ndarray,
    ref: jnp.ndarray,
    comparison: jnp.ndarray,
    *,
    pooling_weight: float = ODDITY_POOLING_WEIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Express a trial in noise units, using its pooled covariance.

    The decision rule measures distance in the pooled covariance
    ``S_bar = pooling_weight * cov_ref + (1 - pooling_weight) * cov_comparison``,
    so every distance carries an ``S_bar^-1``. Changing to coordinates where ``S_bar`` is the
    identity turns those Mahalanobis distances into ordinary Euclidean ones.
    Works in any dimension. This step is also called whitening.

    The change of basis is ``M = inv(chol(S_bar))``, which we apply as triangular
    solves without forming ``M``.

    In the new coordinates the comparison covariance is no longer free. Write
    ``A`` and ``B`` for the 2 covariances there. The constraint
    ``pooling_weight * A + (1 - pooling_weight) * B = I`` gives
    ``B = (I - pooling_weight * A) / (1 - pooling_weight)``, which is ``3I - 2A``
    for the oddity task. That is where ``d(d+1)/2`` of the free numbers go, and it
    also bounds the eigenvalues of ``A`` below ``1 / pooling_weight``.

    Parameters
    ----------
    cov_ref, cov_comparison : jnp.ndarray, shape (d, d)
        Symmetric positive-definite covariances at the two stimulus locations.
    ref, comparison : jnp.ndarray, shape (d,)
        The stimulus locations themselves.
    pooling_weight : float
        Weight of the reference covariance in the pooled covariance. 2/3 for the
        3-AFC oddity task, 1/2 for a two-interval task.

    Returns
    -------
    cov_ref : jnp.ndarray, shape (d, d)
        The reference covariance in noise units.
    displacement : jnp.ndarray, shape (d,)
        The reference minus the comparison, in noise units.
    """
    Sbar = pooling_weight * cov_ref + (1.0 - pooling_weight) * cov_comparison
    chol = jnp.linalg.cholesky(Sbar)

    # M cov_ref M^T with M = chol^-1, via two triangular solves.
    # solve_triangular(chol, X) is chol^-1 X; applying it to the transpose of the
    # first result gives chol^-1 cov_ref chol^-T. cov_ref is symmetric, so no
    # further transpose is needed.
    tmp = solve_triangular(chol, cov_ref, lower=True)
    cov_ref_normalized = solve_triangular(chol, tmp.T, lower=True)

    displacement = solve_triangular(chol, ref - comparison, lower=True)
    return cov_ref_normalized, displacement


def align_displacement_2d(
    cov_ref: jnp.ndarray, displacement: jnp.ndarray
) -> jnp.ndarray:
    r"""Rotate a 2-D trial in noise units so the displacement lies along +y.

    In noise units, rotations are the only coordinate changes left that keep the
    pooled covariance the identity. We use one to put the displacement on the
    +y axis, so only its length remains. That gives the 4 features of a 2-D trial.

    Parameters
    ----------
    cov_ref : jnp.ndarray, shape (2, 2)
        Reference covariance in noise units, from :func:`normalize_by_noise`.
    displacement : jnp.ndarray, shape (2,)
        Displacement in noise units, from :func:`normalize_by_noise`.

    Returns
    -------
    jnp.ndarray, shape (4,)
        ``[||displacement||, a11, a12, a22]`` with ``cov_ref`` expressed in the
        rotated frame.

    Notes
    -----
    When the two stimuli coincide the rotation is undefined, and we use the
    identity. ``||displacement||`` is then 0, and the orientation of ``cov_ref``
    does not change ``P(correct)``.
    """
    sq = jnp.sum(displacement**2)
    degenerate = sq < 1e-30
    # The inner where keeps sqrt away from 0, so its gradient stays finite.
    norm = jnp.where(degenerate, 0.0, jnp.sqrt(jnp.where(degenerate, 1.0, sq)))
    safe = jnp.where(degenerate, 1.0, norm)

    # Rotation taking displacement onto +y; the identity when displacement is zero.
    cos_t = jnp.where(degenerate, 1.0, displacement[1] / safe)
    sin_t = jnp.where(degenerate, 0.0, -displacement[0] / safe)

    a11, a12, a22 = cov_ref[0, 0], cov_ref[0, 1], cov_ref[1, 1]
    c2, s2, cs = cos_t**2, sin_t**2, cos_t * sin_t
    r11 = c2 * a11 + 2.0 * cs * a12 + s2 * a22
    r12 = -cs * a11 + (c2 - s2) * a12 + cs * a22
    r22 = s2 * a11 - 2.0 * cs * a12 + c2 * a22

    return jnp.stack([norm, r11, r12, r22])


def oddity_features(
    cov_ref: jnp.ndarray,
    cov_comparison: jnp.ndarray,
    ref: jnp.ndarray,
    comparison: jnp.ndarray,
) -> jnp.ndarray:
    r"""The 4 features a 2-D network reads.

    We apply :func:`normalize_by_noise`, then :func:`align_displacement_2d`.

    One trial, not a batch. ``BernoulliTaskLikelihood.loglik`` vmaps over trials.

    Parameters
    ----------
    cov_ref, cov_comparison : jnp.ndarray, shape (2, 2)
        Covariances at the reference and at the comparison.
    ref, comparison : jnp.ndarray, shape (2,)
        The reference and comparison stimuli.

    Returns
    -------
    jnp.ndarray, shape (4,)
        ``[||displacement||, a11, a12, a22]``.

    Raises
    ------
    ValueError
        If the stimuli are not 2-D.

    Notes
    -----
    The pooling weight is the oddity task's 2/3, the one the shipped network was
    trained with. For another weight, call :func:`normalize_by_noise` directly.
    """
    d = jnp.shape(ref)[-1]
    if d != 2:
        raise ValueError(f"oddity_features supports input_dim=2, got input_dim={d}.")
    cov_ref_normalized, displacement = normalize_by_noise(
        cov_ref,
        cov_comparison,
        ref,
        comparison,
        pooling_weight=ODDITY_POOLING_WEIGHT,
    )
    return align_displacement_2d(cov_ref_normalized, displacement)
