r"""psyphy.model.likelihood.features

Feature maps for neural surrogate likelihoods: the few numbers per trial that a
surrogate network takes as input.

For a task whose decision rule compares Mahalanobis distances, ``P(correct)``
depends on the two covariances and the two stimulus locations only through a few
numbers. A 2-D trial has 8: 3 per symmetric covariance and 2 for the difference
between the locations. ``P(correct)`` does not change under an invertible affine
change of stimulus coordinates, which removes 4 of them, so a 2-D surrogate takes
4 features. In general the count is ``2 * input_dim``.

We reduce a trial in 2 steps:

``whiten``
    Change coordinates so the pooled covariance becomes the identity. Works in
    any dimension and for any task. The task enters only through the pooling
    weight ``w_ref``. Removes ``d(d+1)/2`` numbers.
``reduce_2d``
    Rotate so the mean difference lies along an axis, which leaves only its
    length. 2-D only. A 3-D version would give 6 features.

These are free functions over arrays, like :mod:`psyphy.viz.geometry`, and need
no model or trained network. To build a feature map for a new task, start from
:func:`whiten` with that task's ``w_ref``. A weight file names the feature map it
expects in its ``feature_map`` metadata.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

__all__ = ["whiten", "reduce_2d", "whitened_oddity_features"]

#: Pooling weight for the 3-AFC oddity task: the reference is presented twice
#: and the comparison once, so the pooled covariance is weighted 2:1.
ODDITY_W_REF = 2.0 / 3.0


def whiten(
    Sigma_ref: jnp.ndarray,
    Sigma_comp: jnp.ndarray,
    mu_ref: jnp.ndarray,
    mu_comp: jnp.ndarray,
    *,
    w_ref: float = ODDITY_W_REF,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Whiten a trial by its pooled covariance.

    The decision rule measures distance in the pooled covariance
    ``S_bar = w_ref * Sigma_ref + (1 - w_ref) * Sigma_comp``, so every distance
    carries an ``S_bar^-1``. Changing to coordinates where ``S_bar`` is the
    identity turns those Mahalanobis distances into ordinary Euclidean ones.
    Works in any dimension.

    The change of basis is ``M = inv(chol(S_bar))``, which we apply as triangular
    solves without forming ``M``.

    In the new coordinates the comparison covariance is no longer free: the
    constraint ``w_ref * A + (1 - w_ref) * B = I`` gives
    ``B = (I - w_ref * A) / (1 - w_ref)``, which is ``3I - 2A`` for the oddity
    task. That is where ``d(d+1)/2`` of the free numbers go, and it also bounds
    the eigenvalues of ``A`` below ``1 / w_ref``.

    Parameters
    ----------
    Sigma_ref, Sigma_comp : jnp.ndarray, shape (d, d)
        Symmetric positive-definite covariances at the two stimulus locations.
    mu_ref, mu_comp : jnp.ndarray, shape (d,)
        The stimulus locations themselves.
    w_ref : float
        Weight of the reference covariance in the pooled covariance. 2/3 for the
        3-AFC oddity task, 1/2 for a two-interval task.

    Returns
    -------
    A : jnp.ndarray, shape (d, d)
        ``Sigma_ref`` in whitened coordinates.
    d_whitened : jnp.ndarray, shape (d,)
        The mean difference in whitened coordinates.
    """
    Sbar = w_ref * Sigma_ref + (1.0 - w_ref) * Sigma_comp
    chol = jnp.linalg.cholesky(Sbar)

    # A = M Sigma_ref M^T with M = chol^-1, via two triangular solves.
    # solve_triangular(chol, X) is chol^-1 X; applying it to the transpose of the
    # first result gives chol^-1 Sigma_ref chol^-T. Sigma_ref is symmetric, so no
    # further transpose is needed.
    tmp = solve_triangular(chol, Sigma_ref, lower=True)
    A = solve_triangular(chol, tmp.T, lower=True)

    d_whitened = solve_triangular(chol, mu_ref - mu_comp, lower=True)
    return A, d_whitened


def reduce_2d(A: jnp.ndarray, d_whitened: jnp.ndarray) -> jnp.ndarray:
    r"""Rotate a whitened 2-D trial so the mean difference lies along +y.

    After whitening, rotations are the only coordinate changes left that keep the
    pooled covariance the identity. We use one to put the mean difference on the
    +y axis, so only its length remains. That gives the 4 features of a 2-D trial.

    Parameters
    ----------
    A : jnp.ndarray, shape (2, 2)
        Whitened reference covariance, from :func:`whiten`.
    d_whitened : jnp.ndarray, shape (2,)
        Whitened mean difference, from :func:`whiten`.

    Returns
    -------
    jnp.ndarray, shape (4,)
        ``[||d||, a11, a12, a22]`` with ``A`` expressed in the rotated frame.

    Notes
    -----
    When the two stimuli coincide the rotation is undefined, and we use the
    identity. ``||d||`` is then 0, and the orientation of ``A`` does not change
    ``P(correct)``.
    """
    norm = jnp.linalg.norm(d_whitened)
    degenerate = norm < 1e-30
    safe = jnp.maximum(norm, 1e-30)

    # Rotation taking d_whitened onto +y; the identity when d_whitened is zero.
    cos_t = jnp.where(degenerate, 1.0, d_whitened[1] / safe)
    sin_t = jnp.where(degenerate, 0.0, -d_whitened[0] / safe)

    a11, a12, a22 = A[0, 0], A[0, 1], A[1, 1]
    c2, s2, cs = cos_t**2, sin_t**2, cos_t * sin_t
    r11 = c2 * a11 + 2.0 * cs * a12 + s2 * a22
    r12 = -cs * a11 + (c2 - s2) * a12 + cs * a22
    r22 = s2 * a11 - 2.0 * cs * a12 + c2 * a22

    return jnp.stack([norm, r11, r12, r22])


def whitened_oddity_features(
    Sigma_ref: jnp.ndarray,
    Sigma_comp: jnp.ndarray,
    mu_ref: jnp.ndarray,
    mu_comp: jnp.ndarray,
    *,
    w_ref: float = ODDITY_W_REF,
) -> jnp.ndarray:
    r"""The four features a 2-D surrogate consumes: :func:`whiten` then :func:`reduce_2d`.

    One trial, not a batch. ``BernoulliTaskLikelihood.loglik`` vmaps over trials.

    Parameters
    ----------
    Sigma_ref, Sigma_comp : jnp.ndarray, shape (2, 2)
        Covariances at the reference and at the comparison.
    mu_ref, mu_comp : jnp.ndarray, shape (2,)
        The reference and comparison stimuli.
    w_ref : float
        See :func:`whiten`. The default is the oddity task's 2/3.

    Returns
    -------
    jnp.ndarray, shape (4,)
        ``[||d_whitened||, a11, a12, a22]``.

    Warnings
    --------
    2-D only, and not checked here. Check ``input_dim`` before calling.
    """
    A, d_whitened = whiten(Sigma_ref, Sigma_comp, mu_ref, mu_comp, w_ref=w_ref)
    return reduce_2d(A, d_whitened)
