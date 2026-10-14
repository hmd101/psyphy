r"""psyphy.model.likelihood.features

Feature maps for neural surrogate likelihoods: the compression step that makes a
trained surrogate small enough to be worth training.

Why this exists
---------------
For a discrimination task whose decision rule is a Mahalanobis comparison, the
probability of a correct response depends on the two covariances and the two
stimulus locations only through a handful of numbers. In 2-D the raw count is
eight (three free entries per symmetric covariance, plus the two-vector mean
difference), and the task is invariant under *any* invertible affine change of
stimulus coordinates, which removes four of them. A surrogate therefore has to
learn a 4-D function rather than an 8-D one.

The reduction happens in two steps, split here because they generalise
differently:

``whiten``
    Change coordinates so the pooled covariance becomes the identity. Works in
    any dimension and for any task; the task enters only through the pooling
    weight ``w_ref``. Removes ``d(d+1)/2`` numbers.
``reduce_2d``
    Spend the remaining rotational freedom aligning the mean difference with an
    axis, so only its length matters. Dimension-specific: a 3-D analogue needs a
    different construction and yields six features rather than four.

The count is ``2 * input_dim`` in general.

Public on purpose
-----------------
These are free functions over arrays, not methods on a model, in the same spirit
as :mod:`psyphy.viz.geometry`: they depend only on JAX, are testable without a
model or a trained network, and are what someone deriving a feature map for their
own task needs to build on. The weight-file metadata records which feature map a
set of weights expects, so the identity of this module is already part of the
serialized format.
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

    The change of basis is ``M = inv(chol(S_bar))``, applied here as triangular
    solves rather than by forming ``M`` (cheaper, and better conditioned).

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
        3-AFC oddity task, 1/2 for a two-interval task. Passing this rather than
        hard-coding it is what lets a different task reuse this function.

    Returns
    -------
    A : jnp.ndarray, shape (d, d)
        ``Sigma_ref`` in whitened coordinates.
    d_whitened : jnp.ndarray, shape (d,)
        The mean difference in whitened coordinates.

    Notes
    -----
    Works in any dimension. The dimension-specific part is the reduction that
    follows, not this.
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
    r"""Spend the residual rotation, giving the four 2-D features.

    After whitening, the only remaining coordinate freedom is rotation, because
    rotations are what preserve the identity covariance. Using it to align the
    mean difference with the +y axis means only the *length* of that vector
    matters, removing one more number.

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
    When the two stimuli coincide the rotation is undefined. The identity is
    used in that case, which is the right limit: ``||d|| = 0`` carries the
    information that the stimuli are identical, and the orientation of ``A`` is
    then irrelevant because every direction is equivalent.
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

    One trial, not a batch: psyphy's ``predict`` contract is per-trial and
    ``BernoulliTaskLikelihood.loglik`` supplies the ``vmap``.

    Parameters
    ----------
    Sigma_ref, Sigma_comp : jnp.ndarray, shape (2, 2)
    mu_ref, mu_comp : jnp.ndarray, shape (2,)
    w_ref : float
        See :func:`whiten`. The default is the oddity task's 2/3.

    Returns
    -------
    jnp.ndarray, shape (4,)
        ``[||d_whitened||, a11, a12, a22]``.

    Warnings
    --------
    2-D only. The rotational reduction in :func:`reduce_2d` is a 2-D
    construction; callers must validate ``input_dim`` themselves.
    """
    A, d_whitened = whiten(Sigma_ref, Sigma_comp, mu_ref, mu_comp, w_ref=w_ref)
    return reduce_2d(A, d_whitened)
