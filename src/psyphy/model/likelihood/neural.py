r"""psyphy.model.likelihood.neural

Trained surrogates for task likelihoods, and the file format they are stored in.

A surrogate replaces a Monte Carlo estimate of ``P(correct)`` with a forward pass
through a small network fitted offline to that same simulation. The payoff is not
only speed: the result is *deterministic*, which is what makes gradient-based
samplers honest and a Laplace approximation possible at all.

This module holds three things: the forward pass, the container and its file
format, and the concrete oddity task that puts them together. The feature maps
that turn a trial into network inputs live in
:mod:`psyphy.model.likelihood.features`.

Why the file format carries metadata
------------------------------------
A surrogate handed the wrong weights does not crash. It returns a plausible
probability and is simply wrong. Measured on a real trial: using the shipped
oddity weights, but building features with the two-interval pooling weight
instead of the oddity one, gives ``0.3469`` where the correct answer is
``0.3627``. Both are in ``(0, 1)``, neither is NaN, and nothing in the stored
arrays distinguishes the two cases, because a 2-D oddity network and a 2-D
two-interval network have identical shapes.

So the weights alone are not a sufficient description of a surrogate. The file
also records *what it is*, and the loader refuses anything that does not match
what the caller asked for.

Required metadata
    ``psyphy_surrogate_version``, ``task``, ``input_dim``, ``n_features``,
    ``feature_map``, ``feature_map_version``, ``w_ref``.
Optional metadata
    ``domain_lo``/``domain_hi``, ``label_mc_samples``, ``label_bandwidth``,
    ``val_max_err``, ``val_rmse``, ``trained_utc``, ``generator_rev``.
    **Absent means unknown, never unbounded.** In particular the weights shipped
    with psyphy carry no training-domain box, because that training set is not
    ours; a box derived from ``input_mean ± k·input_std`` would describe the
    training *distribution* rather than its *support*, and a reader could not
    tell the difference between that and a measured one.

Note that the architecture is deliberately *not* in the metadata: it is already
implicit in ``n_layers`` and the ``W_i`` shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .base import BernoulliTaskLikelihood
from .features import ODDITY_W_REF, whitened_oddity_features

__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_KEYS",
    "ODDITY_2D_REQUIRE",
    "DEFAULT_ODDITY_WEIGHTS",
    "NeuralSurrogate",
    "NeuralOddityTask",
    "mlp_forward",
    "save_surrogate_npz",
]

#: Version of the on-disk metadata schema this module writes and understands.
#: A file declaring a higher version is refused rather than read optimistically.
SCHEMA_VERSION = 1

#: Metadata every surrogate file must carry. See the module docstring.
REQUIRED_KEYS = (
    "psyphy_surrogate_version",
    "task",
    "input_dim",
    "n_features",
    "feature_map",
    "feature_map_version",
    "w_ref",
)

#: Tolerance for comparing float metadata. 2/3 is not representable in binary, so
#: a file written in float64 and a requirement expressed in float32 differ in the
#: last bits while meaning the same thing.
_FLOAT_ATOL = 1e-6

NNParams = list[tuple[jnp.ndarray, jnp.ndarray]]


def mlp_forward(params: NNParams, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: GELU on the hidden layers, sigmoid on the output.

    Parameters
    ----------
    params : list of (W, b)
        Layer weights and biases, outermost first.
    x : jnp.ndarray, shape (..., n_features)
        Standardized features. Batch-polymorphic in the leading axes, so this
        works for a single trial and under ``vmap`` without changes.

    Returns
    -------
    jnp.ndarray, shape (...)
        ``P(correct)`` in ``(0, 1)``. The trailing singleton axis from the output
        layer is squeezed, so a single feature vector gives a scalar.
    """
    h = x
    for W, b in params[:-1]:
        h = jax.nn.gelu(h @ W + b)
    W_last, b_last = params[-1]
    return jax.nn.sigmoid((h @ W_last + b_last)[..., 0])


def _as_python(value: Any) -> Any:
    """Unwrap a 0-d numpy array to a Python scalar, leaving real arrays alone."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        return item.decode() if isinstance(item, bytes) else item
    return arr


@dataclass(frozen=True)
class NeuralSurrogate:
    """A trained surrogate: frozen weights, input standardization, and provenance.

    Weights and standardization statistics are held in **one object** on purpose.
    They are written to one file, they are meaningless apart, and pairing weights
    with the wrong statistics fails silently rather than loudly.

    Not registered as a JAX pytree, deliberately: these are frozen constants that
    are never optimized, and keeping them out of the pytree registry keeps them
    out of any gradient or optimizer state by construction.

    Attributes
    ----------
    params : list of (W, b)
        The network, outermost layer first.
    input_mean, input_std : jnp.ndarray, shape (n_features,)
        Per-feature standardization, applied in :meth:`__call__`.
    meta : dict
        Everything the file declared about itself. Required keys are always
        present; optional ones appear only when the file carried them.
    """

    params: NNParams
    input_mean: jnp.ndarray
    input_std: jnp.ndarray
    meta: dict = field(default_factory=dict)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Standardize the features, then run the network.

        Parameters
        ----------
        x : jnp.ndarray, shape (..., n_features)
            Raw features, as produced by a feature map in
            :mod:`psyphy.model.likelihood.features`.
        """
        return mlp_forward(self.params, (x - self.input_mean) / self.input_std)

    @property
    def domain(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The training-support box, or ``None`` when the file did not record one.

        ``None`` means *unknown*, not *unbounded*. A surrogate extrapolates
        silently outside the range it was fitted on, so the distinction matters:
        callers that want to warn about extrapolation can only do so for files
        that actually carry a measured box.
        """
        if "domain_lo" in self.meta and "domain_hi" in self.meta:
            return np.asarray(self.meta["domain_lo"]), np.asarray(
                self.meta["domain_hi"]
            )
        return None

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        require: dict[str, Any] | None = None,
        dtype: Any = None,
    ) -> NeuralSurrogate:
        """Load a surrogate, verifying that it is the one the caller asked for.

        Parameters
        ----------
        path : str or Path
            A ``.npz`` written by :func:`save_surrogate_npz`.
        require : dict, optional
            What the calling task needs the file to be, for instance
            ``{"task": "oddity", "input_dim": 2, "w_ref": 2/3, ...}``. Every key
            is compared against the file and a mismatch raises, naming **both**
            sides so the message says what the file is as well as what was
            wanted. Passing ``None`` skips this, which is appropriate only for
            inspection.
        dtype : optional
            Cast the weights to this dtype. Defaults to JAX's current default, so
            that float64 files load into a float32 process without forcing x64 on
            everything else.

        Raises
        ------
        ValueError
            If a required key is missing, a required value disagrees with
            ``require``, the declared ``n_features`` disagrees with the width the
            network actually accepts, or the file declares a newer schema
            version than this module understands.
        """
        path = Path(path)
        dtype = dtype or jnp.zeros(()).dtype

        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            meta = {
                k: _as_python(data[k])
                for k in keys
                if not (
                    k.startswith(("W_", "b_"))
                    or k in ("n_layers", "input_mean", "input_std")
                )
            }

            # --- 1. required keys present -------------------------------------
            missing = [k for k in REQUIRED_KEYS if k not in meta]
            if missing:
                raise ValueError(
                    f"{path.name} is missing required metadata {missing}. Files written "
                    f"before the psyphy surrogate schema (v{SCHEMA_VERSION}) cannot be "
                    f"verified and are refused; re-save it with save_surrogate_npz()."
                )

            # --- 2. schema version we understand ------------------------------
            if int(meta["psyphy_surrogate_version"]) > SCHEMA_VERSION:
                raise ValueError(
                    f"{path.name} declares surrogate schema version "
                    f"{meta['psyphy_surrogate_version']}, but this psyphy understands "
                    f"version {SCHEMA_VERSION}. Upgrade psyphy to read it."
                )

            # --- 3. it is the surrogate the caller asked for -------------------
            for key, want in (require or {}).items():
                got = meta.get(key, "<absent>")
                if isinstance(want, float) and isinstance(got, (int, float)):
                    ok = abs(float(got) - want) < _FLOAT_ATOL
                else:
                    ok = got == want
                if not ok:
                    raise ValueError(
                        f"{path.name} is not usable here: {key}={got!r}, but this task "
                        f"requires {key}={want!r}. Shapes alone cannot distinguish "
                        f"surrogates for different tasks or dimensions, which is why "
                        f"this is checked."
                    )

            n_layers = int(data["n_layers"])
            params = [
                (
                    jnp.asarray(data[f"W_{i}"], dtype=dtype),
                    jnp.asarray(data[f"b_{i}"], dtype=dtype),
                )
                for i in range(n_layers)
            ]

            # --- 4. metadata and weights agree about the input width ----------
            actual = int(params[0][0].shape[0])
            if int(meta["n_features"]) != actual:
                raise ValueError(
                    f"{path.name} declares n_features={meta['n_features']} but its first "
                    f"layer accepts {actual} inputs. The metadata and the weights "
                    f"disagree about the network's own input width, so the file is "
                    f"corrupt or was assembled by hand."
                )

            return cls(
                params=params,
                input_mean=jnp.asarray(data["input_mean"], dtype=dtype),
                input_std=jnp.asarray(data["input_std"], dtype=dtype),
                meta=meta,
            )


def save_surrogate_npz(
    path: str | Path,
    *,
    params: NNParams,
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    meta: dict[str, Any],
    schema_version: int = SCHEMA_VERSION,
) -> Path:
    """Write a surrogate and its metadata to a single ``.npz``.

    This is the only supported way to produce a loadable surrogate file. Writing
    one by hand is possible and inadvisable: the point of the format is that a
    file describes itself, and a hand-assembled file is exactly the case the
    verification in :meth:`NeuralSurrogate.from_npz` exists to catch.

    Parameters
    ----------
    path : str or Path
    params : list of (W, b)
    input_mean, input_std : array, shape (n_features,)
    meta : dict
        At minimum ``task``, ``input_dim``, ``n_features``, ``feature_map``,
        ``feature_map_version`` and ``w_ref``; optional keys from the module
        docstring are written through as given. ``psyphy_surrogate_version`` is
        added automatically.
    schema_version : int
        Overridable only so that tests can write a deliberately future file.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    payload: dict[str, Any] = {
        "n_layers": np.array(len(params)),
        "input_mean": np.asarray(input_mean),
        "input_std": np.asarray(input_std),
    }
    for i, (W, b) in enumerate(params):
        payload[f"W_{i}"] = np.asarray(W)
        payload[f"b_{i}"] = np.asarray(b)

    payload["psyphy_surrogate_version"] = np.array(schema_version)
    for key, value in meta.items():
        if key == "psyphy_surrogate_version":
            continue
        payload[key] = np.asarray(value)

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"refusing to write {path.name}: required metadata {missing} not supplied. "
            f"A file that cannot describe itself cannot be verified when loaded."
        )

    np.savez(path, **payload)
    return path


#: Metadata that :class:`NeuralOddityTask` requires of any weight file it loads.
#: Checked by :meth:`NeuralSurrogate.from_npz`, which raises when a file declares
#: anything different. Expressed as a plain dict so that a task defined outside
#: psyphy can state its own requirements the same way.
ODDITY_2D_REQUIRE: dict[str, Any] = {
    "task": "oddity",
    "input_dim": 2,
    "feature_map": "whitened_rotation_reduced",
    "feature_map_version": 1,
    "w_ref": float(ODDITY_W_REF),
}

#: Weight file shipped with psyphy, used when no surrogate is supplied.
DEFAULT_ODDITY_WEIGHTS = "oddity_2d_64x2.npz"

#: Clip applied to the returned probability. This exists so that ``log(p)`` and
#: ``log(1 - p)`` stay finite, and for no other reason. See the note in
#: :meth:`NeuralOddityTask.predict` about why it must not be raised to 1/3.
_P_EPS = 1e-6


def _default_surrogate() -> NeuralSurrogate:
    """Load the weight file that ships with psyphy."""
    path = (
        resources.files("psyphy.model.likelihood") / "weights" / DEFAULT_ODDITY_WEIGHTS
    )
    with resources.as_file(path) as real_path:
        return NeuralSurrogate.from_npz(real_path, require=ODDITY_2D_REQUIRE)


class NeuralOddityTask(BernoulliTaskLikelihood):
    """The 3-AFC oddity likelihood, evaluated by a trained network.

    This is a drop-in replacement for :class:`~psyphy.model.likelihood.OddityTask`.
    Both describe the same observer and the same decision rule; they differ only
    in how ``P(correct)`` is obtained. ``OddityTask`` simulates noisy trials and
    counts how often the rule succeeds. This class evaluates a small network that
    was fitted to that simulation in advance.

    Why that is worth doing
    -----------------------
    The simulation is expensive, because every likelihood evaluation re-simulates
    many trials, and it is *random*, because it is a sample average. The second
    property is the more limiting one. Gradient-based samplers assume the log
    density is a fixed function of the parameters; a stochastic one breaks the
    assumption quietly, and a Laplace approximation, which needs a second
    derivative, becomes meaningless because second derivatives amplify noise.

    A network fitted offline removes both problems at once. It is also, at the
    sample counts normally used, *closer* to a high-accuracy simulation than the
    simulation itself is, so this is not a trade of accuracy for speed.

    How a 4-input network can be enough
    -----------------------------------
    ``P(correct)`` for this task is invariant under any invertible affine change
    of stimulus coordinates. That reduces its arguments from eight numbers to
    four, so the network only has to learn a four-dimensional function. The
    reduction is performed by
    :func:`~psyphy.model.likelihood.features.whitened_oddity_features`, whose
    module docstring explains the counting.

    Limitations, all enforced rather than documented-and-hoped
    ---------------------------------------------------------
    * **Two-dimensional stimuli only.** The reduction's final step is a 2-D
      rotation; three dimensions need six features and separately trained weights.
    * **Gaussian internal noise only.** That assumption is baked into the data the
      network was fitted to.
    * **The weights must match the task.** A file trained for a different task or
      dimension has the same shape as this one, so it would load and silently
      produce wrong numbers. :meth:`NeuralSurrogate.from_npz` refuses it instead.

    Parameters
    ----------
    surrogate : callable, optional
        Maps the four features to ``P(correct)``. Defaults to the weights shipped
        with psyphy. Typed as a plain callable rather than as
        :class:`NeuralSurrogate` so that anything with the right signature works,
        including a network from another framework.

    Examples
    --------
    >>> from psyphy.model import WPPM, Prior, GaussianNoise
    >>> from psyphy.model.likelihood import NeuralOddityTask
    >>> model = WPPM(
    ...     input_dim=2,
    ...     prior=Prior(input_dim=2, basis_degree=3),
    ...     likelihood=NeuralOddityTask(),
    ...     noise=GaussianNoise(),
    ... )

    See Also
    --------
    psyphy.model.likelihood.OddityTask
        The Monte Carlo implementation of the same task. It remains the reference:
        the network was fitted to it, and it is what any surrogate is checked
        against.
    """

    #: This likelihood does not consume randomness, so ``loglik`` returns the same
    #: value for every PRNG key. Samplers can use this to skip workarounds that
    #: exist only for stochastic likelihoods.
    is_stochastic: ClassVar[bool] = False

    #: What this task demands of a weight file. See :data:`ODDITY_2D_REQUIRE`.
    REQUIRED: ClassVar[dict[str, Any]] = ODDITY_2D_REQUIRE

    def __init__(
        self, surrogate: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ) -> None:
        # Annotated rather than inferred, and the annotation is the point: what
        # this class needs is *any* callable mapping the features to a
        # probability. NeuralSurrogate is merely the implementation that ships
        # with psyphy, so the attribute is typed by the capability it requires
        # rather than by the concrete class it usually holds.
        #
        # Filled in by statement rather than by conditional expression, which is
        # not a style choice. A type checker reading ``a if c else b`` must give
        # the whole expression one type, and the only type shared by a bare
        # Callable and NeuralSurrogate is ``object``; the assignment then fails
        # against the annotation above, and calling the attribute fails too. The
        # two-branch form lets each branch be checked against the annotation on
        # its own, which is what was meant.
        if surrogate is None:
            surrogate = _default_surrogate()
        self.surrogate: Callable[[jnp.ndarray], jnp.ndarray] = surrogate

    def predict(
        self,
        params: Any,
        stimuli: jnp.ndarray,
        model: Any,
        *,
        key: Any = None,
    ) -> tuple[jnp.ndarray]:
        """Return ``(P(correct),)`` for one trial.

        Parameters
        ----------
        params : Any
            Model parameters.
        stimuli : jnp.ndarray, shape (2, input_dim)
            Slot 0 is the reference, slot 1 the comparison. The oddity task shows
            three stimuli but stores two: the reference is presented twice, and
            that duplication belongs to the decision rule rather than to the data.
        model : Any
            Supplies the covariance field through ``local_covariance``.
        key : Any, optional
            Accepted for interface compatibility and ignored. This likelihood is
            deterministic, which is the main reason it exists.

        Returns
        -------
        tuple of one jnp.ndarray
            A one-element tuple, matching the convention that ``predict`` returns
            the sufficient statistics of the response distribution. Returning a
            bare array here would not raise; it would make ``loglik`` reuse the
            first trial's probability for the whole batch.
        """
        self._check_applicable(model)

        ref, comp = stimuli[0], stimuli[1]
        # local_covariance is the public accessor for Sigma(x); it already adds
        # the diagonal stabiliser and returns a (d, d) matrix whatever the
        # embedding dimension happens to be.
        sigma_ref = model.local_covariance(params, ref)
        sigma_comp = model.local_covariance(params, comp)

        features = whitened_oddity_features(
            sigma_ref, sigma_comp, ref, comp, w_ref=ODDITY_W_REF
        )
        p = self.surrogate(features)

        # The clip keeps log(p) and log(1 - p) finite. It is deliberately not a
        # floor at the 1/3 guessing rate: this observer's rule is suboptimal, and
        # when the comparison's distribution is much tighter than the reference's
        # it performs reliably *worse* than chance. Values near 0.17 in that
        # regime are correct, and clamping them would replace a right answer with
        # a wrong one while flattening the gradient where it matters most.
        return (jnp.clip(p, _P_EPS, 1.0 - _P_EPS),)

    def _check_applicable(self, model: Any) -> None:
        """Refuse models the shipped weights do not describe.

        These are Python-level checks on static attributes, so they run once when
        the function is traced and cost nothing per call, but they fail loudly
        instead of returning a confident wrong number.
        """
        if int(model.input_dim) != 2:
            raise NotImplementedError(
                f"NeuralOddityTask supports input_dim=2, got {model.input_dim}. "
                f"The feature reduction ends in a two-dimensional rotation, and a "
                f"{model.input_dim}-dimensional task needs both a different "
                f"reduction and its own trained weights. Use OddityTask instead."
            )
        noise_name = type(model.noise).__name__
        if noise_name != "GaussianNoise":
            raise NotImplementedError(
                f"NeuralOddityTask assumes Gaussian internal noise, got "
                f"{noise_name}. The network was fitted to simulations drawn from "
                f"Gaussian distributions, so it has no information about any "
                f"other noise model. Use OddityTask instead."
            )
