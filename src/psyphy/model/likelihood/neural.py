r"""psyphy.model.likelihood.neural

Trained surrogates for task likelihoods, and the file format they are stored in.

A surrogate replaces the Monte Carlo estimate of ``P(correct)`` with a forward pass
through a small network fitted offline to that simulation. It returns the same
value for every PRNG key.

The module holds the forward pass (:func:`mlp_forward`), the network and its file
(:class:`NeuralSurrogate`, :func:`save_surrogate_npz`), and the oddity task that
uses them (:class:`NeuralOddityTask`). The feature maps that turn a trial into
network inputs are in :mod:`psyphy.model.likelihood.features`.

Weight files
------------
A weight file is an ``.npz`` holding the network, its input standardization, and
metadata naming what the network was trained for. Networks for different tasks
can have identical array shapes, so the metadata is what tells them apart.
:meth:`NeuralSurrogate.from_npz` compares it against what the task requires and
raises on a mismatch.

Required metadata
    ``psyphy_surrogate_version``, ``task``, ``input_dim``, ``n_features``,
    ``feature_map``, ``feature_map_version``, ``w_ref``.
Optional metadata
    ``domain_lo``/``domain_hi``, ``label_mc_samples``, ``label_bandwidth``,
    ``val_max_err``, ``val_rmse``, ``trained_utc``, ``generator_rev``.
    An absent key means unknown. The weights shipped with psyphy record no
    training domain.

The layer count and widths are read from ``n_layers`` and the ``W_i`` shapes.
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

#: Version of the metadata schema this module writes and reads.
#: A file declaring a higher version is refused.
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

#: Tolerance for float metadata such as ``w_ref``: 2/3 in float64 and in float32
#: differ in the last bits.
_FLOAT_ATOL = 1e-6

NNParams = list[tuple[jnp.ndarray, jnp.ndarray]]


def mlp_forward(params: NNParams, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: GELU on the hidden layers, sigmoid on the output.

    Parameters
    ----------
    params : list of (W, b), W shape (n_in, n_out), b shape (n_out,)
        Layer weights and biases, input layer first.
    x : jnp.ndarray, shape (..., n_features)
        Standardized features, with any number of leading axes, including none.

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
    """A trained surrogate: frozen weights, input standardization, and metadata.

    Calling it standardizes the features with ``input_mean`` and ``input_std``,
    then runs the network. Load one with :meth:`from_npz`. It is not a JAX pytree,
    so its weights never appear in a gradient or an optimizer state.

    Attributes
    ----------
    params : list of (W, b), W shape (n_in, n_out), b shape (n_out,)
        The network, input layer first.
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

        Returns
        -------
        jnp.ndarray, shape (...)
            ``P(correct)`` in ``(0, 1)``.
        """
        return mlp_forward(self.params, (x - self.input_mean) / self.input_std)

    @property
    def domain(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The training-domain box ``(lo, hi)``, each shape (n_features,), or ``None``.

        ``None`` means the file did not record one, not that the domain is
        unbounded. The network extrapolates without warning outside the range it
        was fitted on.
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
        """Load a surrogate and check its metadata against ``require``.

        Parameters
        ----------
        path : str or Path
            A ``.npz`` written by :func:`save_surrogate_npz`.
        require : dict, optional
            What the calling task needs the file to be, for instance
            ``NeuralOddityTask.REQUIRED``. Every key is compared against the
            file, and a mismatch raises with both values in the message. ``None``
            skips the check, so use it only to inspect a file.
        dtype : optional
            Cast the weights to this dtype. Defaults to JAX's default float dtype,
            so a float64 file loads into a float32 process without enabling x64.

        Returns
        -------
        NeuralSurrogate

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
                    f"{path.name} is missing required metadata {missing}. "
                    f"Re-save it with save_surrogate_npz()."
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
                        f"requires {key}={want!r}."
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
                    f"layer accepts {actual} inputs."
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
    """Write a surrogate and its metadata to one ``.npz``.

    Writes the format :meth:`NeuralSurrogate.from_npz` reads. Write surrogate
    files with this function, since ``from_npz`` refuses a file without the
    required metadata.

    Parameters
    ----------
    path : str or Path
        Where to write the file.
    params : list of (W, b), W shape (n_in, n_out), b shape (n_out,)
        The network, input layer first.
    input_mean, input_std : jnp.ndarray, shape (n_features,)
        Per-feature standardization.
    meta : dict
        At minimum ``task``, ``input_dim``, ``n_features``, ``feature_map``,
        ``feature_map_version`` and ``w_ref``. Optional keys from the module
        docstring are written as given. ``psyphy_surrogate_version`` is added.
    schema_version : int
        Leave at the default. Tests set it higher to check that ``from_npz``
        refuses a newer file.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If ``meta`` lacks a required key.
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
            f"Pass them in meta."
        )

    np.savez(path, **payload)
    return path


#: Metadata :class:`NeuralOddityTask` requires of a weight file, checked by
#: :meth:`NeuralSurrogate.from_npz`. A task defined outside psyphy states its
#: requirements as the same kind of dict.
ODDITY_2D_REQUIRE: dict[str, Any] = {
    "task": "oddity",
    "input_dim": 2,
    "feature_map": "whitened_rotation_reduced",
    "feature_map_version": 1,
    "w_ref": float(ODDITY_W_REF),
}

#: Weight file shipped with psyphy, used when no surrogate is supplied.
DEFAULT_ODDITY_WEIGHTS = "oddity_2d_64x2.npz"

# clip on the returned p, so log(p) and log(1 - p) stay finite. Not 1/3, see predict()
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

    Use it in place of :class:`~psyphy.model.likelihood.OddityTask` for a 2-D
    model with Gaussian noise. Both model the same observer and the same decision
    rule. ``OddityTask`` simulates noisy trials and counts how often the rule
    picks the odd stimulus. ``NeuralOddityTask`` evaluates a small network fitted
    to that simulation in advance, so each call is one forward pass and returns
    the same value for every PRNG key.

    The network takes 4 numbers per trial, computed by
    :func:`~psyphy.model.likelihood.features.whitened_oddity_features`.
    ``P(correct)`` does not change under an invertible affine change of stimulus
    coordinates, and that leaves 4 of the 8 numbers describing a 2-D trial.

    Parameters
    ----------
    surrogate : callable, optional
        Maps the 4 features to ``P(correct)``. Defaults to the weights shipped
        with psyphy. Any callable with that signature works, a
        :class:`NeuralSurrogate` or a network from another framework.

    Notes
    -----
    2-D stimuli and ``GaussianNoise`` only. :meth:`predict` raises
    ``NotImplementedError`` for any other model. A 3-D model needs 6 features
    and its own trained weights.

    A ``surrogate`` passed in is not checked against the task. Load a weight file
    with ``NeuralSurrogate.from_npz(path, require=NeuralOddityTask.REQUIRED)`` to
    check it.

    Examples
    --------
    >>> from psyphy.model import WPPM, Prior, GaussianNoise
    >>> from psyphy.model.likelihood import NeuralOddityTask
    >>> model = WPPM(
    ...     prior=Prior(input_dim=2, basis_degree=3),
    ...     likelihood=NeuralOddityTask(),
    ...     noise=GaussianNoise(),
    ... )

    See Also
    --------
    psyphy.model.likelihood.OddityTask
        The Monte Carlo version of the same task, which the network was fitted to.
    """

    #: ``loglik`` ignores its PRNG key and returns the same value for every key.
    is_stochastic: ClassVar[bool] = False

    #: What this task demands of a weight file. See :data:`ODDITY_2D_REQUIRE`.
    REQUIRED: ClassVar[dict[str, Any]] = ODDITY_2D_REQUIRE

    def __init__(
        self, surrogate: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ) -> None:
        # an if-statement, not `a if c else b`: mypy types that as object
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
            Slot 0 is the reference, slot 1 the comparison. The trial shows the
            reference twice, and the decision rule accounts for that.
        model : Any
            Supplies the covariance field through ``local_covariance``.
        key : Any, optional
            Ignored. Accepted so the signature matches the other tasks.

        Returns
        -------
        tuple of one jnp.ndarray, shape ()
            ``(p_correct,)``, the convention for a Bernoulli task.

        Raises
        ------
        NotImplementedError
            If ``model.input_dim != 2`` or the noise model is not ``GaussianNoise``.
        """
        self._check_applicable(model)

        ref, comp = stimuli[0], stimuli[1]
        # Sigma(x), (2, 2), diagonal term included
        sigma_ref = model.local_covariance(params, ref)
        sigma_comp = model.local_covariance(params, comp)

        features = whitened_oddity_features(
            sigma_ref, sigma_comp, ref, comp, w_ref=ODDITY_W_REF
        )
        p = self.surrogate(features)

        # Clip for finite logs only, no floor at 1/3. When Sigma_comp is much tighter
        # than Sigma_ref this rule does worse than chance (p ~ 0.17), and that p is right
        return (jnp.clip(p, _P_EPS, 1.0 - _P_EPS),)

    def _check_applicable(self, model: Any) -> None:
        """Raise for a model the shipped weights do not describe.

        Checks static attributes, so it runs once at trace time.
        """
        if int(model.input_dim) != 2:
            raise NotImplementedError(
                f"NeuralOddityTask supports input_dim=2, got {model.input_dim}. "
                f"Use OddityTask instead."
            )
        noise_name = type(model.noise).__name__
        if noise_name != "GaussianNoise":
            raise NotImplementedError(
                f"NeuralOddityTask supports GaussianNoise only, got {noise_name}. "
                f"Use OddityTask instead."
            )
