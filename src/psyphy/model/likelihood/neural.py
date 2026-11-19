r"""psyphy.model.likelihood.neural

Trained networks for task likelihoods, and the file format they are stored in. A
trained network is also called a surrogate.

The fast likelihood replaces the Monte Carlo estimate of ``P(correct)`` with a
forward pass through a small network fitted offline to that simulation. The network returns the
logit of ``P(correct)``, the same value for every PRNG key.

The module holds the forward pass (:func:`forward_pass`), the network and its file
(:class:`TrainedNetwork`, :func:`save_network`), and the oddity task that
uses them (:class:`FastOddityTask`). The feature maps that turn a trial into
network inputs are in :mod:`psyphy.model.likelihood.features`.

Weight files
------------
A weight file is an ``.npz`` holding the network, its input standardization, and
metadata naming what the network was trained for. Networks for different tasks
can have identical array shapes, so the metadata is what tells them apart.
:meth:`TrainedNetwork.from_npz` compares it against what the task requires and
raises on a mismatch.

Required metadata
    ``psyphy_network_version``, ``task``, ``input_dim``, ``n_features``,
    ``feature_map``, ``feature_map_version``, ``pooling_weight``.
Optional metadata
    ``domain_lo``/``domain_hi``, ``label_mc_samples``, ``label_bandwidth``,
    ``val_max_err``, ``val_rmse``, ``trained_utc``, ``generator_rev``.
    An absent key means unknown. The weights shipped with psyphy record no
    training domain.

The layer count and widths are read from ``n_layers`` and the ``W_i`` shapes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from ..noise import GaussianNoise
from .base import BernoulliTaskLikelihood, log_probs_from_logit
from .features import ODDITY_POOLING_WEIGHT, oddity_features

__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_KEYS",
    "ODDITY_2D_REQUIRE",
    "DEFAULT_ODDITY_WEIGHTS",
    "TrainedNetwork",
    "FastOddityTask",
    "forward_pass",
    "save_network",
]

#: Version of the metadata schema this module writes and reads.
#: A file declaring a higher version is refused.
SCHEMA_VERSION = 1

#: Metadata every network file must carry. See the module docstring.
REQUIRED_KEYS = (
    "psyphy_network_version",
    "task",
    "input_dim",
    "n_features",
    "feature_map",
    "feature_map_version",
    "pooling_weight",
)

#: Tolerance for float metadata such as ``pooling_weight``: 2/3 in float64 and in float32
#: differ in the last bits.
_FLOAT_ATOL = 1e-6

NNParams = Sequence[tuple[jnp.ndarray, jnp.ndarray]]


def forward_pass(params: NNParams, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: GELU on the hidden layers, a linear output layer.

    Parameters
    ----------
    params : list of (W, b), W shape (n_in, n_out), b shape (n_out,)
        Layer weights and biases, input layer first.
    x : jnp.ndarray, shape (..., n_features)
        Standardized features, with any number of leading axes, including none.

    Returns
    -------
    jnp.ndarray, shape (...)
        The logit of ``P(correct)``, so ``P(correct) = sigmoid(logit)``. The
        trailing singleton axis from the output layer is squeezed, so a single
        feature vector gives a scalar.
    """
    h = x
    for W, b in params[:-1]:
        h = jax.nn.gelu(h @ W + b)
    W_last, b_last = params[-1]
    return (h @ W_last + b_last)[..., 0]


def _as_python(value: Any) -> Any:
    """Unwrap a 0-d numpy array to a Python scalar, leaving real arrays alone."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        return item.decode() if isinstance(item, bytes) else item
    return arr


def _check_requirements(
    metadata: dict[str, Any], require: dict[str, Any], source: str
) -> None:
    """Raise ``ValueError`` for the first key of ``metadata`` that differs from ``require``."""
    for key, want in require.items():
        got = metadata.get(key, "<absent>")
        if isinstance(want, float) and isinstance(got, (int, float)):
            ok = abs(float(got) - want) < _FLOAT_ATOL
        else:
            ok = got == want
        if not ok:
            raise ValueError(
                f"{source} is not usable here: {key}={got!r}, but this task "
                f"requires {key}={want!r}."
            )


@dataclass(frozen=True)
class TrainedNetwork:
    """A trained network: frozen weights, input standardization, and metadata.

    Calling it standardizes the features with ``input_mean`` and ``input_std``,
    then runs the network. Load one with :meth:`from_npz`. It is not a JAX pytree,
    so its weights never appear in a gradient or an optimizer state.

    Attributes
    ----------
    params : tuple of (W, b), W shape (n_in, n_out), b shape (n_out,)
        The network, input layer first. A list passed to the constructor is
        stored as a tuple.
    input_mean, input_std : jnp.ndarray, shape (n_features,)
        Per-feature standardization, applied in :meth:`__call__`.
    metadata : dict
        Everything the file declared about itself. Required keys are always
        present; optional ones appear only when the file carried them.
    """

    params: NNParams
    input_mean: jnp.ndarray
    input_std: jnp.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", tuple(tuple(layer) for layer in self.params))

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
            The logit of ``P(correct)``.
        """
        return forward_pass(self.params, (x - self.input_mean) / self.input_std)

    @property
    def domain(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The training-domain box ``(lo, hi)``, each shape (n_features,), or ``None``.

        ``None`` means the file did not record one, not that the domain is
        unbounded. The network extrapolates without warning outside the range it
        was fitted on.
        """
        if "domain_lo" in self.metadata and "domain_hi" in self.metadata:
            return np.asarray(self.metadata["domain_lo"]), np.asarray(
                self.metadata["domain_hi"]
            )
        return None

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        require: dict[str, Any] | None = None,
        dtype: Any = None,
    ) -> TrainedNetwork:
        """Load a network and check its metadata against ``require``.

        Parameters
        ----------
        path : str or Path
            A ``.npz`` written by :func:`save_network`.
        require : dict, optional
            What the calling task needs the file to be, for instance
            ``FastOddityTask.REQUIRED``. Every key is compared against the
            file, and a mismatch raises with both values in the message. ``None``
            skips the check, so use it only to inspect a file.
        dtype : optional
            Cast the weights to this dtype. Defaults to JAX's default float dtype,
            so a float64 file loads into a float32 process without enabling x64.

        Returns
        -------
        TrainedNetwork

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
            metadata = {
                k: _as_python(data[k])
                for k in keys
                if not (
                    k.startswith(("W_", "b_"))
                    or k in ("n_layers", "input_mean", "input_std")
                )
            }

            # --- 1. required keys present -------------------------------------
            missing = [k for k in REQUIRED_KEYS if k not in metadata]
            if missing:
                raise ValueError(
                    f"{path.name} is missing required metadata {missing}. "
                    f"Re-save it with save_network()."
                )

            # --- 2. schema version we understand ------------------------------
            if int(metadata["psyphy_network_version"]) > SCHEMA_VERSION:
                raise ValueError(
                    f"{path.name} declares network schema version "
                    f"{metadata['psyphy_network_version']}, but this psyphy understands "
                    f"version {SCHEMA_VERSION}. Upgrade psyphy to read it."
                )

            # --- 3. it is the network the caller asked for -------------------
            _check_requirements(metadata, require or {}, path.name)

            n_layers = int(data["n_layers"])
            params = tuple(
                (
                    jnp.asarray(data[f"W_{i}"], dtype=dtype),
                    jnp.asarray(data[f"b_{i}"], dtype=dtype),
                )
                for i in range(n_layers)
            )

            # --- 4. metadata and weights agree about the input width ----------
            actual = int(params[0][0].shape[0])
            if int(metadata["n_features"]) != actual:
                raise ValueError(
                    f"{path.name} declares n_features={metadata['n_features']} but its first "
                    f"layer accepts {actual} inputs."
                )

            return cls(
                params=params,
                input_mean=jnp.asarray(data["input_mean"], dtype=dtype),
                input_std=jnp.asarray(data["input_std"], dtype=dtype),
                metadata=metadata,
            )


def save_network(
    path: str | Path,
    *,
    params: NNParams,
    input_mean: jnp.ndarray,
    input_std: jnp.ndarray,
    metadata: dict[str, Any],
) -> Path:
    """Write a network and its metadata to one ``.npz``.

    Writes the format :meth:`TrainedNetwork.from_npz` reads. Write network
    files with this function, since ``from_npz`` refuses a file without the
    required metadata.

    Parameters
    ----------
    path : str or Path
        Where to write the file.
    params : sequence of (W, b), W shape (n_in, n_out), b shape (n_out,)
        The network, input layer first.
    input_mean, input_std : jnp.ndarray, shape (n_features,)
        Per-feature standardization.
    metadata : dict
        At minimum ``task``, ``input_dim``, ``n_features``, ``feature_map``,
        ``feature_map_version`` and ``pooling_weight``. Optional keys from the module
        docstring are written as given. ``psyphy_network_version`` is added.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If ``metadata`` lacks a required key.
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

    payload["psyphy_network_version"] = np.array(SCHEMA_VERSION)
    for key, value in metadata.items():
        if key == "psyphy_network_version":
            continue
        payload[key] = np.asarray(value)

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"refusing to write {path.name}: required metadata {missing} not supplied. "
            f"Pass them in metadata."
        )

    np.savez(path, **payload)
    return path


#: Metadata :class:`FastOddityTask` requires of a weight file, checked by
#: :meth:`TrainedNetwork.from_npz`. A task defined outside psyphy states its
#: requirements as the same kind of dict.
ODDITY_2D_REQUIRE: dict[str, Any] = {
    "task": "oddity",
    "input_dim": 2,
    "feature_map": "noise_normalized_aligned",
    "feature_map_version": 1,
    "pooling_weight": float(ODDITY_POOLING_WEIGHT),
}

#: Weight file shipped with psyphy, used when no network is supplied.
DEFAULT_ODDITY_WEIGHTS = "oddity_2d_64x2.npz"


def _default_network() -> TrainedNetwork:
    """Load the weight file that ships with psyphy."""
    path = (
        resources.files("psyphy.model.likelihood") / "weights" / DEFAULT_ODDITY_WEIGHTS
    )
    with resources.as_file(path) as real_path:
        return TrainedNetwork.from_npz(real_path, require=ODDITY_2D_REQUIRE)


class FastOddityTask(BernoulliTaskLikelihood):
    """The 3-AFC oddity likelihood, evaluated by a trained network.

    Use it in place of :class:`~psyphy.model.likelihood.OddityTask` for a 2-D
    model with Gaussian noise. Both model the same observer and the same decision
    rule. ``OddityTask`` simulates noisy trials and counts how often the rule
    picks the odd stimulus. ``FastOddityTask`` evaluates a small network fitted
    to that simulation in advance, so each call is one forward pass and returns
    the same value for every PRNG key.

    The network takes 4 numbers per trial, computed by
    :func:`~psyphy.model.likelihood.features.oddity_features`.
    ``P(correct)`` does not change under an invertible affine change of stimulus
    coordinates, and that leaves 4 of the 8 numbers describing a 2-D trial.

    Parameters
    ----------
    network : callable, optional
        Maps the 4 features to the logit of ``P(correct)``, that is
        ``log(p / (1 - p))``. Defaults to the weights shipped with psyphy. Any
        callable with that signature works, a :class:`TrainedNetwork` or a
        network from another framework.

    Raises
    ------
    ValueError
        If ``network.metadata`` disagrees with :attr:`REQUIRED`, for instance a file
        trained for another task. The message names the key and both values.

    Notes
    -----
    2-D stimuli and ``GaussianNoise`` only. :meth:`predict` raises
    ``NotImplementedError`` for any other model. A 3-D model needs 6 features
    and its own trained weights.

    A ``network`` with a ``metadata`` dict, such as a :class:`TrainedNetwork`, is
    checked against :attr:`REQUIRED` at construction. A plain callable has no
    metadata and is not checked.

    Examples
    --------
    >>> from psyphy.model import WPPM, Prior, GaussianNoise
    >>> from psyphy.model.likelihood import FastOddityTask
    >>> model = WPPM(
    ...     prior=Prior(input_dim=2, basis_degree=3),
    ...     likelihood=FastOddityTask(),
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
        self, network: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ) -> None:
        # an if-statement, not `a if c else b`: mypy types that as object
        if network is None:
            network = _default_network()
        metadata = getattr(network, "metadata", None)
        if isinstance(metadata, dict):
            _check_requirements(metadata, self.REQUIRED, "the network")
        self._network: Callable[[jnp.ndarray], jnp.ndarray] = network

    @property
    def network(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """The network this task evaluates. Read-only: pass another to the constructor."""
        return self._network

    def predict(
        self,
        params: Any,
        stimuli: jnp.ndarray,
        model: Any,
        key: Any = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return log P(correct) and log P(incorrect) for one trial.

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
        log_p : jnp.ndarray, shape ()
            log P(correct).
        log_1mp : jnp.ndarray, shape ()
            log P(incorrect).

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

        features = oddity_features(sigma_ref, sigma_comp, ref, comp)
        z = self.network(features)  # logit of P(correct), shape ()

        # no floor at 1/3: when cov_comparison is much tighter than cov_ref this rule
        # does worse than chance (p ~ 0.17), and that p is right
        return log_probs_from_logit(z)

    def _check_applicable(self, model: Any) -> None:
        """Raise for a model the shipped weights do not describe.

        Checks static attributes, so it runs once at trace time.
        """
        if int(model.input_dim) != 2:
            raise NotImplementedError(
                f"FastOddityTask supports input_dim=2, got {model.input_dim}. "
                f"Use OddityTask instead."
            )
        if not isinstance(model.noise, GaussianNoise):
            raise NotImplementedError(
                f"FastOddityTask supports GaussianNoise only, "
                f"got {type(model.noise).__name__}. "
                f"Use OddityTask instead."
            )
