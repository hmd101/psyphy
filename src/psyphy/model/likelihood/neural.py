r"""psyphy.model.likelihood.neural

Trained surrogates for task likelihoods, and the file format they are stored in.

A surrogate replaces a Monte Carlo estimate of ``P(correct)`` with a forward pass
through a small network fitted offline to that same simulation. The payoff is not
only speed: the result is *deterministic*, which is what makes gradient-based
samplers honest and a Laplace approximation possible at all.

This module holds the container and the format. The task classes that use it, and
the feature maps that feed it, live in their own modules.

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
what the caller asked for. The schema is specified in
``psyphy-study/notes/design/nn-surrogate-likelihood.md`` §17.4a; the short version
is below.

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
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_KEYS",
    "NeuralSurrogate",
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
