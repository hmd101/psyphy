"""
dataset.py
-----------

Core data containers for psyphy.

defines:
- ResponseData: container for psychophysical trial data
- TrialBatch: container for a proposed batch of trials

Notes
-----
- Data is stored in standard NumPy (mutable!) arrays or Python lists.
- Use numpy for I/O and analysis.
- Convert to jax.numpy (jnp) (immutable!) arrays only when passing into WPPM
  or inference engines that require JAX/Optax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, slots=True)
class TrialData:
    """Batched trial data for compute.

    This is the canonical, compute-efficient representation of observed trials.

    Shapes
    ------
    refs : (N, d)
    comparisons : (N, d)
    responses : (N,)

    Notes
    -----
    - You can also think of this as a more generic ML-style dataset
      ``X`` with shape (N, 2, d) plus ``y`` with shape (N,). The explicit
      field names (refs/comparisons) are currently native to :class:`OddityTask`.
    - This is intended to be JAX-friendly (PyTree of arrays) so likelihood and
      inference code can be JIT-compiled without touching Python containers.
    """

    refs: jnp.ndarray
    comparisons: jnp.ndarray
    responses: jnp.ndarray

    def __post_init__(self) -> None:
        # Basic shape validation (keep lightweight; raise early for common mistakes).
        if self.refs.ndim != 2:
            raise ValueError(f"refs must be 2D (N,d), got shape {self.refs.shape}")
        if self.comparisons.ndim != 2:
            raise ValueError(
                f"comparisons must be 2D (N,d), got shape {self.comparisons.shape}"
            )
        if self.responses.ndim != 1:
            raise ValueError(
                f"responses must be 1D (N,), got shape {self.responses.shape}"
            )
        if self.refs.shape[0] != self.comparisons.shape[0]:
            raise ValueError(
                "refs and comparisons must have same first dimension; "
                f"got {self.refs.shape[0]} vs {self.comparisons.shape[0]}"
            )
        if self.refs.shape[0] != self.responses.shape[0]:
            raise ValueError(
                "refs and responses must have same first dimension; "
                f"got {self.refs.shape[0]} vs {self.responses.shape[0]}"
            )

    def __len__(self) -> int:
        """Number of trials (N)."""
        return int(self.responses.shape[0])

    @property
    def num_trials(self) -> int:
        """Number of trials (N)."""
        return len(self)


class ResponseData:
    """Python-friendly incremental trial log.

    This container is convenient for adaptive trial placement and I/O (e.g., CSV),
    but it is not a compute-efficient representation for JAX.

    Use :class:`TrialData` for model fitting and likelihood evaluation.
    """

    def __init__(self) -> None:
        self.refs: list[Any] = []
        self.comparisons: list[Any] = []
        self.responses: list[int] = []

    def add_trial(self, ref: Any, comparison: Any, resp: int) -> None:
        """
        append a single trial.

        Parameters
        ----------
        ref : Any
            Reference stimulus (numpy array, list, etc.)
        comparison : Any
            Probe stimulus
        resp : int
            Subject response (binary or categorical)
        """
        self.refs.append(ref)
        self.comparisons.append(comparison)
        self.responses.append(resp)

    def add_batch(self, responses: list[int], trial_batch: TrialBatch) -> None:
        """
        Append responses for a batch of trials.

        Parameters
        ----------
        responses : List[int]
            Responses corresponding to each (ref, comparison) in the trial batch.
        trial_batch : TrialBatch
            The batch of proposed trials.
        """
        for (ref, comparison), resp in zip(trial_batch.stimuli, responses):
            self.add_trial(ref, comparison, resp)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return refs, comparisons, responses as NumPy arrays."""
        return (
            np.asarray(self.refs),
            np.asarray(self.comparisons),
            np.asarray(self.responses),
        )

    def to_trial_data(self) -> TrialData:
        """Convert this log into the canonical JAX batch (:class:`TrialData`)."""
        refs, comparisons, responses = self.to_numpy()
        return TrialData(
            refs=jnp.asarray(refs),
            comparisons=jnp.asarray(comparisons),
            responses=jnp.asarray(responses),
        )

    @property
    def trials(self) -> list[tuple[Any, Any, int]]:
        """
        Return list of (ref, comparison, response) tuples.

        Returns
        -------
        list[tuple]
            Each element is (ref, comparison, resp)
        """
        return list(zip(self.refs, self.comparisons, self.responses))

    def __len__(self) -> int:
        """Return number of trials."""
        return len(self.refs)

    @classmethod
    def from_arrays(
        cls,
        X: jnp.ndarray | np.ndarray,
        y: jnp.ndarray | np.ndarray,
        *,
        comparisons: jnp.ndarray | np.ndarray | None = None,
    ) -> ResponseData:
        """
        Construct ResponseData from arrays.

        Parameters
        ----------
        X : array, shape (n_trials, 2, input_dim) or (n_trials, input_dim)
            Stimuli. If 3D, second axis is [reference, comparison].
            If 2D, comparisons must be provided separately.
        y : array, shape (n_trials,)
            Responses
        comparisons : array, shape (n_trials, input_dim), optional
            Probe stimuli. Only needed if X is 2D.

        Returns
        -------
        ResponseData
            Data container

        Examples
        --------
        >>> # From paired stimuli
        >>> X = jnp.array([[[0, 0], [1, 0]], [[1, 1], [2, 1]]])
        >>> y = jnp.array([1, 0])
        >>> data = ResponseData.from_arrays(X, y)

        >>> # From separate refs and comparisons
        >>> refs = jnp.array([[0, 0], [1, 1]])
        >>> comparisons = jnp.array([[1, 0], [2, 1]])
        >>> data = ResponseData.from_arrays(refs, y, comparisons=comparisons)
        """
        data = cls()

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 3:
            # X is (n_trials, 2, input_dim)
            refs = X[:, 0, :]
            comparisons_arr = X[:, 1, :]
        elif X.ndim == 2 and comparisons is not None:
            refs = X
            comparisons_arr = np.asarray(comparisons)
        else:
            raise ValueError(
                "X must be shape (n_trials, 2, input_dim) or "
                "(n_trials, input_dim) with comparisons argument"
            )

        for ref, comparison, response in zip(refs, comparisons_arr, y):
            data.add_trial(ref, comparison, int(response))

        return data

    @classmethod
    def from_trial_data(cls, data: TrialData) -> ResponseData:
        """Build a ResponseData log from a :class:`TrialData` batch."""
        refs = np.asarray(data.refs)
        comps = np.asarray(data.comparisons)
        ys = np.asarray(data.responses)
        out = cls()
        for r, c, y in zip(refs, comps, ys):
            out.add_trial(r, c, int(y))
        return out

    def merge(self, other: ResponseData) -> None:
        """
        Merge another dataset into this one (in-place).

        Parameters
        ----------
        other : ResponseData
            Dataset to merge
        """
        self.refs.extend(other.refs)
        self.comparisons.extend(other.comparisons)
        self.responses.extend(other.responses)

    def tail(self, n: int) -> ResponseData:
        """
        Return last n trials as a new ResponseData.

        Parameters
        ----------
        n : int
            Number of trials to keep

        Returns
        -------
        ResponseData
            New dataset with last n trials
        """
        new_data = ResponseData()
        new_data.refs = self.refs[-n:]
        new_data.comparisons = self.comparisons[-n:]
        new_data.responses = self.responses[-n:]
        return new_data

    def copy(self) -> ResponseData:
        """
        Create a deep copy of this dataset.

        Returns
        -------
        ResponseData
            New dataset with copied data
        """
        new_data = ResponseData()
        new_data.refs = list(self.refs)
        new_data.comparisons = list(self.comparisons)
        new_data.responses = list(self.responses)
        return new_data


class TrialBatch:
    """
    Container for a proposed batch of trials

    Attributes
    ----------
    stimuli : List[Tuple[Any, Any]]
        Each trial is a (reference, comparison) tuple.
    """

    def __init__(self, stimuli: list[tuple[Any, Any]]) -> None:
        self.stimuli = list(stimuli)

    @classmethod
    def from_stimuli(cls, pairs: list[tuple[Any, Any]]) -> TrialBatch:
        """
        Construct a TrialBatch from a list of stimuli (ref, comparison) pairs.
        """
        return cls(pairs)
