#!/usr/bin/env python3
"""Re-save a reference surrogate weight file with psyphy's metadata schema.

One-off, but version-controlled rather than run-and-forgotten: the provenance of
a shipped weight file should be reproducible, and the *script* is the artefact
worth reviewing. The resulting ``.npz`` is committed as package data.

Context
-------
The surrogate weights come from Philip Greengard's ``nn_march2026`` drop, whose
files hold nine arrays and nothing else: ``W_i``, ``b_i``, ``n_layers``,
``input_mean``, ``input_std``. That is a complete description of a *network* but
not of a *surrogate*, because it does not say which task, dimension or feature
map the network was fitted for. A 2-D oddity network and a 2-D two-interval
network have identical shapes, so nothing stops the wrong one being loaded, and
the result is a plausible probability rather than an error.

``psyphy.model.likelihood.neural.from_npz`` therefore refuses files without
metadata. This script is what makes the reference weights loadable again.

What is asserted here, and what is not
--------------------------------------
``task``, ``input_dim``, ``feature_map``, ``feature_map_version`` and ``w_ref``
are known from how the network was trained: on the Sbar-whitened,
rotation-reduced 4-D features of the 3-AFC oddity task.

``domain_lo``/``domain_hi`` are deliberately **not** written. We do not have the
training set, and a box inferred from ``input_mean ± k·input_std`` would describe
the training distribution rather than its support while looking exactly like a
measured one. Absent means unknown; see the design note, §6.1 of the PR plan.

``label_mc_samples`` and the validation errors are likewise omitted unless given
on the command line, because they are properties of a training run we did not
perform.

Usage
-----
    python scripts/convert_reference_surrogate.py \

"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

from psyphy.model.likelihood.features import ODDITY_W_REF
from psyphy.model.likelihood.neural import SCHEMA_VERSION, save_surrogate_npz

#: Identifier for the feature map these weights expect. Must match what
#: NeuralOddityTask declares, and is bumped whenever the feature definition
#: changes in any way, including a change that leaves the shape the same.
FEATURE_MAP = "whitened_rotation_reduced"
FEATURE_MAP_VERSION = 1


def convert(src: Path, dest: Path, *, label_mc_samples: int | None = None) -> Path:
    with np.load(src, allow_pickle=False) as d:
        n_layers = int(d["n_layers"])
        params = [(d[f"W_{i}"], d[f"b_{i}"]) for i in range(n_layers)]
        input_mean, input_std = d["input_mean"], d["input_std"]

    n_features = int(params[0][0].shape[0])
    input_dim = n_features // 2  # the 2d counting rule; see the design note §17.2

    meta = {
        "task": "oddity",
        "input_dim": input_dim,
        "n_features": n_features,
        "feature_map": FEATURE_MAP,
        "feature_map_version": FEATURE_MAP_VERSION,
        "w_ref": float(ODDITY_W_REF),
        "trained_utc": _dt.datetime.fromtimestamp(src.stat().st_mtime, _dt.timezone.utc)
        .date()
        .isoformat(),
        "generator_rev": f"nn_march2026/{src.name}",
        # NOTE: no domain_lo/domain_hi on purpose. See the module docstring.
    }
    if label_mc_samples is not None:
        meta["label_mc_samples"] = int(label_mc_samples)

    dest.parent.mkdir(parents=True, exist_ok=True)
    save_surrogate_npz(
        dest,
        params=params,
        input_mean=input_mean,
        input_std=input_std,
        meta=meta,
    )
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="reference .npz to convert")
    ap.add_argument(
        "--dest", type=Path, required=True, help="where to write the result"
    )
    ap.add_argument(
        "--label-mc-samples",
        type=int,
        default=None,
        help="MC samples behind each training label, if known. Omitted when not given, "
        "rather than guessed.",
    )
    args = ap.parse_args()

    out = convert(args.src, args.dest, label_mc_samples=args.label_mc_samples)
    size_in, size_out = args.src.stat().st_size, out.stat().st_size
    print(f"schema v{SCHEMA_VERSION}: {args.src.name} -> {out}")
    print(f"  {size_in:,} bytes -> {size_out:,} bytes")

    # Read it back through the real loader, so the script cannot claim success
    # for a file psyphy would refuse.
    from psyphy.model.likelihood.neural import NeuralSurrogate

    with np.load(out, allow_pickle=False) as d:
        input_dim = int(d["input_dim"])

    s = NeuralSurrogate.from_npz(
        out,
        require={
            "task": "oddity",
            "input_dim": input_dim,
            "feature_map": FEATURE_MAP,
            "feature_map_version": FEATURE_MAP_VERSION,
            "w_ref": float(ODDITY_W_REF),
        },
    )
    print(
        f"  verified: {s.meta['task']} {s.meta['input_dim']}-D, "
        f"{s.meta['n_features']} features, domain={'recorded' if s.domain else 'unknown'}"
    )


if __name__ == "__main__":
    main()
