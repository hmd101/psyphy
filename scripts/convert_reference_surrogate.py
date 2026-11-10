#!/usr/bin/env python3
"""Re-save a reference surrogate weight file with psyphy's metadata schema.

We ran it once to produce the weight file psyphy ships,
``src/psyphy/model/likelihood/weights/oddity_2d_64x2.npz``. The weights are
Philip Greengard's, trained outside psyphy. His files hold only the network and
its standardization (``W_i``, ``b_i``, ``n_layers``, ``input_mean``,
``input_std``), and ``NeuralSurrogate.from_npz`` refuses a file without
metadata.

What we write
-------------
``task``, ``input_dim``, ``feature_map``, ``feature_map_version`` and ``w_ref``
follow from how the network was trained: on the whitened, rotation-reduced 4-D
features of the 3-AFC oddity task.

``domain_lo``/``domain_hi`` are left out, since we do not have the training set.
``label_mc_samples`` is written only when passed on the command line, and the
validation errors are not written.

Usage
-----
    python scripts/convert_reference_surrogate.py \
        --src path/to/reference.npz \
        --dest src/psyphy/model/likelihood/weights/oddity_2d_64x2.npz
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
    input_dim = n_features // 2  # 2 * input_dim features, see likelihood.features

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
        # no domain_lo/domain_hi, see the module docstring
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
        help="MC samples behind each training label, if known. Not written when not given.",
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
