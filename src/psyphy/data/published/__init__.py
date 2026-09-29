"""
psyphy.data.published
=====================

Loaders for **published third-party psychophysical datasets**.

These are datasets released alongside papers, used to validate psyphy against
results obtained independently. This is distinct from :mod:`psyphy.data.dataset`,
which provides the containers your *own* experiment data lives in.

psyphy ships no data. Each submodule exposes a ``fetch`` function that downloads
on request into a user cache directory (see ``hong2025.default_data_dir``), and
loaders that return psyphy objects such as
:class:`~psyphy.data.dataset.TrialData`.

Available datasets
------------------
- :mod:`~psyphy.data.published.hong2025` — human color discrimination,
  8 observers, Wishart process fits. Hong et al. (2026), eLife 14:RP108943.

Examples
--------
>>> from psyphy.data.published import hong2025
>>> paths = hong2025.fetch(subject=1)  # doctest: +SKIP
>>> data = hong2025.load_trials(paths["trials"])  # doctest: +SKIP
"""

from . import hong2025 as hong2025

__all__ = ["hong2025"]
