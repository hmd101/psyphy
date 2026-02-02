"""
Central pytest configuration for this project.

This file is automatically discovered by pytest and is intended for:

- **Fixtures**: reusable objects or setup logic shared across multiple test files.
- **Pytest hooks**: project-wide customizations of pytest behavior (e.g., modifying CLI options, adding markers).

Notes
-----
- Contributors should
  install the package in editable mode (`pip install -e .`) so that imports are resolved
  consistently in local dev and CI environments.
- Keep this file focused on test setup. Do not add application logic here.
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import pytest

# Make `import psyphy` work when running tests without an editable install.
#
# Why this is needed:
# - The project uses a `src/` layout (package is in `src/psyphy`).
# - On some systems (e.g., HPC module/venv setups), developers run `pytest`
#   without first doing `pip install -e .`.
# - Adding `src/` to sys.path makes test collection/imports robust.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)


@pytest.fixture
def chebyshev_inputs():
    """Return evenly spaced test points in [-1, 1] for basis function tests."""
    return jnp.linspace(-1, 1, 5)


@pytest.fixture
def default_degree():
    """Default polynomial degree for testing."""
    return 3
