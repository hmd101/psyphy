"""
psyphy.data
==========

submodule for handling psychophysical experiment data.

Includes:
- dataset: ResponseData, TrialBatch, loaders
- transforms: color/model space conversions
- io: save/load datasets
- published: loaders for published third-party datasets (downloaded on request)
"""

from . import published as published
from .dataset import ResponseData, TrialBatch, TrialData

__all__ = ["ResponseData", "TrialBatch", "TrialData", "published"]
