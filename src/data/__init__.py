"""Package entrypoint for data preparation helpers.

This file re-exports the main data-loading and preprocessing helpers
from ``src.data.loader`` so they can be imported directly from
``src.data``.

Example:
    from src.data import prepare_training_data

Instead of importing from ``src.data.loader`` explicitly, this file lets
``src.data`` act like a cleaner package interface.
"""

from .loader import (
    load_spreadsheet_files,
    load_coordinate_lookup,
    merge_training_data,
    build_intersection_dataset,
    save_dataframe,
    prepare_training_data,
)

# The public API for ``src.data``.
# Only names listed here are exported when someone does:
# ``from src.data import *``
__all__ = [
    "load_spreadsheet_files",
    "load_coordinate_lookup",
    "merge_training_data",
    "build_intersection_dataset",
    "save_dataframe",
    "prepare_training_data",
]
