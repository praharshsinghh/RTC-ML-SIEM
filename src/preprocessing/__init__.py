"""
Preprocessing Package — Public API
====================================
Import from here so the rest of the codebase doesn't depend on internal module paths.
"""

from .loader import (
    auto_load,
    load_raw_partitions,
    load_kaggle_splits,
    validate_schema,
    COLUMN_NAMES,
    ATTACK_CATEGORIES,
)
from .cleaner import DataCleaner, LABEL_COL, ATTACK_CAT_COL, CAT_FEATURES
from .splitter import (
    split_dataset,
    get_normal_only,
    get_feature_label_arrays,
    save_splits,
    load_splits,
)

__all__ = [
    "auto_load",
    "load_raw_partitions",
    "load_kaggle_splits",
    "validate_schema",
    "COLUMN_NAMES",
    "ATTACK_CATEGORIES",
    "DataCleaner",
    "LABEL_COL",
    "ATTACK_CAT_COL",
    "CAT_FEATURES",
    "split_dataset",
    "get_normal_only",
    "get_feature_label_arrays",
    "save_splits",
    "load_splits",
]
