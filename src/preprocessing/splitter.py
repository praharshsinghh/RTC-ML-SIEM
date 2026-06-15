"""
Train/Test Splitter
====================
Handles two split scenarios:
  A. Raw partitions: random stratified split (80/20 by attack_cat)
  B. Kaggle pre-split: already split, just validate & return

Also produces the "normal-only" training subset used by the unsupervised models
(Isolation Forest, LSTM-Autoencoder). These models only see normal traffic
during training — they learn what "normal" looks like and flag deviations.

Design note on stratification:
  We stratify on attack_cat (not just label) to ensure each attack category
  is proportionally represented in both splits. Without stratification, rare
  categories like 'Worms' (only ~44 samples in UNSW-NB15) could end up
  entirely in train or entirely in test, making per-category metrics meaningless.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .cleaner import LABEL_COL, ATTACK_CAT_COL

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_col: str = ATTACK_CAT_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split on the combined (already cleaned) DataFrame.

    Parameters
    ----------
    df : cleaned DataFrame (output of DataCleaner.fit_transform)
    test_size : fraction for test set (default 0.2 = 80/20 split)
    random_state : seed for reproducibility
    stratify_col : column to stratify on (default: attack_cat)

    Returns
    -------
    (train_df, test_df)
    """
    stratify = df[stratify_col] if stratify_col in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(
        f"Split: train={len(train_df):,} rows, test={len(test_df):,} rows "
        f"(stratified on '{stratify_col}')"
    )
    _log_class_distribution(train_df, "Train")
    _log_class_distribution(test_df, "Test")
    return train_df, test_df


def get_normal_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only normal (non-attack) rows.
    Used to train unsupervised models (Isolation Forest, LSTM-Autoencoder).

    Design note:
      We use LABEL_COL == 0 (binary label) rather than ATTACK_CAT_COL == 'Normal'
      because in some versions of the dataset there are edge cases where these
      are inconsistent. The binary label is the ground truth.
    """
    if LABEL_COL not in df.columns:
        raise ValueError(f"Column '{LABEL_COL}' not found. Run DataCleaner first.")

    normal_df = df[df[LABEL_COL] == 0].copy()
    total = len(df)
    normal_count = len(normal_df)
    logger.info(
        f"Normal-only subset: {normal_count:,} rows "
        f"({100 * normal_count / total:.1f}% of {total:,} total)"
    )
    return normal_df


def get_feature_label_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    multiclass: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (X, y) arrays ready for sklearn/torch model training.

    Parameters
    ----------
    feature_cols : list of feature column names (from DataCleaner.feature_cols)
    multiclass   : if True, use attack_cat_encoded as y; else use binary label

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    """
    X = df[feature_cols].values.astype(np.float32)
    y_col = "attack_cat_encoded" if multiclass else LABEL_COL
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in DataFrame.")
    y = df[y_col].values
    return X, y


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    normal_only_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> None:
    """Save processed splits to parquet for fast reloading."""
    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    normal_only_df.to_parquet(out_dir / "train_normal_only.parquet", index=False)
    logger.info(f"Splits saved to {out_dir}")


def load_splits(
    out_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved splits from parquet."""
    out_dir = out_dir or PROCESSED_DIR
    train_df = pd.read_parquet(out_dir / "train.parquet")
    test_df = pd.read_parquet(out_dir / "test.parquet")
    normal_df = pd.read_parquet(out_dir / "train_normal_only.parquet")
    return train_df, test_df, normal_df


def _log_class_distribution(df: pd.DataFrame, name: str) -> None:
    if ATTACK_CAT_COL in df.columns:
        dist = df[ATTACK_CAT_COL].value_counts()
        logger.info(f"{name} attack_cat distribution:\n{dist.to_string()}")
