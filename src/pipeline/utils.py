"""
src/pipeline/utils.py — Pipeline Utility Helpers
=================================================
Stateless utility functions shared across pipeline.py and report_generator.py.

Design principle: each function does exactly one thing and is independently
testable. No side-effects; all functions are pure (input → output only).

Responsibilities
----------------
- Loading input data from CSV / Parquet / DataFrame
- Reading the feature_list.txt produced by Phase 1
- Aligning an arbitrary input DataFrame to the exact column set the models
  expect (same order, same columns — missing ones filled with 0)
- Formatting agreement scores to the human-readable '3/4' notation
- Timestamp generation helpers
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of models in the ensemble (RF, XGB, IF, AE)
_N_MODELS = 4


def load_input_data(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    Load input network traffic data from a variety of sources.

    Supported formats
    -----------------
    - str / Path ending in '.csv'     → pd.read_csv()
    - str / Path ending in '.parquet' → pd.read_parquet()
    - pd.DataFrame                    → returned as-is (no copy)

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        Input data source. String paths must end with '.csv' or '.parquet'.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file path does not exist.
    TypeError
        If source is not a str, Path, or DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        logger.info(
            f"[Utils] Input received as DataFrame: "
            f"{len(source):,} rows × {source.shape[1]} columns"
        )
        return source

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        logger.info(f"[Utils] Loaded CSV: {path.name} → {len(df):,} rows")
    elif ext == ".parquet":
        df = pd.read_parquet(path)
        logger.info(f"[Utils] Loaded Parquet: {path.name} → {len(df):,} rows")
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Use '.csv', '.parquet', or pass a DataFrame directly."
        )
    return df


def read_feature_list(feature_list_path: Union[str, Path]) -> list[str]:
    """
    Read the feature names saved by Phase 1 preprocessing.

    The feature_list.txt produced by the DataCleaner contains one feature
    name per line. These names define the exact columns and their ordering
    that the models expect as input.

    Parameters
    ----------
    feature_list_path : str | Path
        Path to feature_list.txt (default: data/processed/feature_list.txt).

    Returns
    -------
    list[str]
        Ordered list of feature column names.

    Raises
    ------
    FileNotFoundError
        If feature_list.txt does not exist.
    """
    path = Path(feature_list_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature list not found: {path}\n"
            "Run Phase 1 (run_preprocessing.py) to regenerate it."
        )

    features = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    logger.info(f"[Utils] Loaded {len(features)} features from {path.name}")
    return features


def align_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Select and reorder columns in df to exactly match feature_cols.

    Columns present in feature_cols but missing from df are filled with 0.
    Columns in df that are not in feature_cols are silently ignored.

    Why is alignment necessary?
    ---------------------------
    The saved models (RF, XGB, IF, AE) were trained on a specific column
    ordering produced by the DataCleaner. If inference input has extra
    columns (e.g., label, attack_cat) or is missing one, passing it
    directly would either raise an error or silently feed wrong data.
    align_features makes the pipeline robust to any DataFrame shape.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame (output of DataCleaner.transform).
    feature_cols : list[str]
        Ordered list of feature names from Phase 1.

    Returns
    -------
    np.ndarray, shape (n_samples, len(feature_cols)), dtype float32
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(
            f"[Utils] {len(missing)} feature(s) missing from input; "
            f"filling with 0: {missing[:10]}{'…' if len(missing) > 10 else ''}"
        )

    aligned = pd.DataFrame(index=df.index)
    for col in feature_cols:
        aligned[col] = df[col] if col in df.columns else 0.0

    return aligned.values.astype(np.float32)


def format_agreement(agreement_score: float, n_models: int = _N_MODELS) -> str:
    """
    Convert a fractional agreement score to the human-readable 'X/N' notation.

    The agreement_score is the sum of MODEL_WEIGHTS for models that agreed
    with the final verdict. Because weights sum to 1.0, we approximate the
    integer count of agreeing models by rounding.

    Examples
    --------
    >>> format_agreement(1.0)   # all 4 models agree
    '4/4'
    >>> format_agreement(0.85)  # RF+XGB+IF agree (0.35+0.35+0.15)
    '3/4'
    >>> format_agreement(0.70)  # RF+XGB agree (0.35+0.35)
    '2/4'

    Parameters
    ----------
    agreement_score : float [0, 1]
    n_models : int
        Total number of models (default 4).

    Returns
    -------
    str, e.g. '3/4'
    """
    agreeing = round(agreement_score * n_models)
    agreeing = max(0, min(n_models, agreeing))
    return f"{agreeing}/{n_models}"


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def extract_model_vote_summary(model_votes: list) -> dict[str, str]:
    """
    Flatten a list of ModelVote (or dicts) into a simple {model: 'ATTACK'/'NORMAL'} map.

    Used by report_generator to populate the 'ensemble' section of each report.

    Parameters
    ----------
    model_votes : list[ModelVote | dict]
        The model_votes list from EnsembleResult or the dicts from EnrichedIncident.

    Returns
    -------
    dict with keys: 'rf', 'xgb', 'iforest', 'lstm'
    """
    name_to_key = {
        "RandomForest":    "rf",
        "XGBoost":         "xgb",
        "IsolationForest": "iforest",
        "DenseAutoencoder": "lstm",   # report schema calls it 'lstm' per spec
    }
    result = {"rf": "N/A", "xgb": "N/A", "iforest": "N/A", "lstm": "N/A"}

    for vote in model_votes:
        # Support both dataclass instances and plain dicts
        if isinstance(vote, dict):
            name = vote.get("model_name", "")
            verdict = vote.get("vote", "N/A")
        else:
            name = getattr(vote, "model_name", "")
            verdict = getattr(vote, "vote", "N/A")

        key = name_to_key.get(name)
        if key:
            result[key] = verdict

    return result


def extract_raw_scores(model_votes: list) -> dict[str, float]:
    """
    Extract raw model scores into the report schema format.

    Parameters
    ----------
    model_votes : list[ModelVote | dict]

    Returns
    -------
    dict with keys: rf_probability, xgb_probability, iforest_score, lstm_error
    """
    name_to_score_key = {
        "RandomForest":    "rf_probability",
        "XGBoost":         "xgb_probability",
        "IsolationForest": "iforest_score",
        "DenseAutoencoder": "lstm_error",
    }
    result: dict[str, float] = {
        "rf_probability": 0.0,
        "xgb_probability": 0.0,
        "iforest_score": 0.0,
        "lstm_error": 0.0,
    }

    for vote in model_votes:
        if isinstance(vote, dict):
            name = vote.get("model_name", "")
            score = float(vote.get("raw_score", 0.0))
        else:
            name = getattr(vote, "model_name", "")
            score = float(getattr(vote, "raw_score", 0.0))

        key = name_to_score_key.get(name)
        if key:
            result[key] = round(score, 6)

    return result
