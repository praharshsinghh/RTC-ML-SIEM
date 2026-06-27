"""
src/dashboard/utils/loaders.py — Cached Resource Loaders
=========================================================
All expensive objects (pipeline, STIX data, feature list) are loaded exactly
once per Streamlit server process using st.cache_resource.

Design
------
- st.cache_resource  → singletons shared across all sessions (models, pipeline)
- st.cache_data      → serialisable results cached per call signature

These loaders are the ONLY place in the dashboard that touch the ML backend.
Pages and components call these functions; they never import model classes directly.
"""
from __future__ import annotations

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ── Ensure project root is on sys.path ────────────────────────────────────────
def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "run_pipeline.py").exists():
            return parent
    return p.parents[3]

_ROOT = _project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_TEST_PARQUET = _ROOT / "data" / "processed" / "test.parquet"
DEFAULT_FEATURE_LIST = _ROOT / "data" / "processed" / "feature_list.txt"


@st.cache_resource(show_spinner="Loading detection pipeline...")
def get_pipeline():
    """
    Load and cache the ThreatDetectionPipeline singleton.

    Uses st.cache_resource so models are loaded ONCE per Streamlit server
    process regardless of how many times the page reruns or users navigate.
    stix_preload=False speeds startup; STIX data is loaded on first enrich call
    and cached inside the pipeline's _enricher attribute.
    """
    from src.pipeline import ThreatDetectionPipeline
    pipeline = ThreatDetectionPipeline(stix_preload=False)
    return pipeline


@st.cache_data(show_spinner="Loading dataset preview...")
def load_parquet_preview(path: str, n_rows: int = 5000) -> pd.DataFrame:
    """Return the first n_rows of a parquet file for display purposes only."""
    return pd.read_parquet(path).head(n_rows)


@st.cache_data(show_spinner=False)
def get_feature_list() -> list[str]:
    """Return the 43-feature list from Phase 1."""
    if not DEFAULT_FEATURE_LIST.exists():
        return []
    return [l.strip() for l in DEFAULT_FEATURE_LIST.read_text().splitlines() if l.strip()]


def run_pipeline(
    source: str | Path | pd.DataFrame,
    max_rows: Optional[int],
) -> tuple[dict, float]:
    """
    Execute the cached pipeline on the given source.

    Returns
    -------
    results : dict   — pipeline.run() return value
    elapsed : float  — wall-clock seconds
    """
    pipeline = get_pipeline()
    t0 = time.perf_counter()
    results = pipeline.run(source=source, max_rows=max_rows)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def check_model_status() -> dict[str, bool]:
    """
    Check which model files exist on disk.
    Returns a dict {model_name: exists}.
    """
    checks = {
        "Random Forest":    _ROOT / "data" / "models" / "rf_detector.joblib",
        "XGBoost":          _ROOT / "data" / "models" / "xgb_detector.joblib",
        "Isolation Forest": _ROOT / "models"           / "isolation_forest.joblib",
        "Autoencoder":      _ROOT / "models"           / "autoencoder.pt",
        "DataCleaner":      _ROOT / "data" / "processed" / "cleaner.joblib",
    }
    return {name: path.exists() for name, path in checks.items()}
