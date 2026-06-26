"""
src/dashboard/utils/exports.py — Download Helpers
===================================================
Pure functions that convert pipeline output to bytes for st.download_button.
No Streamlit calls here — keeps the module independently testable.
"""
from __future__ import annotations

import io
import json
import pandas as pd


def reports_to_json_bytes(reports: list[dict]) -> bytes:
    """Serialise the full incident report list to pretty-printed JSON bytes."""
    return json.dumps(reports, indent=2, ensure_ascii=False).encode("utf-8")


def summary_to_json_bytes(summary: dict) -> bytes:
    """Serialise the session summary dict to JSON bytes."""
    return json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")


def df_to_csv_bytes(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes, optionally dropping internal cols."""
    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def filtered_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export the current (filtered) incident DataFrame."""
    return df_to_csv_bytes(df, exclude_cols=["_report"])
