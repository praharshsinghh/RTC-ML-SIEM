"""
src/dashboard/utils/session.py — Streamlit Session State Management
====================================================================
Centralised helpers for reading/writing st.session_state so that every
page uses the same key names and there are no KeyError surprises.
"""
from __future__ import annotations

from typing import Any, Optional
import pandas as pd
import streamlit as st


# ── Canonical session-state keys ─────────────────────────────────────────────
KEY_RESULTS        = "rtc_results"        # pipeline.run() return dict
KEY_INCIDENTS_DF   = "rtc_incidents_df"   # pre-built DataFrame of attack rows
KEY_ALL_DF         = "rtc_all_df"         # DataFrame of all rows (attack+normal)
KEY_INPUT_SOURCE   = "rtc_input_source"   # str label for the input file
KEY_PROC_TIME      = "rtc_proc_time"      # float seconds
KEY_MAX_ROWS       = "rtc_max_rows"       # int or None


def init_session() -> None:
    """Initialise all session-state keys to safe defaults on first load."""
    defaults: dict[str, Any] = {
        KEY_RESULTS:      None,
        KEY_INCIDENTS_DF: None,
        KEY_ALL_DF:       None,
        KEY_INPUT_SOURCE: "Not loaded",
        KEY_PROC_TIME:    0.0,
        KEY_MAX_ROWS:     None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def has_results() -> bool:
    """True if a pipeline run has been completed this session."""
    return st.session_state.get(KEY_RESULTS) is not None


def get_results() -> Optional[dict]:
    return st.session_state.get(KEY_RESULTS)


def get_incidents_df() -> Optional[pd.DataFrame]:
    return st.session_state.get(KEY_INCIDENTS_DF)


def get_all_df() -> Optional[pd.DataFrame]:
    return st.session_state.get(KEY_ALL_DF)


def get_summary() -> Optional[dict]:
    r = get_results()
    return r["summary"] if r else None


def get_reports() -> list[dict]:
    r = get_results()
    return r["reports"] if r else []


def store_pipeline_output(
    results: dict,
    input_source: str,
    proc_time: float,
    max_rows: Optional[int],
) -> None:
    """Persist pipeline output to session state and pre-build DataFrames."""
    st.session_state[KEY_RESULTS]      = results
    st.session_state[KEY_INPUT_SOURCE] = input_source
    st.session_state[KEY_PROC_TIME]    = proc_time
    st.session_state[KEY_MAX_ROWS]     = max_rows
    st.session_state[KEY_INCIDENTS_DF] = _build_incidents_df(results["reports"])
    st.session_state[KEY_ALL_DF]       = _build_all_df(results["incidents"])


# ── DataFrame builders ────────────────────────────────────────────────────────

def _build_incidents_df(reports: list[dict]) -> pd.DataFrame:
    """Convert the list of attack-only report dicts into a display DataFrame."""
    rows = []
    for r in reports:
        rows.append({
            "Timestamp":      r["timestamp"][:19].replace("T", " "),
            "Category":       r["prediction"]["attack_category"],
            "Severity":       r["severity"],
            "Confidence":     round(r["prediction"]["confidence"], 4),
            "Agreement":      r["ensemble"]["agreement"],
            "Technique ID":   r["mitre"]["technique_id"],
            "Technique":      r["mitre"]["technique_name"],
            "Tactic":         r["mitre"]["tactic"],
            "Tactic ID":      r["mitre"]["tactic_id"],
            "Map Confidence": r["mitre"]["mapping_confidence"],
            "RF":             r["ensemble"]["rf"],
            "XGB":            r["ensemble"]["xgb"],
            "IForest":        r["ensemble"]["iforest"],
            "AE":             r["ensemble"]["lstm"],
            "RF Prob":        round(r["raw_scores"]["rf_probability"], 4),
            "XGB Prob":       round(r["raw_scores"]["xgb_probability"], 4),
            "IF Score":       round(r["raw_scores"]["iforest_score"], 4),
            "AE Error":       round(r["raw_scores"]["lstm_error"], 4),
            "_report":        r,    # raw dict for JSON viewer
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _build_all_df(incidents: list) -> pd.DataFrame:
    """Build a DataFrame for ALL incidents (NORMAL + ATTACK)."""
    rows = []
    for inc in incidents:
        rows.append({
            "Verdict":    getattr(inc, "verdict", "N/A"),
            "Category":   getattr(inc, "attack_category", "unknown"),
            "Severity":   getattr(inc, "severity", "N/A"),
            "Confidence": round(getattr(inc, "confidence", 0.0), 4),
            "Agreement":  round(getattr(inc, "agreement_score", 0.0), 4),
            "Tactic":     getattr(inc, "tactic", "N/A"),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
