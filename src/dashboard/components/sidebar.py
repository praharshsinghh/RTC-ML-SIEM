"""
src/dashboard/components/sidebar.py — Sidebar Component
=========================================================
Renders the persistent sidebar present on every dashboard page.
Import and call render_sidebar() at the top of each page.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── path bootstrap ─────────────────────────────────────────────────────────────
def _root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "run_pipeline.py").exists():
            return parent
    return p.parents[4]

_ROOT = _root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.utils.session import (
    has_results, get_summary,
    KEY_INPUT_SOURCE, KEY_PROC_TIME,
)
from src.dashboard.utils.loaders import check_model_status

# ── Severity badge colours ─────────────────────────────────────────────────────
_SEV_COLOR = {"CRITICAL": "#ff2244", "HIGH": "#ff6b00", "MEDIUM": "#ffd700", "LOW": "#00cc66"}


def render_sidebar() -> None:
    """Render the left sidebar. Call once per page."""
    with st.sidebar:
        # ── Logo / branding ───────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 8px 0 4px 0;'>
            <span style='font-size:2.4rem;'>🛡️</span>
            <div style='font-size:1.2rem; font-weight:700; color:#00d4ff; letter-spacing:1px;'>
                RTC SIEM
            </div>
            <div style='font-size:0.72rem; color:#5a7a9a; margin-top:2px;'>
                ML-Driven Threat Detection
            </div>
        </div>
        <hr style='border-color:#1e3a5f; margin:10px 0;'>
        """, unsafe_allow_html=True)

        # ── Model status ──────────────────────────────────────────────────────
        st.markdown("**🔧 Model Status**")
        status = check_model_status()
        for name, ok in status.items():
            icon = "🟢" if ok else "🔴"
            st.markdown(
                f"<small>{icon} {name}</small>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#1e3a5f;margin:10px 0;'>",
                    unsafe_allow_html=True)

        # ── Pipeline run status ───────────────────────────────────────────────
        st.markdown("**📡 Pipeline Status**")
        if has_results():
            summary = get_summary()
            proc_time = st.session_state.get(KEY_PROC_TIME, 0.0)
            src = st.session_state.get(KEY_INPUT_SOURCE, "—")
            st.success("✅ Results available")
            st.markdown(f"<small>📂 {Path(src).name if src != 'DataFrame' else 'DataFrame'}</small>",
                        unsafe_allow_html=True)
            st.markdown(f"<small>⏱️ {proc_time:.1f}s</small>",
                        unsafe_allow_html=True)
            st.markdown(f"<small>📊 {summary['total_records']:,} records</small>",
                        unsafe_allow_html=True)

            # Mini severity badges
            sev = summary.get("severity_distribution", {})
            badges = " ".join(
                f"<span style='background:{_SEV_COLOR[s]};color:#fff;"
                f"padding:1px 6px;border-radius:4px;font-size:0.65rem;"
                f"margin:1px;'>{s[0]} {sev[s]}</span>"
                for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if sev.get(s, 0) > 0
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("⏳ Awaiting pipeline run")

        st.markdown("<hr style='border-color:#1e3a5f;margin:10px 0;'>",
                    unsafe_allow_html=True)

        # ── GitHub / project info ──────────────────────────────────────────────
        st.markdown("""
        <div style='font-size:0.72rem; color:#5a7a9a; text-align:center;'>
            <div>🐙 <a href='https://github.com' style='color:#00d4ff;'>GitHub Repository</a></div>
            <div style='margin-top:4px;'>UNSW-NB15 Dataset</div>
            <div>MITRE ATT&CK® v14</div>
        </div>
        """, unsafe_allow_html=True)
