"""
src/dashboard/components/metrics.py — KPI Metric Cards
========================================================
Helper functions that render Streamlit metric widgets with
consistent styling using st.metric() and custom HTML cards.
"""
from __future__ import annotations

import streamlit as st


_SEV_COLOR = {
    "CRITICAL": "#ff2244", "HIGH": "#ff6b00",
    "MEDIUM":   "#ffd700", "LOW":  "#00cc66", "N/A": "#5a7a9a",
}


def render_top_metrics(summary: dict, proc_time: float) -> None:
    """Render the 6-card KPI row at the top of the Dashboard page."""
    total   = summary.get("total_records", 0)
    attacks = summary.get("attack_count", 0)
    normals = summary.get("normal_count", 0)
    avg_conf= summary.get("average_confidence", 0.0)
    avg_agr = summary.get("average_agreement_score", 0.0)
    sev     = summary.get("severity_distribution", {})
    critical= sev.get("CRITICAL", 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📊 Total Records",    f"{total:,}")
    c2.metric("🚨 Attacks",           f"{attacks:,}",
              delta=f"{summary.get('attack_rate_pct', 0):.1f}% rate")
    c3.metric("✅ Normal",            f"{normals:,}")
    c4.metric("🎯 Avg Confidence",    f"{avg_conf:.3f}")
    c5.metric("🔴 Critical Alerts",  f"{critical:,}")
    c6.metric("🤝 Avg Agreement",     f"{avg_agr:.3f}")


def severity_badge(severity: str) -> str:
    """Return an HTML severity badge string."""
    color = _SEV_COLOR.get(severity, "#5a7a9a")
    text_color = "#000" if severity == "MEDIUM" else "#fff"
    return (
        f"<span style='background:{color};color:{text_color};"
        f"padding:2px 8px;border-radius:4px;font-size:0.75rem;"
        f"font-weight:600;'>{severity}</span>"
    )


def verdict_badge(verdict: str) -> str:
    """Return an HTML verdict badge (ATTACK / NORMAL)."""
    color = "#ff2244" if verdict == "ATTACK" else "#00cc66"
    return (
        f"<span style='background:{color};color:#fff;"
        f"padding:2px 8px;border-radius:4px;font-size:0.75rem;"
        f"font-weight:600;'>{verdict}</span>"
    )


def render_mini_kpis(label_value_pairs: list[tuple[str, str]]) -> None:
    """Render a row of small KPI cards using equal columns."""
    cols = st.columns(len(label_value_pairs))
    for col, (label, value) in zip(cols, label_value_pairs):
        with col:
            st.markdown(
                f"""<div style='background:#0d1b2a;border:1px solid #1e3a5f;
                border-radius:8px;padding:12px 16px;text-align:center;'>
                <div style='font-size:1.4rem;font-weight:700;color:#00d4ff;'>{value}</div>
                <div style='font-size:0.75rem;color:#5a7a9a;margin-top:2px;'>{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )
