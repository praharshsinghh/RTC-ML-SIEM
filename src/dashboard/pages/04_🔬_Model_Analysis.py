"""
src/dashboard/pages/04_🔬_Model_Analysis.py — Per-Model Performance Analysis
==============================================================================
• Per-model prediction counts (ATTACK / NORMAL) from incident reports
• Raw score distributions per model
• Agreement analysis
• Ensemble vs ground truth (if label column present)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
    init_session, has_results, get_incidents_df, get_all_df, get_summary,
)
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import render_mini_kpis
from src.dashboard.components.charts import (
    model_vote_bar, confidence_histogram, agreement_histogram, _apply,
)

BG, ACCENT, TEXT, BORDER = "#0a0e1a", "#00d4ff", "#e0e0e0", "#1e3a5f"
MODEL_COLORS = {"Random Forest": "#00d4ff", "XGBoost": "#ff6b00",
                "Isolation Forest": "#bf5fff", "Autoencoder": "#00cc66"}

st.set_page_config(page_title="Model Analysis · RTC SIEM", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.stDeployButton{display:none!important}[data-testid='stToolbar']{display:none}</style>",
            unsafe_allow_html=True)
init_session()
render_sidebar()

st.markdown("## 🔬 Model Analysis")
st.markdown("<hr style='border-color:#1e3a5f;margin:4px 0 16px 0;'>", unsafe_allow_html=True)

if not has_results():
    st.warning("⚠️ Run the detection pipeline from the **📊 Dashboard** page first.")
    st.stop()

df   = get_incidents_df()   # attack rows only
adf  = get_all_df()         # all rows
summary = get_summary()

total    = summary.get("total_records", 0)
attacks  = summary.get("attack_count", 0)
normals  = summary.get("normal_count", 0)

# ── KPI strip ──────────────────────────────────────────────────────────────────
render_mini_kpis([
    ("Total Records",      f"{total:,}"),
    ("Ensemble Attacks",   f"{attacks:,}"),
    ("Ensemble Normal",    f"{normals:,}"),
    ("Attack Rate",        f"{summary.get('attack_rate_pct', 0):.1f}%"),
    ("Avg Confidence",     f"{summary.get('average_confidence', 0):.3f}"),
    ("Avg Agreement",      f"{summary.get('average_agreement_score', 0):.3f}"),
])
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗳️ Vote Breakdown", "📈 Score Distributions", "🤝 Agreement", "📊 Summary Table"
])

with tab1:
    st.markdown("#### Per-Model Vote Breakdown (attack incidents only)")
    if df is not None and not df.empty:
        st.plotly_chart(model_vote_bar(df), use_container_width=True)
    else:
        st.info("No attack incidents to display.")

    st.markdown("#### Vote Detail Table")
    if df is not None and not df.empty:
        vote_cols = ["Category", "Severity", "RF", "XGB", "IForest", "AE", "Agreement", "Confidence"]
        avail = [c for c in vote_cols if c in df.columns]
        st.dataframe(df[avail], use_container_width=True, height=320)
    else:
        st.info("No incidents to display.")

with tab2:
    st.markdown("#### Raw Score Distributions (attack incidents only)")
    if df is not None and not df.empty:
        score_cols = {
            "RF Prob":  ("Random Forest Probability",    "#00d4ff"),
            "XGB Prob": ("XGBoost Probability",          "#ff6b00"),
            "IF Score": ("Isolation Forest Anomaly Score","#bf5fff"),
            "AE Error": ("Autoencoder Reconstruction Error","#00cc66"),
        }
        c1, c2 = st.columns(2)
        cols_list = [c for c in score_cols if c in df.columns]
        for i, col in enumerate(cols_list):
            label, color = score_cols[col]
            fig = go.Figure(go.Histogram(
                x=df[col], nbinsx=25,
                marker=dict(color=color, opacity=0.8),
                hovertemplate=f"{label}: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
            ))
            fig.update_layout(title=label, xaxis_title="Score",
                              yaxis_title="Count",
                              paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
                              font=dict(color=TEXT), margin=dict(l=16,r=16,t=40,b=16))
            (c1 if i % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attack incidents to display.")

with tab3:
    st.markdown("#### Model Agreement Analysis")
    if df is not None and not df.empty:
        st.plotly_chart(agreement_histogram(df), use_container_width=True)

        # Agreement breakdown table
        agr_counts = df["Agreement"].value_counts().reset_index()
        agr_counts.columns = ["Agreement", "Count"]
        agr_counts["Percentage"] = (agr_counts["Count"] / len(df) * 100).round(1)
        st.dataframe(agr_counts, use_container_width=True, height=200)
    else:
        st.info("No attack incidents to display.")

with tab4:
    st.markdown("#### Ensemble vs Individual Model Summary")
    # Build summary table from session data
    if df is not None and not df.empty:
        rows = []
        for col, name in [("RF","Random Forest"),("XGB","XGBoost"),
                           ("IForest","Isolation Forest"),("AE","Autoencoder")]:
            if col not in df.columns:
                continue
            a = int((df[col] == "ATTACK").sum())
            n = int((df[col] == "NORMAL").sum())
            rows.append({"Model": name, "Attack Votes": a, "Normal Votes": n,
                         "Attack %": round(a / len(df) * 100, 1)})
        rows.append({"Model": "🔵 Ensemble", "Attack Votes": attacks,
                     "Normal Votes": normals,
                     "Attack %": round(attacks / total * 100, 1) if total else 0})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)
    else:
        st.info("No attack incidents to display.")
