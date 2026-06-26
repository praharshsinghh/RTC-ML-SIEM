"""
src/dashboard/pages/03_🎯_MITRE.py — MITRE ATT&CK Intelligence Page
=====================================================================
• Tactic and technique frequency charts
• Searchable technique table with descriptions
• Clickable ATT&CK reference links
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

def _root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "run_pipeline.py").exists():
            return parent
    return p.parents[4]

_ROOT = _root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.utils.session import init_session, has_results, get_incidents_df
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.charts import mitre_tactic_bar, mitre_technique_bar
from src.dashboard.components.metrics import render_mini_kpis

st.set_page_config(page_title="MITRE · RTC SIEM", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.stDeployButton{display:none!important}[data-testid='stToolbar']{display:none}</style>",
            unsafe_allow_html=True)
init_session()
render_sidebar()

st.markdown("## 🎯 MITRE ATT&CK® Intelligence")
st.markdown("<hr style='border-color:#1e3a5f;margin:4px 0 16px 0;'>", unsafe_allow_html=True)

if not has_results():
    st.warning("⚠️ Run the detection pipeline from the **📊 Dashboard** page first.")
    st.stop()

df = get_incidents_df()
if df is None or df.empty:
    st.success("✅ No attacks detected — no MITRE data to display.")
    st.stop()

# ── KPI strip ──────────────────────────────────────────────────────────────────
render_mini_kpis([
    ("Unique Techniques",  str(df["Technique ID"].nunique())),
    ("Unique Tactics",     str(df["Tactic"].nunique())),
    ("Most Common Tactic", str(df["Tactic"].mode().iloc[0]) if not df["Tactic"].empty else "—"),
    ("Top Technique",      str(df["Technique ID"].mode().iloc[0]) if not df["Technique ID"].empty else "—"),
    ("Mapped Incidents",   str(len(df[df["Technique ID"] != "N/A"]))),
])
st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(mitre_tactic_bar(df), use_container_width=True)
with c2:
    top_n = st.slider("Top N techniques", 5, 20, 10, key="top_n_tech")
    st.plotly_chart(mitre_technique_bar(df, top_n=top_n), use_container_width=True)

# ── Searchable technique table ─────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)
st.markdown("### 📋 Technique Reference Table")

search = st.text_input("🔍 Search techniques / tactics", placeholder="e.g. T1190 or Initial Access")

tech_df = (
    df.groupby(["Technique ID", "Technique", "Tactic", "Tactic ID", "Map Confidence"])
    .size()
    .reset_index(name="Incidents")
    .sort_values("Incidents", ascending=False)
)

if search:
    mask = (
        tech_df["Technique ID"].str.contains(search, case=False, na=False) |
        tech_df["Technique"].str.contains(search, case=False, na=False) |
        tech_df["Tactic"].str.contains(search, case=False, na=False)
    )
    tech_df = tech_df[mask]

# Add ATT&CK URL
tech_df["ATT&CK Link"] = tech_df["Technique ID"].apply(
    lambda t: f"https://attack.mitre.org/techniques/{t}/" if t and t != "N/A" else ""
)

st.dataframe(
    tech_df[["Technique ID", "Technique", "Tactic", "Tactic ID", "Map Confidence", "Incidents", "ATT&CK Link"]],
    use_container_width=True,
    column_config={
        "Incidents": st.column_config.ProgressColumn(
            "Incidents", min_value=0, max_value=int(tech_df["Incidents"].max()),
            format="%d"),
        "ATT&CK Link": st.column_config.LinkColumn("ATT&CK Reference"),
    },
    height=400,
)

# ── Tactic breakdown table ─────────────────────────────────────────────────────
st.markdown("### 🗺️ Tactic Breakdown")
tac_df = (
    df.groupby(["Tactic", "Tactic ID"])
    .agg(Incidents=("Technique ID", "count"),
         Unique_Techniques=("Technique ID", "nunique"))
    .reset_index()
    .sort_values("Incidents", ascending=False)
)
st.dataframe(tac_df, use_container_width=True, height=300)
