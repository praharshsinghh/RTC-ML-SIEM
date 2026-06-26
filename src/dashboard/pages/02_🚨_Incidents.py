"""
src/dashboard/pages/02_🚨_Incidents.py — Incident Investigation Table
=======================================================================
• Filterable / sortable incident table (attack rows only)
• Per-incident detail expander with raw JSON, MITRE info, model votes
• CSV and JSON download buttons
"""
from __future__ import annotations

import sys, json
from pathlib import Path

import pandas as pd
import streamlit as st

def _root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "run_pipeline.py").exists():
            return parent
    return p.parents[4]

_ROOT = _root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.utils.session import init_session, has_results, get_incidents_df, get_results
from src.dashboard.utils.exports import filtered_df_to_csv_bytes, reports_to_json_bytes
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import severity_badge, verdict_badge, render_mini_kpis
from src.dashboard.components.charts import confidence_by_severity, agreement_histogram

st.set_page_config(page_title="Incidents · RTC SIEM", page_icon="🚨",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.stDeployButton{display:none!important}
[data-testid="stToolbar"]{display:none}
.stButton>button{background:linear-gradient(90deg,#00d4ff,#0080aa)!important;
color:#000!important;font-weight:700!important;border:none!important;border-radius:8px!important;}
</style>""", unsafe_allow_html=True)

init_session()
render_sidebar()

st.markdown("## 🚨 Incident Investigation")
st.markdown("<hr style='border-color:#1e3a5f;margin:4px 0 16px 0;'>", unsafe_allow_html=True)

if not has_results():
    st.warning("⚠️ No pipeline results yet. Run the detection pipeline from the **📊 Dashboard** page first.")
    st.stop()

df = get_incidents_df().copy()
results = get_results()

if df.empty:
    st.success("✅ No attacks detected in this run — all traffic classified as normal.")
    st.stop()

# ── Mini KPI strip ─────────────────────────────────────────────────────────────
sev_counts = df["Severity"].value_counts()
render_mini_kpis([
    ("Total Incidents",  str(len(df))),
    ("Critical",         str(sev_counts.get("CRITICAL", 0))),
    ("High",             str(sev_counts.get("HIGH", 0))),
    ("Medium",           str(sev_counts.get("MEDIUM", 0))),
    ("Low",              str(sev_counts.get("LOW", 0))),
    ("Attack Categories", str(df["Category"].nunique())),
])
st.markdown("<br>", unsafe_allow_html=True)

# ── Filters ────────────────────────────────────────────────────────────────────
with st.expander("🔽 Filters", expanded=True):
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sev_opts = ["All"] + sorted(df["Severity"].unique().tolist())
        sev_filter = st.selectbox("Severity", sev_opts)
    with fc2:
        cat_opts = ["All"] + sorted(df["Category"].unique().tolist())
        cat_filter = st.selectbox("Attack Category", cat_opts)
    with fc3:
        tech_opts = ["All"] + sorted(df["Technique ID"].unique().tolist())
        tech_filter = st.selectbox("Technique ID", tech_opts)
    with fc4:
        conf_min = float(df["Confidence"].min())
        conf_max = float(df["Confidence"].max())
        if conf_min < conf_max:
            conf_range = st.slider("Confidence range", conf_min, conf_max,
                                   (conf_min, conf_max), step=0.05)
        else:
            conf_range = (conf_min, conf_max)

# Apply filters
fdf = df.copy()
if sev_filter  != "All": fdf = fdf[fdf["Severity"]    == sev_filter]
if cat_filter  != "All": fdf = fdf[fdf["Category"]    == cat_filter]
if tech_filter != "All": fdf = fdf[fdf["Technique ID"]== tech_filter]
fdf = fdf[(fdf["Confidence"] >= conf_range[0]) & (fdf["Confidence"] <= conf_range[1])]

st.markdown(f"**Showing {len(fdf):,} / {len(df):,} incidents**")

# ── Table ──────────────────────────────────────────────────────────────────────
display_cols = ["Timestamp", "Category", "Severity", "Confidence",
                "Agreement", "Technique ID", "Technique", "Tactic"]
st.dataframe(
    fdf[display_cols],
    use_container_width=True,
    height=380,
    column_config={
        "Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=1, format="%.3f"),
        "Severity": st.column_config.TextColumn("Severity"),
    },
)

# ── Downloads ─────────────────────────────────────────────────────────────────
dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("⬇️ CSV (filtered)",
        data=filtered_df_to_csv_bytes(fdf),
        file_name="incidents_filtered.csv", mime="text/csv",
        use_container_width=True)
with dl2:
    st.download_button("⬇️ CSV (all)",
        data=filtered_df_to_csv_bytes(df),
        file_name="incidents_all.csv", mime="text/csv",
        use_container_width=True)
with dl3:
    st.download_button("⬇️ JSON (all reports)",
        data=reports_to_json_bytes(results["reports"]),
        file_name="incidents.json", mime="application/json",
        use_container_width=True)

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)
cc1, cc2 = st.columns(2)
with cc1:
    st.plotly_chart(confidence_by_severity(fdf), use_container_width=True)
with cc2:
    st.plotly_chart(agreement_histogram(fdf), use_container_width=True)

# ── Per-incident detail viewer ─────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>", unsafe_allow_html=True)
st.markdown("### 🔍 Incident Detail Viewer")

if len(fdf) > 0:
    row_idx = st.selectbox(
        "Select incident (row index)",
        options=range(len(fdf)),
        format_func=lambda i: (
            f"[{fdf.iloc[i]['Severity']}] {fdf.iloc[i]['Category']} "
            f"| Conf: {fdf.iloc[i]['Confidence']:.3f} "
            f"| {fdf.iloc[i]['Technique ID']}"
        ),
    )
    row = fdf.iloc[row_idx]
    raw = row.get("_report", {})

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Prediction**")
        st.markdown(f"Category: `{row['Category']}`")
        st.markdown(f"Severity: {severity_badge(row['Severity'])}", unsafe_allow_html=True)
        st.markdown(f"Confidence: `{row['Confidence']:.4f}`")
        st.markdown(f"Agreement: `{row['Agreement']}`")

        st.markdown("**Model Votes**")
        for m, col in [("RF","RF"), ("XGB","XGB"), ("IF","IForest"), ("AE","AE")]:
            v = row.get(col, "N/A")
            st.markdown(f"{verdict_badge(v)} {m}", unsafe_allow_html=True)

    with c_right:
        st.markdown("**MITRE ATT&CK**")
        st.markdown(f"Technique: `{row['Technique ID']}` — {row['Technique']}")
        st.markdown(f"Tactic: `{row['Tactic ID']}` — {row['Tactic']}")
        st.markdown(f"Map Confidence: `{row.get('Map Confidence', 'N/A')}`")

        st.markdown("**Raw Scores**")
        st.markdown(f"RF Probability: `{row.get('RF Prob', 'N/A')}`")
        st.markdown(f"XGB Probability: `{row.get('XGB Prob', 'N/A')}`")
        st.markdown(f"IF Anomaly Score: `{row.get('IF Score', 'N/A')}`")
        st.markdown(f"AE Reconstruction Error: `{row.get('AE Error', 'N/A')}`")

    with st.expander("📄 Raw JSON Report"):
        if raw:
            raw_copy = {k: v for k, v in raw.items()}
            st.code(json.dumps(raw_copy, indent=2), language="json")
        else:
            st.info("Raw JSON not available for this row.")
