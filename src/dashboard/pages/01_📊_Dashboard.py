"""
src/dashboard/pages/01_📊_Dashboard.py — Main SOC Dashboard
=============================================================
• Upload CSV / use default test.parquet
• Run Detection button → calls ThreatDetectionPipeline exactly once
• KPI metrics row
• Four interactive Plotly charts
• Spinner + progress bar + timing
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import pandas as pd
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
    init_session, has_results, get_summary, get_incidents_df,
    store_pipeline_output, KEY_PROC_TIME,
)
from src.dashboard.utils.loaders import (
    run_pipeline, get_pipeline, check_model_status,
)
from src.dashboard.utils.exports import reports_to_json_bytes, summary_to_json_bytes
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.metrics import render_top_metrics
from src.dashboard.components.charts import (
    attack_vs_normal_donut, severity_bar, category_treemap,
    confidence_histogram, severity_pie, model_vote_bar,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard · RTC SIEM", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.stDeployButton{display:none!important}
[data-testid="stToolbar"]{display:none}
[data-testid="metric-container"]{background:linear-gradient(135deg,#0d1b2a,#111f35)!important;
border:1px solid #1e3a5f!important;border-radius:10px!important;}
.stButton>button{background:linear-gradient(90deg,#00d4ff,#0080aa)!important;
color:#000!important;font-weight:700!important;border:none!important;border-radius:8px!important;}
</style>""", unsafe_allow_html=True)

init_session()
render_sidebar()

st.markdown("## 📊 Detection Dashboard")
st.markdown("<hr style='border-color:#1e3a5f;margin:4px 0 16px 0;'>",
            unsafe_allow_html=True)

# ── Input controls ────────────────────────────────────────────────────────────
with st.expander("⚙️ Pipeline Configuration", expanded=not has_results()):
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        uploaded = st.file_uploader(
            "Upload network traffic CSV",
            type=["csv"],
            help="Upload a CSV with the 43 UNSW-NB15 features. "
                 "Leave empty to use the default test.parquet.",
        )
    with col_b:
        max_rows = st.number_input(
            "Max rows (0 = all)",
            min_value=0, max_value=100_000, value=500, step=100,
            help="Limit rows for quick testing. Set 0 to process everything.",
        )
        max_rows_val = int(max_rows) if max_rows > 0 else None
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Detection", use_container_width=True,
                            type="primary")

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn:
    # Determine source
    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
            src_label = uploaded.name
        except Exception as exc:
            st.error(f"❌ Failed to parse CSV: {exc}")
            st.stop()
    else:
        default = _ROOT / "data" / "processed" / "test.parquet"
        if not default.exists():
            st.error("❌ Default test.parquet not found. Please upload a CSV.")
            st.stop()
        source = default
        src_label = str(default)

    # Run with spinner
    progress = st.progress(0, "Initialising pipeline…")
    try:
        with st.spinner("🔍 Running ensemble detection…"):
            progress.progress(20, "Loading models…")
            results, elapsed = run_pipeline(source, max_rows=max_rows_val)
            progress.progress(80, "Enriching with MITRE ATT&CK…")
            store_pipeline_output(results, src_label, elapsed, max_rows_val)
            progress.progress(100, "✅ Complete!")
            time.sleep(0.3)
        progress.empty()
        st.success(
            f"✅ Detection complete · {results['summary']['total_records']:,} records "
            f"processed in **{elapsed:.1f}s** · "
            f"**{results['summary']['attack_count']}** attacks detected"
        )
        st.rerun()
    except Exception as exc:
        progress.empty()
        st.error(f"❌ Pipeline failed: {exc}")
        import logging; logging.getLogger(__name__).exception("Pipeline error")
        st.stop()

# ── Results display ───────────────────────────────────────────────────────────
if not has_results():
    st.markdown("""
    <div style='text-align:center;padding:60px 0;'>
        <div style='font-size:3rem;'>🔍</div>
        <div style='color:#5a7a9a;font-size:1rem;margin-top:12px;'>
            Configure the pipeline above and click <strong style='color:#00d4ff;'>
            Run Detection</strong> to analyse network traffic.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

summary = get_summary()
results = st.session_state["rtc_results"]
df      = get_incidents_df()
proc_t  = st.session_state.get(KEY_PROC_TIME, 0.0)

# KPI row
render_top_metrics(summary, proc_t)
st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(attack_vs_normal_donut(summary), use_container_width=True)
with c2:
    st.plotly_chart(severity_bar(summary), use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(category_treemap(summary), use_container_width=True)
with c4:
    if df is not None and not df.empty:
        st.plotly_chart(confidence_histogram(df), use_container_width=True)
    else:
        st.plotly_chart(severity_pie(summary), use_container_width=True)

# ── Model vote comparison ─────────────────────────────────────────────────────
if df is not None and not df.empty:
    st.plotly_chart(model_vote_bar(df), use_container_width=True)

# ── Quick downloads ────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0;'>",
            unsafe_allow_html=True)
st.markdown("**📥 Export Results**")
dc1, dc2 = st.columns(2)
with dc1:
    st.download_button(
        "⬇️ Download Incident Reports (JSON)",
        data=reports_to_json_bytes(results["reports"]),
        file_name=f"incidents_{results['run_timestamp']}.json",
        mime="application/json",
        use_container_width=True,
    )
with dc2:
    st.download_button(
        "⬇️ Download Summary (JSON)",
        data=summary_to_json_bytes(summary),
        file_name=f"summary_{results['run_timestamp']}.json",
        mime="application/json",
        use_container_width=True,
    )
