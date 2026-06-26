"""
src/dashboard/pages/05_ℹ️_About.py — Project Information Page
==============================================================
• Architecture overview
• Phase descriptions
• Tech stack
• Data sources
"""
from __future__ import annotations

import sys
from pathlib import Path
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

from src.dashboard.utils.session import init_session
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.utils.loaders import get_feature_list, check_model_status

st.set_page_config(page_title="About · RTC SIEM", page_icon="ℹ️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.stDeployButton{display:none!important}[data-testid='stToolbar']{display:none}</style>",
            unsafe_allow_html=True)
init_session()
render_sidebar()

st.markdown("## ℹ️ About — RTC SIEM Project")
st.markdown("<hr style='border-color:#1e3a5f;margin:4px 0 16px 0;'>", unsafe_allow_html=True)

# ── Project Overview ───────────────────────────────────────────────────────────
st.markdown("""
**RTC (Real-Time Cyber Threat Detection)** is an end-to-end machine learning pipeline that
detects network-layer attacks, classifies them by category, maps them to MITRE ATT&CK® techniques,
and presents actionable intelligence through this SOC-ready dashboard.

> Built on the **UNSW-NB15** dataset — 2.5M real network flow records with 49 features and
> 10 attack categories, collected by the Australian Centre for Cyber Security (ACCS).
""")

# ── Architecture pipeline visual ───────────────────────────────────────────────
st.markdown("### 🏗️ System Architecture")
st.markdown("""
<div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:20px;
font-family:monospace;font-size:0.82rem;color:#e0e0e0;line-height:2;'>
📥 <span style='color:#00d4ff;'>Input</span> (CSV / Parquet / DataFrame)
&nbsp;&nbsp;&nbsp;↓
⚙️ <span style='color:#00d4ff;'>Phase 1: Preprocessing</span> — DataCleaner · LabelEncoder · StandardScaler · Feature Selection
&nbsp;&nbsp;&nbsp;↓
🌲 <span style='color:#ff6b00;'>Phase 2: Supervised Models</span> — Random Forest · XGBoost (multiclass, balanced weights)
&nbsp;&nbsp;&nbsp;↓
🔍 <span style='color:#bf5fff;'>Phase 3: Unsupervised Models</span> — Isolation Forest · Dense Autoencoder (normal-only training)
&nbsp;&nbsp;&nbsp;↓
🗳️ <span style='color:#ffd700;'>Phase 4: Ensemble</span> — Weighted vote (RF=0.35, XGB=0.35, IF=0.15, AE=0.15) · Confidence · Severity
&nbsp;&nbsp;&nbsp;↓
🎯 <span style='color:#00cc66;'>Phase 5: MITRE ATT&amp;CK</span> — STIX 2.1 enrichment · Technique/Tactic mapping · Fallback to T1190
&nbsp;&nbsp;&nbsp;↓
📊 <span style='color:#00d4ff;'>Phase 6: Pipeline</span> — ThreatDetectionPipeline · JSON incident reports · Summary statistics
&nbsp;&nbsp;&nbsp;↓
🖥️ <span style='color:#00d4ff;'>Phase 7: Dashboard</span> — This Streamlit SOC interface
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Phase cards ────────────────────────────────────────────────────────────────
st.markdown("### 📚 Project Phases")
phases = [
    ("1", "Data Preprocessing",
     "UNSW-NB15 ingestion, label normalisation, IQR clipping, LabelEncoding, StandardScaling. "
     "Produces cleaner.joblib and feature_list.txt.",
     "#00d4ff"),
    ("2", "Supervised Detection",
     "Random Forest (500 trees, class_weight='balanced') and XGBoost (multiclass softmax, "
     "balanced sample weights). Trained on 2M+ labelled flows.",
     "#ff6b00"),
    ("3", "Unsupervised Anomaly Detection",
     "Isolation Forest trained on 1.77M normal flows. Dense Autoencoder "
     "(43→32→16→8→16→32→43) trained to reconstruct normal traffic only.",
     "#bf5fff"),
    ("4", "Ensemble Voting",
     "Weighted soft vote fuses all four models. Severity assigned via a 2D "
     "confidence × agreement matrix (CRITICAL / HIGH / MEDIUM / LOW).",
     "#ffd700"),
    ("5", "MITRE ATT&CK Enrichment",
     "Downloads and caches STIX 2.1 Enterprise dataset. Maps 9 attack categories "
     "to ATT&CK techniques/tactics. Fallback: T1190 (Exploit Public-Facing Application).",
     "#00cc66"),
    ("6", "End-to-End Pipeline",
     "ThreatDetectionPipeline orchestrates Phases 1–5 in a single run() call. "
     "Outputs incident JSON reports and summary.json to reports/incidents/.",
     "#ff99bb"),
    ("7", "SOC Dashboard",
     "This Streamlit dashboard. Calls the Phase 6 pipeline without duplicating any "
     "detection logic. Provides analytics, filtering, MITRE views, and exports.",
     "#44ddff"),
]

for num, title, desc, color in phases:
    with st.expander(f"Phase {num} — {title}", expanded=False):
        st.markdown(
            f"<span style='background:{color};color:#000;padding:2px 8px;"
            f"border-radius:4px;font-size:0.8rem;font-weight:700;'>Phase {num}</span>"
            f"&nbsp; **{title}**", unsafe_allow_html=True)
        st.markdown(desc)

# ── Tech stack + model status ──────────────────────────────────────────────────
st.markdown("### 🛠️ Technology Stack")
tc1, tc2 = st.columns(2)
with tc1:
    st.markdown("""
    | Component | Technology |
    |---|---|
    | Language | Python 3.11+ |
    | ML Framework | Scikit-learn, XGBoost |
    | Deep Learning | PyTorch (Dense AE) |
    | Dashboard | Streamlit |
    | Visualisation | Plotly |
    | Threat Intel | MITRE ATT&CK® STIX 2.1 |
    | Dataset | UNSW-NB15 |
    """)
with tc2:
    st.markdown("**Model File Status**")
    for name, ok in check_model_status().items():
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} {name}")

    features = get_feature_list()
    if features:
        st.markdown(f"**Features:** {len(features)} columns loaded from Phase 1")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e3a5f;margin-top:32px;'>
<div style='text-align:center;color:#3a5a7a;font-size:0.75rem;padding:8px 0;'>
    RTC Project · ML-Driven SIEM · MITRE ATT&CK® Enabled ·
    Built with Python · Streamlit · Scikit-learn · XGBoost · PyTorch · MITRE ATT&CK
</div>
""", unsafe_allow_html=True)
