"""
src/dashboard/app.py — RTC SIEM Dashboard Entry Point
=======================================================
Main Streamlit application. Configures the page, injects shared CSS,
initialises session state, and renders the home / landing screen.

Run via:
    streamlit run src/dashboard/app.py
    python run_dashboard.py          (convenience wrapper)
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root on sys.path ──────────────────────────────────────────
def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "run_pipeline.py").exists():
            return parent
    return p.parents[2]

_ROOT = _project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="RTC SIEM Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "RTC — ML-Driven Cyber Threat Detection | MITRE ATT&CK Enabled"},
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide deploy button */
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none; }

/* Metric card styling */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #111f35 100%) !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: 0 2px 12px rgba(0,212,255,0.07) !important;
}
[data-testid="stMetricLabel"] { color: #7a9ab8 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #e0e0e0 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #7a9ab8;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Expander header */
.streamlit-expanderHeader {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #00d4ff, #0080aa) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 24px !important;
    transition: transform 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,212,255,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
from src.dashboard.utils.session import init_session
from src.dashboard.components.sidebar import render_sidebar

init_session()
render_sidebar()

# ── Home / Landing page ───────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 16px 0;'>
    <div style='font-size:3rem;'>🛡️</div>
    <h1 style='color:#00d4ff;margin:8px 0;letter-spacing:2px;'>
        RTC SIEM Dashboard
    </h1>
    <p style='color:#7a9ab8;font-size:1.05rem;max-width:600px;margin:0 auto;'>
        Real-Time Cyber Threat Detection using Machine Learning<br>
        MITRE ATT&amp;CK® Enabled · Ensemble Detection · SOC-Ready
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Feature cards
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("🤖", "Ensemble Detection",
     "RF · XGBoost · Isolation Forest · Autoencoder — weighted vote with confidence scoring"),
    ("🎯", "MITRE ATT&CK",
     "Every attack mapped to an ATT&CK technique and tactic from the STIX knowledge base"),
    ("📊", "Live Analytics",
     "Interactive Plotly dashboards with severity distribution, category breakdown & timelines"),
    ("📋", "SOC Reports",
     "Exportable JSON & CSV incident reports with full audit trails per detection"),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
    col.markdown(f"""
    <div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;
    padding:20px;text-align:center;height:160px;'>
        <div style='font-size:2rem;margin-bottom:8px;'>{icon}</div>
        <div style='color:#00d4ff;font-weight:700;font-size:0.9rem;'>{title}</div>
        <div style='color:#7a9ab8;font-size:0.78rem;margin-top:6px;line-height:1.4;'>{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 **Navigate using the sidebar pages** · Start with **📊 Dashboard** to run the detection pipeline")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e3a5f;margin-top:32px;'>
<div style='text-align:center;color:#3a5a7a;font-size:0.75rem;padding:8px 0 16px 0;'>
    RTC Project · ML-Driven SIEM · MITRE ATT&CK® Enabled ·
    Built with Python · Streamlit · Scikit-learn · XGBoost · PyTorch
</div>
""", unsafe_allow_html=True)
