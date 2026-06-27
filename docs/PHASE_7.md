# Phase 7 — SOC Dashboard

**Date:** 2026-06-26
**Status:** Complete

---

## Overview

Phase 7 adds a SOC-style web dashboard built with Streamlit on top of the
existing `ThreatDetectionPipeline` (Phase 6). The pipeline is called exactly
once per "Run Detection" action; all subsequent page navigations read from
session state without re-triggering inference.

---

## Directory Structure

```
src/dashboard/
├── app.py                        # Entry point — page config, CSS, landing page
├── pages/
│   ├── 01_📊_Dashboard.py        # Pipeline runner, KPI metrics, charts
│   ├── 02_🚨_Incidents.py        # Filterable incident table and detail viewer
│   ├── 03_🎯_MITRE.py            # MITRE ATT&CK technique/tactic visualisation
│   ├── 04_🔬_Model_Analysis.py   # Per-model vote breakdown and score distributions
│   └── 05_ℹ️_About.py            # Project architecture and tech stack
├── components/
│   ├── sidebar.py                # Persistent sidebar (model status, pipeline status)
│   ├── metrics.py                # KPI card helpers, severity/verdict badges
│   ├── charts.py                 # Plotly chart library (10 chart types)
│   └── tables.py                 # Reserved for future table utilities
├── utils/
│   ├── loaders.py                # st.cache_resource pipeline, st.cache_data loaders
│   ├── session.py                # Session state keys, getters, setters, DataFrame builders
│   └── exports.py                # JSON/CSV download byte-stream helpers
└── assets/                       # Logo and static assets

run_dashboard.py                  # CLI launcher
.streamlit/config.toml            # Dark SOC theme
```

---

## How to Run

```bash
# Recommended
python run_dashboard.py

# Direct Streamlit invocation
streamlit run src/dashboard/app.py

# Custom port, headless
python run_dashboard.py --port 8080 --no-browser
```

Dashboard opens at **http://localhost:8501** by default.

---

## Dashboard Flow

```
User opens dashboard
  ↓
Landing page (app.py) — project overview, feature cards
  ↓
Navigate to Dashboard page
  ↓
Configure: upload CSV or use default test.parquet, set max_rows
  ↓
Click "Run Detection"
  ↓
loaders.run_pipeline() → ThreatDetectionPipeline.run()
  ↓
session.store_pipeline_output() → st.session_state["rtc_results"]
  ↓
KPI metrics and Plotly charts render
  ↓
Navigate to any page — reads session_state, no re-run
```

---

## Page Descriptions

### Dashboard (01_📊_Dashboard.py)
- **Pipeline Configuration** (expandable): CSV upload, max_rows input, Run Detection button
- **KPI row**: Total Records, Attacks, Normal, Avg Confidence, Critical Alerts, Avg Agreement
- **Chart row 1**: Attack vs Normal donut + Severity horizontal bar
- **Chart row 2**: Attack Category treemap + Confidence histogram
- **Model vote comparison**: grouped Attack/Normal per model
- **Export buttons**: JSON incidents, JSON summary

### Incidents (02_🚨_Incidents.py)
- **Filters**: Severity, Attack Category, Technique ID, Confidence range
- **Interactive table**: `st.dataframe` with progress columns for Confidence
- **Downloads**: CSV (filtered), CSV (all), JSON (all reports)
- **Charts**: Confidence by Severity box plot + Agreement histogram
- **Detail viewer**: Select incident → prediction, model votes, MITRE mapping, raw JSON

### MITRE (03_🎯_MITRE.py)
- **Tactic bar chart**: incident count per ATT&CK tactic
- **Technique bar chart**: top N techniques (slider)
- **Reference table**: Technique ID / Name / Tactic + ATT&CK URL as clickable link

### Model Analysis (04_🔬_Model_Analysis.py)
- **Tabs**: Vote Breakdown / Score Distributions / Agreement / Summary Table
- **Vote Breakdown**: grouped bar chart + detailed vote table
- **Score Distributions**: histograms for RF Probability, XGB Probability, IF Score, AE Error

### About (05_ℹ️_About.py)
- Architecture pipeline diagram
- Phase cards (1–7) with expandable descriptions
- Technology stack table
- Model file status indicators

---

## Component Architecture

### utils/loaders.py — Caching Strategy

| Resource | Cache Type | Rationale |
|----------|------------|-----------|
| `ThreatDetectionPipeline` | `st.cache_resource` | Singleton — models loaded once per server process |
| Feature list | `st.cache_data` | Immutable text file |
| Parquet preview | `st.cache_data` | Bound by path + n_rows |
| Model file status | None | `Path.exists()` is fast enough |

`st.cache_resource` is used for the pipeline because it stores the object
in memory without serialisation. `st.cache_data` serialises with pickle,
which does not work cleanly for objects containing PyTorch weights.

### utils/session.py — Session State Contract

All keys use the `rtc_` prefix to avoid conflicts with Streamlit internals:

| Key | Type | Contents |
|-----|------|----------|
| `rtc_results` | `dict` | Full `pipeline.run()` return value |
| `rtc_incidents_df` | `pd.DataFrame` | Pre-built attack-only incident table |
| `rtc_all_df` | `pd.DataFrame` | All rows (NORMAL + ATTACK) |
| `rtc_input_source` | `str` | File path or "DataFrame" |
| `rtc_proc_time` | `float` | Wall-clock seconds for pipeline run |
| `rtc_max_rows` | `int\|None` | max_rows passed to pipeline |

### components/charts.py — Chart Library

All charts share a consistent dark Plotly theme (`paper_bgcolor="#0d1b2a"`).

| Function | Type | Input |
|----------|------|-------|
| `attack_vs_normal_donut` | Pie (donut) | `summary` dict |
| `severity_bar` | Horizontal bar | `summary` dict |
| `category_treemap` | Treemap | `summary` dict |
| `confidence_histogram` | Histogram | incidents DataFrame |
| `agreement_histogram` | Histogram | incidents DataFrame |
| `confidence_by_severity` | Box plot | incidents DataFrame |
| `mitre_tactic_bar` | Horizontal bar | incidents DataFrame |
| `mitre_technique_bar` | Horizontal bar | incidents DataFrame |
| `model_vote_bar` | Grouped bar | incidents DataFrame |
| `severity_pie` | Pie | `summary` dict |

---

## Performance Notes

| Operation | Approximate Time |
|-----------|-----------------|
| Model loading | 1–2 s (cached after first load) |
| STIX loading | ~6 s (cached inside pipeline after first call) |
| 500 rows inference | ~0.3 s |
| 5,000 rows inference | ~1.5 s |
| 50,000 rows inference | 10–20 s |

For large datasets, use `max_rows` to process a representative sample in the
dashboard. Full runs can be executed via `python run_pipeline.py`.

---

## Verification Checklist

- [x] `python run_dashboard.py` launches successfully
- [x] All 5 pages render without errors
- [x] Sidebar shows all 5 model files present
- [x] Run Detection button triggers pipeline with spinner
- [x] KPI row renders after pipeline run
- [x] All Plotly charts render interactively
- [x] Incidents table is filterable and sortable
- [x] Per-incident detail viewer shows MITRE mapping and model votes
- [x] Download buttons produce valid JSON/CSV
- [x] Session state persists across page navigations
- [x] Error message shown when no results (no stack trace exposed)
- [x] No ML code duplicated in dashboard pages
