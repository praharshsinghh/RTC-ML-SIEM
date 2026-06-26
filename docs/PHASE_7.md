# Phase 7 — SOC Dashboard

**Date:** 2026-06-26
**Author:** RTC Project
**Status:** ✅ Complete

---

## Overview

Phase 7 adds a professional Security Operations Center (SOC) web dashboard built with Streamlit.
It sits on top of the existing `ThreatDetectionPipeline` (Phase 6) without duplicating any
detection or enrichment logic. The pipeline is called exactly once per "Run Detection" action;
all subsequent page navigations read from session state.

---

## Directory Structure

```
src/dashboard/
├── app.py                         # Entry point — page config, CSS, landing page
├── pages/
│   ├── 01_📊_Dashboard.py         # Pipeline runner + KPI metrics + charts
│   ├── 02_🚨_Incidents.py         # Filterable incident table + detail viewer
│   ├── 03_🎯_MITRE.py             # MITRE ATT&CK technique/tactic visualisation
│   ├── 04_🔬_Model_Analysis.py    # Per-model vote breakdown + score distributions
│   └── 05_ℹ️_About.py             # Project architecture + tech stack
├── components/
│   ├── sidebar.py                 # Persistent sidebar (model status, pipeline status)
│   ├── metrics.py                 # KPI card helpers, severity/verdict badges
│   ├── charts.py                  # Plotly chart library (10 chart types)
│   └── tables.py                  # (reserved for future table utilities)
├── utils/
│   ├── loaders.py                 # st.cache_resource pipeline, st.cache_data data loaders
│   ├── session.py                 # Session state keys, getters, setters, DataFrame builders
│   └── exports.py                 # JSON/CSV download byte-stream helpers
└── assets/                        # Logo and static assets

run_dashboard.py                   # CLI launcher
.streamlit/config.toml             # Dark SOC theme
```

---

## How to Run

```bash
# Recommended — convenience launcher
python run_dashboard.py

# Direct Streamlit invocation
streamlit run src/dashboard/app.py

# Custom port, headless (server mode)
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
Navigate to 📊 Dashboard
    ↓
Configure: upload CSV or use default test.parquet, set max_rows
    ↓
Click "🚀 Run Detection"
    ↓
loaders.run_pipeline() → ThreatDetectionPipeline.run()
    ↓
session.store_pipeline_output() → st.session_state["rtc_results"]
    ↓
KPI metrics + Plotly charts render
    ↓
Navigate to any page — reads session_state, no re-run
```

---

## Page Descriptions

### 📊 Dashboard (01_📊_Dashboard.py)
The primary operations page.

- **Pipeline Configuration** (expandable): CSV upload, max_rows slider, Run Detection button
- **KPI row**: Total Records, Attacks, Normal, Avg Confidence, Critical Alerts, Avg Agreement
- **Chart row 1**: Attack vs Normal donut + Severity horizontal bar
- **Chart row 2**: Attack Category treemap + Confidence histogram
- **Model vote comparison bar**: grouped Attack/Normal per model
- **Export buttons**: JSON incidents, JSON summary

### 🚨 Incidents (02_🚨_Incidents.py)
Investigation interface for attack-only rows.

- **Mini KPI strip**: Total, Critical, High, Medium, Low, Categories
- **Filters**: Severity, Attack Category, Technique ID, Confidence range
- **Interactive table**: `st.dataframe` with progress columns for Confidence
- **Downloads**: CSV (filtered), CSV (all), JSON (all reports)
- **Charts**: Confidence by Severity box plot + Agreement histogram
- **Detail viewer**: Select incident → see prediction, model votes, MITRE mapping, raw JSON

### 🎯 MITRE (03_🎯_MITRE.py)
Threat intelligence enrichment view.

- **KPI strip**: Unique techniques, unique tactics, most common tactic, top technique
- **Tactic bar chart**: incident count per ATT&CK tactic
- **Technique bar chart**: top N techniques (slider)
- **Searchable reference table**: Technique ID / Name / Tactic + ATT&CK URL as clickable link
- **Tactic breakdown table**: incidents and unique techniques per tactic

### 🔬 Model Analysis (04_🔬_Model_Analysis.py)
Per-model forensic analysis.

- **Tabs**: Vote Breakdown / Score Distributions / Agreement / Summary Table
- **Vote Breakdown**: grouped bar chart + detailed vote table
- **Score Distributions**: histograms for RF Probability, XGB Probability, IF Score, AE Error
- **Agreement**: bar chart + agreement count/percentage table
- **Summary Table**: Attack votes / Normal votes per model vs ensemble

### ℹ️ About (05_ℹ️_About.py)
Project documentation embedded in the UI.

- Architecture pipeline diagram (text-based, always visible)
- Phase cards (1–7) with expandable descriptions
- Technology stack table
- Model file status indicators
- Feature list count

---

## Component Architecture

### utils/loaders.py — Caching Strategy

| Resource | Cache Type | Rationale |
|---|---|---|
| `ThreatDetectionPipeline` | `st.cache_resource` | Singleton — models loaded once per server process |
| Feature list | `st.cache_data` | Immutable text file — safe to cache indefinitely |
| Parquet preview | `st.cache_data` | Bound by path + n_rows — safe to cache |
| Model file status | None (cheap) | `Path.exists()` is fast, no caching needed |

`st.cache_resource` is appropriate for the pipeline because:
1. It stores the object in memory (not serialised), so model weights are not re-serialised
2. It is shared across all user sessions (single-user demo scenario)
3. The pipeline's `_enricher` attribute is cached inside the object after first call

### utils/session.py — Session State Contract

All keys follow the prefix `rtc_` to avoid conflicts with Streamlit internals:

| Key | Type | Contents |
|---|---|---|
| `rtc_results` | `dict` | Full `pipeline.run()` return value |
| `rtc_incidents_df` | `pd.DataFrame` | Pre-built attack-only incident table |
| `rtc_all_df` | `pd.DataFrame` | All rows (NORMAL + ATTACK) |
| `rtc_input_source` | `str` | File path or "DataFrame" |
| `rtc_proc_time` | `float` | Wall-clock seconds for pipeline run |
| `rtc_max_rows` | `int\|None` | max_rows passed to pipeline |

Session state is initialised by `init_session()` called at the top of every page.
This prevents `KeyError` on first load before any pipeline run.

### components/charts.py — Chart Library

All charts share a consistent dark Plotly theme (`paper_bgcolor="#0d1b2a"`, `plot_bgcolor="#0d1b2a"`).
The `_DARK_TEMPLATE` dict is applied via `_apply(fig)` which calls `fig.update_layout`.

| Function | Type | Input |
|---|---|---|
| `attack_vs_normal_donut` | Pie (donut) | `summary` dict |
| `severity_bar` | Horizontal bar | `summary` dict |
| `category_treemap` | Treemap | `summary` dict |
| `confidence_histogram` | Histogram | incidents `DataFrame` |
| `agreement_histogram` | Histogram | incidents `DataFrame` |
| `confidence_by_severity` | Box plot | incidents `DataFrame` |
| `mitre_tactic_bar` | Horizontal bar | incidents `DataFrame` |
| `mitre_technique_bar` | Horizontal bar | incidents `DataFrame` |
| `model_vote_bar` | Grouped bar | incidents `DataFrame` |
| `severity_pie` | Pie | `summary` dict |

---

## How Streamlit Interacts with the Pipeline

**Critical principle:** The `ThreatDetectionPipeline` is treated as an opaque black box.
The dashboard calls `pipeline.run(source, max_rows)` exactly once and stores the result.

```python
# In utils/loaders.py
@st.cache_resource(show_spinner="⚙️ Loading detection pipeline…")
def get_pipeline() -> ThreatDetectionPipeline:
    return ThreatDetectionPipeline(stix_preload=False)

# In pages/01_Dashboard.py — only place pipeline.run() is triggered
if run_btn:
    results, elapsed = run_pipeline(source, max_rows=max_rows_val)
    store_pipeline_output(results, src_label, elapsed, max_rows_val)
```

All other pages call `get_results()`, `get_incidents_df()`, etc. from `session.py`.

---

## Interview / Viva Explanation

> **Q: How does the dashboard avoid re-running the pipeline on every page navigation?**
>
> A: Streamlit's `st.session_state` is a dict persisted for the duration of a browser session.
> After the pipeline runs on the Dashboard page, the full `results` dict is stored under the
> key `rtc_results`. Every other page checks `has_results()` (reads `session_state["rtc_results"]`)
> and renders the data from memory — the pipeline is never re-triggered.

> **Q: Why `st.cache_resource` for the pipeline and not `st.cache_data`?**
>
> A: `st.cache_data` serialises the return value with pickle. A `ThreatDetectionPipeline`
> containing PyTorch model weights and sklearn forests cannot be cleanly pickled.
> `st.cache_resource` stores the object reference in memory without serialisation, which is
> the correct pattern for all large in-memory objects (database connections, ML models).

> **Q: How is the dashboard kept independent from the ML backend?**
>
> A: The dashboard imports from `src.pipeline` only inside `utils/loaders.py`.
> Every other module (pages, components) imports from `utils/` only.
> This means the entire visual layer can be developed and tested without the ML backend
> by mocking the `get_pipeline()` and `get_results()` functions.

---

## Performance Notes

- Model loading: ~1–2 seconds (covered by `st.cache_resource`, done once)
- STIX loading: ~6 seconds (covered by pipeline's `_enricher` attribute after first call)
- 500 rows inference: ~0.3 seconds
- 5,000 rows inference: ~1.5 seconds
- 50,000 rows inference: ~10–20 seconds (progress bar recommended)

For large datasets (>10K rows), use `max_rows` to process a representative sample for the dashboard.
The full run can be done via `python run_pipeline.py --input <file>` which writes JSON reports to `reports/incidents/`.

---

## Verification Checklist

- [x] `python run_dashboard.py` launches successfully
- [x] All 5 pages render without errors (HTTP 200)
- [x] Sidebar shows all 5 model files as 🟢 (green)
- [x] "🚀 Run Detection" button triggers pipeline with spinner
- [x] KPI row renders after pipeline run
- [x] All Plotly charts render interactively
- [x] Incidents table is filterable and sortable
- [x] Per-incident detail viewer shows MITRE + model votes + raw JSON
- [x] MITRE page shows tactic + technique charts
- [x] Download buttons produce valid JSON / CSV
- [x] Session state persists across page navigations
- [x] Error message (no stack trace) shown when no results
- [x] No ML code duplicated in dashboard pages

---

## Suggested Git Commit

```
feat(dashboard): add Phase 7 SOC Streamlit dashboard

- src/dashboard/app.py: landing page with dark SOC theme + CSS
- src/dashboard/pages/: 5 pages (Dashboard, Incidents, MITRE, Model Analysis, About)
- src/dashboard/components/: sidebar, metrics, charts (10 Plotly types)
- src/dashboard/utils/: session state, cached loaders, export helpers
- run_dashboard.py: CLI launcher with --port / --no-browser flags
- .streamlit/config.toml: dark theme configuration
- docs/PHASE_7.md: architecture, caching, interview Q&A

Dashboard calls ThreatDetectionPipeline.run() exactly once per user action;
session state persists results across all pages. No ML logic duplicated.
```
