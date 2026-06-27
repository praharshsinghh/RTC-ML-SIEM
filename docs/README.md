# RTC — ML-Driven Network Intrusion Detection with MITRE ATT&CK Enrichment

A machine learning SIEM pipeline for detecting network intrusions in the UNSW-NB15 dataset. Combines supervised and unsupervised detectors in a weighted ensemble, enriches detections with MITRE ATT&CK threat intelligence, and exposes results through an interactive SOC-style dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-orange)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Overview

RTC implements a seven-phase ML security pipeline:

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | Preprocessing | UNSW-NB15 data loading, cleaning, feature encoding, z-score scaling |
| 2 | Supervised Models | Random Forest + XGBoost multiclass classifiers |
| 3 | Unsupervised Models | Isolation Forest + Dense Autoencoder anomaly detectors |
| 4 | Ensemble Layer | Transparent weighted voting with confidence and severity scoring |
| 5 | MITRE ATT&CK | Category-to-technique mapping via STIX knowledge base |
| 6 | Pipeline | End-to-end orchestrator: CSV/Parquet → JSON incident reports |
| 7 | Dashboard | Streamlit SOC dashboard with Plotly analytics |

---

## Architecture

```
Input (CSV / Parquet)
  ↓
Preprocessing (DataCleaner — auto-detects raw vs preprocessed)
  ↓
┌──────────────┬──────────────┬──────────────────┬──────────────────┐
│ Random       │ XGBoost      │ Isolation        │ Dense            │
│ Forest       │ (Supervised) │ Forest           │ Autoencoder      │
│ (Supervised) │              │ (Unsupervised)   │ (Unsupervised)   │
└──────┬───────┴──────┬───────┴────────┬─────────┴────────┬─────────┘
       │              │                │                  │
       └──────────────┴────────────────┴──────────────────┘
                              ↓
                    Weighted Ensemble Vote
                    (RF=0.35, XGB=0.35, IF=0.15, AE=0.15)
                              ↓
                    Confidence + Severity Scoring
                              ↓
                    MITRE ATT&CK Enrichment (STIX)
                              ↓
                    JSON Incident Reports + SOC Dashboard
```

### Model Paradigms

| Model | Type | Training Data | Output |
|-------|------|--------------|--------|
| Random Forest | Supervised | Labelled traffic | Multiclass attack category |
| XGBoost | Supervised | Labelled traffic | Multiclass attack category |
| Isolation Forest | Unsupervised | Normal-only | Anomaly score |
| Dense Autoencoder | Unsupervised | Normal-only | Reconstruction error |

The ensemble design fuses supervised precision (validated attack categories) with unsupervised breadth (zero-day / novel threat detection).

---

## Repository Structure

```
RTC/
├── data/
│   ├── raw/                   # Raw UNSW-NB15 CSV files (not committed)
│   ├── processed/             # Preprocessed Parquet splits + cleaner.joblib
│   ├── models/                # Saved RF and XGBoost models
│   └── download_data.py       # Dataset download script
│
├── models/                    # Saved Isolation Forest + Autoencoder
│
├── src/
│   ├── preprocessing/         # loader.py, cleaner.py, splitter.py
│   ├── models/                # random_forest.py, xgboost_model.py,
│   │                          #   isolation_forest.py, autoencoder.py
│   ├── ensemble/              # detector.py, voting.py, scoring.py, schemas.py
│   ├── attack_mapping/        # enricher.py, loader.py, mappings.yaml, schemas.py
│   ├── pipeline/              # pipeline.py, report_generator.py, utils.py
│   └── dashboard/             # Streamlit app, pages, components, utils
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_ensemble_analysis.ipynb
│   ├── 04_supervised_evaluation.ipynb
│   └── 05_attack_mapping_analysis.ipynb
│
├── reports/
│   ├── incidents/             # JSON incident reports (pipeline output)
│   ├── supervised_evaluation.json
│   ├── ensemble_evaluation.json
│   └── unsupervised_evaluation.json
│
├── docs/
│   └── PHASE_7.md             # Dashboard architecture documentation
│
├── run_preprocessing.py       # Phase 1 entry point
├── run_supervised.py          # Phase 2 entry point
├── run_unsupervised.py        # Phase 3 entry point
├── run_ensemble.py            # Phase 4 entry point
├── run_attack_mapping.py      # Phase 5 entry point
├── run_pipeline.py            # Phase 6 entry point
├── run_dashboard.py           # Phase 7 entry point
└── requirements.txt
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Download Data

```bash
python data/download_data.py
# Alternatively: download from https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
# and place CSVs in data/raw/
```

### 2. Preprocess

```bash
python run_preprocessing.py
# Outputs: data/processed/{train,test,train_normal_only}.parquet
#          data/processed/cleaner.joblib
#          data/processed/feature_list.txt
```

### 3. Train Models

```bash
python run_supervised.py      # RF + XGBoost → data/models/
python run_unsupervised.py    # Isolation Forest + Autoencoder → models/
```

### 4. Build Ensemble + ATT&CK Mapping

```bash
python run_ensemble.py
python run_attack_mapping.py
```

### 5. Run Detection Pipeline

```bash
# Default — uses data/processed/test.parquet
python run_pipeline.py

# Custom input (raw CSV or preprocessed Parquet, auto-detected)
python run_pipeline.py --input traffic.csv
python run_pipeline.py --input data/processed/test.parquet --max-rows 1000

# Reports → reports/incidents/
```

### 6. Launch Dashboard

```bash
python run_dashboard.py
# Opens at http://localhost:8501
```

---

## Pipeline

The `ThreatDetectionPipeline` (Phase 6) supports both raw CSV files and
preprocessed Parquet files as input. It automatically detects whether
categorical columns (proto, state, service) contain strings (raw) or
encoded integers (preprocessed). Raw data is passed through
`DataCleaner.transform()` exactly once; preprocessed data skips that step
and proceeds directly to inference.

```python
from src.pipeline import ThreatDetectionPipeline

pipeline = ThreatDetectionPipeline()
result = pipeline.run("data/processed/test.parquet")  # preprocessed
result = pipeline.run("traffic.csv")                  # raw CSV — auto-detected

print(result["summary"]["attack_count"])
print(result["summary"]["severity_distribution"])
```

---

## Dashboard

The Streamlit dashboard (Phase 7) provides:

- **Detection** — upload CSV or use default test set, configurable row limit
- **KPI Metrics** — total records, attack rate, severity distribution
- **Analytics** — attack vs. normal donut, category treemap, confidence histogram
- **Incident Browser** — filterable table with per-incident detail viewer
- **MITRE ATT&CK View** — tactic/technique breakdown with ATT&CK links
- **Model Analysis** — per-model vote comparison, score distributions

The pipeline runs exactly once per user action; all subsequent page navigations
read from session state without re-triggering inference.

---

## Models

### Random Forest
- `n_estimators=200`, `class_weight='balanced'`, `min_samples_leaf=2`
- `class_weight='balanced'` is required due to severe class imbalance
  (Normal ≈ 1.775M samples, Worms ≈ 139 in the training partition)
- Feature importances use mean decrease in Gini impurity (note: inflated
  for high-cardinality features; use SHAP for production decisions)

### XGBoost
- Multiclass with `scale_pos_weight` per class for imbalance handling
- `eval_metric='mlogloss'`, early stopping on validation set

### Isolation Forest
- Trained on normal-only traffic (`train_normal_only.parquet`)
- `contamination=0.05` (tuned on validation FPR)

### Dense Autoencoder
- PyTorch feedforward encoder-decoder trained on normal-only traffic
- Anomaly detection via reconstruction error threshold (95th percentile
  on training normal traffic)

---

## MITRE ATT&CK Integration

Each detected attack is enriched with the corresponding ATT&CK technique
and tactic from the STIX Enterprise ATT&CK knowledge base.

Mappings are defined in `src/attack_mapping/mappings.yaml` with one primary
technique and optional secondary techniques per UNSW-NB15 attack category.
The mapping rationale and confidence level are included in every incident report.

| Attack Category | Primary Technique | Tactic |
|----------------|------------------|--------|
| Fuzzers | T1595 — Active Scanning | Reconnaissance |
| Exploits | T1190 — Exploit Public-Facing App | Initial Access |
| DoS | T1499 — Endpoint Denial of Service | Impact |
| Generic | T1046 — Network Service Discovery | Discovery |
| Reconnaissance | T1595 — Active Scanning | Reconnaissance |
| Backdoors | T1543 — Create/Modify System Process | Persistence |
| Analysis | T1040 — Network Sniffing | Discovery |
| Shellcode | T1055 — Process Injection | Defence Evasion |
| Worms | T1091 — Replication Through Removable Media | Lateral Movement |

---

## Results

### Ensemble Performance (test set, 175,341 samples)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9684 |
| Precision | 0.9598 |
| Recall | 0.9804 |
| F1 | 0.9700 |
| False Positive Rate | 0.0254 |

### Severity Distribution

| Severity | Description |
|----------|-------------|
| CRITICAL | conf ≥ 0.85 AND agreement ≥ 0.85 |
| HIGH | conf ≥ 0.70 AND agreement ≥ 0.70 |
| MEDIUM | conf ≥ 0.50 AND agreement ≥ 0.50 |
| LOW | any other ATTACK verdict |

---

## Limitations

- Trained and evaluated on UNSW-NB15 only. Performance on real-world traffic
  from different network environments is unknown.
- IP addresses and timestamps are dropped during preprocessing. The pipeline
  cannot correlate alerts across flows by source/destination.
- Unsupervised models may generate false positives on unusual-but-benign
  traffic patterns not present in the training set.
- The MITRE ATT&CK mapping is manually curated and reflects the dominant
  technique for each category, not an exhaustive enumeration.

---

## Future Work

- Add SHAP-based explainability for individual alert justification
- Extend to streaming data sources (Kafka, Zeek log files)
- Implement alert deduplication and correlation across multiple flows
- Add real-time dashboard with auto-refresh for live capture feeds
- Expand MITRE ATT&CK mapping coverage with sub-techniques

---

## Dataset Citation

> Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for
> network intrusion detection systems. *2015 Military Communications and
> Information Systems Conference (MilCIS)*, 1–6. IEEE.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
