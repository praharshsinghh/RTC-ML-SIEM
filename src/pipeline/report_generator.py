"""
src/pipeline/report_generator.py — Incident Report Generator
=============================================================
Transforms enriched incident data (Phase 5 EnrichedIncident + row metadata)
into the canonical Phase 6 JSON incident report schema.

Responsibilities
----------------
1. Build one JSON-serialisable dict per detected attack (IncidentReport schema)
2. Serialize and persist individual report files → reports/incidents/
3. Build and persist the session-level summary.json

Why a separate report_generator module?
----------------------------------------
The pipeline.py orchestrates *what* data to collect; report_generator.py
handles *how* to format and persist it. This separation of concerns means:
  - Report schema changes require editing only this file
  - The pipeline can be used headlessly without writing files (return dict)
  - Unit testing the schema is independent of model loading / inference

Incident Report Schema (one per attack-flagged row)
----------------------------------------------------
See the JSON schema in generate_incident_report() below.

Summary Schema
--------------
See generate_summary() below.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.attack_mapping.schemas import EnrichedIncident
from src.pipeline.utils import (
    format_agreement,
    extract_model_vote_summary,
    extract_raw_scores,
    now_utc_iso,
)

logger = logging.getLogger(__name__)

# Default output directory for incident reports
DEFAULT_INCIDENTS_DIR = Path(__file__).parent.parent.parent / "reports" / "incidents"


def build_incident_report(
    incident: EnrichedIncident,
    row_index: int,
    pipeline_run_id: str,
) -> dict[str, Any]:
    """
    Build a single JSON-serialisable incident report from an EnrichedIncident.

    Only called for rows where final_verdict == 'ATTACK'.

    Schema
    ------
    {
        "id": UUID,
        "pipeline_run_id": str,
        "timestamp": ISO-8601,
        "source": {"row_index": int},
        "prediction": {
            "attack_category": str,
            "binary_label": "ATTACK",
            "confidence": float
        },
        "severity": "CRITICAL"|"HIGH"|"MEDIUM"|"LOW",
        "ensemble": {
            "agreement": "3/4",
            "rf": "ATTACK"|"NORMAL",
            "xgb": "ATTACK"|"NORMAL",
            "iforest": "ATTACK"|"NORMAL",
            "lstm": "ATTACK"|"NORMAL"   # DenseAutoencoder referred to as 'lstm' per spec
        },
        "mitre": {
            "technique_id": str,
            "technique_name": str,
            "tactic": str,
            "tactic_id": str,
            "description": str,
            "mapping_confidence": str
        },
        "raw_scores": {
            "rf_probability": float,
            "xgb_probability": float,
            "iforest_score": float,
            "lstm_error": float
        }
    }

    Parameters
    ----------
    incident : EnrichedIncident
        Fully enriched incident from Phase 5 AttackEnricher.
    row_index : int
        Original row index in the input DataFrame.
    pipeline_run_id : str
        Unique identifier for this pipeline execution session.

    Returns
    -------
    dict — JSON-safe incident report.
    """
    vote_summary = extract_model_vote_summary(incident.model_votes)
    raw_scores   = extract_raw_scores(incident.model_votes)
    agreement_str = format_agreement(incident.agreement_score)

    # MITRE ATT&CK fields — use property accessors for safety
    tech = incident.primary_technique
    report: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "pipeline_run_id": pipeline_run_id,
        "timestamp": incident.timestamp,
        "source": {
            "row_index": row_index,
        },
        "prediction": {
            "attack_category": incident.attack_category,
            "binary_label": incident.verdict,
            "confidence": round(incident.confidence, 6),
        },
        "severity": incident.severity,
        "ensemble": {
            "agreement": agreement_str,
            "rf":      vote_summary["rf"],
            "xgb":     vote_summary["xgb"],
            "iforest": vote_summary["iforest"],
            "lstm":    vote_summary["lstm"],
        },
        "mitre": {
            "technique_id":       tech.technique_id if tech else "N/A",
            "technique_name":     tech.name if tech else "Unknown",
            "tactic":             incident.tactic,
            "tactic_id":          incident.tactic_id,
            "description":        (tech.description[:400] if tech and tech.description else ""),
            "mapping_confidence": incident.mapping_confidence,
        },
        "raw_scores": raw_scores,
    }
    return report


def save_incident_reports(
    reports: list[dict[str, Any]],
    output_dir: Optional[Path] = None,
    run_timestamp: Optional[str] = None,
) -> Path:
    """
    Persist all incident reports to a single JSON file.

    File naming: incident_report_YYYYMMDD_HHMMSS.json
    Location:    reports/incidents/

    Parameters
    ----------
    reports : list[dict]
        List of incident report dicts from build_incident_report().
    output_dir : Path, optional
        Directory to write to. Defaults to reports/incidents/.
    run_timestamp : str, optional
        Timestamp string for the filename. Defaults to current UTC time.

    Returns
    -------
    Path — path to the written JSON file.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_INCIDENTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_timestamp is None:
        run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    filename = f"incident_report_{run_timestamp}.json"
    out_path = output_dir / filename

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2, ensure_ascii=False)

    logger.info(
        f"[Reporter] {len(reports)} incident report(s) saved → {out_path}"
    )
    return out_path


def generate_summary(
    all_incidents: list[EnrichedIncident],
    run_id: str,
    run_timestamp: str,
    input_source: str,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Build and persist the session-level summary.json.

    The summary provides a high-level overview of the pipeline run,
    aggregating counts, distributions, and averages across all records
    (both normal and attack).

    Schema
    ------
    {
        "pipeline_run_id": str,
        "run_timestamp": str,
        "input_source": str,
        "total_records": int,
        "normal_count": int,
        "attack_count": int,
        "severity_distribution": {"CRITICAL": 0, "HIGH": 0, ...},
        "attack_category_distribution": {"Generic": 5, ...},
        "average_confidence": float,
        "average_agreement_score": float,
        "attack_rate_pct": float
    }

    Parameters
    ----------
    all_incidents : list[EnrichedIncident]
        All enriched incidents (both NORMAL and ATTACK).
    run_id : str
        Unique pipeline run identifier.
    run_timestamp : str
        Formatted timestamp string for filename.
    input_source : str
        Description of the input (filename or 'DataFrame').
    output_dir : Path, optional
        Where to write summary.json. Defaults to reports/incidents/.

    Returns
    -------
    dict — the summary dict (also written to disk).
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_INCIDENTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total     = len(all_incidents)
    attacks   = [i for i in all_incidents if i.verdict == "ATTACK"]
    normals   = [i for i in all_incidents if i.verdict == "NORMAL"]

    # Severity distribution (attacks only)
    sev_dist: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for inc in attacks:
        band = inc.severity
        if band in sev_dist:
            sev_dist[band] += 1

    # Attack category distribution (attacks only)
    cat_dist: dict[str, int] = {}
    for inc in attacks:
        cat = inc.attack_category or "Unknown"
        cat_dist[cat] = cat_dist.get(cat, 0) + 1

    # Confidence and agreement averages (all records)
    avg_conf  = round(
        sum(i.confidence for i in all_incidents) / total, 4
    ) if total else 0.0
    avg_agree = round(
        sum(i.agreement_score for i in all_incidents) / total, 4
    ) if total else 0.0

    attack_rate = round(len(attacks) / total * 100, 2) if total else 0.0

    summary: dict[str, Any] = {
        "pipeline_run_id":             run_id,
        "run_timestamp":               run_timestamp,
        "input_source":                input_source,
        "total_records":               total,
        "normal_count":                len(normals),
        "attack_count":                len(attacks),
        "attack_rate_pct":             attack_rate,
        "severity_distribution":       sev_dist,
        "attack_category_distribution": dict(
            sorted(cat_dist.items(), key=lambda x: x[1], reverse=True)
        ),
        "average_confidence":          avg_conf,
        "average_agreement_score":     avg_agree,
    }

    out_path = output_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info(f"[Reporter] Summary saved → {out_path}")
    return summary
