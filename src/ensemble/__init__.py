"""
src/ensemble/__init__.py — Public API for the Ensemble Module
=============================================================
Re-exports the essential classes and constants so downstream code
(notebooks, runners, dashboard) can import with minimal path depth:

    from src.ensemble import EnsembleDetector, EnsembleResult

Design
------
The ensemble module is structured as four focused sub-modules:
  schemas.py  — data contracts (EnsembleResult, ModelVote, constants)
  voting.py   — weighted voting arithmetic
  scoring.py  — confidence and severity calculation
  detector.py — EnsembleDetector (orchestrator)

This __init__.py provides a single import surface that hides the
internal sub-module structure from callers.

Phase 4 module structure summary
----------------------------------
Phase 4: Ensemble Detection Layer
├── src/ensemble/
│   ├── __init__.py      ← this file (public API)
│   ├── schemas.py       ← EnsembleResult, ModelVote, MODEL_WEIGHTS, …
│   ├── voting.py        ← weighted vote aggregation
│   ├── scoring.py       ← confidence + severity rule tables
│   └── detector.py      ← EnsembleDetector (orchestrator)
├── run_ensemble.py      ← CLI entry point (trains, evaluates, saves report)
└── notebooks/
    └── 04_ensemble_analysis.ipynb  ← visualisation & analysis
"""

from src.ensemble.schemas import (  # noqa: F401
    EnsembleResult,
    ModelVote,
    MODEL_WEIGHTS,
    ATTACK_THRESHOLD,
    NORMAL_CAT,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
)
from src.ensemble.voting import (   # noqa: F401
    compute_weighted_attack_score,
    compute_verdict,
    compute_agreement_score,
    build_model_votes,
    resolve_attack_category,
)
from src.ensemble.scoring import (  # noqa: F401
    compute_confidence,
    compute_severity,
)
from src.ensemble.detector import EnsembleDetector  # noqa: F401

__all__ = [
    # Schema
    "EnsembleResult",
    "ModelVote",
    "MODEL_WEIGHTS",
    "ATTACK_THRESHOLD",
    "NORMAL_CAT",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "SEVERITY_HIGH",
    "SEVERITY_CRITICAL",
    # Voting
    "compute_weighted_attack_score",
    "compute_verdict",
    "compute_agreement_score",
    "build_model_votes",
    "resolve_attack_category",
    # Scoring
    "compute_confidence",
    "compute_severity",
    # Orchestrator
    "EnsembleDetector",
]
