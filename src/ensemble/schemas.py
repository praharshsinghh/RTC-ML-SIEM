"""
src/ensemble/schemas.py
========================
Data contracts for the ensemble prediction output.

EnsembleResult is the canonical output of every predict() call.
ModelVote records the per-model contribution for audit and traceability.
Separating schema from logic allows the detection engine and reporting
layer to evolve independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Per-model prediction summary (stored inside EnsembleResult)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelVote:
    """
    Lightweight record of a single model's contribution to the ensemble decision.

    Fields
    ------
    model_name : str
        Identifier of the model (e.g. 'RandomForest', 'XGBoost').
    weight : float
        The model's weight in the weighted vote (0.35 for supervised,
        0.15 for unsupervised). Weights sum to 1.0 across all four models.
    vote : str
        The binary verdict this model cast: 'ATTACK' or 'NORMAL'.
        For unsupervised models, 'ATTACK' = is_anomaly, 'NORMAL' = not.
    attack_cat : str
        The predicted category string from this model.
        Unsupervised models produce 'Anomaly' or 'unknown'.
    confidence : float
        Model-level confidence [0, 1] in its own prediction.
    raw_score : float
        The raw output from the model (class probability, anomaly score,
        or reconstruction error). Preserved for traceability.
    weighted_contribution : float
        weight × confidence — how much this model contributed to the
        ensemble confidence numerically.
    """
    model_name: str
    weight: float
    vote: str               # 'ATTACK' | 'NORMAL'
    attack_cat: str         # 'Generic', 'DoS', 'Anomaly', 'unknown', …
    confidence: float       # [0, 1]
    raw_score: float        # raw model output
    weighted_contribution: float  # weight × confidence

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Main ensemble output — one per inference row
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleResult:
    """
    Ensemble verdict for a single network flow sample.

    Fields
    ------
    timestamp : str
        ISO-8601 UTC timestamp of the inference.
    final_verdict : str
        'ATTACK' or 'NORMAL'. ATTACK when weighted_attack_score >= 0.50.
    confidence : float [0, 1]
        Distance of weighted_attack_score from the decision boundary,
        re-scaled to [0, 1] (see scoring.py).
    severity : str
        SOC alert priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'N/A'.
        Assigned by a rule-based confidence × agreement matrix.
    agreement_score : float [0, 1]
        Sum of weights for models that agree with the final verdict.
    weighted_attack_score : float [0, 1]
        Raw ensemble score: Σ(weight_i × attack_vote_i).
    final_attack_cat : str
        Resolved attack category (see voting.resolve_attack_category).
    model_votes : list[ModelVote]
        Per-model audit trail.
        Order: [RandomForest, XGBoost, IsolationForest, DenseAutoencoder].
    """
    timestamp: str
    final_verdict: str          # 'ATTACK' | 'NORMAL'
    confidence: float           # [0, 1]
    severity: str               # 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
    agreement_score: float      # [0, 1]
    weighted_attack_score: float  # [0, 1] raw score before thresholding
    final_attack_cat: str       # resolved category
    model_votes: list[ModelVote] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe, all Python natives)."""
        d = asdict(self)
        return d

    @staticmethod
    def _now_utc() -> str:
        """ISO-8601 UTC timestamp for the current moment."""
        return datetime.now(tz=timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Constants used across the ensemble module
# ─────────────────────────────────────────────────────────────────────────────

# Supervised models (RF, XGB) receive higher weight: they were trained on
# labelled data and their FPR/recall are validated on the test set.
# Unsupervised models (IF, AE) detect statistical anomalies and are weighted
# lower to limit false positives from unusual-but-benign traffic.
MODEL_WEIGHTS: dict[str, float] = {
    "RandomForest":     0.35,
    "XGBoost":          0.35,
    "IsolationForest":  0.15,
    "DenseAutoencoder": 0.15,
}

# Natural midpoint threshold: combined weight of ATTACK voters >= NORMAL voters.
ATTACK_THRESHOLD: float = 0.50

# NORMAL class label in attack_cat column (matches Phase 2/3 convention)
NORMAL_CAT: str = "unknown"

# Severity band labels (SOC terminology)
SEVERITY_LOW:      str = "LOW"
SEVERITY_MEDIUM:   str = "MEDIUM"
SEVERITY_HIGH:     str = "HIGH"
SEVERITY_CRITICAL: str = "CRITICAL"
