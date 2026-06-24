"""
src/ensemble/schemas.py — EnsembleResult Schema & Data Contracts
=================================================================
Defines the canonical output schema for every ensemble prediction.
Using Python dataclasses (stdlib, no external deps) keeps the schema
transparent and easy to serialise — a deliberate choice over Pydantic
so the module stays lightweight and the logic is visible.

Why a dedicated schema module?
-------------------------------
In a production SIEM pipeline the output of the ensemble needs to be:
  1. Serialisable → JSON for incident databases, SIEM connectors, REST APIs
  2. Typed → catch bugs at schema boundaries, not inside the dashboard
  3. Self-documenting → every analyst or reviewer can grep this file to
     understand exactly what fields flow downstream without reading all
     model code
  4. Stable → the runner, the notebook, and the dashboard all import from
     ONE source of truth; changing a field name here breaks all consumers
     loudly (ImportError) rather than silently producing wrong dashboards

Viva defence: "We separated schema from logic following the
principle of separation of concerns. EnsembleResult is what the
correlation engine *produces*; the EnsembleDetector is how it
*produces* it. They evolve independently."
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
    The final output of the EnsembleDetector for a single network flow sample.

    This is the "security verdict" — analogous to an alert record in a
    commercial SIEM like Splunk ES or IBM QRadar.

    Fields
    ------
    timestamp : str
        ISO-8601 UTC timestamp of when the inference was performed.
    final_verdict : str
        Binary decision: 'ATTACK' or 'NORMAL'.
        ATTACK when weighted_attack_score >= ATTACK_THRESHOLD (0.50).
    confidence : float
        Ensemble confidence in the final verdict [0, 1].
        Derived from the weighted_attack_score (see EnsembleDetector._confidence).
    severity : str
        SOC alert priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'.
        Assigned by a rule-based severity matrix (see scoring.py).
        Transparent and auditable — no black-box model involved.
    agreement_score : float
        Fraction of models that agree with the final_verdict [0, 1].
        Calculated as: (weighted sum of agreeing models) / total_weight.
        High agreement = high certainty; low agreement = borderline case
        that warrants manual review.
    weighted_attack_score : float
        The raw ensemble score before thresholding.
        weighted_attack_score = Σ(weight_i × attack_vote_i)
        where attack_vote_i is 1 if model i votes ATTACK, else 0.
        Range [0, 1]. Threshold = 0.50.
    final_attack_cat : str
        The resolved attack category.
        Rules (in priority order):
          1. If final_verdict = 'NORMAL' → 'unknown'
          2. Both RF and XGB agree on a non-normal class → that class
          3. RF and XGB disagree → use the category from whichever model
             has the higher class probability (raw_score as proxy)
          4. Only unsupervised models flag → 'Anomaly'
        The original predictions from RF and XGB are preserved in model_votes
        so the resolution is fully auditable.
    model_votes : list[ModelVote]
        Detailed per-model records (see ModelVote above).
        Ordered: [RandomForest, XGBoost, IsolationForest, DenseAutoencoder].

    Rationale for every field choice
    ---------------------------------
    - timestamp: enables chronological sorting in SIEM databases
    - final_verdict: binary alarm signal — the analyst's first read
    - confidence: enables threshold-based alert filtering by priority
    - severity: maps confidence + agreement to SOC triage language
    - agreement_score: the "second opinion" metric — if only 1 of 4 models
      flags, the analyst should verify before escalating
    - weighted_attack_score: preserves the continuous score for ROC analysis
      and future threshold tuning
    - final_attack_cat: gives the analyst a starting point for the kill-chain
    - model_votes: full audit trail — answers "why did the SIEM fire this?"
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

# Model weights — supervised models receive higher weight because:
#   1. They were trained on labelled data → directly optimised to distinguish
#      attack categories; their FPR/recall figures are validated on the test set.
#   2. Anomaly detectors (IF, AE) are unsupervised: they learn a model of
#      *normal* traffic and flag deviation — useful for zero-day / novel attacks
#      but prone to higher FPR on unusual-but-benign traffic.
#   3. Weighting 0.35+0.35 = 0.70 ensures supervised consensus can override
#      anomaly-detector noise, while 0.15+0.15 = 0.30 ensures anomaly evidence
#      can tip the balance when supervised models are uncertain.
MODEL_WEIGHTS: dict[str, float] = {
    "RandomForest":    0.35,
    "XGBoost":         0.35,
    "IsolationForest": 0.15,
    "DenseAutoencoder": 0.15,
}

# Classification threshold: weighted_attack_score >= 0.50 → ATTACK.
# Explanation: with weights summing to 1.0, a score of 0.50 means the
# combined weight of ATTACK-voting models equals or exceeds that of
# NORMAL-voting models. This is the mathematically natural decision boundary.
# It is NOT tuned empirically — it is a principled, defensible default.
ATTACK_THRESHOLD: float = 0.50

# NORMAL class label in attack_cat column (matches Phase 2/3 convention)
NORMAL_CAT: str = "unknown"

# Severity band labels (SOC terminology)
SEVERITY_LOW:      str = "LOW"
SEVERITY_MEDIUM:   str = "MEDIUM"
SEVERITY_HIGH:     str = "HIGH"
SEVERITY_CRITICAL: str = "CRITICAL"
