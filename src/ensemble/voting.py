"""
src/ensemble/voting.py
=======================
Transparent weighted voting engine for the ensemble layer.

All aggregation is purely arithmetic — no stacking, no meta-learner.
Every decision can be verified by hand from the model weights and votes.

Voting steps
------------
1. Map each model's is_anomaly flag to a binary attack vote (0 or 1).
2. Compute weighted_attack_score = Σ(weight_i × vote_i).
3. Apply threshold: score >= 0.50 → ATTACK, else NORMAL.
4. Compute agreement_score = sum of weights for models matching the verdict.
5. Resolve the final attack category (see resolve_attack_category).
6. Confidence is derived from the score in scoring.py.
"""

from __future__ import annotations

from typing import Optional

from src.models import PredictionResult
from src.ensemble.schemas import (
    MODEL_WEIGHTS,
    ATTACK_THRESHOLD,
    NORMAL_CAT,
    ModelVote,
)


# ─────────────────────────────────────────────────────────────────────────────
# Expected model name order (must match MODEL_WEIGHTS keys)
# ─────────────────────────────────────────────────────────────────────────────
SUPERVISED_MODELS   = {"RandomForest", "XGBoost"}
UNSUPERVISED_MODELS = {"IsolationForest", "DenseAutoencoder"}


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_weighted_attack_score(
    predictions: list[PredictionResult],
) -> float:
    """
    Compute the weighted attack score from all model predictions.

    Parameters
    ----------
    predictions : list[PredictionResult]
        One PredictionResult per model. The model_name attribute is used
        to look up the weight from MODEL_WEIGHTS.

    Returns
    -------
    float in [0, 1]
        Σ(weight_i × attack_vote_i)
        where attack_vote_i = 1.0 if is_anomaly else 0.0

    Raises
    ------
    KeyError
        If a prediction carries an unrecognised model_name.
    """
    score = 0.0
    for pred in predictions:
        w = MODEL_WEIGHTS[pred.model_name]
        vote = 1.0 if pred.is_anomaly else 0.0
        score += w * vote
    return round(score, 6)


def compute_verdict(weighted_attack_score: float) -> str:
    """
    Apply the decision threshold to produce the binary verdict.

    Parameters
    ----------
    weighted_attack_score : float
        Output of compute_weighted_attack_score().

    Returns
    -------
    str : 'ATTACK' if score >= ATTACK_THRESHOLD else 'NORMAL'

    Rationale
    ---------
    Threshold = 0.50 is the natural midpoint of the [0,1] score range.
    It means: "the combined weight of ATTACK voters equals or exceeds
    the combined weight of NORMAL voters."  No empirical tuning of the
    threshold has been performed; this is a principled default that
    treats the combined model evidence symmetrically.
    """
    return "ATTACK" if weighted_attack_score >= ATTACK_THRESHOLD else "NORMAL"


def compute_agreement_score(
    predictions: list[PredictionResult],
    final_verdict: str,
) -> float:
    """
    Compute the fraction of weighted evidence that agrees with the final verdict.

    Formula
    -------
    agreement = Σ(weight_i  for each model that agrees with final_verdict)

    Because all weights sum to 1.0, agreement ∈ [0, 1].

    Agreement = 1.0 → all four models agree with the verdict.
    Agreement = 0.5 → models worth 50% weight agree (borderline case).
    Agreement = 0.0 → impossible (would flip the verdict).

    SOC interpretation
    ------------------
    A high-agreement ATTACK (≥0.85) should be escalated immediately.
    A low-agreement ATTACK (0.50–0.70) warrants second-tier review.
    This mimics how Tier-1 SOC analysts prioritise alerts.

    Parameters
    ----------
    predictions : list[PredictionResult]
    final_verdict : str — 'ATTACK' or 'NORMAL'

    Returns
    -------
    float [0, 1]
    """
    total_agreeing_weight = 0.0
    for pred in predictions:
        w = MODEL_WEIGHTS[pred.model_name]
        model_vote = "ATTACK" if pred.is_anomaly else "NORMAL"
        if model_vote == final_verdict:
            total_agreeing_weight += w
    return round(total_agreeing_weight, 6)


def build_model_votes(
    predictions: list[PredictionResult],
) -> list[ModelVote]:
    """
    Build the list of ModelVote records for the EnsembleResult.

    Each ModelVote captures the model name, weight, vote, category,
    confidence, raw score, and weighted contribution so the full
    decision audit trail is preserved.

    Parameters
    ----------
    predictions : list[PredictionResult]

    Returns
    -------
    list[ModelVote] in the same order as `predictions`
    """
    votes: list[ModelVote] = []
    for pred in predictions:
        w = MODEL_WEIGHTS[pred.model_name]
        vote = "ATTACK" if pred.is_anomaly else "NORMAL"
        votes.append(
            ModelVote(
                model_name=pred.model_name,
                weight=w,
                vote=vote,
                attack_cat=pred.attack_cat,
                confidence=round(pred.confidence, 6),
                raw_score=round(pred.raw_score, 6),
                weighted_contribution=round(w * pred.confidence, 6),
            )
        )
    return votes


def resolve_attack_category(
    predictions: list[PredictionResult],
    final_verdict: str,
) -> str:
    """
    Resolve the final attack category from all model predictions.

    Resolution rules (applied in priority order)
    ---------------------------------------------
    1. If final_verdict == 'NORMAL' → return NORMAL_CAT ('unknown').
       The ensemble has decided this is benign traffic; no attack category
       should be attributed.

    2. Extract only supervised model predictions (RF and XGB) where the
       model voted ATTACK (is_anomaly=True).

    3a. If BOTH supervised models predict the same non-normal category
        → use that category. (Strong supervised consensus.)

    3b. If supervised models DISAGREE on category (or only one of them
        voted ATTACK) → use the category from whichever supervised model
        has the higher raw_score (class probability).
        Rationale: raw_score for RF/XGB = max class probability.
        The model that is MORE confident in its prediction should win
        the category tiebreak. This avoids arbitrary selection and
        produces a defensible decision.

    4. If NO supervised model voted ATTACK (only unsupervised models
       flagged the sample) → return 'Anomaly'.
       The sample is flagged as anomalous but cannot be categorised
       because the supervised models (which know the categories) did
       not fire. This is the canonical "zero-day / unknown threat" case.

    Parameters
    ----------
    predictions : list[PredictionResult]
    final_verdict : str

    Returns
    -------
    str : attack category string
    """
    # Rule 1 — Normal verdict → no category
    if final_verdict == "NORMAL":
        return NORMAL_CAT

    # Extract supervised model predictions that voted ATTACK
    supervised_attack_preds: list[PredictionResult] = [
        p for p in predictions
        if p.model_name in SUPERVISED_MODELS and p.is_anomaly
    ]

    # Rule 4 — No supervised model fired
    if not supervised_attack_preds:
        return "Anomaly"

    # Rule 3a — Single supervised model fired
    if len(supervised_attack_preds) == 1:
        return supervised_attack_preds[0].attack_cat

    # Rule 3a / 3b — Both supervised models fired
    rf_pred  = next((p for p in supervised_attack_preds if p.model_name == "RandomForest"), None)
    xgb_pred = next((p for p in supervised_attack_preds if p.model_name == "XGBoost"), None)

    if rf_pred and xgb_pred:
        # 3a — Both agree
        if rf_pred.attack_cat == xgb_pred.attack_cat:
            return rf_pred.attack_cat
        # 3b — Disagree → take the more confident model's category
        return rf_pred.attack_cat if rf_pred.raw_score >= xgb_pred.raw_score else xgb_pred.attack_cat

    # Fallback (shouldn't be reached with exactly 2 supervised models)
    return supervised_attack_preds[0].attack_cat
