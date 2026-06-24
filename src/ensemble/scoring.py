"""
src/ensemble/scoring.py — Confidence & Severity Calculation
============================================================
Translates the continuous weighted_attack_score and agreement_score
into human-readable SIEM output: a [0,1] confidence value and a
SOC-priority severity band.

Design philosophy — rule-based transparency
-------------------------------------------
These functions use explicit if/elif rule tables, NOT learned functions.
This is a deliberate design choice:

  1. Explainability: An analyst can read the severity table and immediately
     understand why a particular alert was graded HIGH vs CRITICAL. There
     is no "because the neural network said so."

  2. Auditability: Security policies typically mandate explainable alert
     prioritisation (e.g. PCI-DSS, ISO 27001 log management clauses).
     Rule-based grading satisfies this requirement; a black-box grader
     would not.

  3. Tuneability: Threshold adjustments are one-line edits visible in
     source control, not model retraining events.

  4. Viva defence: "Our severity grading mirrors SOC SLA playbooks where
     CRITICAL alerts require sub-15-minute response and LOW alerts are
     queued for weekly review."

Severity Bands (SOC-aligned)
-----------------------------
  CRITICAL : confidence ≥ 0.85 AND agreement ≥ 0.85
             → near-unanimous high-confidence attack; escalate immediately
  HIGH     : confidence ≥ 0.70 AND agreement ≥ 0.70
             → strong evidence; assign to Tier-2 analyst within 1 hour
  MEDIUM   : confidence ≥ 0.50 AND agreement ≥ 0.50
             → moderate evidence; queue for Tier-1 triage within 4 hours
  LOW      : any other ATTACK verdict
             → weak or borderline evidence; include in daily threat report
  (NORMAL traffic never receives a severity band — field is 'N/A')

Confidence Derivation
---------------------
The ensemble confidence mirrors the distance of the weighted_attack_score
from the decision boundary (0.50):
  - For ATTACK verdicts: confidence = 2 × score − 1  (linear in [0,1])
  - For NORMAL verdicts: confidence = 1 − 2 × score  (linear in [0,1])
  Both formulae produce 0.0 at the boundary and 1.0 at the extremes.

This linear mapping is intentional — it is monotonic, interpretable, and
avoids the distribution assumptions that a sigmoid or softmax would impose.
A confidence of 0.80 literally means the ensemble score is 0.90 on a 0-1
scale (0.20 above the 0.50 midpoint on a [0.5,1.0] attack scale).
"""

from __future__ import annotations

from src.ensemble.schemas import (
    ATTACK_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(
    weighted_attack_score: float,
    final_verdict: str,
) -> float:
    """
    Compute ensemble confidence [0, 1] from the weighted attack score.

    Formula
    -------
    ATTACK: confidence = 2 × score − 1
        At score=0.50 → confidence=0.00  (borderline, not confident)
        At score=1.00 → confidence=1.00  (unanimous ATTACK)

    NORMAL: confidence = 1 − 2 × score
        At score=0.50 → confidence=0.00  (borderline)
        At score=0.00 → confidence=1.00  (unanimous NORMAL)

    The factor-of-2 re-scales the half-range [0.5, 1.0] or [0.0, 0.5]
    back to a full [0, 1] confidence range so analysts can interpret
    0.9 as "very confident" regardless of which verdict was made.

    Parameters
    ----------
    weighted_attack_score : float [0, 1]
    final_verdict : str — 'ATTACK' or 'NORMAL'

    Returns
    -------
    float [0, 1]
    """
    if final_verdict == "ATTACK":
        raw = 2.0 * weighted_attack_score - 1.0
    else:
        raw = 1.0 - 2.0 * weighted_attack_score
    # Clamp to [0, 1] to guard against floating-point imprecision
    return round(max(0.0, min(1.0, raw)), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Severity
# ─────────────────────────────────────────────────────────────────────────────

def compute_severity(
    final_verdict: str,
    confidence: float,
    agreement_score: float,
) -> str:
    """
    Assign a SOC severity band to an ensemble prediction.

    Parameters
    ----------
    final_verdict : str — 'ATTACK' or 'NORMAL'
    confidence : float [0, 1] — from compute_confidence()
    agreement_score : float [0, 1] — from compute_agreement_score()

    Returns
    -------
    str : 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'N/A'
        'N/A' is returned for NORMAL verdicts (no alert to prioritise).

    Severity matrix
    ---------------
    CRITICAL : conf ≥ 0.85 AND agreement ≥ 0.85
    HIGH     : conf ≥ 0.70 AND agreement ≥ 0.70
    MEDIUM   : conf ≥ 0.50 AND agreement ≥ 0.50
    LOW      : any ATTACK not meeting the above thresholds

    Rationale for two-dimensional matrix (confidence + agreement)
    -------------------------------------------------------------
    Using ONLY confidence to grade severity would ignore whether the
    models agreed with each other. Consider two scenarios:
      - Score = 0.85 because BOTH RF and XGB voted ATTACK (full weight 0.70)
        PLUS IF voted ATTACK (0.15 more) → four-model agreement, highly reliable
      - Score = 0.85 because only RF and XGB voted ATTACK (0.70 total weight)
        but IF and AE voted NORMAL → the score is the same but only 2 models agree

    Adding agreement_score to the severity matrix distinguishes these cases:
    the second scenario would score HIGH (agreement=0.70), not CRITICAL,
    ensuring CRITICAL alerts genuinely reflect multi-model consensus.

    Viva defence: "Our severity matrix uses both confidence and agreement
    to prevent a scenario where a single high-weight model produces a
    CRITICAL alert that the anomaly detectors contradict. The dual threshold
    mirrors how real SOC triage playbooks require corroborating evidence
    from multiple data sources before escalating to Tier-2."
    """
    if final_verdict != "ATTACK":
        return "N/A"

    if confidence >= 0.85 and agreement_score >= 0.85:
        return SEVERITY_CRITICAL
    elif confidence >= 0.70 and agreement_score >= 0.70:
        return SEVERITY_HIGH
    elif confidence >= 0.50 and agreement_score >= 0.50:
        return SEVERITY_MEDIUM
    else:
        return SEVERITY_LOW
