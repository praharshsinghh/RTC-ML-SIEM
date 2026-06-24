"""
src/ensemble/detector.py — EnsembleDetector: Main Correlation Engine
======================================================================
The EnsembleDetector is the central coordinator of Phase 4. It:

  1. Holds references to all four trained detector instances
     (RandomForest, XGBoost, IsolationForest, DenseAutoencoder).
  2. Exposes a predict() method that produces one EnsembleResult per
     input row by calling the voting, confidence, and severity modules.
  3. Exposes an evaluate() method that computes ensemble-level metrics
     (accuracy, precision, recall, F1, FPR, confusion matrix) AND
     agreement statistics (full/partial/split vote breakdowns).

Why is this NOT a meta-learner?
---------------------------------
A meta-learner (stacking) trains a secondary model on the outputs of
the base models. This is a black-box: the stacker learns arbitrary
nonlinear combinations of base predictions. We CANNOT explain to an
analyst or auditor why the stacker fired.

Our EnsembleDetector uses only:
  - Hard-coded arithmetic (weighted sum, threshold comparison)
  - Explicit rule tables (severity matrix, category resolution)
  - No parameters learned from data in the ensemble layer itself

This makes every decision fully traceable: given any EnsembleResult,
an analyst can independently verify the verdict by computing the
weighted sum by hand.

Relationship to SIEM correlation engines
------------------------------------------
A modern SIEM correlation engine (Splunk ES "notable events", IBM QRadar
"offenses", Microsoft Sentinel "incidents") aggregates evidence from
multiple data sources and applies rule-based scoring to produce alerts.
Our EnsembleDetector implements the ML analogue of this architecture:
  - Data sources → four trained ML models
  - Correlation rule → weighted vote + threshold
  - Alert → EnsembleResult with severity grading

The key difference from a commercial SIEM is that our models are trained
on UNSW-NB15 and specialised for network intrusion detection. The
correlation logic (voting.py, scoring.py) is identical in principle.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.models import PredictionResult
from src.models.random_forest import RandomForestDetector
from src.models.xgboost_model import XGBoostDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import DenseAutoencoderDetector

from src.ensemble.schemas import (
    EnsembleResult,
    ModelVote,
    NORMAL_CAT,
)
from src.ensemble.voting import (
    compute_weighted_attack_score,
    compute_verdict,
    compute_agreement_score,
    build_model_votes,
    resolve_attack_category,
)
from src.ensemble.scoring import (
    compute_confidence,
    compute_severity,
)

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    SIEM-style correlation engine aggregating four ML detectors.

    Instantiate with four trained detectors, then call predict() to get
    EnsembleResult objects or evaluate() to compute ensemble metrics.

    Parameters
    ----------
    rf : RandomForestDetector
        Trained Random Forest classifier (Phase 2).
    xgb : XGBoostDetector
        Trained XGBoost classifier (Phase 2).
    iso : IsolationForestDetector
        Trained Isolation Forest anomaly detector (Phase 3).
    ae : DenseAutoencoderDetector
        Trained Dense Autoencoder anomaly detector (Phase 3).
    batch_size : int
        Number of samples processed per batch. Reduce if memory-constrained.
        Does not affect results — only throughput.

    Design note — why hold model references?
    -----------------------------------------
    The EnsembleDetector keeps direct references to the four instantiated
    models rather than file paths or class names. This means:
      - The caller controls model instantiation and load-time (dependency
        injection pattern) — EnsembleDetector is not responsible for I/O.
      - Testing: any mock conforming to BaseDetector can be injected.
      - Flexibility: models can be swapped at runtime (e.g., retrained RF)
        without modifying EnsembleDetector.
    """

    def __init__(
        self,
        rf: RandomForestDetector,
        xgb: XGBoostDetector,
        iso: IsolationForestDetector,
        ae: DenseAutoencoderDetector,
        batch_size: int = 2048,
    ) -> None:
        self.rf  = rf
        self.xgb = xgb
        self.iso = iso
        self.ae  = ae
        self.batch_size = batch_size

        # Ordered list of (model, name) for iteration
        self._detectors = [
            (self.rf,  "RandomForest"),
            (self.xgb, "XGBoost"),
            (self.iso, "IsolationForest"),
            (self.ae,  "DenseAutoencoder"),
        ]

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        timestamps: Optional[list[str]] = None,
    ) -> list[EnsembleResult]:
        """
        Run ensemble inference on X and return one EnsembleResult per row.

        Parameters
        ----------
        X : np.ndarray shape (n_samples, n_features)
            Feature matrix (float32, scaled) — same preprocessing as used
            to train the individual models.
        timestamps : list[str], optional
            ISO-8601 timestamps for each row. If None, the current UTC time
            is used for all rows. Useful for notebook demos; in live SIEM
            deployment the actual packet capture timestamp would be passed.

        Returns
        -------
        list[EnsembleResult] : length n_samples
        """
        n = len(X)
        now_ts = datetime.now(tz=timezone.utc).isoformat()
        if timestamps is None:
            timestamps = [now_ts] * n

        logger.info(f"[Ensemble] Running inference on {n:,} samples …")
        t0 = time.time()

        # --- Step 1: Collect all per-model predictions ----------------------
        # Each model returns list[PredictionResult] of length n.
        all_preds: dict[str, list[PredictionResult]] = {}
        for model, name in self._detectors:
            logger.debug(f"[Ensemble]   → {name}.predict({n} rows)")
            all_preds[name] = model.predict(X)

        # --- Step 2: Assemble per-sample EnsembleResult ---------------------
        results: list[EnsembleResult] = []
        for i in range(n):
            # Collect this sample's predictions from all four models
            row_preds = [all_preds[name][i] for _, name in self._detectors]

            # Weighted voting
            w_score   = compute_weighted_attack_score(row_preds)
            verdict   = compute_verdict(w_score)
            agreement = compute_agreement_score(row_preds, verdict)

            # Confidence and severity
            conf      = compute_confidence(w_score, verdict)
            severity  = compute_severity(verdict, conf, agreement)

            # Category resolution
            cat       = resolve_attack_category(row_preds, verdict)

            # Build audit trail
            votes = build_model_votes(row_preds)

            results.append(
                EnsembleResult(
                    timestamp=timestamps[i],
                    final_verdict=verdict,
                    confidence=conf,
                    severity=severity,
                    agreement_score=agreement,
                    weighted_attack_score=w_score,
                    final_attack_cat=cat,
                    model_votes=votes,
                )
            )

        elapsed = time.time() - t0
        logger.info(
            f"[Ensemble] Inference complete: {n:,} samples in {elapsed:.2f}s "
            f"({n/elapsed:,.0f} samples/sec)"
        )
        return results

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_binary: np.ndarray,
    ) -> dict:
        """
        Compute ensemble-level evaluation metrics.

        Parameters
        ----------
        X_test : np.ndarray (n_samples, n_features)
        y_binary : np.ndarray (n_samples,)
            Ground-truth binary labels: 0 = normal, 1 = attack.
            Derived from the 'label' column of the UNSW-NB15 test parquet.

        Returns
        -------
        dict with keys (suitable for reports/ensemble_evaluation.json):
            accuracy, precision, recall, f1, false_positive_rate,
            confusion_matrix, confusion_matrix_labels,
            tp, tn, fp, fn,
            agreement_stats (dict: full/partial/split vote distributions),
            severity_distribution (dict),
            confidence_stats (mean, std, p25, p50, p75),
            n_samples, n_attack, n_normal
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        logger.info(f"[Ensemble] Evaluating on {len(X_test):,} test samples …")
        results = self.predict(X_test)

        # Binary predictions from ensemble
        y_pred_binary = np.array([
            1 if r.final_verdict == "ATTACK" else 0
            for r in results
        ])

        # --- Core metrics ---------------------------------------------------
        acc  = round(float(accuracy_score(y_binary, y_pred_binary)), 4)
        prec = round(float(precision_score(y_binary, y_pred_binary, zero_division=0)), 4)
        rec  = round(float(recall_score(y_binary, y_pred_binary, zero_division=0)), 4)
        f1   = round(float(f1_score(y_binary, y_pred_binary, zero_division=0)), 4)

        from sklearn.metrics import confusion_matrix as sk_cm
        cm   = sk_cm(y_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0

        # --- Agreement statistics -------------------------------------------
        # Full agreement: all 4 models agree (agreement_score ∈ {0.15, 0.85, 1.0})
        # In practice: agreement=1.0 → all 4 agree; agreement≥0.85 → 3+ agree
        agreement_scores = np.array([r.agreement_score for r in results])
        full_agreement   = int((agreement_scores >= 0.99).sum())      # all 4 agree
        high_agreement   = int(((agreement_scores >= 0.85) &
                                (agreement_scores < 0.99)).sum())     # 3 agree
        split_vote       = int(((agreement_scores >= 0.50) &
                                (agreement_scores < 0.85)).sum())     # slim majority
        low_agreement    = int((agreement_scores < 0.50).sum())       # minority verdict

        # --- Severity distribution (ATTACK samples only) --------------------
        sev_counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "N/A": 0}
        for r in results:
            sev_counts[r.severity] = sev_counts.get(r.severity, 0) + 1

        # --- Confidence statistics -------------------------------------------
        confidences = np.array([r.confidence for r in results])
        conf_stats  = {
            "mean": round(float(confidences.mean()), 4),
            "std":  round(float(confidences.std()),  4),
            "p25":  round(float(np.percentile(confidences, 25)), 4),
            "p50":  round(float(np.percentile(confidences, 50)), 4),
            "p75":  round(float(np.percentile(confidences, 75)), 4),
        }

        # --- Individual model performance comparison ------------------------
        individual_metrics = self._compute_individual_metrics(
            results, y_binary, y_pred_binary
        )

        n_samples = len(results)
        n_attack  = int(y_binary.sum())
        n_normal  = n_samples - n_attack

        logger.info(
            f"[Ensemble] Results: accuracy={acc}, precision={prec}, "
            f"recall={rec}, f1={f1}, FPR={fpr}"
        )

        return {
            "model_name": "EnsembleDetector",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_positive_rate": fpr,
            "fpr_note": (
                f"FPR = FP/(FP+TN) on binary normal-vs-attack. "
                f"FP={fp}, TN={tn}. "
                f"Means {fpr*100:.2f}% of normal traffic triggers a false alarm."
            ),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": ["normal", "attack"],
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "agreement_stats": {
                "full_agreement_all4":  full_agreement,
                "high_agreement_3of4":  high_agreement,
                "split_vote_slim_majority": split_vote,
                "low_agreement_minority": low_agreement,
                "mean_agreement": round(float(agreement_scores.mean()), 4),
                "std_agreement":  round(float(agreement_scores.std()), 4),
            },
            "severity_distribution": sev_counts,
            "confidence_stats": conf_stats,
            "individual_model_comparison": individual_metrics,
            "n_samples": n_samples,
            "n_attack": n_attack,
            "n_normal": n_normal,
        }

    def _compute_individual_metrics(
        self,
        results: list[EnsembleResult],
        y_binary: np.ndarray,
        y_ensemble_pred: np.ndarray,
    ) -> dict:
        """
        Extract per-model binary predictions from ModelVote records and
        compute precision/recall/F1/FPR for each model individually.
        Used in the evaluation report to compare ensemble vs each model.

        Parameters
        ----------
        results : list[EnsembleResult]
        y_binary : np.ndarray — ground truth (0=normal, 1=attack)
        y_ensemble_pred : np.ndarray — ensemble binary predictions

        Returns
        -------
        dict keyed by model name, each containing precision/recall/f1/fpr
        """
        from sklearn.metrics import (
            confusion_matrix as sk_cm,
            f1_score,
            precision_score,
            recall_score,
        )

        # Collect per-model binary predictions from vote audit trail
        model_preds: dict[str, list[int]] = {
            "RandomForest": [], "XGBoost": [],
            "IsolationForest": [], "DenseAutoencoder": [],
        }
        for r in results:
            for vote in r.model_votes:
                model_preds[vote.model_name].append(
                    1 if vote.vote == "ATTACK" else 0
                )

        individual: dict[str, dict] = {}
        for name, preds in model_preds.items():
            y_pred = np.array(preds)
            cm   = sk_cm(y_binary, y_pred, labels=[0, 1])
            tn_m, fp_m = int(cm[0,0]), int(cm[0,1])
            fpr_m = round(fp_m / (fp_m + tn_m), 6) if (fp_m + tn_m) > 0 else 0.0
            individual[name] = {
                "precision": round(float(precision_score(y_binary, y_pred, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_binary, y_pred, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_binary, y_pred, zero_division=0)), 4),
                "false_positive_rate": fpr_m,
            }

        # Add ensemble entry for side-by-side comparison
        cm_ens = sk_cm(y_binary, y_ensemble_pred, labels=[0, 1])
        tn_e, fp_e = int(cm_ens[0,0]), int(cm_ens[0,1])
        individual["Ensemble"] = {
            "precision": round(float(precision_score(y_binary, y_ensemble_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_binary, y_ensemble_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_binary, y_ensemble_pred, zero_division=0)), 4),
            "false_positive_rate": round(fp_e / (fp_e + tn_e), 6) if (fp_e + tn_e) > 0 else 0.0,
        }
        return individual

    # ── Convenience: load all four models from standard paths ─────────────────

    @classmethod
    def from_disk(
        cls,
        rf_path: Path,
        xgb_path: Path,
        iso_path: Path,
        ae_path: Path,
        batch_size: int = 2048,
    ) -> "EnsembleDetector":
        """
        Convenience factory: load all four models from disk and return a
        ready-to-use EnsembleDetector.

        Parameters
        ----------
        rf_path  : Path to rf_detector.joblib
        xgb_path : Path to xgb_detector.joblib
        iso_path : Path to isolation_forest.joblib
        ae_path  : Path to autoencoder.pt

        Returns
        -------
        EnsembleDetector

        Example
        -------
        >>> ensemble = EnsembleDetector.from_disk(
        ...     rf_path  = Path("data/models/rf_detector.joblib"),
        ...     xgb_path = Path("data/models/xgb_detector.joblib"),
        ...     iso_path = Path("models/isolation_forest.joblib"),
        ...     ae_path  = Path("models/autoencoder.pt"),
        ... )
        """
        logger.info("[Ensemble] Loading all four models from disk …")
        rf  = RandomForestDetector.load_model(rf_path)
        xgb = XGBoostDetector.load_model(xgb_path)
        iso = IsolationForestDetector.load_model(iso_path)
        ae  = DenseAutoencoderDetector.load_model(ae_path)
        logger.info("[Ensemble] All models loaded.")
        return cls(rf=rf, xgb=xgb, iso=iso, ae=ae, batch_size=batch_size)
