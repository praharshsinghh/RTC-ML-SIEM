"""
src/ensemble/detector.py
=========================
SIEM-style correlation engine aggregating four ML detectors.

The EnsembleDetector holds references to the four trained model instances
and exposes predict() and evaluate() methods. Aggregation uses transparent,
rule-based weighted voting — no stacking or meta-learner — so every
decision is fully traceable by hand computation.

Relationship to SIEM correlation engines
-----------------------------------------
Commercial SIEM engines (Splunk ES, IBM QRadar, Microsoft Sentinel) aggregate
evidence from multiple sources and apply rule-based scoring to produce alerts.
The EnsembleDetector is the ML analogue: four trained models serve as data
sources, weighted voting replaces correlation rules, and EnsembleResult is
the alert. The transparency principle is identical — every alert has a
traceable, arithmetic cause.
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

    Instantiate with four trained detectors, then call predict() or
    evaluate(). Models are injected as dependencies so the ensemble layer
    controls no I/O and is independently testable.

    Parameters
    ----------
    rf : RandomForestDetector
    xgb : XGBoostDetector
    iso : IsolationForestDetector
    ae : DenseAutoencoderDetector
    batch_size : int
        Samples per inference batch. Affects throughput, not results.
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

        self._detectors = [
            (self.rf,  "RandomForest"),
            (self.xgb, "XGBoost"),
            (self.iso, "IsolationForest"),
            (self.ae,  "DenseAutoencoder"),
        ]

    def predict(
        self,
        X: np.ndarray,
        timestamps: Optional[list[str]] = None,
    ) -> list[EnsembleResult]:
        """
        Run ensemble inference on X and return one EnsembleResult per row.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Float32 feature matrix, preprocessed identically to training data.
        timestamps : list[str], optional
            ISO-8601 timestamps per row. Defaults to current UTC for all rows.

        Returns
        -------
        list[EnsembleResult], length n_samples
        """
        n = len(X)
        now_ts = datetime.now(tz=timezone.utc).isoformat()
        if timestamps is None:
            timestamps = [now_ts] * n

        logger.info(f"[Ensemble] Running inference on {n:,} samples")
        t0 = time.time()

        all_preds: dict[str, list[PredictionResult]] = {}
        for model, name in self._detectors:
            logger.debug(f"[Ensemble]   {name}.predict({n} rows)")
            all_preds[name] = model.predict(X)

        results: list[EnsembleResult] = []
        for i in range(n):
            row_preds = [all_preds[name][i] for _, name in self._detectors]

            w_score   = compute_weighted_attack_score(row_preds)
            verdict   = compute_verdict(w_score)
            agreement = compute_agreement_score(row_preds, verdict)
            conf      = compute_confidence(w_score, verdict)
            severity  = compute_severity(verdict, conf, agreement)
            cat       = resolve_attack_category(row_preds, verdict)
            votes     = build_model_votes(row_preds)

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

    def evaluate(
        self,
        X_test: np.ndarray,
        y_binary: np.ndarray,
    ) -> dict:
        """
        Compute ensemble-level evaluation metrics.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_samples, n_features)
        y_binary : np.ndarray, shape (n_samples,)
            Ground-truth binary labels: 0 = normal, 1 = attack.

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1, false_positive_rate,
        confusion_matrix, tp, tn, fp, fn, agreement_stats, severity_distribution,
        confidence_stats, individual_model_comparison, n_samples, n_attack, n_normal.
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        logger.info(f"[Ensemble] Evaluating on {len(X_test):,} test samples")
        results = self.predict(X_test)

        y_pred_binary = np.array([
            1 if r.final_verdict == "ATTACK" else 0
            for r in results
        ])

        acc  = round(float(accuracy_score(y_binary, y_pred_binary)), 4)
        prec = round(float(precision_score(y_binary, y_pred_binary, zero_division=0)), 4)
        rec  = round(float(recall_score(y_binary, y_pred_binary, zero_division=0)), 4)
        f1   = round(float(f1_score(y_binary, y_pred_binary, zero_division=0)), 4)

        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0

        agreement_scores = np.array([r.agreement_score for r in results])
        full_agreement   = int((agreement_scores >= 0.99).sum())
        high_agreement   = int(((agreement_scores >= 0.85) & (agreement_scores < 0.99)).sum())
        split_vote       = int(((agreement_scores >= 0.50) & (agreement_scores < 0.85)).sum())
        low_agreement    = int((agreement_scores < 0.50).sum())

        sev_counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "N/A": 0}
        for r in results:
            sev_counts[r.severity] = sev_counts.get(r.severity, 0) + 1

        confidences = np.array([r.confidence for r in results])
        conf_stats  = {
            "mean": round(float(confidences.mean()), 4),
            "std":  round(float(confidences.std()),  4),
            "p25":  round(float(np.percentile(confidences, 25)), 4),
            "p50":  round(float(np.percentile(confidences, 50)), 4),
            "p75":  round(float(np.percentile(confidences, 75)), 4),
        }

        individual_metrics = self._compute_individual_metrics(
            results, y_binary, y_pred_binary
        )

        n_samples = len(results)
        n_attack  = int(y_binary.sum())
        n_normal  = n_samples - n_attack

        logger.info(
            f"[Ensemble] accuracy={acc}, precision={prec}, "
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
                f"FPR = FP/(FP+TN). FP={fp}, TN={tn}. "
                f"{fpr*100:.2f}% of normal traffic triggers a false alarm."
            ),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": ["normal", "attack"],
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "agreement_stats": {
                "full_agreement_all4":       full_agreement,
                "high_agreement_3of4":       high_agreement,
                "split_vote_slim_majority":  split_vote,
                "low_agreement_minority":    low_agreement,
                "mean_agreement": round(float(agreement_scores.mean()), 4),
                "std_agreement":  round(float(agreement_scores.std()),  4),
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
        Compute precision/recall/F1/FPR for each model individually.

        Extracts per-model binary predictions from the ModelVote audit trail
        embedded in each EnsembleResult. Also adds an 'Ensemble' entry for
        side-by-side comparison.

        Parameters
        ----------
        results : list[EnsembleResult]
        y_binary : np.ndarray — ground truth
        y_ensemble_pred : np.ndarray — ensemble binary predictions

        Returns
        -------
        dict keyed by model name
        """
        from sklearn.metrics import (
            confusion_matrix as sk_cm,
            f1_score,
            precision_score,
            recall_score,
        )

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
            cm     = sk_cm(y_binary, y_pred, labels=[0, 1])
            tn_m, fp_m = int(cm[0,0]), int(cm[0,1])
            fpr_m = round(fp_m / (fp_m + tn_m), 6) if (fp_m + tn_m) > 0 else 0.0
            individual[name] = {
                "precision": round(float(precision_score(y_binary, y_pred, zero_division=0)), 4),
                "recall":    round(float(recall_score(y_binary, y_pred, zero_division=0)), 4),
                "f1":        round(float(f1_score(y_binary, y_pred, zero_division=0)), 4),
                "false_positive_rate": fpr_m,
            }

        cm_ens = sk_cm(y_binary, y_ensemble_pred, labels=[0, 1])
        tn_e, fp_e = int(cm_ens[0,0]), int(cm_ens[0,1])
        individual["Ensemble"] = {
            "precision": round(float(precision_score(y_binary, y_ensemble_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_binary, y_ensemble_pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(y_binary, y_ensemble_pred, zero_division=0)), 4),
            "false_positive_rate": round(fp_e / (fp_e + tn_e), 6) if (fp_e + tn_e) > 0 else 0.0,
        }
        return individual

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
        Load all four models from disk and return a ready-to-use EnsembleDetector.

        Parameters
        ----------
        rf_path  : Path to rf_detector.joblib
        xgb_path : Path to xgb_detector.joblib
        iso_path : Path to isolation_forest.joblib
        ae_path  : Path to autoencoder.pt

        Returns
        -------
        EnsembleDetector
        """
        logger.info("[Ensemble] Loading all four models from disk")
        rf  = RandomForestDetector.load_model(rf_path)
        xgb = XGBoostDetector.load_model(xgb_path)
        iso = IsolationForestDetector.load_model(iso_path)
        ae  = DenseAutoencoderDetector.load_model(ae_path)
        logger.info("[Ensemble] All models loaded")
        return cls(rf=rf, xgb=xgb, iso=iso, ae=ae, batch_size=batch_size)
