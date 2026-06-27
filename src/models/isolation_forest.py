"""
src/models/isolation_forest.py
===============================
Unsupervised anomaly detector trained on normal-only traffic.

Design decisions
----------------
Training on normal traffic only: Isolation Forest learns a model of normal
behaviour. Training on mixed normal + attack data blends the distributions,
making anomalies harder to isolate. Normal-only training is standard for
one-class classification in intrusion detection.

Why Isolation Forest for network flows: tabular, high-dimensional (43 features),
dominated by normal traffic. IF isolates anomalies near the root of random
binary trees (short path length = anomalous). It scales to millions of rows
O(n log n) and requires no pairwise distance computation.

Percentile thresholding: after fitting, anomaly scores are computed on the
normal training data. The threshold is the Nth percentile of those scores.
At the default N=95, 95% of known normal traffic scores below the threshold;
the remaining 5% represents the accepted false-positive budget on training data.
This approach is fully unsupervised — no labelled attack samples are needed
for threshold calibration.

Anomalous score convention: sklearn's score_samples() returns negative anomaly
scores. We negate them so higher values indicate more anomalous samples,
matching the convention used by the Autoencoder (reconstruction error).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models import BaseDetector, PredictionResult

logger = logging.getLogger(__name__)

MODEL_NAME = "IsolationForest"

# Percentile used to select the anomaly threshold from training normal scores.
# The threshold represents the top 5% most unusual observations among known
# normal traffic.  Any test sample scoring above this is flagged as anomalous.
THRESHOLD_PERCENTILE = 95


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest unsupervised anomaly detector.

    Trained exclusively on normal traffic; no labels are used during fitting.
    A percentile-based threshold on training anomaly scores determines the
    decision boundary at inference time.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.  300 gives robust score estimation while
        remaining practical on large datasets.
    contamination : str | float
        'auto' lets sklearn use the expected contamination implied by the
        average path length formula.  Since we train on normal-only data,
        'auto' is appropriate — we do NOT know the true contamination fraction
        and instead derive our own threshold via percentile thresholding.
    random_state : int
        Reproducibility seed.
    n_jobs : int
        -1 = use all available CPUs.
    threshold_percentile : int
        Percentile of training anomaly scores used to set the decision
        boundary.  Default 95 → top 5% of normal traffic triggers alarm.
    """

    name = MODEL_NAME

    def __init__(
        self,
        n_estimators: int = 300,
        contamination: str | float = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
        threshold_percentile: int = THRESHOLD_PERCENTILE,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.threshold_percentile = threshold_percentile

        self._model: Optional[IsolationForest] = None
        self._threshold: Optional[float] = None       # set after train()
        self._feature_names: Optional[list[str]] = None

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,             # ignored — unsupervised; only normal rows expected
        feature_names: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Fit the IsolationForest on normal-traffic-only data.

        Parameters
        ----------
        X_train : array (n_samples, n_features)
            Should contain ONLY normal (label=0) samples from the
            train_normal_only.parquet file.  Labels are NOT used.
        y_train : array (n_samples,)
            Ignored during fitting (kept to satisfy BaseDetector ABC).
            Pass np.zeros(n_samples) as a placeholder.
        feature_names : list[str], optional
            Stored for diagnostics / logging.
        """
        logger.info(
            f"[IF] Training on {X_train.shape[0]:,} normal samples, "
            f"{X_train.shape[1]} features  (labels ignored — unsupervised)"
        )

        self._feature_names = feature_names

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_train)

        # Set threshold from training anomaly scores (no labels needed).
        raw_scores = self._model.score_samples(X_train)
        anomaly_scores = -raw_scores  # higher = more anomalous
        self._threshold = float(np.percentile(anomaly_scores, self.threshold_percentile))

        logger.info(
            f"[IF] Training complete. "
            f"Threshold (p{self.threshold_percentile}) = {self._threshold:.6f}"
        )

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> list[PredictionResult]:
        """
        Return a PredictionResult per row in X.

        Anomaly score = -score_samples() so higher = more anomalous.
        A sample is flagged (is_anomaly=True) if its score exceeds the
        training-derived percentile threshold.

        Parameters
        ----------
        X : array (n_samples, n_features)
        """
        if self._model is None or self._threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")

        raw_scores = self._model.score_samples(X)
        anomaly_scores = -raw_scores                       # higher = more anomalous

        results: list[PredictionResult] = []
        for i in range(len(X)):
            score = float(anomaly_scores[i])
            is_anom = score > self._threshold
            # Normalise confidence to [0, 1] via sigmoid-like scaling around threshold.
            # This is approximate; the raw_score carries the full information.
            confidence = float(1 / (1 + np.exp(-(score - self._threshold) * 10)))
            results.append(
                PredictionResult(
                    is_anomaly=bool(is_anom),
                    attack_cat="unknown" if not is_anom else "Anomaly",
                    confidence=confidence,
                    raw_score=score,
                    model_name=MODEL_NAME,
                )
            )
        return results

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """
        Evaluate on labelled test data.

        Parameters
        ----------
        X_test : array (n_samples, n_features)
        y_test : array (n_samples,)
            Binary labels: 0 = normal, 1 = attack.

        Returns
        -------
        dict with keys:
            model_name, precision, recall, f1, roc_auc,
            false_positive_rate, confusion_matrix, threshold
        """
        if self._model is None or self._threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")

        raw_scores = self._model.score_samples(X_test)
        anomaly_scores = -raw_scores                       # higher = more anomalous
        y_pred = (anomaly_scores > self._threshold).astype(int)

        # ── Core metrics ──────────────────────────────────────────────────────
        prec  = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
        rec   = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
        f1    = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

        try:
            roc_auc = round(float(roc_auc_score(y_test, anomaly_scores)), 4)
        except Exception as exc:
            logger.warning(f"[IF] ROC-AUC computation failed: {exc}")
            roc_auc = None

        # ── False Positive Rate ───────────────────────────────────────────────
        # FPR = FP / (FP + TN): fraction of NORMAL samples incorrectly flagged.
        # Critical metric for SIEM deployment — see module docstring.
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0

        logger.info(
            f"[IF] Evaluation — precision={prec}, recall={rec}, f1={f1}, "
            f"roc_auc={roc_auc}, FPR={fpr}"
        )

        return {
            "model_name": MODEL_NAME,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "false_positive_rate": fpr,
            "fpr_note": (
                f"FPR = FP/(FP+TN) on binary normal-vs-attack. "
                f"FP={fp}, TN={tn}. "
                f"Means {fpr*100:.2f}% of normal traffic is mis-flagged as attack."
            ),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": ["normal", "attack"],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "threshold": round(self._threshold, 6),
            "threshold_note": (
                f"Threshold = {self._threshold:.6f} "
                f"(p{self.threshold_percentile} of training normal anomaly scores). "
                f"The threshold represents the top {100 - self.threshold_percentile}% "
                f"most unusual observations among known normal traffic."
            ),
        }

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save_model(self, path: Path) -> None:
        """Save trained detector (model + threshold + metadata) via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "threshold": self._threshold,
            "feature_names": self._feature_names,
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "random_state": self.random_state,
                "threshold_percentile": self.threshold_percentile,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"[IF] Model saved → {path}")

    def save(self, path: Path) -> None:
        """Alias to satisfy BaseDetector ABC (delegates to save_model)."""
        self.save_model(path)

    @classmethod
    def load_model(cls, path: Path) -> "IsolationForestDetector":
        """Load a previously saved IsolationForestDetector from disk."""
        path = Path(path)
        payload = joblib.load(path)
        hp = payload["hyperparams"]
        instance = cls(
            n_estimators=hp["n_estimators"],
            contamination=hp["contamination"],
            random_state=hp["random_state"],
            threshold_percentile=hp["threshold_percentile"],
        )
        instance._model = payload["model"]
        instance._threshold = payload["threshold"]
        instance._feature_names = payload["feature_names"]
        logger.info(f"[IF] Model loaded ← {path}")
        return instance

    @classmethod
    def load(cls, path: Path) -> "IsolationForestDetector":
        """Alias to satisfy BaseDetector ABC (delegates to load_model)."""
        return cls.load_model(path)
