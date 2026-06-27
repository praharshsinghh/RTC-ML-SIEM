"""
src/models/random_forest.py
============================
Random Forest multiclass detector for UNSW-NB15 attack categories.

Design decisions
----------------
class_weight='balanced': UNSW-NB15 is severely imbalanced
(Normal ~1.775M samples, Worms ~139 in train). Without reweighting the
classifier maximises accuracy by ignoring minority classes. 'balanced'
applies sklearn's n_samples / (n_classes * class_count) reweighting.

FPR as primary evaluation metric: a SIEM analyst investigates every alert.
False positives waste analyst time and cause alert fatigue. FPR =
FP / (FP + TN) quantifies the fraction of benign connections that trigger
false alarms. Overall accuracy is misleading here; a model that labels
everything as Normal achieves 87%+ accuracy with zero attack detection.

Impurity-based feature importance: mean decrease in Gini impurity tends to
inflate the importance of high-cardinality numeric features (e.g., sport,
dsport). Use SHAP or permutation importance for reliable production rankings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from src.models import BaseDetector, PredictionResult

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NORMAL_CAT = "unknown"   # how normal traffic is labelled in attack_cat col
MODEL_NAME = "RandomForest"


class RandomForestDetector(BaseDetector):
    """
    Random Forest multiclass detector.

    Parameters
    ----------
    n_estimators : int
        Number of trees. 200 is a good default — more trees reduce variance
        at the cost of memory and inference time; beyond ~500 improvements plateau.
    max_depth : None | int
        None = fully grown trees. Allows the forest to capture complex decision
        boundaries; overfitting is controlled by min_samples_leaf and bagging.
    min_samples_leaf : int
        Prevents fitting noise. 2 means a leaf must cover at least 2 samples,
        which meaningfully prunes the tree for rare classes like Worms.
    class_weight : str | dict
        'balanced' — see module docstring for why this is non-negotiable.
    n_jobs : int
        -1 = use all available CPUs.
    random_state : int
        Reproducibility seed.
    """

    name = MODEL_NAME

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._model: Optional[RandomForestClassifier] = None
        self._classes: Optional[np.ndarray] = None      # ordered class labels (strings)
        self._feature_names: Optional[list[str]] = None

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Fit the RandomForestClassifier.

        Parameters
        ----------
        X_train : array (n_samples, n_features)
        y_train : array (n_samples,) — string attack category labels
        feature_names : list[str], optional — stored for importance export
        """
        logger.info(
            f"[RF] Training on {X_train.shape[0]:,} samples, "
            f"{X_train.shape[1]} features, "
            f"{len(np.unique(y_train))} classes"
        )

        self._feature_names = feature_names
        self._classes = np.unique(y_train)

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0,
        )
        self._model.fit(X_train, y_train)
        logger.info("[RF] Training complete.")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> list[PredictionResult]:
        """
        Return a PredictionResult per row in X.

        confidence = max class probability (the model's certainty in its top pick)
        is_anomaly = True if predicted class is NOT 'unknown' (Normal)
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        proba = self._model.predict_proba(X)                  # (n, n_classes)
        pred_idx = np.argmax(proba, axis=1)
        pred_cats = self._model.classes_[pred_idx]
        confidences = proba[np.arange(len(proba)), pred_idx]

        results: list[PredictionResult] = []
        for i in range(len(X)):
            cat = str(pred_cats[i])
            results.append(
                PredictionResult(
                    is_anomaly=(cat != NORMAL_CAT),
                    attack_cat=cat,
                    confidence=float(confidences[i]),
                    raw_score=float(confidences[i]),
                    model_name=MODEL_NAME,
                )
            )
        return results

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        binary_labels: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute comprehensive evaluation metrics.

        Parameters
        ----------
        X_test : array (n_samples, n_features)
        y_test : array (n_samples,) — string attack_cat labels
        binary_labels : array (n_samples,) — 0=normal, 1=attack
            Used ONLY for FPR. If None, derived from y_test (unknown → 0).

        Returns
        -------
        dict with guaranteed keys (same schema as XGBoostDetector.evaluate):
            macro_f1, weighted_f1, roc_auc,
            fpr (false positive rate on binary task),
            confusion_matrix (list-of-lists),
            per_class (dict keyed by class name → precision/recall/f1/support),
            model_name
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        y_pred = self._model.predict(X_test)
        proba = self._model.predict_proba(X_test)
        classes = self._model.classes_

        # ── Per-class metrics ─────────────────────────────────────────────────
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, zero_division=0
        )
        per_class = {
            cls: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }
            for i, cls in enumerate(classes)
        }

        # ── Aggregate metrics ─────────────────────────────────────────────────
        macro_f1 = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
        weighted_f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

        # ROC-AUC one-vs-rest (requires binarised labels)
        try:
            y_bin = label_binarize(y_test, classes=classes)
            if y_bin.shape[1] == 1:
                # Binary edge case — shouldn't happen here
                roc_auc = float(roc_auc_score(y_bin, proba[:, 1]))
            else:
                roc_auc = round(
                    float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")),
                    4,
                )
        except Exception as exc:
            logger.warning(f"[RF] ROC-AUC computation failed: {exc}")
            roc_auc = None

        # ── False Positive Rate (binary) ──────────────────────────────────────
        # FPR = FP / (FP + TN): "what fraction of NORMAL traffic was flagged?"
        # This is the metric SIEM operators care about most — see module docstring.
        if binary_labels is None:
            binary_labels = (y_test != NORMAL_CAT).astype(int)  # 0=normal, 1=attack

        binary_pred = (y_pred != NORMAL_CAT).astype(int)
        cm_binary = confusion_matrix(binary_labels, binary_pred, labels=[0, 1])
        tn = int(cm_binary[0, 0])
        fp = int(cm_binary[0, 1])
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0

        # ── Confusion matrix (multiclass) ─────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        return {
            "model_name": MODEL_NAME,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "roc_auc": roc_auc,
            "fpr": fpr,
            "fpr_note": (
                f"FPR = FP/(FP+TN) on binary normal-vs-attack. "
                f"FP={fp}, TN={tn}. "
                f"Means {fpr*100:.2f}% of normal traffic is mis-flagged as attack."
            ),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": classes.tolist(),
            "per_class": per_class,
        }

    # ── Feature Importances ───────────────────────────────────────────────────

    def get_feature_importances(self) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame of (feature, importance) sorted descending.
        Importance = mean decrease in Gini impurity across all trees.

        Limitation: impurity-based importance overestimates high-cardinality
        features. Use SHAP or permutation importance for production decisions.
        """
        if self._model is None:
            return None
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save_feature_importances(self, path: Path) -> None:
        df = self.get_feature_importances()
        if df is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"[RF] Feature importances saved → {path}")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save_model(self, path: Path) -> None:
        """Save trained model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "classes": self._classes,
            "feature_names": self._feature_names,
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"[RF] Model saved → {path}")

    def save(self, path: Path) -> None:
        """Alias to satisfy BaseDetector ABC (delegates to save_model)."""
        self.save_model(path)

    @classmethod
    def load_model(cls, path: Path) -> "RandomForestDetector":
        """Load a previously saved RandomForestDetector from disk."""
        path = Path(path)
        payload = joblib.load(path)
        hp = payload["hyperparams"]
        instance = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"],
            class_weight=hp["class_weight"],
            random_state=hp["random_state"],
        )
        instance._model = payload["model"]
        instance._classes = payload["classes"]
        instance._feature_names = payload["feature_names"]
        logger.info(f"[RF] Model loaded ← {path}")
        return instance

    @classmethod
    def load(cls, path: Path) -> "RandomForestDetector":
        """Alias to satisfy BaseDetector ABC (delegates to load_model)."""
        return cls.load_model(path)
