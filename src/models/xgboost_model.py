"""
src/models/xgboost_model.py — XGBoost Detector
================================================
Multiclass supervised detector trained on UNSW-NB15 attack_cat labels.

Design decisions
----------------
**Why XGBoost uses compute_sample_weight instead of scale_pos_weight?**
    scale_pos_weight is designed for BINARY classification only — it's the
    ratio of negative to positive class counts and adjusts the gradient for
    the positive class. For multiclass (10 categories), XGBoost does not
    natively support scale_pos_weight per class. Instead, we use
    sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)
    to produce per-sample weights, which are passed as the sample_weight
    argument to model.fit(). This achieves the same balancing effect as
    RF's class_weight='balanced' but works across all XGBoost multiclass
    objectives (softmax, softprob).

**Why gain-based feature importance differs from RF impurity importance?**
    XGBoost records importance_type='gain' by default: the average training
    loss reduction (gain) when a feature is used in a split, averaged across
    all splits and trees. This favours features that are *useful* when selected,
    not just frequently selected. RF's mean-decrease-in-Gini counts how often
    a feature is used weighted by node purity. The two measures frequently
    disagree — a feature that splits occasionally but decisively ranks high
    in gain, low in impurity frequency. Use SHAP TreeExplainer for a unified
    view that agrees with both causal and predictive interpretations.

**Why eval_metric='mlogloss' and not 'merror'?**
    mlogloss (multiclass log-loss) penalises confident wrong predictions more
    harshly than merror (misclassification rate), which pushes the model to
    be well-calibrated. For a SIEM we care about confidence scores (used in
    the ensemble layer), so calibration matters more than raw accuracy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.models import BaseDetector, PredictionResult

logger = logging.getLogger(__name__)

NORMAL_CAT = "unknown"
MODEL_NAME = "XGBoost"


class XGBoostDetector(BaseDetector):
    """
    XGBoost multiclass detector.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth per round. 6 is XGBoost's default and performs well
        on tabular network-flow data.
    learning_rate : float
        Shrinkage factor applied to each tree's contribution. Lower = more trees
        needed but better generalisation; 0.1 is a solid starting point.
    subsample : float
        Fraction of training rows sampled per tree (without replacement). 0.8
        adds stochasticity → reduces overfitting on majority classes.
    colsample_bytree : float
        Fraction of features sampled per tree. 0.8 acts like RF's max_features,
        diversifying individual trees.
    eval_metric : str
        Loss metric tracked internally during training. 'mlogloss' = multiclass
        log-loss — see module docstring.
    random_state : int
        Reproducibility seed (mapped to XGBoost's seed parameter).
    n_jobs : int
        -1 = all CPUs.
    """

    name = MODEL_NAME

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "mlogloss",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._model: Optional[XGBClassifier] = None
        self._classes: Optional[np.ndarray] = None       # string class labels
        self._label_to_int: Optional[dict] = None        # str → int encoding
        self._int_to_label: Optional[dict] = None        # int → str decoding
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
        Fit the XGBClassifier.

        XGBoost requires integer class labels for multiclass, so we encode
        y_train → int, then decode predictions back to category strings.
        Class imbalance is handled via compute_sample_weight('balanced').
        """
        logger.info(
            f"[XGB] Training on {X_train.shape[0]:,} samples, "
            f"{X_train.shape[1]} features"
        )

        self._feature_names = feature_names
        self._classes = np.unique(y_train)

        # Encode string labels → integers (XGBoost requirement for multiclass)
        self._label_to_int = {lbl: i for i, lbl in enumerate(self._classes)}
        self._int_to_label = {i: lbl for lbl, i in self._label_to_int.items()}
        y_int = np.array([self._label_to_int[lbl] for lbl in y_train])

        # Compute per-sample weights to handle class imbalance
        # This is the XGBoost-appropriate substitute for RF's class_weight='balanced'
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_int)

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=len(self._classes),
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
        )
        self._model.fit(X_train, y_int, sample_weight=sample_weights)
        logger.info("[XGB] Training complete.")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> list[PredictionResult]:
        """
        Return a PredictionResult per row in X.
        Internally XGBoost predicts integer class indices; we decode to strings.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        proba = self._model.predict_proba(X)                 # (n, n_classes)
        pred_int = np.argmax(proba, axis=1)
        confidences = proba[np.arange(len(proba)), pred_int]
        pred_cats = np.array([self._int_to_label[i] for i in pred_int])

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
        Evaluate the model. Returns IDENTICAL key schema to RandomForestDetector
        so Phase 4 ensemble can compare both without special-casing.

        Parameters
        ----------
        X_test : array (n_samples, n_features)
        y_test : array (n_samples,) — string attack_cat labels
        binary_labels : array (n_samples,) — 0=normal, 1=attack
            Used ONLY for FPR. Derived from y_test if None.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Encode y_test to int for XGBoost, handling unseen labels gracefully
        y_int = np.array([
            self._label_to_int.get(lbl, -1) for lbl in y_test
        ])

        proba = self._model.predict_proba(X_test)              # (n, n_classes)
        pred_int = np.argmax(proba, axis=1)

        # Decode predictions back to string labels
        y_pred_str = np.array([self._int_to_label[i] for i in pred_int])
        classes = self._classes

        # ── Per-class metrics ─────────────────────────────────────────────────
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_test, y_pred_str, labels=classes, zero_division=0
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
        macro_f1 = round(float(f1_score(y_test, y_pred_str, average="macro", zero_division=0)), 4)
        weighted_f1 = round(float(f1_score(y_test, y_pred_str, average="weighted", zero_division=0)), 4)

        # ROC-AUC one-vs-rest
        try:
            y_bin = label_binarize(y_test, classes=classes)
            if y_bin.shape[1] == 1:
                roc_auc = float(roc_auc_score(y_bin, proba[:, 1]))
            else:
                roc_auc = round(
                    float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")),
                    4,
                )
        except Exception as exc:
            logger.warning(f"[XGB] ROC-AUC computation failed: {exc}")
            roc_auc = None

        # ── False Positive Rate (binary) ──────────────────────────────────────
        if binary_labels is None:
            binary_labels = (y_test != NORMAL_CAT).astype(int)

        binary_pred = (y_pred_str != NORMAL_CAT).astype(int)
        cm_binary = confusion_matrix(binary_labels, binary_pred, labels=[0, 1])
        tn = int(cm_binary[0, 0])
        fp = int(cm_binary[0, 1])
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0

        # ── Confusion matrix (multiclass) ─────────────────────────────────────
        cm = confusion_matrix(y_test, y_pred_str, labels=classes)

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

    def get_feature_importances(self, importance_type: str = "gain") -> Optional[pd.DataFrame]:
        """
        Return a DataFrame of (feature, importance) sorted descending.

        Parameters
        ----------
        importance_type : str
            'gain' (default) — average loss reduction per split (recommended)
            'weight' — number of times feature is used in splits
            'cover' — average sample coverage per split

        Note
        ----
        Gain-based importance is generally more reliable than weight-based.
        Still, it can disagree with SHAP values; treat as a diagnostic, not truth.
        """
        if self._model is None:
            return None
        importances = self._model.feature_importances_   # default = gain
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save_feature_importances(self, path: Path) -> None:
        df = self.get_feature_importances()
        if df is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"[XGB] Feature importances saved → {path}")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save_model(self, path: Path) -> None:
        """Save the trained detector (model + label maps + metadata) via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "classes": self._classes,
            "label_to_int": self._label_to_int,
            "int_to_label": self._int_to_label,
            "feature_names": self._feature_names,
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "eval_metric": self.eval_metric,
                "random_state": self.random_state,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"[XGB] Model saved → {path}")

    def save(self, path: Path) -> None:
        """Alias to satisfy BaseDetector ABC."""
        self.save_model(path)

    @classmethod
    def load_model(cls, path: Path) -> "XGBoostDetector":
        """Load a previously saved XGBoostDetector from disk."""
        path = Path(path)
        payload = joblib.load(path)
        hp = payload["hyperparams"]
        instance = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"],
            subsample=hp["subsample"],
            colsample_bytree=hp["colsample_bytree"],
            eval_metric=hp["eval_metric"],
            random_state=hp["random_state"],
        )
        instance._model = payload["model"]
        instance._classes = payload["classes"]
        instance._label_to_int = payload["label_to_int"]
        instance._int_to_label = payload["int_to_label"]
        instance._feature_names = payload["feature_names"]
        logger.info(f"[XGB] Model loaded ← {path}")
        return instance

    @classmethod
    def load(cls, path: Path) -> "XGBoostDetector":
        """Alias to satisfy BaseDetector ABC."""
        return cls.load_model(path)
