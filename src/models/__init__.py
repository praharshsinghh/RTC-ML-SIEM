"""
Shared Base Interfaces for All Detection Models
================================================
Every model module (random_forest.py, xgboost_model.py, etc.) must subclass
BaseDetector and implement train(), predict(), evaluate().

Why a base class?
  - The ensemble layer calls model.predict() on all 4 detectors uniformly
  - The pipeline can swap models without changing any downstream code
  - evaluate() always returns the same dict schema → easy comparison table
  - Forces consistent model serialization (save/load)

Consistent interface pattern is a standard design choice in ML systems
(cf. sklearn's BaseEstimator). Defend this as "loose coupling for extensibility."
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np


class PredictionResult:
    """
    Standardized output from any model's predict() call.
    The ensemble layer expects this format from all 4 detectors.
    """
    __slots__ = (
        "is_anomaly",      # bool: True if flagged as attack/anomaly
        "attack_cat",      # str: predicted category ('Normal', 'DoS', etc.) — supervised only
        "confidence",      # float [0,1]: model's confidence in this prediction
        "raw_score",       # float: raw output (probability, anomaly score, recon error)
        "model_name",      # str: which model produced this
    )

    def __init__(
        self,
        is_anomaly: bool,
        attack_cat: str,
        confidence: float,
        raw_score: float,
        model_name: str,
    ):
        self.is_anomaly = is_anomaly
        self.attack_cat = attack_cat
        self.confidence = confidence
        self.raw_score = raw_score
        self.model_name = model_name

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "is_anomaly": self.is_anomaly,
            "attack_cat": self.attack_cat,
            "confidence": round(self.confidence, 4),
            "raw_score": round(float(self.raw_score), 6),
        }


class BaseDetector(ABC):
    """Abstract base class for all 4 detection models."""

    name: str = "BaseDetector"   # override in subclass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """Fit the model. y_train is binary for unsupervised (all 0s), multiclass for supervised."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> list[PredictionResult]:
        """
        Return a PredictionResult for each row in X.
        X shape: (n_samples, n_features)
        """

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Compute and return evaluation metrics.
        Minimum required keys:
          precision, recall, f1, fpr (false positive rate), roc_auc
        May also include per-class breakdown.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseDetector":
        """Load model from disk."""
