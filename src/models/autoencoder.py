"""
src/models/autoencoder.py — Dense Feedforward Autoencoder Detector
====================================================================
Unsupervised anomaly detector based on reconstruction error.  Trained ONLY
on normal traffic; attack samples are expected to reconstruct poorly.

Design decisions
----------------
**Why a Dense Autoencoder instead of an LSTM?**
    UNSW-NB15 consists of individual network flow records — tabular feature
    vectors with no inherent temporal ordering within the dataset as stored.
    Each row is a BIHOURLY aggregated flow summary (sport, sbytes, dur, etc.),
    not a time-series of packet payloads.

    LSTM autoencoders are designed for SEQUENTIAL data where the order of
    observations carries information (e.g., packet byte streams, HTTP sessions
    concatenated over time).  Applying an LSTM here would require:
      1. Artificially constructing sequences (imposing a fake temporal axis).
      2. Choosing an arbitrary window length — a free parameter with no
         principled basis in the UNSW-NB15 feature set.
      3. Significantly higher training time with no accuracy benefit.

    A Dense Feedforward Autoencoder is the appropriate choice because:
      - It treats each flow record as an independent, fixed-size feature vector.
      - The bottleneck (8-dimensional) forces the encoder to learn a compact
        representation of normal flow behaviour.
      - Attacks deviate from the learned manifold and are poorly reconstructed,
        producing high MSE → high anomaly score.
      - Standard practice in network anomaly detection literature
        (e.g., Mirsky et al. 2018 "Kitsune").

**Why only normal traffic for training?**
    The autoencoder learns to reconstruct the distribution it was trained on.
    Training on normal-only data means the decoder is optimised to reconstruct
    normal flows precisely.  Attack flows lie off the learned manifold and
    produce large reconstruction errors.  Including attacks in training would
    teach the autoencoder to reconstruct them too, destroying the anomaly signal.

**Why percentile thresholding at the 95th percentile?**
    Same rationale as Isolation Forest: we calibrate on training reconstruction
    errors so that 95% of known-normal samples score below the threshold.
    This gives a 5% false-positive budget on training data and requires no
    labelled attack samples — preserving the unsupervised nature of the method.
    The threshold represents the top 5% most unusual observations among
    known normal traffic.

**Why MSE as reconstruction loss?**
    MSE (Mean Squared Error) penalises large deviations heavily (quadratic
    penalty), which is desirable: a single highly anomalous feature dimension
    will dominate the reconstruction error and trigger an alert.  MAE is more
    robust to outliers in loss optimisation (might suppress gradients for
    extreme features) and therefore less suitable here.

**Architecture (43 → 32 → 16 → 8 → 16 → 32 → 43):**
    The bottleneck is 8 dimensions for 43 input features (~5:1 compression).
    This compression forces the encoder to discard noise and retain only the
    principal structure of normal traffic.  A deeper or wider bottleneck would
    memorise more of the input variance, reducing discrimination power.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models import BaseDetector, PredictionResult

logger = logging.getLogger(__name__)

MODEL_NAME = "DenseAutoencoder"
THRESHOLD_PERCENTILE = 95


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class _AutoencoderNet(nn.Module):
    """
    Symmetric Dense Feedforward Autoencoder.

    Architecture
    ------------
    Encoder: input_dim → 32 → 16 → 8   (Linear + ReLU each layer)
    Decoder: 8 → 16 → 32 → input_dim   (Linear + ReLU hidden; Linear output)

    The final decoder layer has NO activation — outputs are real-valued
    reconstructions comparable to the scaled input features.

    Parameters
    ----------
    input_dim : int
        Number of input features (43 for UNSW-NB15 feature set).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()

        # Encoder: compress input to 8-dimensional bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        # Decoder: reconstruct from bottleneck back to input space
        # Hidden layers use ReLU; final output layer is linear (no activation)
        # so reconstruction is unbounded — matching the StandardScaler-scaled
        # input range which is not constrained to [0, 1].
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),   # final layer: Linear, no activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return bottleneck representation only."""
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# Detector Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DenseAutoencoderDetector(BaseDetector):
    """
    Dense Feedforward Autoencoder anomaly detector.

    Trained exclusively on normal traffic; anomaly score is the per-sample
    mean squared reconstruction error.  A percentile-based threshold on
    training errors determines the decision boundary.

    Parameters
    ----------
    hidden_dims : list[int]
        Encoder hidden layer sizes.  Fixed at [32, 16, 8] per spec.
    batch_size : int
        Mini-batch size for SGD.  1024 is efficient on 1.7M normal rows.
    epochs : int
        Maximum number of training epochs.
    learning_rate : float
        Adam optimiser learning rate.
    val_fraction : float
        Fraction of training data held out as validation for early stopping.
    patience : int
        Early stopping: number of epochs without validation loss improvement
        before halting.  Prevents overfitting and reduces training time.
    threshold_percentile : int
        Percentile of training reconstruction errors used to set the
        anomaly detection threshold.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    random_state : int
        Reproducibility seed for PyTorch and numpy operations.
    """

    name = MODEL_NAME

    def __init__(
        self,
        batch_size: int = 1024,
        epochs: int = 30,
        learning_rate: float = 1e-3,
        val_fraction: float = 0.1,
        patience: int = 5,
        threshold_percentile: int = THRESHOLD_PERCENTILE,
        device: Optional[str] = None,
        random_state: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_fraction = val_fraction
        self.patience = patience
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state

        # Device selection: prefer CUDA if available, fall back to CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._net: Optional[_AutoencoderNet] = None
        self._threshold: Optional[float] = None
        self._input_dim: Optional[int] = None
        self._feature_names: Optional[list[str]] = None
        self._train_history: list[dict] = []   # epoch-level loss log

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,              # ignored — unsupervised
        feature_names: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Train the autoencoder on normal-traffic-only data.

        Parameters
        ----------
        X_train : array (n_samples, n_features)
            Should contain ONLY normal (label=0) samples from
            train_normal_only.parquet.  Labels are not used.
        y_train : array (n_samples,)
            Ignored (kept to satisfy BaseDetector ABC).
        feature_names : list[str], optional
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._feature_names = feature_names
        self._input_dim = X_train.shape[1]
        n_samples = X_train.shape[0]

        logger.info(
            f"[AE] Training on {n_samples:,} normal samples, "
            f"{self._input_dim} features  (labels ignored — unsupervised)"
        )
        logger.info(
            f"[AE] Device={self.device}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, lr={self.learning_rate}"
        )

        # ── Build network ──────────────────────────────────────────────────────
        self._net = _AutoencoderNet(input_dim=self._input_dim).to(self.device)

        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)

        # ── Train / validation split ───────────────────────────────────────────
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)

        n_val = max(1, int(n_samples * self.val_fraction))
        n_train = n_samples - n_val
        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.random_state),
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        logger.info(
            f"[AE] Train split: {n_train:,} | Val split: {n_val:,}"
        )

        # ── Training loop with early stopping ─────────────────────────────────
        best_val_loss = float("inf")
        best_state_dict = None
        epochs_no_improve = 0
        self._train_history = []

        for epoch in range(1, self.epochs + 1):
            # -- Train --
            self._net.train()
            train_loss_sum = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                reconstructed = self._net(batch_x)
                loss = criterion(reconstructed, batch_x)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                train_loss_sum += loss.item() * batch_x.size(0)

            train_loss = train_loss_sum / n_train

            # -- Validate --
            self._net.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(self.device)
                    reconstructed = self._net(batch_x)
                    loss = criterion(reconstructed, batch_x)
                    val_loss_sum += loss.item() * batch_x.size(0)

            val_loss = val_loss_sum / n_val

            self._train_history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            logger.info(
                f"[AE] Epoch {epoch:>3}/{self.epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

            # -- Early stopping & best checkpoint --
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Deep copy the state dict so later epochs don't overwrite it
                best_state_dict = {k: v.clone() for k, v in self._net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        f"[AE] Early stopping triggered at epoch {epoch} "
                        f"(no val improvement for {self.patience} consecutive epochs). "
                        f"Best val_loss={best_val_loss:.6f}"
                    )
                    break

        # Restore best checkpoint
        if best_state_dict is not None:
            self._net.load_state_dict(best_state_dict)
            logger.info(f"[AE] Best checkpoint restored (val_loss={best_val_loss:.6f})")

        # ── Threshold selection ────────────────────────────────────────────────
        # Compute reconstruction errors on ALL training normal data.
        # Threshold = Nth percentile of these errors.
        # The threshold represents the top 5% most unusual observations among
        # known normal traffic.
        train_errors = self._reconstruction_errors(X_train)
        self._threshold = float(np.percentile(train_errors, self.threshold_percentile))

        logger.info(
            f"[AE] Training complete. "
            f"Threshold (p{self.threshold_percentile}) = {self._threshold:.6f}"
        )

    # ── Reconstruction error helpers ──────────────────────────────────────────

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample mean squared reconstruction error.

        Parameters
        ----------
        X : array (n_samples, n_features)

        Returns
        -------
        errors : array (n_samples,)  — higher = more anomalous
        """
        if self._net is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self._net.eval()
        errors = []
        X_tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                reconstructed = self._net(batch_x)
                # Per-sample MSE: mean over feature dimension
                mse = torch.mean((reconstructed - batch_x) ** 2, dim=1)
                errors.extend(mse.cpu().numpy().tolist())

        return np.array(errors, dtype=np.float64)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> list[PredictionResult]:
        """
        Return a PredictionResult per row in X.

        Anomaly score = mean squared reconstruction error.
        A sample is flagged if its error exceeds the training-derived threshold.

        Parameters
        ----------
        X : array (n_samples, n_features)
        """
        if self._net is None or self._threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")

        errors = self._reconstruction_errors(X)
        results: list[PredictionResult] = []

        for i in range(len(X)):
            err = float(errors[i])
            is_anom = err > self._threshold
            # Soft confidence via sigmoid centred on threshold
            confidence = float(1 / (1 + np.exp(-(err - self._threshold) * 100)))
            results.append(
                PredictionResult(
                    is_anomaly=bool(is_anom),
                    attack_cat="unknown" if not is_anom else "Anomaly",
                    confidence=confidence,
                    raw_score=err,
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
        if self._net is None or self._threshold is None:
            raise RuntimeError("Model not trained. Call train() first.")

        errors = self._reconstruction_errors(X_test)
        y_pred = (errors > self._threshold).astype(int)

        # ── Core metrics ──────────────────────────────────────────────────────
        prec  = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
        rec   = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
        f1    = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

        try:
            roc_auc = round(float(roc_auc_score(y_test, errors)), 4)
        except Exception as exc:
            logger.warning(f"[AE] ROC-AUC computation failed: {exc}")
            roc_auc = None

        # ── False Positive Rate ───────────────────────────────────────────────
        # FPR = FP / (FP + TN)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0

        logger.info(
            f"[AE] Evaluation — precision={prec}, recall={rec}, f1={f1}, "
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
                f"(p{self.threshold_percentile} of training normal reconstruction errors). "
                f"The threshold represents the top {100 - self.threshold_percentile}% "
                f"most unusual observations among known normal traffic."
            ),
            "train_history": self._train_history,
        }

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save_model(self, path: Path) -> None:
        """
        Save trained autoencoder to disk.

        Saves PyTorch weights as a .pt file.  The threshold and hyperparameters
        are stored in the same payload dict so a single file is self-contained.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._net.state_dict() if self._net else None,
            "threshold": self._threshold,
            "input_dim": self._input_dim,
            "feature_names": self._feature_names,
            "train_history": self._train_history,
            "hyperparams": {
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "val_fraction": self.val_fraction,
                "patience": self.patience,
                "threshold_percentile": self.threshold_percentile,
                "random_state": self.random_state,
                "device": self.device,
            },
        }
        torch.save(payload, path)
        logger.info(f"[AE] Model saved → {path}")

    def save(self, path: Path) -> None:
        """Alias to satisfy BaseDetector ABC (delegates to save_model)."""
        self.save_model(path)

    @classmethod
    def load_model(cls, path: Path) -> "DenseAutoencoderDetector":
        """Load a previously saved DenseAutoencoderDetector from disk."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        hp = payload["hyperparams"]
        instance = cls(
            batch_size=hp["batch_size"],
            epochs=hp["epochs"],
            learning_rate=hp["learning_rate"],
            val_fraction=hp["val_fraction"],
            patience=hp["patience"],
            threshold_percentile=hp["threshold_percentile"],
            random_state=hp["random_state"],
            device=hp.get("device", "cpu"),
        )
        instance._input_dim = payload["input_dim"]
        instance._threshold = payload["threshold"]
        instance._feature_names = payload["feature_names"]
        instance._train_history = payload.get("train_history", [])

        if payload["state_dict"] is not None:
            instance._net = _AutoencoderNet(input_dim=instance._input_dim)
            instance._net.load_state_dict(payload["state_dict"])
            instance._net.to(instance.device)
            instance._net.eval()

        logger.info(f"[AE] Model loaded ← {path}")
        return instance

    @classmethod
    def load(cls, path: Path) -> "DenseAutoencoderDetector":
        """Alias to satisfy BaseDetector ABC (delegates to load_model)."""
        return cls.load_model(path)
