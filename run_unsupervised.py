#!/usr/bin/env python3
"""
run_unsupervised.py — Phase 3: Unsupervised Anomaly Detection Entry Point
==========================================================================
Trains Isolation Forest and Dense Autoencoder detectors on UNSW-NB15
preprocessed data, evaluates both on the held-out test set, and writes
comparison reports.

Models
------
1. Isolation Forest  — trained on train_normal_only.parquet (normal only)
2. Dense Autoencoder — trained on train_normal_only.parquet (normal only)

Both models are unsupervised: no labels are used during training.
Labels are only used at evaluation time to compute precision/recall/F1/ROC-AUC.

Usage
-----
    python run_unsupervised.py [--skip-train] [--skip-if] [--skip-ae]

    --skip-train   Load existing models from models/ instead of retraining.
                   Useful for re-running evaluation without retraining.
    --skip-if      Skip Isolation Forest (train Autoencoder only).
    --skip-ae      Skip Autoencoder (train Isolation Forest only).

Outputs
-------
    models/isolation_forest.joblib   — trained IsolationForestDetector
    models/autoencoder.pt            — trained DenseAutoencoderDetector
    reports/unsupervised_evaluation.json — full metrics for both models

Expected runtime (full dataset, CPU)
--------------------------------------
    IF training  : ~5-20 min   (300 trees, n_jobs=-1, ~1.7M normal rows)
    AE training  : ~3-10 min   (30 epochs, batch_size=1024, ~1.7M normal rows)
    Evaluation   : ~2-5 min    (ROC-AUC on 508K test rows)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import DenseAutoencoderDetector

# ── Directories ───────────────────────────────────────────────────────────────
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"          # Phase 3 saves here (not data/models/)
REPORTS_DIR    = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
NORMAL_TRAIN_PATH  = PROCESSED_DIR / "train_normal_only.parquet"
TEST_PATH          = PROCESSED_DIR / "test.parquet"
FEATURE_LIST_PATH  = PROCESSED_DIR / "feature_list.txt"

IF_MODEL_PATH  = MODELS_DIR / "isolation_forest.joblib"
AE_MODEL_PATH  = MODELS_DIR / "autoencoder.pt"
EVAL_PATH      = REPORTS_DIR / "unsupervised_evaluation.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_normal_train(
    feature_names: list[str],
) -> np.ndarray:
    """
    Load normal-only training data.

    Returns
    -------
    X_normal : array (n_samples, n_features) — dtype float32
    """
    logger.info(f"Reading train_normal_only.parquet ({NORMAL_TRAIN_PATH})…")
    df = pd.read_parquet(NORMAL_TRAIN_PATH, columns=feature_names)
    X = df.values.astype(np.float32)
    logger.info(f"Normal train: {X.shape[0]:,} rows × {X.shape[1]} features")
    return X


def load_test(
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test set and binary labels.

    Binary labels: 0 = normal, 1 = attack
    (derived from the 'label' column as specified in the Phase 3 requirement)

    Returns
    -------
    X_test : array (n_samples, n_features) — float32
    y_test : array (n_samples,) — int (0 = normal, 1 = attack)
    """
    logger.info(f"Reading test.parquet ({TEST_PATH})…")
    df = pd.read_parquet(TEST_PATH, columns=feature_names + ["label"])
    X_test = df[feature_names].values.astype(np.float32)

    # y_true = (label == 1).astype(int) — label column: 0=normal, 1=attack
    y_test = (df["label"] == 1).astype(int).values

    logger.info(
        f"Test: {X_test.shape[0]:,} rows × {X_test.shape[1]} features  "
        f"| attack={int(y_test.sum()):,}  normal={int((y_test==0).sum()):,}"
    )
    return X_test, y_test


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load all required datasets.

    Returns
    -------
    X_normal, X_test, y_test, feature_names
    """
    console.print("\n[bold cyan]━━━ Loading preprocessed data ━━━[/bold cyan]")

    feature_names = FEATURE_LIST_PATH.read_text().strip().splitlines()
    logger.info(f"Features ({len(feature_names)}): {feature_names}")

    X_normal = load_normal_train(feature_names)
    X_test, y_test = load_test(feature_names)

    console.print(
        f"  Normal train : [green]{X_normal.shape[0]:,}[/green] rows × {X_normal.shape[1]} features"
    )
    console.print(
        f"  Test         : [green]{X_test.shape[0]:,}[/green] rows | "
        f"attack={int(y_test.sum()):,}, normal={int((y_test==0).sum()):,}"
    )

    return X_normal, X_test, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_serialisable(obj):
    """Recursively convert numpy types to Python natives for JSON dumping."""
    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serialisable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def print_comparison_table(if_metrics: dict, ae_metrics: dict) -> None:
    """Pretty-print side-by-side Isolation Forest vs Autoencoder comparison."""
    table = Table(
        title="📊  Unsupervised Model Comparison — Isolation Forest vs Autoencoder",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="bold", min_width=24)
    table.add_column("Isolation Forest", justify="right", min_width=18)
    table.add_column("Dense Autoencoder", justify="right", min_width=18)

    def fmt(val):
        if val is None:
            return "[dim]N/A[/dim]"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    for key, label in [
        ("precision",           "Precision"),
        ("recall",              "Recall"),
        ("f1",                  "F1 Score"),
        ("roc_auc",             "ROC-AUC"),
        ("false_positive_rate", "False Positive Rate"),
        ("threshold",           "Anomaly Threshold"),
    ]:
        table.add_row(
            label,
            fmt(if_metrics.get(key)),
            fmt(ae_metrics.get(key)),
        )

    # Confusion matrix breakdown
    table.add_row("", "", "")
    table.add_row("[bold]── CONFUSION MATRIX ──[/bold]", "", "")
    for key, label in [("tp", "True Positives"), ("tn", "True Negatives"),
                       ("fp", "False Positives"), ("fn", "False Negatives")]:
        table.add_row(
            f"  {label}",
            fmt(if_metrics.get(key)),
            fmt(ae_metrics.get(key)),
        )

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Train & evaluate unsupervised anomaly detectors"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Load existing models from models/ instead of retraining",
    )
    parser.add_argument(
        "--skip-if",
        action="store_true",
        help="Skip Isolation Forest (Autoencoder only)",
    )
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip Autoencoder (Isolation Forest only)",
    )
    args = parser.parse_args()

    t_total = time.time()
    console.print(
        "\n[bold white on blue]  UNSW-NB15  ·  Phase 3: Unsupervised Anomaly Detection  [/bold white on blue]\n"
    )

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_normal, X_test, y_test, feature_names = load_data()

    # Placeholder y for train() signature (unsupervised; not used in fitting)
    y_placeholder = np.zeros(X_normal.shape[0], dtype=np.int32)

    results = {}

    # ── 2. Isolation Forest ───────────────────────────────────────────────────
    if not args.skip_if:
        console.print("\n[bold cyan]━━━ Isolation Forest ━━━[/bold cyan]")

        if args.skip_train and IF_MODEL_PATH.exists():
            console.print(f"  ↩  Loading from {IF_MODEL_PATH}")
            ifd = IsolationForestDetector.load_model(IF_MODEL_PATH)
        else:
            console.print(
                "  🌲 Training IsolationForestDetector "
                "(n_estimators=300, contamination='auto')…"
            )
            t0 = time.time()
            ifd = IsolationForestDetector(
                n_estimators=300,
                contamination="auto",
                random_state=42,
                n_jobs=-1,
                threshold_percentile=95,
            )
            ifd.train(X_normal, y_placeholder, feature_names=feature_names)
            elapsed = time.time() - t0
            console.print(
                f"  ✅ IF trained in [green]{elapsed/60:.1f} min[/green]  "
                f"threshold={ifd._threshold:.6f}"
            )

            console.print(f"  💾 Saving model → {IF_MODEL_PATH}")
            ifd.save_model(IF_MODEL_PATH)

        console.print("  🔍 Evaluating Isolation Forest on test set…")
        t0 = time.time()
        if_metrics = ifd.evaluate(X_test, y_test)
        elapsed = time.time() - t0
        console.print(
            f"  ✅ IF evaluation in [green]{elapsed:.1f}s[/green] — "
            f"precision={if_metrics['precision']}, "
            f"recall={if_metrics['recall']}, "
            f"f1={if_metrics['f1']}, "
            f"FPR={if_metrics['false_positive_rate']}"
        )
        results["isolation_forest"] = if_metrics

    # ── 3. Dense Autoencoder ──────────────────────────────────────────────────
    if not args.skip_ae:
        console.print("\n[bold cyan]━━━ Dense Autoencoder ━━━[/bold cyan]")

        if args.skip_train and AE_MODEL_PATH.exists():
            console.print(f"  ↩  Loading from {AE_MODEL_PATH}")
            ae = DenseAutoencoderDetector.load_model(AE_MODEL_PATH)
        else:
            console.print(
                "  🧠 Training DenseAutoencoderDetector "
                "(43→32→16→8→16→32→43, epochs=30, batch=1024)…"
            )
            t0 = time.time()
            ae = DenseAutoencoderDetector(
                batch_size=1024,
                epochs=30,
                learning_rate=1e-3,
                val_fraction=0.1,
                patience=5,
                threshold_percentile=95,
                random_state=42,
            )
            ae.train(X_normal, y_placeholder, feature_names=feature_names)
            elapsed = time.time() - t0
            console.print(
                f"  ✅ AE trained in [green]{elapsed/60:.1f} min[/green]  "
                f"threshold={ae._threshold:.6f}"
            )

            console.print(f"  💾 Saving model → {AE_MODEL_PATH}")
            ae.save_model(AE_MODEL_PATH)

        console.print("  🔍 Evaluating Autoencoder on test set…")
        t0 = time.time()
        ae_metrics = ae.evaluate(X_test, y_test)
        elapsed = time.time() - t0
        # Strip train_history from console log (it's saved in JSON)
        console.print(
            f"  ✅ AE evaluation in [green]{elapsed:.1f}s[/green] — "
            f"precision={ae_metrics['precision']}, "
            f"recall={ae_metrics['recall']}, "
            f"f1={ae_metrics['f1']}, "
            f"FPR={ae_metrics['false_positive_rate']}"
        )
        results["autoencoder"] = ae_metrics

    # ── 4. Comparison table ───────────────────────────────────────────────────
    if "isolation_forest" in results and "autoencoder" in results:
        console.print()
        print_comparison_table(results["isolation_forest"], results["autoencoder"])

    # ── 5. Save evaluation JSON ───────────────────────────────────────────────
    console.print(f"\n[bold cyan]━━━ Saving evaluation report ━━━[/bold cyan]")

    report = make_serialisable({
        "phase": "3_unsupervised_models",
        "models": results,
        "metadata": {
            "normal_train_rows": int(X_normal.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "test_attack_rows": int(y_test.sum()),
            "test_normal_rows": int((y_test == 0).sum()),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "threshold_percentile": 95,
            "threshold_rationale": (
                "Threshold = p95 of anomaly scores / reconstruction errors "
                "on training normal data. Represents the top 5% most unusual "
                "normal observations. Chosen to limit FPR while maintaining "
                "reasonable recall without requiring any labelled attack samples."
            ),
            "dense_ae_rationale": (
                "Dense Feedforward Autoencoder chosen over LSTM because "
                "UNSW-NB15 is tabular network flow data (not sequential time-series). "
                "Each row is an independent aggregated flow record. "
                "Imposing an LSTM sequence structure would require an arbitrary "
                "window length with no principled basis. "
                "Dense AE treats each flow as a fixed-size feature vector, "
                "learning a compact manifold of normal traffic in the bottleneck, "
                "producing high reconstruction error for attacks."
            ),
            "elapsed_total_seconds": round(time.time() - t_total, 1),
        },
    })

    EVAL_PATH.write_text(json.dumps(report, indent=2))
    console.print(f"  ✅ Report saved → {EVAL_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    console.print(
        f"\n[bold white on green]  ✅  Phase 3 Complete in {elapsed_total/60:.1f} min  [/bold white on green]"
    )
    console.print("\nOutputs:")
    if not args.skip_if:
        console.print(f"  🌲  {IF_MODEL_PATH}")
    if not args.skip_ae:
        console.print(f"  🧠  {AE_MODEL_PATH}")
    console.print(f"  📄  {EVAL_PATH}")
    console.print(
        "\nNext: open [bold]notebooks/03_unsupervised_models.ipynb[/bold] to visualise results."
    )


if __name__ == "__main__":
    main()
