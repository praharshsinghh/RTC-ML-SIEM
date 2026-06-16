#!/usr/bin/env python3
"""
run_supervised.py — Phase 2: Supervised Models Entry Point
===========================================================
Trains Random Forest and XGBoost classifiers on UNSW-NB15 preprocessed data,
evaluates both on the held-out test set, and writes comparison reports.

Usage:
    python run_supervised.py [--skip-train]

    --skip-train   Skip training and load existing models from data/models/.
                   Useful for re-running evaluation or regenerating reports
                   without retraining (training takes 5-20 min on full data).

Outputs
-------
    data/models/rf_detector.joblib          — trained RandomForestDetector
    data/models/xgb_detector.joblib         — trained XGBoostDetector
    reports/rf_feature_importances.csv      — RF impurity-based feature ranks
    reports/xgb_feature_importances.csv     — XGBoost gain-based feature ranks
    reports/supervised_evaluation.json      — full metrics for both models

Expected runtime (full 2M-row dataset)
---------------------------------------
    RF training  : ~10-25 min  (200 trees, n_jobs=-1, parallelised by sklearn)
    XGB training : ~5-15 min   (200 rounds, n_jobs=-1)
    Evaluation   : ~2-5 min    (ROC-AUC on 508K test rows is the bottleneck)
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

from src.models.random_forest import RandomForestDetector
from src.models.xgboost_model import XGBoostDetector

# ── Directories ───────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
FEATURE_LIST_PATH = PROCESSED_DIR / "feature_list.txt"

RF_MODEL_PATH = MODELS_DIR / "rf_detector.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgb_detector.joblib"
RF_IMP_PATH = REPORTS_DIR / "rf_feature_importances.csv"
XGB_IMP_PATH = REPORTS_DIR / "xgb_feature_importances.csv"
EVAL_PATH = REPORTS_DIR / "supervised_evaluation.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load train and test parquet files.

    Returns
    -------
    X_train, y_train, X_test, y_test, feature_names
    y arrays are string attack_cat labels.
    """
    console.print("\n[bold cyan]━━━ Loading preprocessed data ━━━[/bold cyan]")

    feature_names = FEATURE_LIST_PATH.read_text().strip().splitlines()

    logger.info(f"Reading train.parquet ({TRAIN_PATH})…")
    train_df = pd.read_parquet(TRAIN_PATH, columns=feature_names + ["attack_cat", "label"])

    logger.info(f"Reading test.parquet ({TEST_PATH})…")
    test_df = pd.read_parquet(TEST_PATH, columns=feature_names + ["attack_cat", "label"])

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["attack_cat"].values
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["attack_cat"].values

    # Binary labels for FPR computation (passed separately to evaluate())
    y_binary_test = train_df["label"].values   # 0=normal,1=attack — train not used
    y_binary_test = test_df["label"].values

    console.print(f"  Train : [green]{X_train.shape[0]:,}[/green] rows × {X_train.shape[1]} features")
    console.print(f"  Test  : [green]{X_test.shape[0]:,}[/green] rows × {X_test.shape[1]} features")
    console.print(f"  Classes : {sorted(set(y_train))}")
    return X_train, y_train, X_test, y_test, y_binary_test, feature_names


def print_comparison_table(rf_metrics: dict, xgb_metrics: dict) -> None:
    """Pretty-print side-by-side RF vs XGBoost comparison table."""
    table = Table(
        title="📊  Supervised Model Comparison — RF vs XGBoost",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric / Category", style="bold", min_width=22)
    table.add_column("Random Forest", justify="right", min_width=16)
    table.add_column("XGBoost", justify="right", min_width=16)

    def fmt(val):
        if val is None:
            return "[dim]N/A[/dim]"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    # ── Overall metrics ───────────────────────────────────────────────────────
    table.add_row("[bold]── OVERALL ──[/bold]", "", "")
    for key, label in [
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("roc_auc", "ROC-AUC (OvR, macro)"),
        ("fpr", "FPR (binary, normal)"),
    ]:
        table.add_row(label, fmt(rf_metrics.get(key)), fmt(xgb_metrics.get(key)))

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    table.add_row("", "", "")
    table.add_row("[bold]── PER-CATEGORY F1 ──[/bold]", "", "")
    all_cats = sorted(
        set(rf_metrics["per_class"].keys()) | set(xgb_metrics["per_class"].keys())
    )
    for cat in all_cats:
        rf_f1 = rf_metrics["per_class"].get(cat, {}).get("f1")
        xgb_f1 = xgb_metrics["per_class"].get(cat, {}).get("f1")
        # Highlight 0.0 F1 in red (likely rare class issue)
        rf_str = f"[red]{rf_f1:.4f}[/red]" if rf_f1 == 0.0 else fmt(rf_f1)
        xgb_str = f"[red]{xgb_f1:.4f}[/red]" if xgb_f1 == 0.0 else fmt(xgb_f1)
        table.add_row(f"  {cat}", rf_str, xgb_str)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Train & evaluate supervised detectors")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Load existing models instead of retraining",
    )
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Skip training/loading Random Forest (XGBoost only)",
    )
    parser.add_argument(
        "--skip-xgb",
        action="store_true",
        help="Skip training/loading XGBoost (RF only)",
    )
    args = parser.parse_args()

    t_total = time.time()
    console.print("\n[bold white on blue]  UNSW-NB15  ·  Phase 2: Supervised Detection  [/bold white on blue]\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test, y_binary_test, feature_names = load_data()

    results = {}

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    if not args.skip_rf:
        console.print("\n[bold cyan]━━━ Random Forest ━━━[/bold cyan]")

        if args.skip_train and RF_MODEL_PATH.exists():
            console.print(f"  ↩  Loading from {RF_MODEL_PATH}")
            rf = RandomForestDetector.load_model(RF_MODEL_PATH)
        else:
            console.print("  🌳 Training RandomForestDetector (n_estimators=200, balanced)…")
            t0 = time.time()
            rf = RandomForestDetector(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            rf.train(X_train, y_train, feature_names=feature_names)
            elapsed = time.time() - t0
            console.print(f"  ✅ RF trained in [green]{elapsed/60:.1f} min[/green]")

            console.print(f"  💾 Saving model → {RF_MODEL_PATH}")
            rf.save_model(RF_MODEL_PATH)

            console.print(f"  📊 Saving feature importances → {RF_IMP_PATH}")
            rf.save_feature_importances(RF_IMP_PATH)

        console.print("  🔍 Evaluating RF on test set…")
        t0 = time.time()
        rf_metrics = rf.evaluate(X_test, y_test, binary_labels=y_binary_test)
        elapsed = time.time() - t0
        console.print(f"  ✅ RF evaluation in [green]{elapsed:.1f}s[/green] — "
                      f"macro_F1={rf_metrics['macro_f1']}, FPR={rf_metrics['fpr']}")
        results["random_forest"] = rf_metrics

    # ── 3. XGBoost ────────────────────────────────────────────────────────────
    if not args.skip_xgb:
        console.print("\n[bold cyan]━━━ XGBoost ━━━[/bold cyan]")

        if args.skip_train and XGB_MODEL_PATH.exists():
            console.print(f"  ↩  Loading from {XGB_MODEL_PATH}")
            xgb = XGBoostDetector.load_model(XGB_MODEL_PATH)
        else:
            console.print("  ⚡ Training XGBoostDetector (n_estimators=200, compute_sample_weight)…")
            t0 = time.time()
            xgb = XGBoostDetector(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
            xgb.train(X_train, y_train, feature_names=feature_names)
            elapsed = time.time() - t0
            console.print(f"  ✅ XGB trained in [green]{elapsed/60:.1f} min[/green]")

            console.print(f"  💾 Saving model → {XGB_MODEL_PATH}")
            xgb.save_model(XGB_MODEL_PATH)

            console.print(f"  📊 Saving feature importances → {XGB_IMP_PATH}")
            xgb.save_feature_importances(XGB_IMP_PATH)

        console.print("  🔍 Evaluating XGBoost on test set…")
        t0 = time.time()
        xgb_metrics = xgb.evaluate(X_test, y_test, binary_labels=y_binary_test)
        elapsed = time.time() - t0
        console.print(f"  ✅ XGB evaluation in [green]{elapsed:.1f}s[/green] — "
                      f"macro_F1={xgb_metrics['macro_f1']}, FPR={xgb_metrics['fpr']}")
        results["xgboost"] = xgb_metrics

    # ── 4. Comparison table ───────────────────────────────────────────────────
    if "random_forest" in results and "xgboost" in results:
        console.print()
        print_comparison_table(results["random_forest"], results["xgboost"])

    # ── 5. Save evaluation JSON ───────────────────────────────────────────────
    console.print(f"\n[bold cyan]━━━ Saving evaluation report ━━━[/bold cyan]")

    # Convert numpy types for JSON serialisation
    def make_serialisable(obj):
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

    report = make_serialisable({
        "phase": "2_supervised_models",
        "models": results,
        "metadata": {
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "classes": sorted(set(y_train).union(set(y_test))),
            "elapsed_total_seconds": round(time.time() - t_total, 1),
        },
    })

    EVAL_PATH.write_text(json.dumps(report, indent=2))
    console.print(f"  ✅ Report saved → {EVAL_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    console.print(f"\n[bold white on green]  ✅  Phase 2 Complete in {elapsed_total/60:.1f} min  [/bold white on green]")
    console.print(f"\nOutputs:")
    if not args.skip_rf:
        console.print(f"  🌳  {RF_MODEL_PATH}")
        console.print(f"  📊  {RF_IMP_PATH}")
    if not args.skip_xgb:
        console.print(f"  ⚡  {XGB_MODEL_PATH}")
        console.print(f"  📊  {XGB_IMP_PATH}")
    console.print(f"  📄  {EVAL_PATH}")
    console.print(f"\nNext: open [bold]notebooks/02_supervised_models.ipynb[/bold] to visualise results.")


if __name__ == "__main__":
    main()
