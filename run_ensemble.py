#!/usr/bin/env python3
"""
run_ensemble.py — Phase 4: Ensemble Detection Layer Entry Point
================================================================
Loads all four trained detectors (Phase 2 + Phase 3), assembles the
EnsembleDetector, evaluates it on the UNSW-NB15 test set, and writes
a comprehensive evaluation report to reports/ensemble_evaluation.json.

This script intentionally does NOT retrain any model — ensemble construction
is a zero-training operation (only arithmetic weight combination). It only
requires that the four model artefacts exist on disk.

Usage
-----
    python run_ensemble.py

    Optional flags:
    --sample N      Run on a random N-row sample of the test set
                    (useful for quick smoke-testing; full set takes ~5 min)
    --batch-size N  Batch size for model inference (default: 2048)

Outputs
-------
    reports/ensemble_evaluation.json   — full metrics + agreement stats
    (Notebook 04_ensemble_analysis.ipynb is separate and reads this JSON)

Expected runtime
----------------
    Full test set (508K rows): ~5-15 min on CPU depending on AE inference.
    Sample (10K rows):         ~30-90 seconds.

Model artefact paths (must exist from previous phases)
-------------------------------------------------------
    data/models/rf_detector.joblib      (Phase 2)
    data/models/xgb_detector.joblib     (Phase 2)
    models/isolation_forest.joblib      (Phase 3)
    models/autoencoder.pt               (Phase 3)

Design note — why no training in run_ensemble.py?
--------------------------------------------------
The weighted voting ensemble has NO learnable parameters. There is nothing
to train. The weights (0.35, 0.35, 0.15, 0.15) are domain-informed
constants, not data-fitted values. Running evaluate() on the test set
measures how well the *combination* of the four pre-trained models performs
— it does NOT involve any fitting on test data.

This is a strict separation: the test set is touched ONLY for evaluation,
never for any parameter selection or threshold tuning.
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

from src.ensemble import EnsembleDetector, MODEL_WEIGHTS

# ── Directories & Paths ───────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUPERVISED_MODELS_DIR   = PROJECT_ROOT / "data" / "models"
UNSUPERVISED_MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH  = SUPERVISED_MODELS_DIR   / "rf_detector.joblib"
XGB_PATH = SUPERVISED_MODELS_DIR   / "xgb_detector.joblib"
ISO_PATH = UNSUPERVISED_MODELS_DIR / "isolation_forest.joblib"
AE_PATH  = UNSUPERVISED_MODELS_DIR / "autoencoder.pt"

TEST_PATH         = PROCESSED_DIR / "test.parquet"
FEATURE_LIST_PATH = PROCESSED_DIR / "feature_list.txt"

EVAL_PATH = REPORTS_DIR / "ensemble_evaluation.json"

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


def check_model_paths() -> bool:
    """Verify all four model artefacts exist before attempting to load."""
    all_ok = True
    for path, desc in [
        (RF_PATH,  "Random Forest (Phase 2)"),
        (XGB_PATH, "XGBoost (Phase 2)"),
        (ISO_PATH, "Isolation Forest (Phase 3)"),
        (AE_PATH,  "Dense Autoencoder (Phase 3)"),
    ]:
        exists = path.exists()
        status = "✅" if exists else "❌"
        console.print(f"  {status} {desc:35s} → {path}")
        if not exists:
            all_ok = False
    return all_ok


def load_test_data(
    feature_names: list[str],
    sample_n: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the test set and binary labels.

    Parameters
    ----------
    feature_names : list[str]
    sample_n : int | None — if set, sample this many rows (for speed)
    random_state : int

    Returns
    -------
    X_test : (n_samples, n_features) float32
    y_binary : (n_samples,) int — 0=normal, 1=attack
    """
    logger.info(f"Reading test.parquet ({TEST_PATH}) …")
    df = pd.read_parquet(TEST_PATH, columns=feature_names + ["label"])

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=random_state)
        logger.info(f"Sampled {sample_n:,} rows from test set.")

    X_test   = df[feature_names].values.astype(np.float32)
    y_binary = (df["label"] == 1).astype(int).values

    console.print(
        f"  Test  : [green]{X_test.shape[0]:,}[/green] rows × {X_test.shape[1]} features  "
        f"| attack={int(y_binary.sum()):,}  normal={int((y_binary==0).sum()):,}"
    )
    return X_test, y_binary


def print_weights_table() -> None:
    """Display the ensemble weight configuration."""
    table = Table(
        title="⚖️  Ensemble Weight Configuration",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Model", style="bold", min_width=22)
    table.add_column("Type", min_width=14)
    table.add_column("Weight", justify="right", min_width=10)
    table.add_column("Rationale", min_width=50)

    rows = [
        ("RandomForest",    "Supervised",   0.35, "Trained on labelled data; validated FPR/recall; high reliability"),
        ("XGBoost",         "Supervised",   0.35, "Gradient-boosted; highest macro-F1 in Phase 2; gain-based importance"),
        ("IsolationForest", "Unsupervised", 0.15, "Zero-day coverage; path-length anomaly; higher FPR"),
        ("DenseAutoencoder","Unsupervised", 0.15, "Reconstruction-error anomaly; captures nonlinear normal manifold"),
    ]
    for name, mtype, w, rationale in rows:
        table.add_row(name, mtype, f"{w:.2f}", rationale)

    table.add_row("", "", "──────", "")
    table.add_row("[bold]TOTAL[/bold]", "", "[bold]1.00[/bold]", "")
    console.print(table)


def print_results_table(metrics: dict) -> None:
    """Print a summary table of ensemble evaluation metrics."""
    table = Table(
        title="📊  Ensemble Evaluation — Test Set Results",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="bold", min_width=26)
    table.add_column("Ensemble", justify="right", min_width=12)
    table.add_column("RandomForest", justify="right", min_width=14)
    table.add_column("XGBoost", justify="right", min_width=10)
    table.add_column("IsolationForest", justify="right", min_width=16)
    table.add_column("DenseAutoencoder", justify="right", min_width=18)

    indiv = metrics.get("individual_model_comparison", {})

    def fmt(val):
        if val is None:
            return "[dim]N/A[/dim]"
        return f"{val:.4f}"

    for key, label in [
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1 Score"),
        ("false_positive_rate", "FPR"),
    ]:
        table.add_row(
            label,
            fmt(metrics.get(key)),
            fmt(indiv.get("RandomForest",    {}).get(key)),
            fmt(indiv.get("XGBoost",         {}).get(key)),
            fmt(indiv.get("IsolationForest", {}).get(key)),
            fmt(indiv.get("DenseAutoencoder",{}).get(key)),
        )

    table.add_row("", "", "", "", "", "")
    table.add_row("[bold]Accuracy[/bold]", fmt(metrics.get("accuracy")), "", "", "", "")

    ag = metrics.get("agreement_stats", {})
    table.add_row("", "", "", "", "", "")
    table.add_row("[bold]── AGREEMENT ──[/bold]", "", "", "", "", "")
    table.add_row("All 4 models agree", str(ag.get("full_agreement_all4", "")), "", "", "", "")
    table.add_row("3 models agree",      str(ag.get("high_agreement_3of4", "")), "", "", "", "")
    table.add_row("Slim majority",        str(ag.get("split_vote_slim_majority", "")), "", "", "", "")
    table.add_row("Minority verdict",     str(ag.get("low_agreement_minority", "")), "", "", "", "")

    sev = metrics.get("severity_distribution", {})
    table.add_row("", "", "", "", "", "")
    table.add_row("[bold]── SEVERITY ──[/bold]", "", "", "", "", "")
    for sev_label in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "N/A"]:
        table.add_row(f"  {sev_label}", str(sev.get(sev_label, 0)), "", "", "", "")

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Evaluate EnsembleDetector on UNSW-NB15 test set"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on a random N-row sample (default: full test set)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Inference batch size (default: 2048)",
    )
    args = parser.parse_args()

    t_total = time.time()
    console.print(
        "\n[bold white on blue]  UNSW-NB15  ·  Phase 4: Ensemble Detection Layer  [/bold white on blue]\n"
    )

    # ── 1. Check model artefacts ──────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Checking model artefacts ━━━[/bold cyan]")
    if not check_model_paths():
        console.print(
            "\n[bold red]ERROR:[/bold red] One or more model artefacts are missing. "
            "Please run run_supervised.py (Phase 2) and run_unsupervised.py (Phase 3) first."
        )
        sys.exit(1)

    # ── 2. Print weight configuration ─────────────────────────────────────────
    console.print()
    print_weights_table()

    # ── 3. Load test data ──────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Loading test data ━━━[/bold cyan]")
    feature_names = FEATURE_LIST_PATH.read_text().strip().splitlines()
    X_test, y_binary = load_test_data(
        feature_names,
        sample_n=args.sample,
    )

    # ── 4. Load ensemble ───────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Loading ensemble models ━━━[/bold cyan]")
    t0 = time.time()
    ensemble = EnsembleDetector.from_disk(
        rf_path=RF_PATH,
        xgb_path=XGB_PATH,
        iso_path=ISO_PATH,
        ae_path=AE_PATH,
        batch_size=args.batch_size,
    )
    console.print(f"  ✅ All models loaded in [green]{time.time()-t0:.1f}s[/green]")

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Running ensemble evaluation ━━━[/bold cyan]")
    t0 = time.time()
    metrics = ensemble.evaluate(X_test, y_binary)
    elapsed = time.time() - t0
    console.print(f"  ✅ Evaluation complete in [green]{elapsed:.1f}s[/green]")

    # ── 6. Print results table ─────────────────────────────────────────────────
    console.print()
    print_results_table(metrics)

    # ── 7. Save evaluation report ─────────────────────────────────────────────
    console.print(f"\n[bold cyan]━━━ Saving evaluation report ━━━[/bold cyan]")

    report = make_serialisable({
        "phase": "4_ensemble_detection",
        "ensemble": metrics,
        "configuration": {
            "model_weights": MODEL_WEIGHTS,
            "attack_threshold": 0.50,
            "weight_rationale": (
                "Supervised models (RF, XGB) receive weight 0.35 each because "
                "they are trained on labelled attack categories and optimised "
                "directly for classification precision/recall. Anomaly detectors "
                "(IF, AE) receive 0.15 each because they are unsupervised: they "
                "detect deviation from normal patterns but cannot categorise "
                "attacks and typically have higher false positive rates. "
                "The 0.70/0.30 split ensures supervised consensus can override "
                "anomaly detector noise while preserving the zero-day detection "
                "capability of the unsupervised layer."
            ),
            "threshold_rationale": (
                "ATTACK threshold = 0.50 is the natural midpoint of [0,1]. "
                "It means the combined weight of ATTACK-voting models equals "
                "or exceeds the NORMAL-voting weight. No test-set calibration "
                "was performed — this is a principled default preserving "
                "strict train/test separation."
            ),
            "severity_matrix": {
                "CRITICAL": "confidence >= 0.85 AND agreement >= 0.85",
                "HIGH":     "confidence >= 0.70 AND agreement >= 0.70",
                "MEDIUM":   "confidence >= 0.50 AND agreement >= 0.50",
                "LOW":      "any ATTACK not meeting the above thresholds",
            },
            "category_resolution": {
                "rule_1": "NORMAL verdict → 'unknown'",
                "rule_2": "Both RF & XGB agree on same non-normal cat → use that cat",
                "rule_3": "RF & XGB disagree → use category from model with higher raw_score",
                "rule_4": "No supervised model voted ATTACK → 'Anomaly' (zero-day)",
            },
        },
        "metadata": {
            "test_rows":   metrics["n_samples"],
            "n_attack":    metrics["n_attack"],
            "n_normal":    metrics["n_normal"],
            "n_features":  len(feature_names),
            "feature_names": feature_names,
            "sampled": args.sample is not None,
            "sample_n": args.sample,
            "elapsed_total_seconds": round(time.time() - t_total, 1),
        },
    })

    EVAL_PATH.write_text(json.dumps(report, indent=2))
    console.print(f"  ✅ Report saved → [bold]{EVAL_PATH}[/bold]")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    console.print(
        f"\n[bold white on green]  ✅  Phase 4 Complete in {elapsed_total/60:.1f} min  [/bold white on green]"
    )
    console.print("\nOutputs:")
    console.print(f"  📄  {EVAL_PATH}")
    console.print(
        "\nNext: open [bold]notebooks/04_ensemble_analysis.ipynb[/bold] to visualise results."
    )


if __name__ == "__main__":
    main()
