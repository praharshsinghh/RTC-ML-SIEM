#!/usr/bin/env python3
"""
run_preprocessing.py — Phase 1 Preprocessing Orchestrator
===========================================================
Run this script after downloading the data to produce clean, split datasets
that all subsequent model training scripts will consume.

Usage:
    python run_preprocessing.py

Output (written to data/processed/):
    train.parquet           — 80% stratified split, all attack categories
    test.parquet            — 20% stratified split
    train_normal_only.parquet — normal traffic only (for unsupervised models)
    cleaner.joblib          — fitted DataCleaner (reused at inference time)
    feature_list.txt        — ordered list of feature column names
    preprocessing_report.json — summary stats for verification

Takes ~2-5 min on full 2.5M-row dataset, ~30s on Kaggle pre-split subset.
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    auto_load,
    validate_schema,
    DataCleaner,
    split_dataset,
    get_normal_only,
    save_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("\n" + "=" * 65)
    print("  UNSW-NB15 Preprocessing Pipeline")
    print("=" * 65)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Step 1/5: Loading data...")
    data = auto_load()

    if isinstance(data, tuple):
        # Kaggle pre-split format
        train_raw, test_raw = data
        logger.info(f"  Kaggle format: train={train_raw.shape}, test={test_raw.shape}")

        # Validate schema before cleaning
        logger.info("  Validating train schema...")
        schema_info = validate_schema(train_raw)
        logger.info(f"  Attack cat distribution:\n{schema_info['attack_cat_distribution']}")

        # ── 2. Clean + encode ─────────────────────────────────────────────────
        logger.info("Step 2/5: Fitting cleaner on train, transforming test...")
        cleaner = DataCleaner(clip_iqr_multiplier=3.0)
        train_clean = cleaner.fit_transform(train_raw)
        test_clean = cleaner.transform(test_raw)
        
        # ── 3. Normal-only subset ─────────────────────────────────────────────
        logger.info("Step 3/5: Extracting normal-only training subset...")
        normal_only = get_normal_only(train_clean)

    else:
        # Raw 4-partition format → need to split ourselves
        df_raw = data
        logger.info(f"  Raw partitions: {df_raw.shape}")

        schema_info = validate_schema(df_raw)
        logger.info(f"  Attack cat distribution:\n{schema_info['attack_cat_distribution']}")

        # ── 2. Clean entire dataset ────────────────────────────────────────────
        logger.info("Step 2/5: Fitting cleaner on combined data (will split after)...")
        cleaner = DataCleaner(clip_iqr_multiplier=3.0)
        df_clean = cleaner.fit_transform(df_raw)

        # ── 3. Split after cleaning ────────────────────────────────────────────
        logger.info("Step 3/5: Stratified train/test split (80/20)...")
        train_clean, test_clean = split_dataset(
            df_clean, test_size=0.2, random_state=42
        )
        normal_only = get_normal_only(train_clean)

    # ── 4. Save all artifacts ──────────────────────────────────────────────────
    logger.info("Step 4/5: Saving processed splits...")
    save_splits(train_clean, test_clean, normal_only, PROCESSED_DIR)

    cleaner.save(PROCESSED_DIR / "cleaner.joblib")

    # Save feature list for downstream models
    feature_list_path = PROCESSED_DIR / "feature_list.txt"
    feature_list_path.write_text("\n".join(cleaner.feature_cols))
    logger.info(f"  Feature list ({len(cleaner.feature_cols)} features) → {feature_list_path}")

    # ── 5. Write preprocessing report ──────────────────────────────────────────
    logger.info("Step 5/5: Writing preprocessing report...")
    report = {
        "train_rows": int(len(train_clean)),
        "test_rows": int(len(test_clean)),
        "normal_only_rows": int(len(normal_only)),
        "n_features": int(len(cleaner.feature_cols)),
        "feature_names": cleaner.feature_cols,
        "attack_cat_classes": list(cleaner.attack_cat_encoder.classes_),
        "train_attack_distribution": (
            train_clean["attack_cat"].value_counts().to_dict()
        ),
        "test_attack_distribution": (
            test_clean["attack_cat"].value_counts().to_dict()
        ),
        "imbalance_ratio": float(
            (train_clean["label"] == 0).sum() / max((train_clean["label"] == 1).sum(), 1)
        ),
        "processing_time_seconds": round(time.time() - t0, 1),
    }
    report_path = PROCESSED_DIR / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  ✅  Preprocessing Complete")
    print("=" * 65)
    print(f"  Train rows       : {report['train_rows']:,}")
    print(f"  Test rows        : {report['test_rows']:,}")
    print(f"  Normal-only rows : {report['normal_only_rows']:,}")
    print(f"  Feature count    : {report['n_features']}")
    print(f"  Normal:Attack ratio (train): {report['imbalance_ratio']:.2f}:1")
    print(f"  Elapsed time     : {elapsed:.1f}s")
    print(f"\n  Outputs in: {PROCESSED_DIR.resolve()}")
    print("=" * 65)
    print("\nNext: open notebooks/01_eda.ipynb to explore the data.")


if __name__ == "__main__":
    main()
