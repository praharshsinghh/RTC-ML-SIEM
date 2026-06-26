#!/usr/bin/env python3
"""
run_pipeline.py — Phase 6: End-to-End Threat Detection Pipeline Runner
========================================================================

Reads network traffic data from a CSV or Parquet file (or uses the
default test.parquet produced by Phase 1), runs the full detection
pipeline, and writes incident reports + summary to reports/incidents/.

Usage
-----
# Default — runs on data/processed/test.parquet
python run_pipeline.py

# Custom input file
python run_pipeline.py --input data/test.parquet
python run_pipeline.py --input traffic.csv

# Limit rows (useful for quick smoke-tests)
python run_pipeline.py --input data/processed/test.parquet --max-rows 500

# Quiet output (WARNING+ only)
python run_pipeline.py --log-level WARNING

Arguments
---------
--input       Path to the input CSV or Parquet file.
              Default: data/processed/test.parquet
--max-rows    Process only the first N rows. Optional.
--log-level   Logging verbosity: DEBUG | INFO | WARNING | ERROR.
              Default: INFO
--no-stix-preload
              Skip eager STIX loading (slightly faster startup, slow on
              first enrich call instead). Useful in CI environments.

Output
------
reports/incidents/
  incident_report_YYYYMMDD_HHMMSS.json   — all attack incident reports
  summary.json                           — session-level statistics

Exit codes
----------
0 — pipeline completed successfully
1 — pipeline failed (error logged to stderr)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _configure_logging(level: str) -> None:
    """Configure root logger with a clean timestamped format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTC Phase 6 — End-to-End Threat Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/test.parquet",
        help="Path to input CSV or Parquet file (default: data/processed/test.parquet)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows (optional, for quick tests)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--no-stix-preload",
        action="store_true",
        default=False,
        help="Disable eager STIX data loading (lazy load on first enrich call)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point — returns 0 on success, 1 on failure."""
    args = _parse_args()
    _configure_logging(args.log_level)
    logger = logging.getLogger("run_pipeline")

    # ── Import after logging is configured so module-level loggers fire ───────
    from src.pipeline import ThreatDetectionPipeline

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    logger.info(f"Input:     {input_path}")
    logger.info(f"Max rows:  {args.max_rows or 'all'}")
    logger.info(f"STIX:      {'lazy' if args.no_stix_preload else 'eager'}")

    try:
        pipeline = ThreatDetectionPipeline(
            stix_preload=not args.no_stix_preload,
        )
        result = pipeline.run(
            source=input_path,
            max_rows=args.max_rows,
        )

        summary = result["summary"]
        print()
        print("─" * 60)
        print("  PIPELINE SUMMARY")
        print("─" * 60)
        print(f"  Run ID      : {result['run_id']}")
        print(f"  Timestamp   : {result['run_timestamp']}")
        print(f"  Total rows  : {summary['total_records']:,}")
        print(f"  Normal      : {summary['normal_count']:,}")
        print(f"  Attack      : {summary['attack_count']:,}  "
              f"({summary['attack_rate_pct']:.1f}%)")
        print(f"  Severity    : CRITICAL={summary['severity_distribution']['CRITICAL']}, "
              f"HIGH={summary['severity_distribution']['HIGH']}, "
              f"MEDIUM={summary['severity_distribution']['MEDIUM']}, "
              f"LOW={summary['severity_distribution']['LOW']}")
        print(f"  Avg Conf    : {summary['average_confidence']:.4f}")
        print(f"  Avg Agree   : {summary['average_agreement_score']:.4f}")
        print(f"  Reports     : {len(result['reports'])} incident(s) saved")
        print("─" * 60)
        print()
        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        return 1
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
