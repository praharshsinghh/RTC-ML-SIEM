#!/usr/bin/env python3
"""
run_attack_mapping.py — Phase 5: ATT&CK Integration Entry Point
================================================================
Loads Phase 4 ensemble results (or generates them fresh), enriches
each detection with MITRE ATT&CK context, validates all mappings
against the STIX dataset, and writes:

  reports/attack_mapping_validation.json  — mapping coverage + validation
  reports/incidents/sample_enriched.json  — 5 sample enriched incidents
  reports/attack_mapping_validation.json  — tactic/technique distributions

Usage
-----
    python run_attack_mapping.py
    python run_attack_mapping.py --sample 5000    # quicker run
    python run_attack_mapping.py --download-only  # just fetch STIX data

Expected runtime
----------------
  STIX data download  : ~15-30 sec (47 MB)
  STIX parsing        : ~10-15 sec
  Enrichment (508K)   : ~30 sec (pure Python, no ML inference)
  Enrichment (5K)     : ~3 sec

Note: ATT&CK data is loaded ONCE and cached; subsequent enrichments
in the same session use the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attack_mapping import (
    AttackEnricher,
    get_attack_data,
    validate_mapping_config,
    get_coverage_summary,
    stix_info,
)
from src.ensemble import EnsembleDetector

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
STIX_DIR       = PROJECT_ROOT / "data" / "attack"
STIX_PATH      = STIX_DIR / "enterprise-attack.json"
REPORTS_DIR    = PROJECT_ROOT / "reports"
INCIDENTS_DIR  = REPORTS_DIR / "incidents"
MAPPINGS_PATH  = PROJECT_ROOT / "src" / "attack_mapping" / "mappings.yaml"

RF_PATH  = PROJECT_ROOT / "data" / "models" / "rf_detector.joblib"
XGB_PATH = PROJECT_ROOT / "data" / "models" / "xgb_detector.joblib"
ISO_PATH = PROJECT_ROOT / "models" / "isolation_forest.joblib"
AE_PATH  = PROJECT_ROOT / "models" / "autoencoder.pt"

VALIDATION_PATH = REPORTS_DIR / "attack_mapping_validation.json"
SAMPLES_PATH    = INCIDENTS_DIR / "sample_enriched.json"

STIX_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


def make_serialisable(obj):
    if isinstance(obj, dict):  return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serialisable(i) for i in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# STIX download
# ─────────────────────────────────────────────────────────────────────────────

def download_stix(force: bool = False) -> None:
    if STIX_PATH.exists() and not force:
        console.print(f"  ✅ ATT&CK STIX already present: {STIX_PATH} ({STIX_PATH.stat().st_size//1_000_000}MB)")
        return

    console.print(f"  ⬇️  Downloading enterprise-attack.json from GitHub …")
    t0 = time.time()
    urllib.request.urlretrieve(ATTACK_STIX_URL, STIX_PATH)
    elapsed = time.time() - t0
    console.print(
        f"  ✅ Downloaded {STIX_PATH.stat().st_size//1_000_000}MB in {elapsed:.1f}s → {STIX_PATH}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation report
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(enricher: AttackEnricher) -> dict:
    import yaml
    with open(MAPPINGS_PATH) as f:
        raw = yaml.safe_load(f)
    mapping_config = raw.get("mappings", [])

    console.print("\n[bold cyan]━━━ Validating ATT&CK Mappings ━━━[/bold cyan]")
    val_result = validate_mapping_config(mapping_config)
    coverage   = get_coverage_summary(val_result)

    table = Table(title="📋  ATT&CK Mapping Coverage", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Metric", style="bold", min_width=30)
    table.add_column("Value", justify="right", min_width=12)

    table.add_row("Total categories", str(coverage["total_categories"]))
    table.add_row("Mapped categories", str(coverage["mapped_categories"]))
    table.add_row("Coverage %", f"{coverage['mapping_coverage_pct']}%")
    table.add_row("Valid technique IDs", str(coverage["valid_technique_count"]))
    table.add_row("Invalid technique IDs", str(coverage["invalid_technique_count"]))
    status = "[green]PASSED ✅[/green]" if coverage["validation_passed"] else "[red]FAILED ❌[/red]"
    table.add_row("Validation status", status)
    table.add_row("", "")
    table.add_row("[bold]Confidence breakdown[/bold]", "")
    for conf, cnt in coverage["confidence_breakdown"].items():
        table.add_row(f"  {conf}", str(cnt))

    console.print(table)
    return {"validation": val_result, "coverage": coverage}


# ─────────────────────────────────────────────────────────────────────────────
# Sample enriched incidents
# ─────────────────────────────────────────────────────────────────────────────

def build_sample_incidents(enricher: AttackEnricher, ensemble: EnsembleDetector,
                            feature_names: list[str]) -> list[dict]:
    """
    Generate one representative sample incident for each attack category
    plus one NORMAL sample. Returns list of dicts for JSON saving.
    """
    console.print("\n[bold cyan]━━━ Generating Sample Enriched Incidents ━━━[/bold cyan]")

    target_cats = ["unknown", "Reconnaissance", "Exploits", "DoS", "Backdoors",
                   "Shellcode", "Generic", "Fuzzers", "Worms", "Analysis"]

    TEST_PATH = PROCESSED_DIR / "test.parquet"
    df = pd.read_parquet(TEST_PATH, columns=feature_names + ["attack_cat", "label"])

    samples = []
    for cat in target_cats:
        subset = df[df["attack_cat"] == cat]
        if len(subset) == 0:
            console.print(f"  ⚠️  No test rows for category '{cat}' — skipping")
            continue

        row = subset.sample(n=1, random_state=42).iloc[0]
        X   = row[feature_names].values.astype("float32").reshape(1, -1)

        results = ensemble.predict(X)
        result  = results[0]

        incident = enricher.enrich(result)
        d = make_serialisable(incident.to_dict())

        # Add ground truth for the sample notebook
        d["_ground_truth_category"] = cat
        d["_ground_truth_label"]    = int(row["label"])

        samples.append(d)
        tech_id   = d.get("primary_technique", {}).get("technique_id", "N/A") if d.get("primary_technique") else "N/A"
        tactic    = d.get("tactic", "N/A")
        console.print(
            f"  {'✅' if result.final_verdict=='ATTACK' else '⬜'} "
            f"[bold]{cat:17s}[/bold] → verdict={result.final_verdict:6s} "
            f"tech={tech_id:12s} tactic={tactic}"
        )

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Batch enrichment stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_enrichment_stats(enricher: AttackEnricher, ensemble: EnsembleDetector,
                              feature_names: list[str], sample_n: int | None) -> dict:
    """Run ensemble + enrichment on the test set and compute distribution stats."""
    console.print("\n[bold cyan]━━━ Computing Enrichment Statistics ━━━[/bold cyan]")

    TEST_PATH = PROCESSED_DIR / "test.parquet"
    df = pd.read_parquet(TEST_PATH, columns=feature_names + ["label", "attack_cat"])

    if sample_n:
        df = df.sample(n=sample_n, random_state=42)
        console.print(f"  Using {sample_n:,}-row sample")
    else:
        console.print(f"  Using full test set ({len(df):,} rows)")

    X = df[feature_names].values.astype("float32")
    t0 = time.time()
    results   = ensemble.predict(X)
    incidents = enricher.enrich_batch(results)
    elapsed   = time.time() - t0
    console.print(f"  ✅ {len(incidents):,} records enriched in {elapsed:.1f}s")

    # Tactic distribution
    tactic_dist: dict[str, int] = {}
    technique_dist: dict[str, int] = {}
    severity_dist: dict[str, int] = {}
    confidence_dist: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for inc in incidents:
        if inc.verdict == "ATTACK":
            tactic_dist[inc.tactic] = tactic_dist.get(inc.tactic, 0) + 1
            if inc.primary_technique:
                tid = inc.primary_technique.technique_id
                technique_dist[tid] = technique_dist.get(tid, 0) + 1
            confidence_dist[inc.mapping_confidence] = (
                confidence_dist.get(inc.mapping_confidence, 0) + 1
            )
        severity_dist[inc.severity] = severity_dist.get(inc.severity, 0) + 1

    # Print tactic table
    tac_table = Table(title="🎯 Tactic Distribution (ATTACK alerts)", box=box.SIMPLE,
                      header_style="bold magenta")
    tac_table.add_column("Tactic", style="bold", min_width=28)
    tac_table.add_column("Count", justify="right", min_width=10)
    for tac, cnt in sorted(tactic_dist.items(), key=lambda x: -x[1]):
        tac_table.add_row(tac, f"{cnt:,}")
    console.print(tac_table)

    return {
        "tactic_distribution": tactic_dist,
        "technique_distribution": technique_dist,
        "severity_distribution": severity_dist,
        "mapping_confidence_distribution": confidence_dist,
        "n_total": len(incidents),
        "n_attack": sum(1 for i in incidents if i.verdict == "ATTACK"),
        "n_normal": sum(1 for i in incidents if i.verdict == "NORMAL"),
        "elapsed_seconds": round(elapsed, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: MITRE ATT&CK enrichment")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Sample N rows for statistics (default: full test set)")
    parser.add_argument("--download-only", action="store_true",
                        help="Download ATT&CK STIX data then exit")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download STIX even if file exists")
    args = parser.parse_args()

    t_total = time.time()
    console.print("\n[bold white on blue]  UNSW-NB15  ·  Phase 5: ATT&CK Enrichment  [/bold white on blue]\n")

    # ── 1. Ensure STIX data ───────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ ATT&CK STIX Data ━━━[/bold cyan]")
    download_stix(force=args.force_download)

    if args.download_only:
        console.print("\nDownload-only mode — exiting.")
        return

    # ── 2. Load ATT&CK data ───────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Loading ATT&CK Knowledge Base ━━━[/bold cyan]")
    t0 = time.time()
    get_attack_data(stix_path=str(STIX_PATH))
    info = stix_info()
    console.print(
        f"  ✅ Loaded {info['n_techniques']} techniques, {info['n_tactics']} tactics "
        f"in {time.time()-t0:.1f}s"
    )

    # ── 3. Validate mappings ───────────────────────────────────────────────────
    enricher   = AttackEnricher(mappings_path=MAPPINGS_PATH, stix_path=str(STIX_PATH), preload=False)
    val_data   = run_validation(enricher)

    # ── 4. Load ensemble models ────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Loading Ensemble Models ━━━[/bold cyan]")
    ensemble = EnsembleDetector.from_disk(RF_PATH, XGB_PATH, ISO_PATH, AE_PATH)
    console.print("  ✅ All models loaded")

    feature_names = (PROCESSED_DIR / "feature_list.txt").read_text().strip().splitlines()

    # ── 5. Build sample incidents ──────────────────────────────────────────────
    samples = build_sample_incidents(enricher, ensemble, feature_names)
    SAMPLES_PATH.write_text(json.dumps(make_serialisable(samples), indent=2))
    console.print(f"\n  ✅ {len(samples)} sample incidents saved → {SAMPLES_PATH}")

    # ── 6. Compute enrichment statistics ──────────────────────────────────────
    stats = compute_enrichment_stats(enricher, ensemble, feature_names, args.sample)

    # ── 7. Save validation report ─────────────────────────────────────────────
    console.print(f"\n[bold cyan]━━━ Saving Validation Report ━━━[/bold cyan]")
    report = make_serialisable({
        "phase": "5_attack_mapping",
        "stix_info": info,
        "validation": val_data,
        "enrichment_statistics": stats,
        "metadata": {
            "stix_path": str(STIX_PATH),
            "mappings_path": str(MAPPINGS_PATH),
            "sample_n": args.sample,
            "elapsed_total_seconds": round(time.time() - t_total, 1),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    })
    VALIDATION_PATH.write_text(json.dumps(report, indent=2))
    console.print(f"  ✅ Report saved → [bold]{VALIDATION_PATH}[/bold]")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    console.print(
        f"\n[bold white on green]  ✅  Phase 5 Complete in {elapsed_total/60:.1f} min  [/bold white on green]"
    )
    console.print("\nOutputs:")
    console.print(f"  📄  {VALIDATION_PATH}")
    console.print(f"  🔍  {SAMPLES_PATH}")
    console.print("\nNext: open [bold]notebooks/05_attack_mapping_analysis.ipynb[/bold]")


if __name__ == "__main__":
    main()
