"""
src/attack_mapping/validator.py — ATT&CK Mapping Validation
=============================================================
Validates every ATT&CK technique ID in mappings.yaml against the live
STIX dataset to ensure no typos, deprecated techniques, or invalid IDs
slip into the enrichment layer.

Why validate?
-------------
MITRE regularly revokes or deprecates techniques (e.g. T1086 was
merged into T1059.001 in ATT&CK v8). If our mappings.yaml references
a revoked ID, the enricher would silently return incomplete results.
Pre-validation catches these issues at load time.

This module is also the canonical source for the coverage statistics
reported in reports/attack_mapping_validation.json:
  - How many categories are mapped?
  - How many mapped IDs are valid?
  - Which categories have HIGH vs LOW mapping confidence?
  - Which techniques could not be found in the STIX dataset?
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.attack_mapping.loader import get_attack_data, get_technique_by_id, list_all_technique_ids

logger = logging.getLogger(__name__)


def validate_mapping_config(mapping_config: list[dict]) -> dict:
    """
    Validate all technique IDs in the mapping configuration against the ATT&CK STIX data.

    Parameters
    ----------
    mapping_config : list[dict]
        The list of category mapping dicts loaded from mappings.yaml.

    Returns
    -------
    dict with structure:
        {
            "total_categories": int,
            "mapped_categories": int,       # categories with non-null primary_technique_id
            "unmapped_categories": int,     # null primary (e.g. 'unknown')
            "valid_ids": list[str],         # IDs found in STIX
            "invalid_ids": list[str],       # IDs NOT found in STIX (warnings)
            "categories_by_confidence": dict,
            "per_category": dict           # per-category validation result
        }
    """
    # Ensure ATT&CK data is loaded
    get_attack_data()

    total = len(mapping_config)
    mapped = 0
    unmapped = 0
    valid_ids = set()
    invalid_ids = set()
    by_confidence = {"HIGH": [], "MEDIUM": [], "LOW": []}
    per_category = {}

    for entry in mapping_config:
        cat = entry["category"]
        primary = entry.get("primary_technique_id")
        secondaries = entry.get("secondary_technique_ids", [])
        confidence = entry.get("mapping_confidence", "LOW")

        by_confidence.setdefault(confidence, []).append(cat)

        cat_result = {
            "category": cat,
            "primary_technique_id": primary,
            "mapping_confidence": confidence,
            "primary_valid": None,
            "secondary_results": {},
            "warnings": [],
        }

        if primary is None:
            unmapped += 1
            cat_result["primary_valid"] = "N/A (intentionally unmapped)"
            per_category[cat] = cat_result
            continue

        mapped += 1

        # Validate primary
        obj = get_technique_by_id(primary)
        if obj is not None:
            valid_ids.add(primary)
            cat_result["primary_valid"] = True
        else:
            invalid_ids.add(primary)
            cat_result["primary_valid"] = False
            cat_result["warnings"].append(
                f"Primary technique {primary} NOT found in ATT&CK STIX data. "
                f"It may be revoked, deprecated, or a typo."
            )
            logger.warning(
                f"[Validate] Category '{cat}': primary technique {primary} NOT found in STIX"
            )

        # Validate secondaries
        for tid in secondaries:
            obj2 = get_technique_by_id(tid)
            if obj2 is not None:
                valid_ids.add(tid)
                cat_result["secondary_results"][tid] = True
            else:
                invalid_ids.add(tid)
                cat_result["secondary_results"][tid] = False
                cat_result["warnings"].append(
                    f"Secondary technique {tid} NOT found in ATT&CK STIX data."
                )
                logger.warning(
                    f"[Validate] Category '{cat}': secondary technique {tid} NOT found in STIX"
                )

        per_category[cat] = cat_result

    result = {
        "total_categories": total,
        "mapped_categories": mapped,
        "unmapped_categories": unmapped,
        "valid_technique_ids": sorted(valid_ids),
        "invalid_technique_ids": sorted(invalid_ids),
        "categories_by_confidence": by_confidence,
        "per_category": per_category,
        "validation_passed": len(invalid_ids) == 0,
    }

    if invalid_ids:
        logger.warning(
            f"[Validate] {len(invalid_ids)} technique IDs not found in STIX: "
            f"{sorted(invalid_ids)}"
        )
    else:
        logger.info(
            f"[Validate] All {len(valid_ids)} mapped technique IDs validated successfully."
        )

    return result


def get_coverage_summary(validation_result: dict) -> dict:
    """
    Produce a concise coverage summary for the reports/attack_mapping_validation.json.

    Parameters
    ----------
    validation_result : dict — output of validate_mapping_config()

    Returns
    -------
    dict
    """
    per_cat = validation_result.get("per_category", {})
    total = validation_result["total_categories"]
    mapped = validation_result["mapped_categories"]

    return {
        "total_categories": total,
        "mapped_categories": mapped,
        "unmapped_categories": validation_result["unmapped_categories"],
        "mapping_coverage_pct": round(mapped / total * 100, 1) if total else 0,
        "valid_technique_count": len(validation_result["valid_technique_ids"]),
        "invalid_technique_count": len(validation_result["invalid_technique_ids"]),
        "invalid_technique_ids": validation_result["invalid_technique_ids"],
        "validation_passed": validation_result["validation_passed"],
        "confidence_breakdown": {
            conf: len(cats)
            for conf, cats in validation_result["categories_by_confidence"].items()
        },
        "categories_needing_review": [
            cat for cat, res in per_cat.items()
            if res.get("warnings")
        ],
    }
