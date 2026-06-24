"""
src/attack_mapping/__init__.py — Public API for the Attack Mapping Module
=========================================================================
Phase 5: MITRE ATT&CK Integration and Threat Enrichment Layer.

Re-exports all public classes and functions so downstream code can import
from a single surface:

    from src.attack_mapping import AttackEnricher, EnrichedIncident

Module structure
----------------
  schemas.py   — AttackTechnique, AttackTactic, EnrichedIncident dataclasses
  loader.py    — Singleton STIX data loader + lookup functions
  mappings.yaml — Category→technique mapping config (edit to update mappings)
  validator.py — Validates all mapped technique IDs against STIX data
  enricher.py  — AttackEnricher: Phase 4 output → EnrichedIncident
"""

from src.attack_mapping.schemas import (   # noqa: F401
    AttackTechnique,
    AttackTactic,
    EnrichedIncident,
)
from src.attack_mapping.loader import (    # noqa: F401
    get_attack_data,
    get_technique_by_id,
    extract_technique_metadata,
    extract_tactic_metadata,
    stix_info,
)
from src.attack_mapping.validator import (  # noqa: F401
    validate_mapping_config,
    get_coverage_summary,
)
from src.attack_mapping.enricher import AttackEnricher  # noqa: F401

__all__ = [
    "AttackTechnique",
    "AttackTactic",
    "EnrichedIncident",
    "AttackEnricher",
    "get_attack_data",
    "get_technique_by_id",
    "extract_technique_metadata",
    "extract_tactic_metadata",
    "validate_mapping_config",
    "get_coverage_summary",
    "stix_info",
]
