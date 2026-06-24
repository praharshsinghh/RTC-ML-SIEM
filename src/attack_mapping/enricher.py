"""
src/attack_mapping/enricher.py — AttackEnricher: ATT&CK Enrichment Engine
===========================================================================
The AttackEnricher accepts Phase 4 EnsembleResult objects and transforms
them into EnrichedIncident records by:

  1. Looking up the UNSW-NB15 attack category in the mapping configuration
  2. Fetching ATT&CK technique metadata from the STIX dataset
  3. Building AttackTechnique and EnrichedIncident dataclass instances
  4. Gracefully handling unmapped categories, STIX lookup failures, and
     NORMAL verdicts (which do not receive ATT&CK enrichment)

Design decisions
----------------
Fail-safe by default: enrichment never raises exceptions at the per-record
level. Unknown categories receive a generic fallback mapping with LOW
confidence. STIX lookup failures produce a partial AttackTechnique with
found_in_stix=False. NORMAL verdicts produce a minimal EnrichedIncident
with is_mapped=False and no technique attached.

Caching within a session: the STIX data is already cached by loader.py.
The enricher additionally caches built AttackTechnique objects per
technique_id to avoid rebuilding dataclasses for repeated IDs.

Integration with Phase 6
-------------------------
EnrichedIncident.to_dict() produces a flat JSON-serialisable dict.
The Phase 6 incident reporter consumes this directly — no transformation
needed. The to_dict() output is also the format stored in
reports/incidents/*.json.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from src.attack_mapping.schemas import AttackTechnique, AttackTactic, EnrichedIncident
from src.attack_mapping.loader import (
    get_attack_data,
    extract_technique_metadata,
    extract_tactic_metadata,
)

logger = logging.getLogger(__name__)

# Default fallback technique for ATTACK verdicts with no category mapping
_FALLBACK_TECHNIQUE_ID = "T1190"
_FALLBACK_RATIONALE = (
    "Fallback mapping: attack category not found in mapping configuration. "
    "T1190 (Exploit Public-Facing Application) is assigned as the generic "
    "attacker-initiated network threat. Review and refine in mappings.yaml."
)


class AttackEnricher:
    """
    ATT&CK enrichment engine for Phase 4 EnsembleResult objects.

    Parameters
    ----------
    mappings_path : Path | str, optional
        Path to mappings.yaml. Defaults to src/attack_mapping/mappings.yaml.
    stix_path : str, optional
        Override path to enterprise-attack.json.
    preload : bool
        If True, load ATT&CK STIX data immediately on instantiation.
        If False, load lazily on first enrichment call.

    Usage
    -----
    >>> from src.attack_mapping import AttackEnricher
    >>> enricher = AttackEnricher()
    >>> from src.ensemble import EnsembleResult  # Phase 4 output
    >>> incident = enricher.enrich(ensemble_result)
    >>> print(incident.technique_id, incident.tactic)
    """

    def __init__(
        self,
        mappings_path: Optional[Path | str] = None,
        stix_path: Optional[str] = None,
        preload: bool = True,
    ) -> None:
        # Resolve mappings.yaml path
        if mappings_path is None:
            mappings_path = Path(__file__).parent / "mappings.yaml"
        self._mappings_path = Path(mappings_path)

        self._stix_path = stix_path
        self._mappings: dict[str, dict] = {}        # category → mapping config entry
        self._technique_cache: dict[str, AttackTechnique] = {}  # id → dataclass

        self._load_mappings()

        if preload:
            get_attack_data(stix_path=stix_path)

    # ── Config loading ────────────────────────────────────────────────────────

    def _load_mappings(self) -> None:
        """
        Parse mappings.yaml and build a category→entry lookup dict.
        """
        if not self._mappings_path.exists():
            raise FileNotFoundError(
                f"Mappings config not found: {self._mappings_path}\n"
                f"This file should be present at src/attack_mapping/mappings.yaml"
            )

        with open(self._mappings_path) as f:
            raw = yaml.safe_load(f)

        for entry in raw.get("mappings", []):
            cat = entry["category"]
            self._mappings[cat] = entry

        logger.info(
            f"[Enricher] Loaded {len(self._mappings)} category mappings "
            f"from {self._mappings_path.name}"
        )

    # ── Technique building ────────────────────────────────────────────────────

    def _build_technique(self, technique_id: str) -> AttackTechnique:
        """
        Build an AttackTechnique dataclass from the STIX data.
        Results are cached so repeated lookups for the same ID are free.
        """
        if technique_id in self._technique_cache:
            return self._technique_cache[technique_id]

        meta = extract_technique_metadata(technique_id)
        tech = AttackTechnique(
            technique_id=meta["technique_id"],
            name=meta["name"],
            tactics=meta["tactics"],
            description=meta["description"],
            platforms=meta["platforms"],
            url=meta["url"],
            is_subtechnique=meta["is_subtechnique"],
            found_in_stix=meta["found_in_stix"],
        )
        self._technique_cache[technique_id] = tech
        return tech

    def _build_tactic(self, phase_name: str) -> AttackTactic:
        """Build an AttackTactic dataclass from the STIX data."""
        meta = extract_tactic_metadata(phase_name)
        return AttackTactic(
            tactic_id=meta["tactic_id"],
            name=meta["name"],
            short_name=meta["short_name"],
            url=meta["url"],
        )

    # ── Core enrichment ───────────────────────────────────────────────────────

    def enrich(self, ensemble_result) -> EnrichedIncident:
        """
        Enrich one EnsembleResult into an EnrichedIncident.

        Parameters
        ----------
        ensemble_result : EnsembleResult (Phase 4 schema)
            The output from EnsembleDetector.predict()[i].

        Returns
        -------
        EnrichedIncident

        Behaviour by verdict
        --------------------
        NORMAL: returns EnrichedIncident with is_mapped=False, no technique.
        ATTACK: looks up the attack category in mappings.yaml, fetches STIX
                metadata, builds AttackTechnique and resolves tactic.
        Unknown category: uses fallback technique T1190 with LOW confidence.
        """
        # Build base model_votes list
        model_votes = [v.to_dict() for v in ensemble_result.model_votes]

        # Base incident (Phase 4 fields)
        incident = EnrichedIncident(
            timestamp=ensemble_result.timestamp,
            verdict=ensemble_result.final_verdict,
            confidence=ensemble_result.confidence,
            severity=ensemble_result.severity,
            agreement_score=ensemble_result.agreement_score,
            attack_category=ensemble_result.final_attack_cat,
            weighted_score=ensemble_result.weighted_attack_score,
            model_votes=model_votes,
        )

        # NORMAL traffic — no ATT&CK enrichment needed
        if ensemble_result.final_verdict == "NORMAL":
            incident.tactic = "N/A"
            incident.tactic_id = "N/A"
            incident.mapping_rationale = "Normal traffic — no ATT&CK technique applicable."
            incident.mapping_confidence = "HIGH"
            incident.is_mapped = False
            return incident

        # Lookup category mapping
        cat = ensemble_result.final_attack_cat
        mapping_entry = self._mappings.get(cat)

        if mapping_entry is None:
            logger.warning(
                f"[Enricher] No mapping found for category '{cat}'. "
                f"Using fallback technique {_FALLBACK_TECHNIQUE_ID}."
            )
            return self._apply_fallback(incident, cat)

        primary_id = mapping_entry.get("primary_technique_id")

        # If primary_id is null (e.g. 'unknown' category) but verdict is ATTACK
        # (shouldn't happen normally — ensemble only returns ATTACK for non-normal)
        if primary_id is None:
            return self._apply_fallback(incident, cat)

        # Build primary technique
        try:
            primary_tech = self._build_technique(primary_id)
        except Exception as exc:
            logger.error(f"[Enricher] Failed to build technique {primary_id}: {exc}")
            return self._apply_fallback(incident, cat)

        # Build secondary techniques
        secondary_ids = mapping_entry.get("secondary_technique_ids", [])
        secondaries = []
        for sid in secondary_ids:
            try:
                secondaries.append(self._build_technique(sid))
            except Exception as exc:
                logger.warning(f"[Enricher] Failed to build secondary {sid}: {exc}")

        # Resolve primary tactic
        tactic_override = mapping_entry.get("tactic_override")
        if tactic_override:
            primary_phase = tactic_override
        elif primary_tech.tactics:
            primary_phase = primary_tech.tactics[0]
        else:
            primary_phase = "unknown"

        # Build tactic metadata
        tactic_meta = self._build_tactic(primary_phase)

        # Populate enriched incident
        incident.primary_technique    = primary_tech
        incident.secondary_techniques = secondaries
        incident.tactic               = tactic_meta.name
        incident.tactic_id            = tactic_meta.tactic_id
        incident.mapping_rationale    = mapping_entry.get("rationale", "").strip()
        incident.mapping_confidence   = mapping_entry.get("mapping_confidence", "LOW")
        incident.is_mapped            = True

        logger.debug(
            f"[Enricher] '{cat}' → {primary_id} ({primary_tech.name}) "
            f"| tactic={tactic_meta.name} | conf={incident.mapping_confidence}"
        )

        return incident

    def _apply_fallback(self, incident: EnrichedIncident, cat: str) -> EnrichedIncident:
        """
        Apply fallback enrichment for unmapped or problematic categories.
        Uses T1190 with LOW mapping_confidence and a warning note.
        """
        try:
            fallback_tech = self._build_technique(_FALLBACK_TECHNIQUE_ID)
            tactic_phase  = fallback_tech.tactics[0] if fallback_tech.tactics else "initial-access"
            tactic_meta   = self._build_tactic(tactic_phase)

            incident.primary_technique  = fallback_tech
            incident.tactic             = tactic_meta.name
            incident.tactic_id          = tactic_meta.tactic_id
        except Exception as exc:
            logger.error(f"[Enricher] Fallback enrichment failed: {exc}")

        incident.mapping_rationale  = (
            f"{_FALLBACK_RATIONALE} Original category: '{cat}'."
        )
        incident.mapping_confidence = "LOW"
        incident.is_mapped          = False
        return incident

    # ── Batch enrichment ──────────────────────────────────────────────────────

    def enrich_batch(self, ensemble_results: list) -> list[EnrichedIncident]:
        """
        Enrich a list of EnsembleResult objects.

        Parameters
        ----------
        ensemble_results : list[EnsembleResult]

        Returns
        -------
        list[EnrichedIncident] — same length as input
        """
        logger.info(f"[Enricher] Enriching {len(ensemble_results):,} ensemble results …")
        incidents = []
        errors = 0
        for res in ensemble_results:
            try:
                incidents.append(self.enrich(res))
            except Exception as exc:
                logger.error(f"[Enricher] Unexpected error enriching result: {exc}")
                errors += 1
                # Return a minimal incident rather than crashing the batch
                incidents.append(EnrichedIncident(
                    timestamp=getattr(res, "timestamp", datetime.now(tz=timezone.utc).isoformat()),
                    verdict=getattr(res, "final_verdict", "UNKNOWN"),
                    confidence=0.0, severity="N/A",
                    agreement_score=0.0, attack_category="error",
                    weighted_score=0.0,
                    mapping_rationale=f"Enrichment failed: {exc}",
                ))

        logger.info(
            f"[Enricher] Enrichment complete: {len(incidents):,} records "
            f"({errors} errors)"
        )
        return incidents

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_mapping_for_category(self, category: str) -> Optional[dict]:
        """Return the raw mapping config entry for a given category."""
        return self._mappings.get(category)

    def list_mapped_categories(self) -> list[str]:
        """Return all category names present in mappings.yaml."""
        return list(self._mappings.keys())

    def cache_stats(self) -> dict:
        """Return cache hit stats for debugging."""
        return {
            "cached_techniques": len(self._technique_cache),
            "cached_ids": sorted(self._technique_cache.keys()),
        }
