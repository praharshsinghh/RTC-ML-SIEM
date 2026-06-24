"""
src/attack_mapping/schemas.py — ATT&CK Enrichment Data Schemas
================================================================
Defines the data contracts for the MITRE ATT&CK enrichment layer.
All structs are plain Python dataclasses (stdlib only) — transparent,
serialisable, and viva-defensible without external schema libraries.

Design principle
----------------
Each schema is a separate concern:
  AttackTechnique  — represents one ATT&CK technique record
  AttackTactic     — represents one ATT&CK tactic
  EnrichedIncident — the final output combining ensemble + ATT&CK context

The EnrichedIncident is the primary output fed to Phase 6 (reporting).
It is intentionally self-contained: every field a SOC analyst or incident
responder needs is present in a single flat-ish structure.

Viva defence
------------
"We use dataclasses rather than Pydantic here because the schema is
static and small. The to_dict() pattern ensures consistent JSON
serialisation without runtime schema overhead. For a production system
ingesting external data with validation requirements, Pydantic would be
preferable."
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class AttackTactic:
    """
    One MITRE ATT&CK tactic (a high-level adversary objective phase).

    In the ATT&CK framework, tactics represent the *why* — the adversary's
    goal at a particular stage of the attack lifecycle. Examples:
    Reconnaissance, Initial Access, Execution, Persistence, Impact.
    """
    tactic_id:   str    # e.g. 'TA0043'
    name:        str    # e.g. 'Reconnaissance'
    short_name:  str    # e.g. 'reconnaissance' (ATT&CK phase_name)
    url:         str    # e.g. 'https://attack.mitre.org/tactics/TA0043/'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttackTechnique:
    """
    One MITRE ATT&CK technique mapped to an UNSW-NB15 attack category.

    A technique represents the *how* — the specific method an adversary
    uses to achieve a tactic. Each UNSW-NB15 category maps to one
    PRIMARY technique and optionally several secondary ones.

    Fields
    ------
    technique_id   : ATT&CK ID, e.g. 'T1595'
    name           : Human-readable technique name
    tactics        : list of tactic short-names this technique belongs to
    description    : First 400 chars of ATT&CK description (truncated for reports)
    platforms      : list of target platforms (Windows, Linux, Network, etc.)
    url            : Direct link to technique on attack.mitre.org
    is_subtechnique: True if ID contains a '.' (e.g. T1595.001)
    found_in_stix  : True if this ID was validated against the live STIX dataset
    """
    technique_id:    str
    name:            str
    tactics:         list[str]       = field(default_factory=list)
    description:     str             = ""
    platforms:       list[str]       = field(default_factory=list)
    url:             str             = ""
    is_subtechnique: bool            = False
    found_in_stix:   bool            = True  # set False if lookup fails

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnrichedIncident:
    """
    A fully enriched security incident combining ensemble ML output with
    MITRE ATT&CK threat intelligence context.

    This is the primary output of Phase 5 and the input to Phase 6
    (incident reporting). The schema is designed to be:
      - Self-contained: everything a SOC analyst needs in one record
      - Serialisable: to_dict() produces clean JSON
      - Auditable: mapping_rationale explains every enrichment decision

    Fields (ML from Phase 4)
    ------------------------
    verdict          : 'ATTACK' or 'NORMAL'
    confidence       : ensemble confidence [0,1]
    severity         : 'CRITICAL'|'HIGH'|'MEDIUM'|'LOW'|'N/A'
    agreement_score  : fraction of model weight agreeing with verdict [0,1]
    attack_category  : UNSW-NB15 category string ('Generic', 'DoS', etc.)
    weighted_score   : raw ensemble weighted attack score [0,1]

    Fields (ATT&CK from Phase 5)
    ----------------------------
    primary_technique: primary mapped ATT&CK technique
    secondary_techniques: additional relevant techniques
    tactic           : primary tactic name (e.g. 'Impact')
    tactic_id        : tactic ATT&CK ID (e.g. 'TA0040')
    mapping_rationale: 1-2 sentence explanation of why this mapping was chosen
    mapping_confidence: 'HIGH'|'MEDIUM'|'LOW' — certainty of the category→technique mapping
    is_mapped        : True if a specific ATT&CK mapping exists for this category

    Fields (metadata)
    -----------------
    timestamp        : ISO-8601 UTC timestamp
    model_votes      : per-model vote summary dicts (from EnsembleResult.model_votes)
    """
    # ── Phase 4 ensemble fields ───────────────────────────────────────────────
    timestamp:       str
    verdict:         str             # 'ATTACK' | 'NORMAL'
    confidence:      float
    severity:        str
    agreement_score: float
    attack_category: str
    weighted_score:  float

    # ── Phase 5 ATT&CK enrichment fields ─────────────────────────────────────
    primary_technique:    Optional[AttackTechnique] = None
    secondary_techniques: list[AttackTechnique]     = field(default_factory=list)
    tactic:               str                        = "Unknown"
    tactic_id:            str                        = "N/A"
    mapping_rationale:    str                        = ""
    mapping_confidence:   str                        = "LOW"   # 'HIGH'|'MEDIUM'|'LOW'
    is_mapped:            bool                       = False

    # ── Audit trail ────────────────────────────────────────────────────────────
    model_votes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to plain dict (JSON-safe)."""
        d = asdict(self)
        return d

    @property
    def technique_id(self) -> str:
        """Convenience accessor — primary technique ID or 'N/A'."""
        return self.primary_technique.technique_id if self.primary_technique else "N/A"

    @property
    def technique_name(self) -> str:
        """Convenience accessor — primary technique name or 'Unknown'."""
        return self.primary_technique.name if self.primary_technique else "Unknown"
