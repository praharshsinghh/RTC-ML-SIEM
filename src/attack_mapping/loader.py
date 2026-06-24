"""
src/attack_mapping/loader.py — ATT&CK STIX Data Loader
========================================================
Loads and caches the MITRE ATT&CK Enterprise STIX dataset, exposing
clean lookup functions for techniques, tactics, and related metadata.

Architecture: Singleton cache
------------------------------
ATT&CK data (47 MB JSON) is expensive to parse. We implement a
module-level singleton: the first call to get_attack_data() loads and
parses the STIX bundle; subsequent calls return the cached instance.
This means models can call lookup functions repeatedly without incurring
repeated file I/O or JSON parsing overhead.

STIX file location
-------------------
Default path: {PROJECT_ROOT}/data/attack/enterprise-attack.json
This file is downloaded by run_attack_mapping.py on first run.
The path can be overridden by ATTACK_STIX_PATH environment variable.

Data freshness
--------------
The ATT&CK data file is a snapshot. MITRE releases quarterly updates.
For a production system, integrate with the ATT&CK TAXII server or
GitHub releases. For this project, the file is pinned to the version
downloaded during setup and noted in reports/attack_mapping_validation.json.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level singleton ────────────────────────────────────────────────────
_attack_data = None          # MitreAttackData instance
_stix_path: Optional[Path] = None
_techniques_cache: Optional[dict] = None   # technique_id → stix object
_tactics_cache: Optional[dict]    = None   # phase_name → tactic object

# Default STIX file path (relative to project root)
_DEFAULT_STIX_FILENAME = "data/attack/enterprise-attack.json"


def _resolve_stix_path(override: Optional[str] = None) -> Path:
    """
    Resolve the path to the enterprise-attack.json STIX file.

    Priority order:
    1. `override` argument (direct path)
    2. ATTACK_STIX_PATH environment variable
    3. Default: {PROJECT_ROOT}/data/attack/enterprise-attack.json
    """
    if override:
        return Path(override)

    env_path = os.environ.get("ATTACK_STIX_PATH")
    if env_path:
        return Path(env_path)

    # Walk up from this file's location to find project root
    here = Path(__file__).parent
    for ancestor in [here] + list(here.parents):
        candidate = ancestor / _DEFAULT_STIX_FILENAME
        if candidate.exists():
            return candidate

    # Absolute fallback
    return Path(_DEFAULT_STIX_FILENAME)


def get_attack_data(stix_path: Optional[str] = None, force_reload: bool = False):
    """
    Return the singleton MitreAttackData instance, loading if necessary.

    Parameters
    ----------
    stix_path : str, optional
        Override path to enterprise-attack.json STIX file.
    force_reload : bool
        If True, discard the cached instance and reload from disk.

    Returns
    -------
    MitreAttackData

    Raises
    ------
    FileNotFoundError
        If the STIX file does not exist at the resolved path.
    RuntimeError
        If the mitreattack-python library fails to parse the file.
    """
    global _attack_data, _stix_path, _techniques_cache, _tactics_cache

    if _attack_data is not None and not force_reload:
        return _attack_data

    from mitreattack.stix20 import MitreAttackData

    resolved = _resolve_stix_path(stix_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"ATT&CK STIX file not found: {resolved}\n"
            f"Run: python run_attack_mapping.py --download-only\n"
            f"Or set ATTACK_STIX_PATH environment variable."
        )

    logger.info(f"[ATT&CK] Loading STIX data from {resolved} …")
    try:
        _attack_data = MitreAttackData(str(resolved))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse ATT&CK STIX file: {exc}") from exc

    _stix_path = resolved

    # Pre-build technique lookup cache (id → stix object)
    techs = _attack_data.get_techniques(remove_revoked_deprecated=True)
    _techniques_cache = {}
    for t in techs:
        tid = _attack_data.get_attack_id(t.id)
        if tid:
            _techniques_cache[tid] = t

    # Pre-build tactic lookup cache (phase_name → tactic object)
    tactics = _attack_data.get_tactics()
    _tactics_cache = {}
    for tac in tactics:
        phase = tac.get("x_mitre_shortname", "")
        if phase:
            _tactics_cache[phase] = tac

    logger.info(
        f"[ATT&CK] Loaded {len(_techniques_cache)} techniques, "
        f"{len(_tactics_cache)} tactics from {resolved.name}"
    )
    return _attack_data


def get_technique_by_id(technique_id: str) -> Optional[object]:
    """
    Retrieve a STIX attack-pattern object by ATT&CK technique ID.

    Parameters
    ----------
    technique_id : str
        ATT&CK ID, e.g. 'T1595' or 'T1595.001'

    Returns
    -------
    STIX object dict-like, or None if not found / revoked.
    """
    if _techniques_cache is None:
        get_attack_data()
    return _techniques_cache.get(technique_id)


def get_tactic_by_phase(phase_name: str) -> Optional[object]:
    """
    Retrieve a STIX x-mitre-tactic object by ATT&CK phase short name.

    Parameters
    ----------
    phase_name : str
        Short name, e.g. 'reconnaissance', 'initial-access', 'impact'

    Returns
    -------
    STIX object or None.
    """
    if _tactics_cache is None:
        get_attack_data()
    return _tactics_cache.get(phase_name)


def get_technique_url(technique_id: str) -> str:
    """
    Construct the canonical ATT&CK URL for a technique.

    Parameters
    ----------
    technique_id : str — e.g. 'T1595' or 'T1595.001'

    Returns
    -------
    str — URL, e.g. 'https://attack.mitre.org/techniques/T1595/'
    """
    if "." in technique_id:
        parent, sub = technique_id.split(".", 1)
        return f"https://attack.mitre.org/techniques/{parent}/{sub}/"
    return f"https://attack.mitre.org/techniques/{technique_id}/"


def get_tactic_url(tactic_id: str) -> str:
    """
    Construct the canonical ATT&CK URL for a tactic.

    Parameters
    ----------
    tactic_id : str — e.g. 'TA0043'

    Returns
    -------
    str — URL
    """
    return f"https://attack.mitre.org/tactics/{tactic_id}/"


def extract_technique_metadata(technique_id: str) -> dict:
    """
    Extract a structured metadata dict for one technique from the STIX data.

    Combines: name, tactics, platforms, description (truncated), URL.
    Returns a dict suitable for building an AttackTechnique schema object.

    Parameters
    ----------
    technique_id : str

    Returns
    -------
    dict with keys: technique_id, name, tactics, platforms, description,
                    url, is_subtechnique, found_in_stix
    """
    obj = get_technique_by_id(technique_id)
    if obj is None:
        logger.warning(f"[ATT&CK] Technique {technique_id} not found in STIX data")
        return {
            "technique_id": technique_id,
            "name": f"Unknown ({technique_id})",
            "tactics": [],
            "platforms": [],
            "description": "",
            "url": get_technique_url(technique_id),
            "is_subtechnique": "." in technique_id,
            "found_in_stix": False,
        }

    # Extract tactics from kill chain phases
    tactics = []
    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") == "mitre-attack":
            tactics.append(phase.get("phase_name", ""))

    # Platforms
    platforms = list(obj.get("x_mitre_platforms", []))

    # Description — first 500 characters (full description can be multi-KB)
    desc_full = obj.get("description", "")
    description = desc_full[:500].rstrip() + ("…" if len(desc_full) > 500 else "")

    return {
        "technique_id": technique_id,
        "name": obj.get("name", ""),
        "tactics": tactics,
        "platforms": platforms,
        "description": description,
        "url": get_technique_url(technique_id),
        "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
        "found_in_stix": True,
    }


def extract_tactic_metadata(phase_name: str) -> dict:
    """
    Extract structured metadata for one tactic from the STIX data.

    Parameters
    ----------
    phase_name : str — e.g. 'reconnaissance'

    Returns
    -------
    dict with keys: tactic_id, name, short_name, url
    """
    obj = get_tactic_by_phase(phase_name)
    if obj is None:
        # Build a minimal record from the phase name alone
        name = phase_name.replace("-", " ").title()
        return {
            "tactic_id": "N/A",
            "name": name,
            "short_name": phase_name,
            "url": "",
        }

    # Extract ATT&CK tactic ID from external_references
    tactic_id = "N/A"
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            tactic_id = ref.get("external_id", "N/A")
            break

    return {
        "tactic_id": tactic_id,
        "name": obj.get("name", phase_name.replace("-", " ").title()),
        "short_name": phase_name,
        "url": get_tactic_url(tactic_id) if tactic_id != "N/A" else "",
    }


def list_all_technique_ids() -> list[str]:
    """Return all non-revoked technique IDs currently loaded."""
    if _techniques_cache is None:
        get_attack_data()
    return sorted(_techniques_cache.keys())


def stix_info() -> dict:
    """Return metadata about the loaded STIX file (version, stats)."""
    if _attack_data is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "stix_path": str(_stix_path),
        "n_techniques": len(_techniques_cache or {}),
        "n_tactics": len(_tactics_cache or {}),
    }
