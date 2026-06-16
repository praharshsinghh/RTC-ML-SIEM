"""
src/models/base.py — Canonical re-export shim
===============================================
BaseDetector and PredictionResult live in src/models/__init__.py.
This module re-exports them so downstream code can import from either:

    from src.models import BaseDetector, PredictionResult   # preferred
    from src.models.base import BaseDetector, PredictionResult  # also fine

It also defines the save_model / load_model naming convention that Phase 2
models expose as public aliases (the abstract base uses save / load internally
to keep the ABC surface minimal, but user-facing scripts call save_model /
load_model for clarity).
"""

from src.models import BaseDetector, PredictionResult  # noqa: F401

__all__ = ["BaseDetector", "PredictionResult"]
