"""
src/pipeline/__init__.py — Public API for the Pipeline Module
=============================================================
Phase 6: End-to-End Detection Pipeline.

Re-exports the primary public interface so downstream code can import
from a single surface:

    from src.pipeline import ThreatDetectionPipeline
"""

from src.pipeline.pipeline import ThreatDetectionPipeline  # noqa: F401

__all__ = ["ThreatDetectionPipeline"]
