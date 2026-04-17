"""
OMR (Optical Mark Recognition) module for the Vision-Accelerated Exam Data Entry System.

This module provides the OMREngine class that implements pixel density-based
answer extraction from exam sheets.

Exports:
    OMREngine: Main OMR processing engine
    OMRConfig: Configuration dataclass
    OMRResult: Result dataclass
    OMRQuestionResult: Single question result dataclass
    create_omr_engine: Factory function
    extract_omr: Convenience function
    DEFAULT_TEMPLATE: Default template coordinates
"""

from app.core.omr.omr_engine import (
    OMREngine,
    OMRConfig,
    OMRResult,
    OMRQuestionResult,
    create_omr_engine,
    extract_omr,
    DEFAULT_TEMPLATE
)

__all__ = [
    "OMREngine",
    "OMRConfig",
    "OMRResult",
    "OMRQuestionResult",
    "create_omr_engine",
    "extract_omr",
    "DEFAULT_TEMPLATE",
]