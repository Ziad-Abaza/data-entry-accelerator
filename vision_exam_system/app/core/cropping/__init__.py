"""
Dynamic Cropping module for the Vision-Accelerated Exam Data Entry System.

This module provides the CropEngine class that extracts structured image
snippets for UI-driven human verification.

Exports:
    CropEngine: Main cropping engine
    CropConfig: Configuration dataclass
    CropResult: Single crop result dataclass
    CropCollection: Collection of all crops
    create_crop_engine: Factory function
    extract_all_crops: Convenience function
    DEFAULT_CROP_BOXES: Default crop coordinates
"""

from app.core.cropping.crop_engine import (
    CropEngine,
    CropConfig,
    CropResult,
    CropCollection,
    create_crop_engine,
    extract_all_crops,
    DEFAULT_CROP_BOXES,
    ALT_CROP_BOXES
)

__all__ = [
    "CropEngine",
    "CropConfig",
    "CropResult",
    "CropCollection",
    "create_crop_engine",
    "extract_all_crops",
    "DEFAULT_CROP_BOXES",
    "ALT_CROP_BOXES",
]