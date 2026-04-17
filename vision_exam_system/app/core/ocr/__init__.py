"""
OCR (Optical Character Recognition) module for the Vision-Accelerated Exam Data Entry System.

This module provides the OCREngine class that implements handwritten digit
extraction using PyTesseract.

Exports:
    OCREngine: Main OCR processing engine
    OCRConfig: Configuration dataclass
    OCRResult: Result dataclass
    create_ocr_engine: Factory function
    extract_academic_id: Convenience function
    ACADEMIC_ID_BOX: Default ID region coordinates
"""

from app.core.ocr.ocr_engine import (
    OCREngine,
    OCRConfig,
    OCRResult,
    create_ocr_engine,
    extract_academic_id,
    ACADEMIC_ID_BOX,
    ACADEMIC_ID_BOX_ALT
)

__all__ = [
    "OCREngine",
    "OCRConfig",
    "OCRResult",
    "create_ocr_engine",
    "extract_academic_id",
    "ACADEMIC_ID_BOX",
    "ACADEMIC_ID_BOX_ALT",
]