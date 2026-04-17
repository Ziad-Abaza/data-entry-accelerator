"""
Preprocessing module for the Vision-Accelerated Exam Data Entry System.

This module provides the ImagePreprocessor class that implements
Phase A: Image Pre-processing pipeline.

Exports:
    ImagePreprocessor: Main preprocessing class
    PreprocessingConfig: Configuration dataclass
    create_preprocessor: Factory function
    preprocess_image: Convenience function
"""

from app.core.preprocessing.preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
    create_preprocessor,
    preprocess_image
)

__all__ = [
    "ImagePreprocessor",
    "PreprocessingConfig",
    "create_preprocessor",
    "preprocess_image",
]