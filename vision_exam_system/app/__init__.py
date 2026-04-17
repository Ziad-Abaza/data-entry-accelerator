"""
App package for the Vision-Accelerated Exam Data Entry System.

This package contains the core application modules:
- core: Computer vision processing (preprocessing, OMR, OCR, cropping)
- services: Pipeline orchestration and business logic
- models: Data structures and representations
- ui: PySide6 graphical user interface
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "Vision System Team"

from app.models import ExamRecord, ProcessingResult, PipelineStats
from app.services import PipelineService, ExportService, ValidationService
from config import get_config, SystemConfig

__all__ = [
    "ExamRecord",
    "ProcessingResult", 
    "PipelineStats",
    "PipelineService",
    "ExportService",
    "ValidationService",
    "get_config",
    "SystemConfig",
]