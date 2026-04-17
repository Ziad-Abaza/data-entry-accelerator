"""
Utility functions and helpers for the Vision-Accelerated Exam Data Entry System.

This module contains common utilities, constants, and helper functions
used across the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
import hashlib


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        log_format: Optional custom log format
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# File System Helpers
# ============================================================================

def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_filename(prefix: str, extension: str) -> str:
    """
    Generate a filename with timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension (with or without dot)
        
    Returns:
        Generated filename
    """
    if not extension.startswith("."):
        extension = f".{extension}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"


def get_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of the file hash
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


# ============================================================================
# Image Helpers
# ============================================================================

def validate_image_path(path: Path) -> bool:
    """
    Validate that a path points to a valid image file.
    
    Args:
        path: Path to validate
        
    Returns:
        True if valid image path
    """
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    return path.suffix.lower() in valid_extensions


def get_image_files(directory: Path, recursive: bool = False) -> list[Path]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    if recursive:
        return [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
    else:
        return [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]


# ============================================================================
# Data Helpers
# ============================================================================

def format_confidence(value: float) -> str:
    """
    Format a confidence value as percentage string.
    
    Args:
        value: Confidence value (0.0 to 1.0)
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.1f}%"


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# Constants
# ============================================================================

class Constants:
    """Application-wide constants."""
    
    # Version
    VERSION = "1.0.0"
    APP_NAME = "Vision-Accelerated Exam Data Entry"
    
    # MCQ Options
    MCQ_OPTIONS = ["A", "B", "C", "D"]
    MCQ_OPTION_COUNT = 4
    MCQ_QUESTION_COUNT = 30
    
    # Status Colors (for UI)
    STATUS_COLOR_HIGH = "#28a745"    # Green
    STATUS_COLOR_MEDIUM = "#ffc107"  # Yellow
    STATUS_COLOR_LOW = "#dc3545"     # Red
    STATUS_COLOR_DEFAULT = "#6c757d" # Gray
    
    # UI Keys
    KEY_NEXT = "Next"
    KEY_PREV = "Previous"
    KEY_SAVE = "Save"
    KEY_EXPORT = "Export"
    KEY_CLEAR = "Clear"


# ============================================================================
# Error Classes
# ============================================================================

class VisionSystemError(Exception):
    """Base exception for the vision system."""
    pass


class ImageLoadError(VisionSystemError):
    """Exception raised when image loading fails."""
    pass


class ProcessingError(VisionSystemError):
    """Exception raised when processing fails."""
    pass


class ValidationError(VisionSystemError):
    """Exception raised when validation fails."""
    pass


class ExportError(VisionSystemError):
    """Exception raised when export fails."""
    pass