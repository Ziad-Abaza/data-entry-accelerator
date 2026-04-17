"""
Configuration management for the Vision-Accelerated Exam Data Entry System.

This module provides centralized configuration with type-safe access to all
system parameters including image processing, OMR thresholds, OCR settings,
and directory paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImageConfig:
    """Image processing configuration."""
    master_height: int = 2000
    master_width: Optional[int] = None  # Auto-calculated if None
    interpolation: str = "lanczos"  # cv2 interpolation method
    supported_formats: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


@dataclass
class OMRConfig:
    """Optical Mark Recognition configuration."""
    density_threshold: float = 0.15  # Minimum pixel density to consider marked
    ambiguity_threshold: float = 0.10  # Threshold for marking as ambiguous
    min_box_width: int = 30
    min_box_height: int = 20
    options_count: int = 4  # A, B, C, D
    questions_count: int = 30


@dataclass
class OCRConfig:
    """Optical Character Recognition configuration."""
    tesseract_path: Optional[str] = None
    psm_mode: int = 6  # Page segmentation mode (single uniform block)
    oem_mode: int = 3  # OCR engine mode (default + LSTM)
    whitelist_chars: str = "0123456789"  # For academic ID
    language: str = "eng"
    confidence_threshold: float = 60.0


@dataclass
class CropConfig:
    """Image cropping configuration for UI display."""
    name_box: tuple = (100, 50, 400, 100)  # (x, y, width, height)
    id_box: tuple = (500, 50, 300, 100)
    mcq_grid_start: tuple = (50, 200)
    mcq_box_size: tuple = (40, 30)
    question_regions: dict = None  # Dynamically populated

    def __post_init__(self):
        if self.question_regions is None:
            self.question_regions = {
                "q2": (50, 800, 600, 200),
                "q3": (50, 1050, 600, 200),
                "q4": (50, 1300, 600, 200),
            }


@dataclass
class DirectoryConfig:
    """Directory path configuration."""
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    temp_dir: str = "data/temp"
    log_dir: str = "data/logs"
    template_dir: str = "config/templates"

    def __post_init__(self):
        """Convert string paths to Path objects and ensure directories exist."""
        for attr in ["input_dir", "output_dir", "temp_dir", "log_dir", "template_dir"]:
            path = Path(getattr(self, attr))
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, attr, path)


@dataclass
class UIConfig:
    """User interface configuration."""
    window_title: str = "Vision-Accelerated Exam Data Entry"
    window_width: int = 1400
    window_height: int = 900
    thumbnail_size: tuple = (150, 200)
    zoom_factor: float = 2.0
    auto_advance: bool = True
    theme: str = "light"


@dataclass
class ValidationConfig:
    """Data validation configuration."""
    allow_duplicate_ids: bool = False
    require_all_mcq: bool = True
    max_name_length: int = 100
    max_text_answer_length: int = 2000


@dataclass
class SystemConfig:
    """Main configuration container aggregating all sub-configs."""
    image: ImageConfig = None
    omr: OMRConfig = None
    ocr: OCRConfig = None
    crop: CropConfig = None
    directories: DirectoryConfig = None
    ui: UIConfig = None
    validation: ValidationConfig = None

    def __post_init__(self):
        """Initialize all sub-configs with defaults if not provided."""
        self.image = self.image or ImageConfig()
        self.omr = self.omr or OMRConfig()
        self.ocr = self.ocr or OCRConfig()
        self.crop = self.crop or CropConfig()
        self.directories = self.directories or DirectoryConfig()
        self.ui = self.ui or UIConfig()
        self.validation = self.validation or ValidationConfig()

    @classmethod
    def from_file(cls, config_path: Path) -> "SystemConfig":
        """
        Load configuration from a JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            SystemConfig instance with loaded values
        """
        # Placeholder for future implementation
        # Will support JSON/YAML loading
        return cls()

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "image": self.image.__dict__,
            "omr": self.omr.__dict__,
            "ocr": self.ocr.__dict__,
            "crop": {
                "name_box": self.crop.name_box,
                "id_box": self.crop.id_box,
                "question_regions": self.crop.question_regions,
            },
            "directories": {k: str(v) for k, v in self.directories.__dict__.items()},
            "ui": self.ui.__dict__,
            "validation": self.validation.__dict__,
        }


# Global configuration instance
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """
    Get the global configuration instance.
    
    Returns:
        The global SystemConfig instance
    """
    global _config
    if _config is None:
        _config = SystemConfig()
    return _config


def set_config(config: SystemConfig) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: SystemConfig instance to use globally
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _config
    _config = SystemConfig()