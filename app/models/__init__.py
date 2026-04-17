"""
Data models for the Vision-Accelerated Exam Data Entry System.

This module contains all structured data representations including
dataclasses for exam records, processing results, and validation errors.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime


class RecordStatus(Enum):
    """Status of an exam record in the processing pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class ConfidenceLevel(Enum):
    """Confidence level for extracted data."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class OMRResult:
    """Result from OMR processing for a single question."""
    question_number: int
    selected_option: Optional[str]  # 'A', 'B', 'C', 'D' or None if unanswered
    confidence: float  # 0.0 to 1.0
    densities: Dict[str, float] = field(default_factory=dict)  # {'A': 0.2, 'B': 0.8, ...}
    is_ambiguous: bool = False
    is_unanswered: bool = False

    def __post_init__(self):
        if self.selected_option is None:
            self.is_unanswered = True


@dataclass
class OCRResult:
    """Result from OCR processing for text fields."""
    text: str
    confidence: float  # 0.0 to 1.0
    bounding_box: Optional[tuple] = None  # (x, y, w, h)
    is_empty: bool = False

    def __post_init__(self):
        if not self.text or not self.text.strip():
            self.is_empty = True
            self.text = ""


@dataclass
class CropRegion:
    """Represents a cropped image region for UI display."""
    name: str  # e.g., 'student_name', 'q2_answer'
    image: Optional[Any] = None  # numpy array
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    @property
    def dimensions(self) -> tuple:
        """Return (width, height) tuple."""
        return (self.width, self.height)

    @property
    def is_valid(self) -> bool:
        """Check if crop region has valid dimensions."""
        return self.width > 0 and self.height > 0


@dataclass
class ExamRecord:
    """
    Main data model representing a single exam paper's extracted data.
    
    This is the core data structure that flows through the entire pipeline
    from image processing to UI display to Excel export.
    """
    # Identity fields
    student_name: str = ""
    academic_id: str = ""
    
    # MCQ answers (list of 30 OMR results)
    mcq_answers: List[OMRResult] = field(default_factory=list)
    
    # Open-ended question answers
    q2_text: str = ""
    q3_text: str = ""
    q4_text: str = ""
    
    # Metadata
    source_image_path: str = ""
    processed_at: Optional[datetime] = None
    status: RecordStatus = RecordStatus.PENDING
    
    # Confidence tracking
    overall_confidence: float = 0.0
    name_confidence: float = 0.0
    id_confidence: float = 0.0
    
    # Cropped regions for UI
    crops: Dict[str, CropRegion] = field(default_factory=dict)
    
    # Validation
    validation_errors: List[str] = field(default_factory=list)
    is_valid: bool = True

    def __post_init__(self):
        """Initialize MCQ answers list if empty."""
        if not self.mcq_answers:
            self.mcq_answers = [
                OMRResult(question_number=i, selected_option=None, confidence=0.0)
                for i in range(1, 31)
            ]
        if self.processed_at is None:
            self.processed_at = datetime.now()

    def get_mcq_answer(self, question_number: int) -> Optional[str]:
        """
        Get the answer for a specific MCQ question.
        
        Args:
            question_number: Question number (1-30)
            
        Returns:
            Selected option ('A', 'B', 'C', 'D') or None
        """
        if 1 <= question_number <= len(self.mcq_answers):
            return self.mcq_answers[question_number - 1].selected_option
        return None

    def set_mcq_answer(self, question_number: int, option: str) -> None:
        """
        Set the answer for a specific MCQ question.
        
        Args:
            question_number: Question number (1-30)
            option: Selected option ('A', 'B', 'C', 'D')
        """
        if 1 <= question_number <= len(self.mcq_answers):
            self.mcq_answers[question_number - 1].selected_option = option.upper()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert record to dictionary for export.
        
        Returns:
            Dictionary suitable for DataFrame export
        """
        mcq_dict = {f"Q{i+1}": ans.selected_answer or "" for i, ans in enumerate(self.mcq_answers)}
        
        return {
            "Student Name": self.student_name,
            "Academic ID": self.academic_id,
            **mcq_dict,
            "Q2_Text": self.q2_text,
            "Q3_Text": self.q3_text,
            "Q4_Text": self.q4_text,
            "Status": self.status.value,
            "Confidence": f"{self.overall_confidence:.2%}",
            "Processed At": self.processed_at.isoformat() if self.processed_at else "",
        }

    def validate(self) -> List[str]:
        """
        Validate the record according to system rules.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for empty academic ID
        if not self.academic_id.strip():
            errors.append("Academic ID is required")
        
        # Check for unanswered MCQs if required
        if self.validation_rules.get("require_all_mcq", True):
            unanswered = [i+1 for i, ans in enumerate(self.mcq_answers) if ans.selected_option is None]
            if unanswered:
                errors.append(f"Unanswered MCQs: {unanswered}")
        
        # Check name length
        if len(self.student_name) > self.validation_rules.get("max_name_length", 100):
            errors.append(f"Student name exceeds maximum length")
        
        self.validation_errors = errors
        self.is_valid = len(errors) == 0
        
        return errors

    # Validation rules (can be injected from config)
    validation_rules: Dict[str, Any] = field(default_factory=lambda: {
        "require_all_mcq": True,
        "allow_duplicate_id": False,
        "max_name_length": 100,
    })


@dataclass
class ProcessingResult:
    """Result from processing a single image through the pipeline."""
    success: bool
    record: Optional[ExamRecord] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0  # seconds
    stages_completed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if all pipeline stages were completed."""
        expected_stages = ["load", "preprocess", "omr", "ocr", "crops"]
        return all(stage in self.stages_completed for stage in expected_stages)


@dataclass
class PipelineStats:
    """Statistics about pipeline processing."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    review_required: int = 0
    average_processing_time: float = 0.0
    high_confidence_count: int = 0
    low_confidence_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful / self.total_processed) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "review_required": self.review_required,
            "success_rate": f"{self.success_rate:.1f}%",
            "average_time": f"{self.average_processing_time:.2f}s",
        }