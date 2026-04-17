"""
Orchestrator Service - Central System Controller

Coordinates the entire pipeline and serves as the single source of truth for the UI.
Acts as the bridge between CV modules and the PySide6 dashboard.

Design Principles:
- UI = dumb renderer
- Orchestrator = system brain
- CV modules = stateless tools
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import threading

from app.core.preprocessing import ImagePreprocessor
from app.core.omr import OMREngine
from app.core.ocr import OCREngine
from app.core.cropping import CropEngine
from app.models import ExamRecord, RecordStatus

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Types of fields in the exam."""
    STUDENT_NAME = "student_name"
    ACADEMIC_ID = "academic_id"
    MCQ = "mcq"
    TEXT_QUESTION = "text_question"


@dataclass
class FieldInfo:
    """Information about a single field."""
    name: str
    label: str
    field_type: FieldType
    crop_key: str
    question_number: Optional[int] = None


@dataclass
class ValidationResult:
    """Result of validation checks."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class UIPayload:
    """
    Structured payload for UI display.
    
    Contains all data needed by the UI to render the current state.
    """
    # Crop images for display
    student_name_img: Optional[np.ndarray] = None
    academic_id_img: Optional[np.ndarray] = None
    q_crops: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # OMR results
    omr_results: Dict[str, Any] = field(default_factory=dict)
    
    # OCR results
    ocr_result: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source_path: str = ""
    processing_time_ms: float = 0.0


class Orchestrator:
    """
    Central controller for the entire exam data entry system.
    
    Responsibilities:
    - Coordinate CV pipeline execution
    - Manage UI state and field navigation
    - Validate data and enforce business rules
    - Produce final output rows for export
    
    Thread-safe for concurrent UI access.
    """
    
    # Field order for turbo input workflow
    FIELD_ORDER: List[FieldInfo] = [
        FieldInfo("student_name", "Student Name", FieldType.STUDENT_NAME, "student_name"),
        FieldInfo("academic_id", "Academic ID", FieldType.ACADEMIC_ID, "academic_id"),
        FieldInfo("q2", "Question 2", FieldType.MCQ, "q2", 2),
        FieldInfo("q3", "Question 3", FieldType.MCQ, "q3", 3),
        FieldInfo("q4", "Question 4", FieldType.MCQ, "q4", 4),
        FieldInfo("q5", "Question 5", FieldType.MCQ, "q5", 5),
        FieldInfo("q6", "Question 6", FieldType.MCQ, "q6", 6),
        FieldInfo("q7", "Question 7", FieldType.MCQ, "q7", 7),
        FieldInfo("q8", "Question 8", FieldType.MCQ, "q8", 8),
        FieldInfo("q9", "Question 9", FieldType.MCQ, "q9", 9),
        FieldInfo("q10", "Question 10", FieldType.MCQ, "q10", 10),
        FieldInfo("q11", "Question 11", FieldType.MCQ, "q11", 11),
        FieldInfo("q12", "Question 12", FieldType.MCQ, "q12", 12),
        FieldInfo("q13", "Question 13", FieldType.MCQ, "q13", 13),
        FieldInfo("q14", "Question 14", FieldType.MCQ, "q14", 14),
        FieldInfo("q15", "Question 15", FieldType.MCQ, "q15", 15),
        FieldInfo("q16", "Question 16", FieldType.MCQ, "q16", 16),
        FieldInfo("q17", "Question 17", FieldType.MCQ, "q17", 17),
        FieldInfo("q18", "Question 18", FieldType.MCQ, "q18", 18),
        FieldInfo("q19", "Question 19", FieldType.MCQ, "q19", 19),
        FieldInfo("q20", "Question 20", FieldType.MCQ, "q20", 20),
        FieldInfo("q21", "Question 21", FieldType.MCQ, "q21", 21),
        FieldInfo("q22", "Question 22", FieldType.MCQ, "q22", 22),
        FieldInfo("q23", "Question 23", FieldType.MCQ, "q23", 23),
        FieldInfo("q24", "Question 24", FieldType.MCQ, "q24", 24),
        FieldInfo("q25", "Question 25", FieldType.MCQ, "q25", 25),
        FieldInfo("q26", "Question 26", FieldType.MCQ, "q26", 26),
        FieldInfo("q27", "Question 27", FieldType.MCQ, "q27", 27),
        FieldInfo("q28", "Question 28", FieldType.MCQ, "q28", 28),
        FieldInfo("q29", "Question 29", FieldType.MCQ, "q29", 29),
        FieldInfo("q30", "Question 30", FieldType.MCQ, "q30", 30),
    ]
    
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        omr_engine: OMREngine,
        ocr_engine: OCREngine,
        crop_engine: CropEngine
    ):
        """
        Initialize the orchestrator with CV module dependencies.
        
        Args:
            preprocessor: Image preprocessing engine
            omr_engine: OMR extraction engine
            ocr_engine: OCR extraction engine
            crop_engine: Crop extraction engine
        """
        # CV module dependencies
        self._preprocessor = preprocessor
        self._omr_engine = omr_engine
        self._ocr_engine = ocr_engine
        self._crop_engine = crop_engine
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Session state (reset on new session)
        self._session_ids: set = set()
        self._processed_count: int = 0
        
        # Current record state
        self._current_payload: Optional[UIPayload] = None
        self._current_record: Optional[ExamRecord] = None
        self._current_field_index: int = 0
        self._mcq_answers: Dict[int, str] = {}  # question_number -> answer
        self._student_name: str = ""
        
        # Validation state
        self._validation_errors: List[str] = []
        self._validation_warnings: List[str] = []
        
        logger.info("Orchestrator initialized")
    
    # =========================================================================
    # Pipeline Execution
    # =========================================================================
    
    def load_and_process(self, image_source: Any) -> Tuple[bool, Optional[UIPayload], Optional[str]]:
        """
        Load and process an image through the full CV pipeline.
        
        Args:
            image_source: Path to image file or numpy array
            
        Returns:
            Tuple of (success, payload, error_message)
        """
        with self._lock:
            start_time = time.perf_counter()
            
            try:
                # Import here to avoid circular imports
                import cv2
                
                # Step 1: Load image
                if isinstance(image_source, (str, Path)):
                    image = cv2.imread(str(image_source))
                    if image is None:
                        return False, None, f"Failed to load image: {image_source}"
                    source_path = str(image_source)
                elif isinstance(image_source, np.ndarray):
                    image = image_source
                    source_path = "memory"
                else:
                    return False, None, f"Invalid image source type: {type(image_source)}"
                
                # Step 2: Preprocessing
                processed = self._preprocessor.process(image)
                
                # Step 3: Crop extraction
                crops = self._crop_engine.extract_all(processed)
                crops_dict = crops.to_dict() if hasattr(crops, 'to_dict') else {}
                
                # Step 4: OMR extraction
                omr_result = self._omr_engine.extract(processed)
                
                # Step 5: OCR extraction
                ocr_result = self._ocr_engine.extract_academic_id(processed)
                
                # Build UI payload
                processing_time = (time.perf_counter() - start_time) * 1000
                
                payload = UIPayload(
                    student_name_img=crops_dict.get("student_name"),
                    academic_id_img=crops_dict.get("academic_id"),
                    q_crops={k: v for k, v in crops_dict.items() if k.startswith("q")},
                    omr_results={
                        "answers": omr_result.question_results if hasattr(omr_result, 'question_results') else [],
                        "confidence": omr_result.confidence if hasattr(omr_result, 'confidence') else 0.0,
                        "flags": omr_result.flags if hasattr(omr_result, 'flags') else []
                    },
                    ocr_result={
                        "text": ocr_result.text if hasattr(ocr_result, 'text') else "",
                        "confidence": ocr_result.confidence if hasattr(ocr_result, 'confidence') else 0.0,
                        "valid": ocr_result.valid if hasattr(ocr_result, 'valid') else False
                    },
                    source_path=source_path,
                    processing_time_ms=processing_time
                )
                
                # Create ExamRecord
                self._current_record = ExamRecord(
                    student_name="",
                    academic_id=ocr_result.text if hasattr(ocr_result, 'text') else "",
                    mcq_answers=omr_result.question_results if hasattr(omr_result, 'question_results') else {},
                    source_image_path=source_path,
                    id_confidence=ocr_result.confidence if hasattr(ocr_result, 'confidence') else 0.0,
                    crops=crops_dict
                )
                
                # Reset field state
                self._current_field_index = 0
                self._mcq_answers = {}
                self._student_name = ""
                self._current_payload = payload
                
                # Auto-populate MCQ answers from OMR
                if hasattr(omr_result, 'question_results') and omr_result.question_results:
                    for q_num, answer in omr_result.question_results.items():
                        if isinstance(q_num, int) and answer:
                            self._mcq_answers[q_num] = answer
                
                logger.info(f"Processed image in {processing_time:.1f}ms")
                return True, payload, None
                
            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)
                return False, None, str(e)
    
    # =========================================================================
    # UI Binding Layer
    # =========================================================================
    
    def get_current_field(self) -> Optional[FieldInfo]:
        """
        Get information about the current field.
        
        Returns:
            FieldInfo for current field, or None if no record loaded
        """
        with self._lock:
            if not self._current_payload or self._current_field_index >= len(self.FIELD_ORDER):
                return None
            return self.FIELD_ORDER[self._current_field_index]
    
    def get_current_crop_image(self) -> Optional[np.ndarray]:
        """
        Get the crop image for the current field.
        
        Returns:
            numpy array for current field, or None
        """
        with self._lock:
            if not self._current_payload:
                return None
            
            field = self.get_current_field()
            if not field:
                return None
            
            # Return appropriate crop based on field type
            if field.field_type == FieldType.STUDENT_NAME:
                return self._current_payload.student_name_img
            elif field.field_type == FieldType.ACADEMIC_ID:
                return self._current_payload.academic_id_img
            elif field.field_type == FieldType.MCQ:
                return self._current_payload.q_crops.get(field.crop_key)
            
            return None
    
    def get_field_count(self) -> int:
        """Get total number of fields."""
        return len(self.FIELD_ORDER)
    
    def get_current_field_index(self) -> int:
        """Get current field index (0-based)."""
        with self._lock:
            return self._current_field_index
    
    def get_current_answer(self) -> str:
        """
        Get the current answer for the displayed field.
        
        Returns:
            Current answer (A/B/C/D) or "-" if not answered
        """
        with self._lock:
            field = self.get_current_field()
            if field and field.field_type == FieldType.MCQ and field.question_number:
                return self._mcq_answers.get(field.question_number, "-")
            return "-"
    
    def submit_answer(self, value: str) -> bool:
        """
        Receive answer input from UI.
        
        Args:
            value: Answer value (A/B/C/D for MCQ, text for other fields)
            
        Returns:
            True if answer accepted
        """
        with self._lock:
            field = self.get_current_field()
            if not field:
                return False
            
            if field.field_type == FieldType.MCQ and field.question_number:
                # Validate MCQ answer
                value_upper = value.upper()
                if value_upper in ["A", "B", "C", "D"]:
                    self._mcq_answers[field.question_number] = value_upper
                    logger.debug(f"Set Q{field.question_number} = {value_upper}")
                    return True
            elif field.field_type == FieldType.STUDENT_NAME:
                self._student_name = value
                return True
            elif field.field_type == FieldType.ACADEMIC_ID:
                if self._current_record:
                    self._current_record.academic_id = value
                return True
            
            return False
    
    def navigate_next(self) -> bool:
        """
        Move to the next field.
        
        Returns:
            True if navigation successful, False if at end
        """
        with self._lock:
            if self._current_field_index < len(self.FIELD_ORDER) - 1:
                self._current_field_index += 1
                return True
            return False
    
    def navigate_prev(self) -> bool:
        """
        Move to the previous field.
        
        Returns:
            True if navigation successful, False if at start
        """
        with self._lock:
            if self._current_field_index > 0:
                self._current_field_index -= 1
                return True
            return False
    
    def navigate_to(self, field_index: int) -> bool:
        """
        Navigate to a specific field index.
        
        Args:
            field_index: Target field index (0-based)
            
        Returns:
            True if navigation successful
        """
        with self._lock:
            if 0 <= field_index < len(self.FIELD_ORDER):
                self._current_field_index = field_index
                return True
            return False
    
    # =========================================================================
    # Validation Engine
    # =========================================================================
    
    def validate_current_record(self) -> ValidationResult:
        """
        Validate the current record against business rules.
        
        Returns:
            ValidationResult with is_valid, errors, and warnings
        """
        with self._lock:
            errors = []
            warnings = []
            
            if not self._current_record:
                return ValidationResult(False, ["No record loaded"])
            
            # Rule 1: Check for duplicate Academic ID
            academic_id = self._current_record.academic_id
            if academic_id:
                if academic_id in self._session_ids:
                    errors.append(f"Duplicate Academic ID: {academic_id}")
                elif not self._is_valid_academic_id(academic_id):
                    warnings.append(f"Academic ID format may be invalid: {academic_id}")
            
            # Rule 2: Check for incomplete MCQ set
            answered_count = len(self._mcq_answers)
            if answered_count < 30:
                missing = 30 - answered_count
                warnings.append(f"Incomplete MCQ: {missing} questions unanswered")
            
            # Rule 3: Flag low-confidence OCR
            ocr_conf = self._current_record.id_confidence
            if ocr_conf < 0.5:
                warnings.append(f"Low OCR confidence: {ocr_conf:.0%}")
            elif ocr_conf < 0.7:
                warnings.append(f"Medium OCR confidence: {ocr_conf:.0%}")
            
            # Rule 4: Detect unanswered questions (0% density)
            if self._current_payload and self._current_payload.omr_results:
                flags = self._current_payload.omr_results.get("flags", [])
                if flags:
                    warnings.append(f"OMR flags: {', '.join(flags[:3])}")
            
            is_valid = len(errors) == 0
            
            # Store for UI access
            self._validation_errors = errors
            self._validation_warnings = warnings
            
            return ValidationResult(is_valid, errors, warnings)
    
    def _is_valid_academic_id(self, academic_id: str) -> bool:
        """
        Check if academic ID has valid format.
        
        Args:
            academic_id: The academic ID string
            
        Returns:
            True if format appears valid
        """
        # Basic validation: should be numeric, 6-12 digits
        if not academic_id:
            return False
        return academic_id.isdigit() and 6 <= len(academic_id) <= 12
    
    # =========================================================================
    # Finalization & Output
    # =========================================================================
    
    def finalize_row(self) -> Optional[Dict[str, Any]]:
        """
        Produce the final structured row for DataFrame/export.
        
        Returns:
            Dictionary with final row data, or None if no record
        """
        with self._lock:
            if not self._current_record:
                return None
            
            # Validate before finalization
            validation = self.validate_current_record()
            
            # Build output row
            row = {
                "ID": self._processed_count + 1,
                "Student Name": self._student_name or "",
                "Academic ID": self._current_record.academic_id or "",
            }
            
            # Add MCQ answers (Q1_1 through Q1_30)
            for q_num in range(1, 31):
                key = f"Q1_{q_num}"
                row[key] = self._mcq_answers.get(q_num, "")
            
            # Add text questions (Q2, Q3, etc. if applicable)
            # Currently not implemented in this version
            
            # Add metadata
            row["Status"] = "COMPLETED" if validation.is_valid else "REVIEW_REQUIRED"
            row["Confidence"] = self._current_record.overall_confidence
            row["Source"] = self._current_payload.source_path if self._current_payload else ""
            
            # Add to session tracking
            if self._current_record.academic_id:
                self._session_ids.add(self._current_record.academic_id)
            
            self._processed_count += 1
            
            logger.info(f"Finalized row: ID={row['Academic ID']}, Status={row['Status']}")
            
            return row
    
    def get_omr_status_color(self) -> str:
        """
        Get the OMR status color for UI indicator.
        
        Returns:
            "GREEN", "YELLOW", or "RED"
        """
        with self._lock:
            if not self._current_record:
                return "GRAY"
            
            conf = self._current_record.overall_confidence
            if conf >= 0.7:
                return "GREEN"
            elif conf >= 0.4:
                return "YELLOW"
            else:
                return "RED"
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def clear_session(self) -> None:
        """Clear all session data."""
        with self._lock:
            self._session_ids.clear()
            self._processed_count = 0
            self._current_payload = None
            self._current_record = None
            self._current_field_index = 0
            self._mcq_answers = {}
            self._student_name = ""
            self._validation_errors = []
            self._validation_warnings = []
            logger.info("Session cleared")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        with self._lock:
            return {
                "processed": self._processed_count,
                "unique_ids": len(self._session_ids),
                "current_field": self._current_field_index,
                "answered_questions": len(self._mcq_answers)
            }
    
    # =========================================================================
    # Reprocessing Support
    # =========================================================================
    
    def reprocess_current(self) -> Tuple[bool, Optional[str]]:
        """
        Reprocess the current image.
        
        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            if not self._current_payload:
                return False, "No image loaded"
            
            source = self._current_payload.source_path
            return self.load_and_process(source)
    
    # =========================================================================
    # Quick Access Properties
    # =========================================================================
    
    @property
    def current_academic_id(self) -> str:
        """Get current academic ID."""
        with self._lock:
            return self._current_record.academic_id if self._current_record else ""
    
    @property
    def current_ocr_confidence(self) -> float:
        """Get current OCR confidence."""
        with self._lock:
            return self._current_record.id_confidence if self._current_record else 0.0
    
    @property
    def has_record(self) -> bool:
        """Check if a record is currently loaded."""
        with self._lock:
            return self._current_record is not None
    
    @property
    def is_at_last_field(self) -> bool:
        """Check if at the last field."""
        with self._lock:
            return self._current_field_index >= len(self.FIELD_ORDER) - 1
    
    @property
    def validation_errors(self) -> List[str]:
        """Get current validation errors."""
        with self._lock:
            return self._validation_errors.copy()
    
    @property
    def validation_warnings(self) -> List[str]:
        """Get current validation warnings."""
        with self._lock:
            return self._validation_warnings.copy()