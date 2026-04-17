"""
Services module for the Vision-Accelerated Exam Data Entry System.

This module contains pipeline orchestration and business logic services
that coordinate the flow of data through the application.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

from app.core import CVProcessor, ImageLoader
from app.models import (
    ExamRecord, ProcessingResult, PipelineStats, RecordStatus, ConfidenceLevel
)
from app.services.orchestrator import Orchestrator, UIPayload, ValidationResult, FieldInfo, FieldType
from app.services.session_manager import SessionManager
from app.services.export_engine import ExportEngine, ExportValidationError
from config import get_config

logger = logging.getLogger(__name__)


class PipelineService:
    """
    Main pipeline orchestration service.
    
    Coordinates the flow of exam sheet images through the computer vision
    pipeline and manages the processing queue.
    """
    
    def __init__(self):
        self.config = get_config()
        self.cv_processor = CVProcessor()
        self._stats = PipelineStats()
        self._processed_records: List[ExamRecord] = []
        self._id_cache: Dict[str, bool] = {}  # For duplicate detection
    
    def process_image(self, image_path: Path) -> ProcessingResult:
        """
        Process a single exam sheet image through the full pipeline.
        
        Args:
            image_path: Path to the exam sheet image
            
        Returns:
            ProcessingResult with extracted data or error information
        """
        logger.info(f"Processing image: {image_path.name}")
        
        result = self.cv_processor.process(image_path)
        
        if result.success and result.record:
            # Post-process the record
            self._post_process_record(result.record)
            
            # Update statistics
            self._update_stats(result)
            
            # Add to processed records
            self._processed_records.append(result.record)
        
        return result
    
    def process_batch(self, image_paths: List[Path]) -> List[ProcessingResult]:
        """
        Process multiple exam sheet images in batch.
        
        Args:
            image_paths: List of paths to exam sheet images
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        results = []
        for i, path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {path.name}")
            result = self.process_image(path)
            results.append(result)
        
        logger.info(f"Batch processing complete: {len(results)} images processed")
        return results
    
    def process_directory(self, directory: Path) -> List[ProcessingResult]:
        """
        Process all valid images in a directory.
        
        Args:
            directory: Path to directory containing exam sheet images
            
        Returns:
            List of ProcessingResult objects
        """
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Invalid directory: {directory}")
            return []
        
        # Get all supported image files
        supported_formats = self.config.image.supported_formats
        image_paths = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in supported_formats
        ]
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        return self.process_batch(image_paths)
    
    def _post_process_record(self, record: ExamRecord) -> None:
        """
        Perform post-processing on a completed record.
        
        Includes:
        - Duplicate ID detection
        - Confidence level assignment
        - Status determination
        
        Args:
            record: The processed ExamRecord
        """
        # Check for duplicate academic ID
        if record.academic_id in self._id_cache:
            record.validation_errors.append(
                f"Duplicate Academic ID: {record.academic_id}"
            )
            record.is_valid = False
            record.status = RecordStatus.REVIEW_REQUIRED
        else:
            self._id_cache[record.academic_id] = True
        
        # Determine status based on confidence
        if record.overall_confidence >= 0.8:
            record.status = RecordStatus.COMPLETED
        elif record.overall_confidence >= 0.5:
            record.status = RecordStatus.REVIEW_REQUIRED
        else:
            record.status = RecordStatus.REVIEW_REQUIRED
        
        # Check for validation errors
        if record.validation_errors:
            record.status = RecordStatus.REVIEW_REQUIRED
    
    def _update_stats(self, result: ProcessingResult) -> None:
        """Update pipeline statistics with a processing result."""
        self._stats.total_processed += 1
        
        if result.success:
            self._stats.successful += 1
            
            if result.record:
                if result.record.overall_confidence >= 0.8:
                    self._stats.high_confidence_count += 1
                elif result.record.overall_confidence < 0.5:
                    self._stats.low_confidence_count += 1
                
                if result.record.status == RecordStatus.REVIEW_REQUIRED:
                    self._stats.review_required += 1
        else:
            self._stats.failed += 1
        
        # Update average processing time
        if self._stats.total_processed > 1:
            self._stats.average_processing_time = (
                (self._stats.average_processing_time * (self._stats.total_processed - 1) +
                 result.processing_time) / self._stats.total_processed
            )
        else:
            self._stats.average_processing_time = result.processing_time
    
    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics."""
        return self._stats
    
    def get_records(self) -> List[ExamRecord]:
        """Get all processed records."""
        return self._processed_records
    
    def get_record_by_id(self, academic_id: str) -> Optional[ExamRecord]:
        """Get a processed record by academic ID."""
        for record in self._processed_records:
            if record.academic_id == academic_id:
                return record
        return None
    
    def clear_session(self) -> None:
        """Clear the current processing session."""
        self._processed_records.clear()
        self._id_cache.clear()
        self._stats = PipelineStats()
        logger.info("Session cleared")


class ExportService:
    """
    Service for exporting processed exam data to various formats.
    
    Supports export to Excel, CSV, and JSON formats.
    """
    
    def __init__(self):
        self.config = get_config()
    
    def to_dataframe(self, records: List[ExamRecord]) -> "pd.DataFrame":
        """
        Convert records to a Pandas DataFrame.
        
        Args:
            records: List of ExamRecord objects
            
        Returns:
            Pandas DataFrame
        """
        import pandas as pd
        
        data = [record.to_dict() for record in records]
        return pd.DataFrame(data)
    
    def to_excel(self, records: List[ExamRecord], output_path: Path) -> bool:
        """
        Export records to an Excel file.
        
        Args:
            records: List of ExamRecord objects
            output_path: Path for the output Excel file
            
        Returns:
            True if export successful
        """
        import pandas as pd
        
        try:
            df = self.to_dataframe(records)
            df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"Exported {len(records)} records to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
            return False
    
    def to_csv(self, records: List[ExamRecord], output_path: Path) -> bool:
        """
        Export records to a CSV file.
        
        Args:
            records: List of ExamRecord objects
            output_path: Path for the output CSV file
            
        Returns:
            True if export successful
        """
        import pandas as pd
        
        try:
            df = self.to_dataframe(records)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(records)} records to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def to_json(self, records: List[ExamRecord], output_path: Path) -> bool:
        """
        Export records to a JSON file.
        
        Args:
            records: List of ExamRecord objects
            output_path: Path for the output JSON file
            
        Returns:
            True if export successful
        """
        import json
        
        try:
            data = [record.to_dict() for record in records]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(records)} records to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False


class ValidationService:
    """
    Service for validating exam records.
    
    Provides validation logic that can be applied both during
    processing and for manual review.
    """
    
    def __init__(self):
        self.config = get_config().validation
    
    def validate(self, record: ExamRecord) -> List[str]:
        """
        Validate a single exam record.
        
        Args:
            record: ExamRecord to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check academic ID is present
        if not record.academic_id or not record.academic_id.strip():
            errors.append("Academic ID is required")
        
        # Check MCQ completeness if required
        if self.config.require_all_mcq:
            unanswered = [
                i + 1 for i, ans in enumerate(record.mcq_answers)
                if ans.selected_option is None
            ]
            if unanswered:
                errors.append(f"Unanswered MCQs: {unanswered}")
        
        # Check name length
        if len(record.student_name) > self.config.max_name_length:
            errors.append(f"Student name exceeds {self.config.max_name_length} characters")
        
        # Check text answer lengths
        for q_name in ["q2_text", "q3_text", "q4_text"]:
            text = getattr(record, q_name, "")
            if len(text) > self.config.max_text_answer_length:
                errors.append(f"{q_name} exceeds {self.config.max_text_answer_length} characters")
        
        return errors
    
    def check_duplicate(self, academic_id: str, existing_ids: List[str]) -> bool:
        """
        Check if an academic ID is a duplicate.
        
        Args:
            academic_id: The ID to check
            existing_ids: List of existing IDs to compare against
            
        Returns:
            True if duplicate detected
        """
        if self.config.allow_duplicate_ids:
            return False
        
        return academic_id in existing_ids


class QueueService:
    """
    Service for managing the processing queue.
    
    Handles queue operations including adding, removing, and
    reordering items in the processing queue.
    """
    
    def __init__(self):
        self._queue: List[Path] = []
        self._processed: List[Path] = []
        self._failed: List[Path] = []
    
    def add(self, path: Path) -> None:
        """Add a path to the processing queue."""
        if path not in self._queue and path not in self._processed:
            self._queue.append(path)
            logger.debug(f"Added to queue: {path.name}")
    
    def add_batch(self, paths: List[Path]) -> None:
        """Add multiple paths to the processing queue."""
        for path in paths:
            self.add(path)
        logger.info(f"Added {len(paths)} items to queue")
    
    def get_next(self) -> Optional[Path]:
        """Get the next item from the queue."""
        if self._queue:
            return self._queue.pop(0)
        return None
    
    def mark_processed(self, path: Path) -> None:
        """Mark a path as successfully processed."""
        if path not in self._processed:
            self._processed.append(path)
    
    def mark_failed(self, path: Path) -> None:
        """Mark a path as failed."""
        if path not in self._failed:
            self._failed.append(path)
    
    def get_pending(self) -> List[Path]:
        """Get all pending items in the queue."""
        return self._queue.copy()
    
    def get_processed(self) -> List[Path]:
        """Get all processed items."""
        return self._processed.copy()
    
    def get_failed(self) -> List[Path]:
        """Get all failed items."""
        return self._failed.copy()
    
    def clear(self) -> None:
        """Clear the queue."""
        self._queue.clear()
        self._processed.clear()
        self._failed.clear()
    
    @property
    def pending_count(self) -> int:
        """Get count of pending items."""
        return len(self._queue)
    
    @property
    def processed_count(self) -> int:
        """Get count of processed items."""
        return len(self._processed)
    
    @property
    def failed_count(self) -> int:
        """Get count of failed items."""
        return len(self._failed)