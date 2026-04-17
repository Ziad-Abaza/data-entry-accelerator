"""
Tests for the services module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from app.services import (
    PipelineService, ExportService, ValidationService, QueueService
)
from app.models import (
    ExamRecord, ProcessingResult, RecordStatus, OMRResult
)


class TestPipelineService:
    """Tests for PipelineService class."""
    
    def test_initialization(self):
        """Test pipeline service initialization."""
        service = PipelineService()
        assert service.cv_processor is not None
        assert service._stats.total_processed == 0
    
    def test_get_stats(self):
        """Test getting pipeline statistics."""
        service = PipelineService()
        stats = service.get_stats()
        
        assert stats.total_processed == 0
        assert stats.successful == 0
    
    def test_clear_session(self):
        """Test clearing session."""
        service = PipelineService()
        service._processed_records.append(ExamRecord())
        service._id_cache["123"] = True
        
        service.clear_session()
        
        assert len(service._processed_records) == 0
        assert len(service._id_cache) == 0


class TestQueueService:
    """Tests for QueueService class."""
    
    def test_initialization(self):
        """Test queue service initialization."""
        service = QueueService()
        assert service.pending_count == 0
        assert service.processed_count == 0
    
    def test_add_single(self):
        """Test adding single item to queue."""
        service = QueueService()
        path = Path("/test/image.jpg")
        
        service.add(path)
        
        assert service.pending_count == 1
    
    def test_add_batch(self):
        """Test adding multiple items."""
        service = QueueService()
        paths = [Path(f"/test/image{i}.jpg") for i in range(5)]
        
        service.add_batch(paths)
        
        assert service.pending_count == 5
    
    def test_get_next(self):
        """Test getting next item from queue."""
        service = QueueService()
        path = Path("/test/image.jpg")
        service.add(path)
        
        next_item = service.get_next()
        
        assert next_item == path
        assert service.pending_count == 0
    
    def test_mark_processed(self):
        """Test marking item as processed."""
        service = QueueService()
        path = Path("/test/image.jpg")
        service.add(path)
        service.get_next()
        
        service.mark_processed(path)
        
        assert service.processed_count == 1
        assert path in service.get_processed()
    
    def test_mark_failed(self):
        """Test marking item as failed."""
        service = QueueService()
        path = Path("/test/image.jpg")
        
        service.mark_failed(path)
        
        assert service.failed_count == 1
        assert path in service.get_failed()
    
    def test_duplicate_prevention(self):
        """Test that duplicates are prevented."""
        service = QueueService()
        path = Path("/test/image.jpg")
        
        service.add(path)
        service.add(path)  # Duplicate
        
        assert service.pending_count == 1
    
    def test_clear(self):
        """Test clearing the queue."""
        service = QueueService()
        service.add(Path("/test1.jpg"))
        service.add(Path("/test2.jpg"))
        service.mark_failed(Path("/test3.jpg"))
        
        service.clear()
        
        assert service.pending_count == 0
        assert service.processed_count == 0
        assert service.failed_count == 0


class TestExportService:
    """Tests for ExportService class."""
    
    def test_initialization(self):
        """Test export service initialization."""
        service = ExportService()
        assert service.config is not None
    
    def test_to_dataframe(self):
        """Test converting records to DataFrame."""
        service = ExportService()
        records = [
            ExamRecord(student_name="John", academic_id="001"),
            ExamRecord(student_name="Jane", academic_id="002"),
        ]
        
        df = service.to_dataframe(records)
        
        assert len(df) == 2
        assert "Student Name" in df.columns
        assert "Academic ID" in df.columns
    
    def test_to_excel_creates_file(self):
        """Test Excel export creates file."""
        service = ExportService()
        records = [ExamRecord(student_name="John", academic_id="001")]
        
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            success = service.to_excel(records, tmp_path)
            assert success is True
            assert tmp_path.exists()
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    def test_to_csv_creates_file(self):
        """Test CSV export creates file."""
        service = ExportService()
        records = [ExamRecord(student_name="John", academic_id="001")]
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            success = service.to_csv(records, tmp_path)
            assert success is True
            assert tmp_path.exists()
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    def test_to_json_creates_file(self):
        """Test JSON export creates file."""
        service = ExportService()
        records = [ExamRecord(student_name="John", academic_id="001")]
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            success = service.to_json(records, tmp_path)
            assert success is True
            assert tmp_path.exists()
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)


class TestValidationService:
    """Tests for ValidationService class."""
    
    def test_initialization(self):
        """Test validation service initialization."""
        service = ValidationService()
        assert service.config is not None
    
    def test_validate_empty_id(self):
        """Test validation catches empty ID."""
        service = ValidationService()
        record = ExamRecord(academic_id="")
        
        errors = service.validate(record)
        
        assert any("Academic ID is required" in e for e in errors)
    
    def test_validate_unanswered_mcq(self):
        """Test validation catches unanswered MCQs."""
        service = ValidationService()
        record = ExamRecord(academic_id="12345")
        
        errors = service.validate(record)
        
        assert any("Unanswered MCQs" in e for e in errors)
    
    def test_validate_name_too_long(self):
        """Test validation catches name exceeding max length."""
        service = ValidationService()
        record = ExamRecord(
            academic_id="12345",
            student_name="A" * 150
        )
        
        errors = service.validate(record)
        
        assert any("exceeds" in e and "characters" in e for e in errors)
    
    def test_check_duplicate(self):
        """Test duplicate ID detection."""
        service = ValidationService()
        
        assert service.check_duplicate("123", ["123", "456"]) is True
        assert service.check_duplicate("789", ["123", "456"]) is False
    
    def test_allow_duplicates_config(self):
        """Test duplicate checking respects config."""
        service = ValidationService()
        service.config.allow_duplicate_ids = True
        
        assert service.check_duplicate("123", ["123", "456"]) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])