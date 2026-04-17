"""
Tests for the Vision-Accelerated Exam Data Entry System.

This package contains unit and integration tests for all modules.
"""

import pytest
from pathlib import Path
import numpy as np

from app.models import (
    ExamRecord, OMRResult, OCRResult, CropRegion,
    RecordStatus, ProcessingResult, PipelineStats
)
from app.services import PipelineService, ExportService, ValidationService
from config import SystemConfig, get_config


class TestExamRecord:
    """Tests for the ExamRecord model."""
    
    def test_default_initialization(self):
        """Test default initialization of ExamRecord."""
        record = ExamRecord()
        
        assert record.student_name == ""
        assert record.academic_id == ""
        assert len(record.mcq_answers) == 30
        assert record.status == RecordStatus.PENDING
    
    def test_get_mcq_answer(self):
        """Test getting MCQ answer by question number."""
        record = ExamRecord()
        record.mcq_answers[0].selected_option = "A"
        record.mcq_answers[4].selected_option = "C"
        
        assert record.get_mcq_answer(1) == "A"
        assert record.get_mcq_answer(5) == "C"
        assert record.get_mcq_answer(31) is None
    
    def test_set_mcq_answer(self):
        """Test setting MCQ answer."""
        record = ExamRecord()
        record.set_mcq_answer(1, "B")
        record.set_mcq_answer(15, "D")
        
        assert record.get_mcq_answer(1) == "B"
        assert record.get_mcq_answer(15) == "D"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = ExamRecord(
            student_name="John Doe",
            academic_id="12345"
        )
        record.mcq_answers[0].selected_option = "A"
        
        data = record.to_dict()
        
        assert data["Student Name"] == "John Doe"
        assert data["Academic ID"] == "12345"
        assert data["Q1"] == "A"


class TestOMRResult:
    """Tests for the OMRResult model."""
    
    def test_default_values(self):
        """Test default values."""
        result = OMRResult(question_number=1, selected_option=None, confidence=0.0)
        
        assert result.question_number == 1
        assert result.selected_option is None
        assert result.is_unanswered is True
    
    def test_selected_option_sets_unanswered(self):
        """Test that setting selected option marks as not unanswered."""
        result = OMRResult(question_number=1, selected_option="A", confidence=0.9)
        
        assert result.is_unanswered is False


class TestOCRResult:
    """Tests for the OCRResult model."""
    
    def test_empty_text_detection(self):
        """Test detection of empty text."""
        result = OCRResult(text="", confidence=0.0)
        assert result.is_empty is True
        
        result = OCRResult(text="   ", confidence=0.0)
        assert result.is_empty is True
        
        result = OCRResult(text="ABC123", confidence=0.8)
        assert result.is_empty is False


class TestCropRegion:
    """Tests for the CropRegion model."""
    
    def test_dimensions_property(self):
        """Test dimensions property."""
        crop = CropRegion(name="test", x=10, y=20, width=100, height=50)
        
        assert crop.dimensions == (100, 50)
    
    def test_is_valid(self):
        """Test is_valid property."""
        crop_valid = CropRegion(name="test", width=100, height=50)
        crop_invalid = CropRegion(name="test", width=0, height=50)
        
        assert crop_valid.is_valid is True
        assert crop_invalid.is_valid is False


class TestPipelineStats:
    """Tests for the PipelineStats model."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = PipelineStats()
        stats.total_processed = 10
        stats.successful = 8
        
        assert stats.success_rate == 80.0
    
    def test_empty_success_rate(self):
        """Test success rate with no processed items."""
        stats = PipelineStats()
        
        assert stats.success_rate == 0.0


class TestValidationService:
    """Tests for the ValidationService."""
    
    def test_validate_empty_id(self):
        """Test validation catches empty academic ID."""
        service = ValidationService()
        record = ExamRecord(academic_id="")
        
        errors = service.validate(record)
        
        assert "Academic ID is required" in errors
    
    def test_validate_unanswered_mcq(self):
        """Test validation catches unanswered MCQs."""
        service = ValidationService()
        record = ExamRecord(academic_id="12345")
        # All MCQs are unanswered by default
        
        errors = service.validate(record)
        
        assert any("Unanswered MCQs" in err for err in errors)
    
    def test_validate_valid_record(self):
        """Test validation passes for valid record."""
        service = ValidationService()
        record = ExamRecord(academic_id="12345", student_name="John Doe")
        
        # Set all MCQ answers
        for i in range(30):
            record.mcq_answers[i].selected_option = "A"
        
        errors = service.validate(record)
        
        assert len(errors) == 0


class TestExportService:
    """Tests for the ExportService."""
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        service = ExportService()
        records = [
            ExamRecord(student_name="John", academic_id="001"),
            ExamRecord(student_name="Jane", academic_id="002"),
        ]
        
        df = service.to_dataframe(records)
        
        assert len(df) == 2
        assert "Student Name" in df.columns
        assert "Academic ID" in df.columns


class TestConfig:
    """Tests for configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SystemConfig()
        
        assert config.image.master_height == 2000
        assert config.omr.questions_count == 30
        assert config.omr.options_count == 4
        assert config.ui.window_title == "Vision-Accelerated Exam Data Entry"
    
    def test_get_config_singleton(self):
        """Test get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])