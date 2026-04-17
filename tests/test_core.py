"""
Tests for the core computer vision module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.core import (
    ImageLoader, ImagePreprocessor, OMRProcessor, 
    OCRProcessor, ImageCropper, CVProcessor
)
from app.models import ProcessingResult, OMRResult, CropRegion


class TestImageLoader:
    """Tests for ImageLoader class."""
    
    def test_validate_valid_image(self):
        """Test validation of valid image."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        assert ImageLoader.validate(image) is True
    
    def test_validate_none_image(self):
        """Test validation of None image."""
        assert ImageLoader.validate(None) is False
    
    def test_validate_empty_image(self):
        """Test validation of empty image."""
        image = np.array([])
        assert ImageLoader.validate(image) is False
    
    def test_validate_grayscale_image(self):
        """Test validation of grayscale image."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        assert ImageLoader.validate(image) is False


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = ImagePreprocessor()
        assert preprocessor.config is not None
    
    def test_preprocess_returns_image(self):
        """Test preprocess returns an image."""
        preprocessor = ImagePreprocessor()
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        
        result = preprocessor.preprocess(image)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_resize_to_master_height(self):
        """Test resize to master height."""
        preprocessor = ImagePreprocessor()
        image = np.ones((500, 400, 3), dtype=np.uint8) * 255
        
        result = preprocessor.preprocess(image)
        
        # Should be resized to master height (2000)
        assert result.shape[0] == 2000


class TestOMRProcessor:
    """Tests for OMRProcessor class."""
    
    def test_initialization(self):
        """Test OMR processor initialization."""
        processor = OMRProcessor()
        assert processor.config is not None
        assert processor.config.questions_count == 30
    
    def test_extract_returns_list(self):
        """Test extract returns list of results."""
        processor = OMRProcessor()
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255
        
        results = processor.extract(image)
        
        assert isinstance(results, list)
        assert len(results) == 30
        assert all(isinstance(r, OMRResult) for r in results)
    
    def test_calculate_density(self):
        """Test pixel density calculation."""
        processor = OMRProcessor()
        
        # White region should have low density
        white_region = np.ones((10, 10), dtype=np.uint8) * 255
        density = processor._calculate_density(white_region)
        assert density < 0.1
        
        # Black region should have high density
        black_region = np.zeros((10, 10), dtype=np.uint8)
        density = processor._calculate_density(black_region)
        assert density > 0.9


class TestOCRProcessor:
    """Tests for OCRProcessor class."""
    
    def test_initialization(self):
        """Test OCR processor initialization."""
        processor = OCRProcessor()
        assert processor.config is not None
    
    def test_extract_name_returns_ocr_result(self):
        """Test extract_name returns OCRResult."""
        processor = OCRProcessor()
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        result = processor.extract_name(image)
        
        assert result is not None
        assert hasattr(result, 'text')
        assert hasattr(result, 'confidence')
    
    def test_extract_id_returns_ocr_result(self):
        """Test extract_id returns OCRResult."""
        processor = OCRProcessor()
        image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        
        result = processor.extract_id(image)
        
        assert result is not None
        assert hasattr(result, 'text')


class TestImageCropper:
    """Tests for ImageCropper class."""
    
    def test_initialization(self):
        """Test cropper initialization."""
        cropper = ImageCropper()
        assert cropper.config is not None
    
    def test_crop_all_returns_dict(self):
        """Test crop_all returns dictionary of crops."""
        cropper = ImageCropper()
        image = np.ones((2000, 1500, 3), dtype=np.uint8) * 255
        
        crops = cropper.crop_all(image)
        
        assert isinstance(crops, dict)
        assert "student_name" in crops
        assert "academic_id" in crops
        assert "q2" in crops
        assert "q3" in crops
        assert "q4" in crops
    
    def test_crop_region_bounds_checking(self):
        """Test crop region handles out of bounds."""
        cropper = ImageCropper()
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Request crop larger than image
        crop = cropper._crop_region(image, "test", (0, 0, 200, 200))
        
        assert crop.width <= 100
        assert crop.height <= 100


class TestCVProcessor:
    """Tests for CVProcessor class."""
    
    def test_initialization(self):
        """Test CV processor initialization."""
        processor = CVProcessor()
        assert processor.loader is not None
        assert processor.preprocessor is not None
        assert processor.omr_processor is not None
        assert processor.ocr_processor is not None
        assert processor.cropper is not None
    
    @patch('app.core.CVProcessor.cv2')
    def test_process_handles_invalid_path(self, mock_cv2):
        """Test process handles invalid image path."""
        processor = CVProcessor()
        
        result = processor.process(Path("/nonexistent/image.jpg"))
        
        assert result.success is False
        assert "load" in result.stages_completed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])