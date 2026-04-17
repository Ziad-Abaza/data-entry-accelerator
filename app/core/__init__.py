"""
Core computer vision module for the Vision-Accelerated Exam Data Entry System.

This module contains all image processing functionality including:
- Image loading and validation
- Preprocessing (rotation, deskewing, resizing) - via app.core.preprocessing
- OMR (Optical Mark Recognition) processing - via app.core.omr
- OCR (Optical Character Recognition) processing - via app.core.ocr
- Image cropping for UI display - via app.core.cropping
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
import app.core.preprocessing
import app.core.omr
import app.core.ocr
import app.core.cropping

from app.models import (
    OMRResult as ModelOMRResult, OCRResult as ModelOCRResult, CropRegion, ExamRecord, ProcessingResult
)
from app.core.preprocessing import ImagePreprocessor as PreprocessorModule, PreprocessingConfig
from app.core.omr import OMREngine, OMRConfig, DEFAULT_TEMPLATE
from app.core.ocr import OCREngine, OCRConfig, ACADEMIC_ID_BOX
from app.core.cropping import CropEngine, CropConfig, DEFAULT_CROP_BOXES
from config import get_config

logger = logging.getLogger(__name__)


# For backwards compatibility, import the production preprocessor
ImagePreprocessor = PreprocessorModule


class ImageLoader:
    """
    Handles loading and initial validation of exam sheet images.
    
    Supports multiple image formats and performs basic validation
    to ensure the image is suitable for processing.
    """
    
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    @staticmethod
    def load(image_path: Path) -> Optional[np.ndarray]:
        """
        Load an image from the specified path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array (BGR format) or None if loading fails
        """
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None
        
        if image_path.suffix.lower() not in ImageLoader.SUPPORTED_FORMATS:
            logger.error(f"Unsupported image format: {image_path.suffix}")
            return None
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to decode image: {image_path}")
                return None
            
            logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def validate(image: np.ndarray) -> bool:
        """
        Validate that an image is suitable for processing.
        
        Args:
            image: Input image array
            
        Returns:
            True if image is valid
        """
        if image is None or image.size == 0:
            return False
        
        if len(image.shape) != 3:
            return False
        
        return True


class ImagePreprocessor:
    """
    Handles image preprocessing operations to prepare exam sheets
    for OMR and OCR processing.
    
    This class wraps the production-grade ImagePreprocessor from
    app.core.preprocessing module which implements:
    - Auto-rotation based on template matching
    - Deskewing (rotation correction)
    - Adaptive scaling to master resolution
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            config: Optional PreprocessingConfig. If None, uses defaults.
        """
        # Get image config from global config
        image_config = get_config().image
        
        # Create preprocessing config from global settings
        if config is None:
            config = PreprocessingConfig(
                master_height=image_config.master_height
            )
        
        # Use the production preprocessing module
        self._preprocessor = app.core.preprocessing.ImagePreprocessor(config)
        self.config = image_config
        self._rotation_cache: Dict[str, int] = {}
    
    def preprocess(self, image: np.ndarray, image_path: Optional[Path] = None) -> np.ndarray:
        """
        Apply all preprocessing steps to an image.
        
        Args:
            image: Input image (BGR format)
            image_path: Optional path for caching rotation state
            
        Returns:
            Preprocessed image
        """
        logger.info("Starting image preprocessing")
        
        # Use the production preprocessing pipeline
        processed = self._preprocessor.process(image)
        
        logger.info(f"Preprocessing complete: {processed.shape[1]}x{processed.shape[0]}")
        return processed
    
    def _auto_rotate(self, image: np.ndarray, image_path: Optional[Path] = None) -> np.ndarray:
        """
        Automatically rotate image based on content analysis.
        
        This is now handled by the preprocessing module.
        
        Args:
            image: Input image
            image_path: Optional path for caching
            
        Returns:
            Rotated image if needed, otherwise original
        """
        logger.debug("Auto-rotate: handled by preprocessing module")
        return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew/rotation in the image.
        
        This is now handled by the preprocessing module.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        logger.debug("Deskew: handled by preprocessing module")
        return image
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to master resolution for consistent coordinate processing.
        
        This is now handled by the preprocessing module.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        logger.debug("Resize: handled by preprocessing module")
        return image
    
    def _get_interpolation(self) -> int:
        """Get OpenCV interpolation constant from config."""
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        return interp_map.get(self.config.interpolation, cv2.INTER_LANCZOS4)


class OMRProcessor:
    """
    Optical Mark Recognition processor for MCQ sections.
    
    This class wraps the production-grade OMREngine from
    app.core.omr module which implements:
    - Pixel density analysis for answer detection
    - Confidence scoring
    - Ambiguity detection
    - Configurable template coordinates
    """
    
    def __init__(self):
        self.config = get_config().omr
        self.crop_config = get_config().crop
        
        # Create the production OMR engine
        omr_config = OMRConfig(
            questions_count=self.config.questions_count,
            min_mark_threshold=self.config.density_threshold,
            ambiguity_threshold=self.config.ambiguity_threshold
        )
        self._engine = OMREngine(omr_config, DEFAULT_TEMPLATE)
    
    def extract(self, image: np.ndarray) -> List[ModelOMRResult]:
        """
        Extract MCQ answers from the processed image.
        
        Args:
            image: Preprocessed exam sheet image
            
        Returns:
            List of OMRResult objects for all questions
        """
        logger.info(f"Starting OMR extraction for {self.config.questions_count} questions")
        
        # Use the production OMR engine
        result = self._engine.extract(image)
        
        # Convert to the model's OMRResult format
        model_results = []
        for qr in result.question_results:
            model_result = ModelOMRResult(
                question_number=qr.question_number,
                selected_option=qr.selected_answer,
                confidence=qr.confidence,
                densities=qr.densities,
                is_ambiguous=(qr.flag == "AMBIGUOUS"),
                is_unanswered=(qr.flag == "EMPTY")
            )
            model_results.append(model_result)
        
        logger.info(f"OMR extraction complete: {len(model_results)} questions processed")
        return model_results
    
    def _process_question(
        self, 
        image: np.ndarray, 
        question_num: int,
        grid_start: Tuple[int, int],
        box_size: Tuple[int, int]
    ) -> ModelOMRResult:
        """
        Process a single MCQ question to determine the selected answer.
        
        This is now handled by the OMR engine.
        
        Args:
            image: Exam sheet image
            question_num: Question number (1-30)
            grid_start: Starting position of MCQ grid
            box_size: Size of each answer option box
            
        Returns:
            OMRResult for the question
        """
        logger.debug("OMR processing: handled by OMR engine")
        return ModelOMRResult(
            question_number=question_num,
            selected_option=None,
            confidence=0.0,
            densities={},
            is_ambiguous=False,
            is_unanswered=True
        )
    
    def _calculate_density(self, region: np.ndarray) -> float:
        """
        Calculate pixel density (ratio of dark pixels) in a region.
        
        This is now handled by the OMR engine.
        
        Args:
            region: Image region to analyze
            
        Returns:
            Density ratio (0.0 to 1.0)
        """
        if region.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # Calculate density (dark pixels / total pixels)
        dark_pixels = np.sum(binary < 128)
        total_pixels = binary.size
        
        return dark_pixels / total_pixels


class OCRProcessor:
    """
    Optical Character Recognition processor for handwritten fields.
    
    This class wraps the production-grade OCREngine from
    app.core.ocr module which implements:
    - Handwritten digit extraction for Academic ID
    - PyTesseract with digit-only whitelist
    - Robust preprocessing for handwriting
    - Validation and confidence estimation
    """
    
    def __init__(self):
        self.config = get_config().ocr
        self.crop_config = get_config().crop
        
        # Create the production OCR engine
        ocr_config = OCRConfig(
            tesseract_path=self.config.tesseract_path,
            psm_mode=self.config.psm_mode,
            oem_mode=self.config.oem_mode,
            whitelist=self.config.whitelist_chars,
            expected_id_length=8,  # Default, can be configured
            min_confidence=self.config.confidence_threshold / 100.0
        )
        self._engine = OCREngine(ocr_config, ACADEMIC_ID_BOX)
    
    def extract_name(self, image: np.ndarray) -> ModelOCRResult:
        """
        Extract student name from the image.
        
        Note: This is a placeholder. Full implementation would require
        a different OCR configuration (not digit-only) and different
        region coordinates.
        
        Args:
            image: Preprocessed exam sheet image
            
        Returns:
            OCRResult with extracted text
        """
        # Placeholder: Student name OCR would need different config
        logger.debug("OCR: extracting student name (placeholder)")
        return ModelOCRResult(text="", confidence=0.0, is_empty=True)
    
    def extract_id(self, image: np.ndarray) -> ModelOCRResult:
        """
        Extract academic ID from the image.
        
        Args:
            image: Preprocessed exam sheet image
            
        Returns:
            OCRResult with extracted numeric ID
        """
        logger.info("Extracting Academic ID")
        
        # Use the production OCR engine
        result = self._engine.extract_academic_id(image)
        
        # Convert to model's OCRResult format
        return ModelOCRResult(
            text=result.text,
            confidence=result.confidence,
            bounding_box=ACADEMIC_ID_BOX,
            is_empty=not result.valid
        )
    
    def extract_text(self, image: np.ndarray) -> ModelOCRResult:
        """
        Extract text from a general image region.
        
        This is a placeholder for general text extraction.
        
        Args:
            image: Image region to process
            
        Returns:
            OCRResult with extracted text
        """
        logger.debug("OCR: general text extraction (placeholder)")
        return ModelOCRResult(text="", confidence=0.0, is_empty=True)


class ImageCropper:
    """
    Generates cropped image regions for UI display.
    
    This class wraps the production-grade CropEngine from
    app.core.cropping module which implements:
    - High-precision crop extraction
    - Image enhancement (sharpening, contrast)
    - UI-normalized sizing
    - Debug visualization
    """
    
    def __init__(self):
        self.config = get_config().crop
        
        # Create the production Crop engine
        crop_config = CropConfig(
            target_width=400,
            enhance_sharpen=True,
            enhance_contrast=False
        )
        self._engine = CropEngine(crop_config, DEFAULT_CROP_BOXES)
    
    def crop_all(self, image: np.ndarray) -> Dict[str, CropRegion]:
        """
        Generate all required crop regions from an exam sheet.
        
        Args:
            image: Preprocessed exam sheet image
            
        Returns:
            Dictionary mapping region names to CropRegion objects
        """
        logger.info("Extracting crop regions")
        
        # Use the production Crop engine
        result = self._engine.extract_all(image)
        
        # Convert to CropRegion format
        crops = {}
        
        for name in ["student_name", "academic_id", "q2", "q3", "q4"]:
            crop_img = result.get(name)
            box = DEFAULT_CROP_BOXES.get(name, (0, 0, 0, 0))
            
            crops[name] = CropRegion(
                name=name,
                image=crop_img,
                x=box[0],
                y=box[1],
                width=box[2] - box[0],
                height=box[3] - box[1]
            )
        
        logger.info(f"Generated {len(crops)} crop regions")
        return crops
    
    def _crop_region(
        self, 
        image: np.ndarray, 
        name: str, 
        box: Tuple[int, int, int, int]
    ) -> CropRegion:
        """
        Crop a specific region from the image.
        
        This is now handled by the Crop engine.
        
        Args:
            image: Source image
            name: Region name
            box: (x, y, width, height) tuple
            
        Returns:
            CropRegion object
        """
        logger.debug(f"Crop region: handled by Crop engine")
        
        # Delegate to engine
        crop_img = self._engine.extract_single(image, name)
        
        x, y, w, h = box
        
        return CropRegion(
            name=name,
            image=crop_img,
            x=x, y=y, width=w, height=h
        )


class CVProcessor:
    """
    Main computer vision processor coordinating all CV operations.
    
    This is the primary interface for the services layer to interact
    with the core computer vision functionality.
    """
    
    def __init__(self):
        self.loader = ImageLoader()
        self.preprocessor = ImagePreprocessor()
        self.omr_processor = OMRProcessor()
        self.ocr_processor = OCRProcessor()
        self.cropper = ImageCropper()
    
    def process(self, image_path: Path) -> ProcessingResult:
        """
        Process a single exam sheet image through the full CV pipeline.
        
        Args:
            image_path: Path to the exam sheet image
            
        Returns:
            ProcessingResult containing the extracted ExamRecord
        """
        import time
        start_time = time.time()
        
        stages = []
        warnings = []
        
        # Stage 1: Load image
        image = self.loader.load(image_path)
        if image is None:
            return ProcessingResult(
                success=False,
                error_message="Failed to load image",
                stages_completed=["load"]
            )
        stages.append("load")
        
        # Stage 2: Preprocess
        if not self.loader.validate(image):
            return ProcessingResult(
                success=False,
                error_message="Invalid image",
                stages_completed=["load", "preprocess"]
            )
        image = self.preprocessor.preprocess(image, image_path)
        stages.append("preprocess")
        
        # Stage 3: OMR extraction
        mcq_results = self.omr_processor.extract(image)
        stages.append("omr")
        
        # Stage 4: OCR extraction
        name_result = self.ocr_processor.extract_name(image)
        id_result = self.ocr_processor.extract_id(image)
        stages.append("ocr")
        
        # Stage 5: Generate crops
        crops = self.cropper.crop_all(image)
        stages.append("crops")
        
        # Build exam record
        record = ExamRecord(
            student_name=name_result.text,
            academic_id=id_result.text,
            mcq_answers=mcq_results,
            source_image_path=str(image_path),
            name_confidence=name_result.confidence,
            id_confidence=id_result.confidence,
            crops=crops
        )
        
        # Calculate overall confidence
        record.overall_confidence = self._calculate_confidence(record)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=True,
            record=record,
            processing_time=processing_time,
            stages_completed=stages,
            warnings=warnings
        )
    
    def _calculate_confidence(self, record: ExamRecord) -> float:
        """Calculate overall confidence score for a record."""
        confidences = []
        
        if record.name_confidence > 0:
            confidences.append(record.name_confidence)
        if record.id_confidence > 0:
            confidences.append(record.id_confidence)
        
        # Add MCQ confidences
        for mcq in record.mcq_answers:
            if mcq.confidence > 0:
                confidences.append(mcq.confidence)
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)