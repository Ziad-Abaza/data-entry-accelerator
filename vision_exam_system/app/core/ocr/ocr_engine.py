"""
OCR (Optical Character Recognition) Engine for the Vision-Accelerated Exam Data Entry System.

This module implements Phase B: Feature Extraction - OCR processing for handwritten
academic ID extraction using PyTesseract.

Core Concept:
- Extract Academic ID from a predefined region
- Use pixel preprocessing optimized for handwritten digits
- Apply strict digit-only whitelist
- Validate results and estimate confidence

Author: Vision System Team
"""

import cv2
import numpy as np
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import time

# Try to import pytesseract, handle if not available
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logging.warning("PyTesseract not available. OCR functionality will be limited.")

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OCRConfig:
    """
    Configuration dataclass for OCR engine parameters.
    
    Attributes:
        tesseract_path: Path to Tesseract executable (optional)
        psm_mode: Page segmentation mode (default: 6 - single uniform block)
        oem_mode: OCR engine mode (default: 3 - default + LSTM)
        whitelist: Allowed characters (default: digits only)
        expected_id_length: Expected length of academic ID (default: 8)
        min_confidence: Minimum confidence threshold (default: 0.5)
        debug_mode: Whether to save debug visualizations
        debug_output_dir: Directory for debug images
    """
    tesseract_path: Optional[str] = None
    psm_mode: int = 6
    oem_mode: int = 3
    whitelist: str = "0123456789"
    expected_id_length: int = 8
    min_confidence: float = 0.5
    debug_mode: bool = False
    debug_output_dir: Optional[Path] = None


# Default Academic ID region coordinates (assuming normalized 2000px height)
# Format: (x1, y1, x2, y2)
ACADEMIC_ID_BOX = (500, 50, 800, 150)

# Alternative positions for different templates
ACADEMIC_ID_BOX_ALT = (600, 80, 900, 160)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OCRResult:
    """Result from OCR extraction."""
    text: str
    confidence: float  # 0.0 to 1.0
    valid: bool
    raw_text: str = ""  # Before post-processing
    processing_time: float = 0.0


# ============================================================================
# OCR Engine
# ============================================================================

class OCREngine:
    """
    Production-grade OCR engine for handwritten digit extraction.
    
    Uses PyTesseract with strict configuration to extract academic IDs
    from preprocessed exam sheet images. Optimized for handwritten digits
    with robust preprocessing and validation.
    
    Example:
        >>> config = OCRConfig(expected_id_length=8, debug_mode=True)
        >>> engine = OCREngine(config)
        >>> result = engine.extract_academic_id(preprocessed_image)
        >>> print(result.text)  # '12345678'
    """
    
    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        id_box: Tuple[int, int, int, int] = ACADEMIC_ID_BOX
    ):
        """
        Initialize the OCR engine.
        
        Args:
            config: OCRConfig instance. If None, uses defaults.
            id_box: Academic ID region coordinates (x1, y1, x2, y2)
        """
        self.config = config or OCRConfig()
        self.id_box = id_box
        self._debug_counter: int = 0
        
        # Configure Tesseract if available
        if PYTESSERACT_AVAILABLE:
            if self.config.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_path
        
        logger.info(f"OCR Engine initialized: expected_length={self.config.expected_id_length}")
    
    def extract_academic_id(self, image: np.ndarray) -> OCRResult:
        """
        Extract Academic ID from the preprocessed exam sheet image.
        
        Args:
            image: Preprocessed image (numpy array)
                   Should be the output of the preprocessing phase
            
        Returns:
            OCRResult containing:
            - text: Extracted digits (e.g., "12345678")
            - confidence: Confidence score (0.0 to 1.0)
            - valid: Whether the result passes validation
            - raw_text: Original OCR output before cleaning
            
        Raises:
            ValueError: If input image is None or empty
        """
        start_time = time.time()
        
        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        logger.info("Starting Academic ID extraction")
        
        # Step 1: Crop the Academic ID region
        roi = self._crop(image)
        
        if roi is None or roi.size == 0:
            logger.warning("Failed to crop Academic ID region")
            return OCRResult(
                text="",
                confidence=0.0,
                valid=False,
                processing_time=time.time() - start_time
            )
        
        # Save debug image
        if self.config.debug_mode:
            self._save_debug(roi, "01_roi")
        
        # Step 2: Preprocess for OCR
        processed = self._preprocess(roi)
        
        if self.config.debug_mode:
            self._save_debug(processed, "02_processed")
        
        # Step 3: Run OCR
        raw_text = self._run_ocr(processed)
        
        # Step 4: Post-process
        text, cleaned = self._postprocess(raw_text)
        
        # Step 5: Validate and estimate confidence
        valid, confidence = self._validate(text)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Academic ID extraction: '{text}' (confidence: {confidence:.2f}, "
            f"valid: {valid}) in {processing_time:.3f}s"
        )
        
        return OCRResult(
            text=text,
            confidence=confidence,
            valid=valid,
            raw_text=raw_text,
            processing_time=processing_time
        )
    
    def _crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop the Academic ID region from the image.
        
        Args:
            image: Input image
            
        Returns:
            Cropped ROI or None if invalid
        """
        x1, y1, x2, y2 = self.id_box
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        
        # Handle invalid boxes
        if x2 <= x1 or y2 <= y1:
            logger.error(f"Invalid ID box coordinates: {self.id_box}")
            return None
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        logger.debug(f"Cropped ID region: {roi.shape}")
        return roi
    
    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI for optimal OCR results.
        
        Steps:
        1. Convert to grayscale
        2. Gaussian blur
        3. Adaptive threshold
        4. Morphological operations
        
        Args:
            roi: Cropped region of interest
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive threshold for lighting invariance
        # Using smaller block size for small text
        thresholded = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Morphological operations to enhance digits
        # CLOSE operation to fill gaps in broken digits
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        
        # Optional: slight dilation to enhance strokes
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        logger.debug("Applied OCR preprocessing")
        return dilated
    
    def _run_ocr(self, processed: np.ndarray) -> str:
        """
        Run Tesseract OCR on the preprocessed image.
        
        Args:
            processed: Preprocessed binary image
            
        Returns:
            Raw OCR output text
        """
        if not PYTESSERACT_AVAILABLE:
            logger.warning("PyTesseract not available, returning empty result")
            return ""
        
        try:
            # Build Tesseract config
            config = (
                f"--psm {self.config.psm_mode} "
                f"-c tessedit_char_whitelist={self.config.whitelist}"
            )
            
            # Run OCR
            text = pytesseract.image_to_string(
                processed,
                config=config,
                lang='eng'
            )
            
            logger.debug(f"Raw OCR output: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _postprocess(self, raw_text: str) -> Tuple[str, str]:
        """
        Post-process OCR output to extract clean digits.
        
        Args:
            raw_text: Raw output from Tesseract
            
        Returns:
            Tuple of (cleaned_text, original_cleaned)
        """
        # Remove non-digit characters
        cleaned = re.sub(r"\D", "", raw_text)
        
        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Remove leading zeros if desired (optional)
        # cleaned = cleaned.lstrip('0')
        # if not cleaned:
        #     cleaned = "0"
        
        logger.debug(f"Post-processed: '{raw_text}' -> '{cleaned}'")
        return cleaned, cleaned
    
    def _validate(self, text: str) -> Tuple[bool, float]:
        """
        Validate OCR result and estimate confidence.
        
        Validation rules:
        - Not empty
        - Length matches expected (or within tolerance)
        - Contains only digits
        
        Confidence estimation:
        - Length match score
        - Character consistency (heuristic)
        
        Args:
            text: Cleaned OCR text
            
        Returns:
            Tuple of (valid, confidence)
        """
        # Check if empty
        if not text:
            return False, 0.0
        
        # Check if contains only digits
        if not text.isdigit():
            return False, 0.0
        
        # Check length
        length = len(text)
        expected = self.config.expected_id_length
        
        # Length match score (0.0 to 1.0)
        if length == expected:
            length_score = 1.0
        elif length == expected - 1 or length == expected + 1:
            length_score = 0.7
        elif length == expected - 2 or length == expected + 2:
            length_score = 0.4
        else:
            length_score = 0.1
        
        # Character consistency heuristic
        # If all characters are similar (e.g., all thin strokes), might be noise
        # This is a simple heuristic - could be enhanced
        
        # Final confidence
        confidence = length_score
        
        # Check against minimum threshold
        valid = confidence >= self.config.min_confidence
        
        logger.debug(f"Validation: text='{text}', confidence={confidence:.2f}, valid={valid}")
        return valid, confidence
    
    def _save_debug(self, image: np.ndarray, stage: str) -> None:
        """
        Save debug images if debug mode is enabled.
        
        Args:
            image: Image to save
            stage: Processing stage name
        """
        if not self.config.debug_mode or not self.config.debug_output_dir:
            return
        
        try:
            self.config.debug_output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"ocr_{stage}_{self._debug_counter:03d}.png"
            filepath = self.config.debug_output_dir / filename
            
            cv2.imwrite(str(filepath), image)
            self._debug_counter += 1
            
            logger.debug(f"Saved OCR debug image: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug image: {e}")
    
    def set_id_box(self, box: Tuple[int, int, int, int]) -> None:
        """
        Set new Academic ID region coordinates.
        
        Args:
            box: (x1, y1, x2, y2) coordinates
        """
        self.id_box = box
        logger.info(f"ID box updated: {box}")
    
    def get_id_box(self) -> Tuple[int, int, int, int]:
        """Get current Academic ID region coordinates."""
        return self.id_box


# ============================================================================
# Factory Functions
# ============================================================================

def create_ocr_engine(
    tesseract_path: Optional[str] = None,
    expected_id_length: int = 8,
    min_confidence: float = 0.5,
    debug_mode: bool = False,
    debug_output_dir: Optional[Path] = None,
    id_box: Tuple[int, int, int, int] = ACADEMIC_ID_BOX
) -> OCREngine:
    """
    Factory function to create an OCR engine with common settings.
    
    Args:
        tesseract_path: Path to Tesseract executable
        expected_id_length: Expected length of academic ID
        min_confidence: Minimum confidence threshold
        debug_mode: Whether to save debug visualizations
        debug_output_dir: Directory for debug images
        id_box: Academic ID region coordinates
        
    Returns:
        Configured OCREngine instance
    """
    config = OCRConfig(
        tesseract_path=tesseract_path,
        expected_id_length=expected_id_length,
        min_confidence=min_confidence,
        debug_mode=debug_mode,
        debug_output_dir=debug_output_dir
    )
    return OCREngine(config, id_box)


def extract_academic_id(
    image: np.ndarray,
    expected_id_length: int = 8
) -> OCRResult:
    """
    Convenience function to extract Academic ID with default settings.
    
    Args:
        image: Preprocessed image
        expected_id_length: Expected length of ID
        
    Returns:
        OCRResult
    """
    engine = create_ocr_engine(expected_id_length=expected_id_length)
    return engine.extract_academic_id(image)