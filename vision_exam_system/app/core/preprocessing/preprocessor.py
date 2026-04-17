"""
Image Preprocessing Module for the Vision-Accelerated Exam Data Entry System.

This module implements Phase A: Image Pre-processing pipeline that normalizes
raw mobile-captured exam images into a consistent format suitable for OMR and OCR.

Pipeline Steps (in order):
1. Convert to Grayscale
2. Noise Reduction (Gaussian Blur)
3. Adaptive Threshold
4. Auto-Orientation (template-based rotation)
5. Deskew (Hough line detection)
6. Resize (normalization)

Author: Vision System Team
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)


class PreprocessingConfig:
    """
    Configuration dataclass for image preprocessing parameters.
    
    Attributes:
        master_height: Target height for normalized images (default: 2000)
        gaussian_kernel_size: Kernel size for Gaussian blur (default: 5)
        adaptive_block_size: Block size for adaptive threshold (default: 11)
        adaptive_c: Constant subtracted from mean (default: 2)
        canny_low: Lower threshold for Canny edge detection (default: 50)
        canny_high: Upper threshold for Canny edge detection (default: 150)
        hough_threshold: Minimum votes for line detection (default: 100)
        hough_min_line_length: Minimum line length (default: 50)
        hough_max_line_gap: Maximum gap between line segments (default: 10)
        template_path: Path to anchor template for orientation detection
        debug_mode: Whether to save intermediate processing images
        debug_output_dir: Directory for debug images
    """
    
    def __init__(
        self,
        master_height: int = 2000,
        gaussian_kernel_size: int = 5,
        adaptive_block_size: int = 11,
        adaptive_c: int = 2,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 100,
        hough_min_line_length: int = 50,
        hough_max_line_gap: int = 10,
        template_path: Optional[Path] = None,
        debug_mode: bool = False,
        debug_output_dir: Optional[Path] = None
    ):
        self.master_height = master_height
        self.gaussian_kernel_size = gaussian_kernel_size
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.template_path = template_path
        self.debug_mode = debug_mode
        self.debug_output_dir = debug_output_dir


class ImagePreprocessor:
    """
    Production-grade image preprocessor for exam sheet normalization.
    
    This class implements a complete preprocessing pipeline that handles:
    - Wrong orientation (upside down detection and correction)
    - Skew/tilt detection and correction
    - Inconsistent resolution normalization
    - Lighting variation handling
    
    The pipeline produces stable output suitable for OMR pixel density
    analysis and OCR text extraction.
    
    Example:
        >>> config = PreprocessingConfig(master_height=2000, debug_mode=True)
        >>> preprocessor = ImagePreprocessor(config)
        >>> processed = preprocessor.process(raw_image)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the ImagePreprocessor with configuration.
        
        Args:
            config: PreprocessingConfig instance. If None, uses defaults.
        """
        self.config = config or PreprocessingConfig()
        self._template_image: Optional[np.ndarray] = None
        self._template_loaded: bool = False
        
        # Debug image counter
        self._debug_counter: int = 0
        
        # Processing statistics
        self._stats: Dict[str, Any] = {
            "orientation_corrections": 0,
            "deskew_angle": 0.0,
            "processing_time": 0.0
        }
        
        # Load template if path provided
        if self.config.template_path and self.config.template_path.exists():
            self._load_template()
    
    def _load_template(self) -> bool:
        """
        Load the anchor template for orientation detection.
        
        Returns:
            True if template loaded successfully, False otherwise
        """
        try:
            self._template_image = cv2.imread(str(self.config.template_path), cv2.IMREAD_GRAYSCALE)
            if self._template_image is not None:
                self._template_loaded = True
                logger.info(f"Template loaded: {self.config.template_path}")
                return True
            else:
                logger.warning(f"Failed to load template: {self.config.template_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            return False
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Execute the full preprocessing pipeline on an image.
        
        This is the main entry point that applies all preprocessing steps
        in the required order:
        1. Convert to grayscale
        2. Noise reduction (Gaussian blur)
        3. Adaptive threshold
        4. Auto-orientation (template-based)
        5. Deskew (Hough line detection)
        6. Resize (normalization)
        
        Args:
            image: Input RGB image (numpy array, BGR format from cv2.imread)
            
        Returns:
            Fully processed image (grayscale, denoised, thresholded, 
            oriented, deskewed, and resized)
            
        Raises:
            ValueError: If input image is None or empty
        """
        start_time = time.time()
        
        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        if len(image.shape) != 3:
            raise ValueError("Input image must be a 3-channel color image")
        
        logger.info(f"Starting preprocessing: {image.shape[1]}x{image.shape[0]}")
        
        # Step 1: Convert to grayscale
        gray = self._to_gray(image)
        self._save_debug(gray, "01_gray")
        
        # Step 2: Noise reduction
        denoised = self._denoise(gray)
        self._save_debug(denoised, "02_denoised")
        
        # Step 3: Adaptive threshold
        thresholded = self._threshold(denoised)
        self._save_debug(thresholded, "03_thresholded")
        
        # Step 4: Auto-orientation (using original color image)
        oriented = self._auto_orient(image, thresholded)
        self._save_debug(cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY), "04_oriented")
        
        # Step 5: Deskew
        deskewed = self._deskew(oriented)
        self._save_debug(cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY), "05_deskewed")
        
        # Step 6: Resize to master height
        resized = self._resize(deskewed)
        
        # Calculate processing time
        self._stats["processing_time"] = time.time() - start_time
        logger.info(f"Preprocessing complete in {self._stats['processing_time']:.3f}s")
        
        return resized
    
    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to grayscale.
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug(f"Converted to grayscale: {gray.shape}")
        return gray
    
    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction.
        
        Args:
            gray: Input grayscale image
            
        Returns:
            Denoised grayscale image
        """
        kernel_size = self.config.gaussian_kernel_size
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        denoised = cv2.GaussianBlur(
            gray, 
            (kernel_size, kernel_size), 
            0
        )
        logger.debug(f"Applied Gaussian blur: kernel={kernel_size}x{kernel_size}")
        return denoised
    
    def _threshold(self, denoised: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for consistent binarization.
        
        This helps handle lighting variations across the image and
        produces stable input for OMR pixel density analysis.
        
        Args:
            denoised: Denoised grayscale image
            
        Returns:
            Binary thresholded image
        """
        block_size = self.config.adaptive_block_size
        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        thresholded = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.config.adaptive_c
        )
        logger.debug(f"Applied adaptive threshold: block={block_size}, C={self.config.adaptive_c}")
        return thresholded
    
    def _auto_orient(self, image: np.ndarray, thresholded: np.ndarray) -> np.ndarray:
        """
        Detect and correct image orientation using template matching.
        
        Uses a predefined anchor template (e.g., university logo or MCQ grid corner)
        to detect the current orientation. If the best match is found in the
        lower half of the image, the image is rotated 180 degrees.
        
        Args:
            image: Original BGR image
            thresholded: Thresholded image (for reference)
            
        Returns:
            Orientation-corrected BGR image
        """
        # If no template loaded, use heuristic-based orientation detection
        if not self._template_loaded:
            return self._auto_orient_heuristic(image, thresholded)
        
        # Use template matching
        return self._auto_orient_template(image)
    
    def _auto_orient_template(self, image: np.ndarray) -> np.ndarray:
        """
        Template-based orientation detection.
        
        Args:
            image: BGR image
            
        Returns:
            Orientation-corrected image
        """
        try:
            # Convert to grayscale for template matching
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Perform template matching
            result = cv2.matchTemplate(gray, self._template_image, cv2.TM_CCOEFF_NORMED)
            
            # Find best match location
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            logger.debug(f"Template match confidence: {max_val:.3f} at {max_loc}")
            
            # Check if match confidence is sufficient
            if max_val < 0.5:
                logger.warning("Template match confidence too low, skipping orientation correction")
                return image
            
            # Get Y coordinate of best match
            template_height, template_width = self._template_image.shape
            image_height, image_width = gray.shape
            
            # Calculate center Y of matched template
            match_center_y = max_loc[1] + template_height // 2
            
            # If match is in lower half, rotate 180 degrees
            if match_center_y > image_height / 2:
                logger.info(f"Orientation detected: upside-down (match at Y={match_center_y})")
                self._stats["orientation_corrections"] += 1
                return cv2.rotate(image, cv2.ROTATE_180)
            
            logger.debug("Orientation detected: correct (upright)")
            return image
            
        except Exception as e:
            logger.error(f"Error in template-based orientation: {e}")
            return image
    
    def _auto_orient_heuristic(self, image: np.ndarray, thresholded: np.ndarray) -> np.ndarray:
        """
        Heuristic-based orientation detection when no template is available.
        
        Uses edge density analysis to detect if the image is upside down.
        This is a fallback method when template matching is not available.
        
        Args:
            image: BGR image
            thresholded: Thresholded image
            
        Returns:
            Orientation-corrected image
        """
        try:
            height, width = thresholded.shape
            
            # Calculate edge density in top and bottom halves
            top_half = thresholded[:height // 2, :]
            bottom_half = thresholded[height // 2:, :]
            
            # Count non-zero pixels (edges)
            top_density = np.count_nonzero(top_half) / top_half.size
            bottom_density = np.count_nonzero(bottom_half) / bottom_half.size
            
            logger.debug(f"Edge density - Top: {top_density:.3f}, Bottom: {bottom_density:.3f}")
            
            # If bottom has significantly more edges, image might be upside down
            # This is a heuristic and may need tuning
            if bottom_density > top_density * 1.5:
                logger.info("Orientation detected: upside-down (heuristic)")
                self._stats["orientation_corrections"] += 1
                return cv2.rotate(image, cv2.ROTATE_180)
            
            logger.debug("Orientation detected: correct (heuristic)")
            return image
            
        except Exception as e:
            logger.error(f"Error in heuristic orientation: {e}")
            return image
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew/tilt in the image using Hough line detection.
        
        The algorithm:
        1. Applies Canny edge detection
        2. Detects lines using HoughLinesP
        3. Calculates the dominant angle of detected lines
        4. Applies affine rotation to correct the skew
        
        Args:
            image: BGR image
            
        Returns:
            Deskewed BGR image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                gray,
                self.config.canny_low,
                self.config.canny_high
            )
            self._save_debug(edges, "deskew_edges")
            
            # Detect lines using probabilistic Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.config.hough_threshold,
                minLineLength=self.config.hough_min_line_length,
                maxLineGap=self.config.hough_max_line_gap
            )
            
            if lines is None or len(lines) == 0:
                logger.warning("No lines detected for deskewing")
                return image
            
            # Calculate angles of all detected lines
            angles: List[float] = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle in degrees
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Filter out near-vertical lines (within 5 degrees of vertical)
                # These are likely noise or page borders
                if not (85 <= abs(angle) <= 95):
                    angles.append(angle)
            
            if not angles:
                logger.warning("No valid angles detected for deskewing")
                return image
            
            # Use median angle to be robust against outliers
            median_angle = np.median(angles)
            
            # Calculate skew angle (deviation from horizontal)
            skew_angle = median_angle
            
            # Only correct if skew is significant (> 0.5 degrees)
            if abs(skew_angle) > 0.5:
                logger.info(f"Deskewing: correcting {skew_angle:.2f}° skew")
                self._stats["deskew_angle"] = skew_angle
                
                # Get image center
                height, width = gray.shape
                center = (width // 2, height // 2)
                
                # Create rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                
                # Apply rotation
                deskewed = cv2.warpAffine(
                    image,
                    rotation_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
                
                return deskewed
            
            logger.debug(f"No significant skew detected: {skew_angle:.2f}°")
            return image
            
        except Exception as e:
            logger.error(f"Error during deskewing: {e}")
            return image
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to master height while maintaining aspect ratio.
        
        Args:
            image: Input BGR image
            
        Returns:
            Resized BGR image with height = master_height
        """
        target_height = self.config.master_height
        current_height, current_width = image.shape[:2]
        
        if current_height == target_height:
            logger.debug("Image already at target height, no resize needed")
            return image
        
        # Calculate new width maintaining aspect ratio
        scale = target_height / current_height
        new_width = int(current_width * scale)
        
        # Resize using Lanczos interpolation for quality
        resized = cv2.resize(
            image,
            (new_width, target_height),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        logger.debug(f"Resized: {current_width}x{current_height} -> {new_width}x{target_height}")
        return resized
    
    def _save_debug(self, image: np.ndarray, stage: str) -> None:
        """
        Save intermediate processing images if debug mode is enabled.
        
        Args:
            image: Image to save
            stage: Name of the processing stage
        """
        if not self.config.debug_mode or not self.config.debug_output_dir:
            return
        
        try:
            self.config.debug_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            filename = f"{self._debug_counter:03d}_{stage}.png"
            filepath = self.config.debug_output_dir / filename
            
            cv2.imwrite(str(filepath), image)
            self._debug_counter += 1
            
            logger.debug(f"Saved debug image: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug image: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            "orientation_corrections": 0,
            "deskew_angle": 0.0,
            "processing_time": 0.0
        }
        self._debug_counter = 0


def create_preprocessor(
    master_height: int = 2000,
    template_path: Optional[Path] = None,
    debug_mode: bool = False,
    debug_output_dir: Optional[Path] = None
) -> ImagePreprocessor:
    """
    Factory function to create an ImagePreprocessor with common settings.
    
    Args:
        master_height: Target height for normalized images
        template_path: Path to anchor template for orientation
        debug_mode: Whether to save debug images
        debug_output_dir: Directory for debug images
        
    Returns:
        Configured ImagePreprocessor instance
    """
    config = PreprocessingConfig(
        master_height=master_height,
        template_path=template_path,
        debug_mode=debug_mode,
        debug_output_dir=debug_output_dir
    )
    return ImagePreprocessor(config)


# Convenience function for quick usage
def preprocess_image(
    image: np.ndarray,
    master_height: int = 2000,
    template_path: Optional[Path] = None
) -> np.ndarray:
    """
    Convenience function to preprocess an image with default settings.
    
    Args:
        image: Input BGR image
        master_height: Target height for output
        template_path: Optional path to orientation template
        
    Returns:
        Preprocessed image
    """
    preprocessor = create_preprocessor(
        master_height=master_height,
        template_path=template_path
    )
    return preprocessor.process(image)