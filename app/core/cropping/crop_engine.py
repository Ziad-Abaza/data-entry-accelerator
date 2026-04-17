"""
Dynamic Cropping Engine for the Vision-Accelerated Exam Data Entry System.

This module implements the cropping functionality that extracts structured
image snippets for UI-driven human verification.

Core Concept:
- Extract high-precision crops for key regions
- Apply enhancement for better readability
- Normalize sizes for consistent UI display

Author: Vision System Team
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CropConfig:
    """
    Configuration dataclass for Crop engine parameters.
    
    Attributes:
        target_width: Target width for normalized crops (default: 400)
        enhance_sharpen: Whether to apply sharpening (default: True)
        enhance_contrast: Whether to apply contrast normalization (default: False)
        sharpen_kernel: Kernel size for sharpening (default: 3)
        debug_mode: Whether to save debug crops
        debug_output_dir: Directory for debug images
    """
    target_width: int = 400
    enhance_sharpen: bool = True
    enhance_contrast: bool = False
    sharpen_kernel: int = 3
    debug_mode: bool = False
    debug_output_dir: Optional[Path] = None


# Default crop coordinates (adjusted for ~1152px height images)
# Format: (x1, y1, x2, y2)
DEFAULT_CROP_BOXES: Dict[str, Tuple[int, int, int, int]] = {
    # Identity fields - scaled for 1152px height
    "student_name": (30, 30, 250, 70),     # Name field
    "academic_id": (280, 30, 450, 70),     # ID field
    
    # Open-ended question answers - scaled for 1152px height
    "q2": (30, 460, 380, 580),             # Q2 answer area
    "q3": (30, 610, 380, 730),             # Q3 answer area  
    "q4": (30, 760, 380, 880),             # Q4 answer area
}

# Alternative coordinates for different templates
ALT_CROP_BOXES: Dict[str, Tuple[int, int, int, int]] = {
    "student_name": (50, 35, 280, 80),
    "academic_id": (320, 35, 480, 80),
    "q2": (50, 490, 400, 610),
    "q3": (50, 640, 400, 760),
    "q4": (50, 790, 400, 910),
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CropResult:
    """Result for a single crop operation."""
    name: str
    image: Optional[np.ndarray]
    original_box: Tuple[int, int, int, int]
    success: bool
    error_message: Optional[str] = None


@dataclass
class CropCollection:
    """Collection of all crop results."""
    student_name: Optional[np.ndarray] = None
    academic_id: Optional[np.ndarray] = None
    q2: Optional[np.ndarray] = None
    q3: Optional[np.ndarray] = None
    q4: Optional[np.ndarray] = None
    
    # Metadata
    processing_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    
    def to_dict(self) -> Dict[str, Optional[np.ndarray]]:
        """Convert to dictionary of numpy arrays."""
        return {
            "student_name": self.student_name,
            "academic_id": self.academic_id,
            "q2": self.q2,
            "q3": self.q3,
            "q4": self.q4,
        }
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get crop by name."""
        return getattr(self, key, None)


# ============================================================================
# Crop Engine
# ============================================================================

class CropEngine:
    """
    Production-grade Dynamic Cropping Engine.
    
    Extracts high-precision image crops from preprocessed exam sheets
    for UI-driven human verification. Optimized for speed and quality.
    
    Example:
        >>> config = CropConfig(target_width=400, debug_mode=True)
        >>> engine = CropEngine(config)
        >>> crops = engine.extract_all(preprocessed_image)
        >>> cv2.imshow("Q2", crops.q2)
    """
    
    def __init__(
        self,
        config: Optional[CropConfig] = None,
        crop_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    ):
        """
        Initialize the Crop Engine.
        
        Args:
            config: CropConfig instance. If None, uses defaults.
            crop_boxes: Custom crop coordinates. If None, uses DEFAULT_CROP_BOXES.
        """
        self.config = config or CropConfig()
        self.crop_boxes = crop_boxes or DEFAULT_CROP_BOXES
        self._debug_counter: int = 0
        
        logger.info(f"Crop Engine initialized: {len(self.crop_boxes)} regions")
    
    def extract_all(self, image: np.ndarray) -> CropCollection:
        """
        Extract all crop regions from the image.
        
        Args:
            image: Preprocessed image (numpy array)
                   Should be the output of the preprocessing phase
            
        Returns:
            CropCollection containing:
            - student_name: Cropped name region
            - academic_id: Cropped ID region
            - q2, q3, q4: Cropped answer regions
            
        Raises:
            ValueError: If input image is None or empty
        """
        start_time = time.time()
        
        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        logger.info(f"Starting crop extraction for {len(self.crop_boxes)} regions")
        
        # Initialize collection
        collection = CropCollection()
        
        # Extract each crop
        for name, box in self.crop_boxes.items():
            crop = self._extract_single(image, name, box)
            
            # Store in collection
            if crop.success and crop.image is not None:
                setattr(collection, name, crop.image)
                collection.success_count += 1
            else:
                collection.failure_count += 1
        
        collection.processing_time = time.time() - start_time
        
        logger.info(
            f"Crop extraction complete: {collection.success_count} successful, "
            f"{collection.failure_count} failed in {collection.processing_time:.3f}s"
        )
        
        # Save debug crops if enabled
        if self.config.debug_mode:
            self._save_debug_crops(collection)
        
        return collection
    
    def _extract_single(
        self,
        image: np.ndarray,
        name: str,
        box: Tuple[int, int, int, int]
    ) -> CropResult:
        """
        Extract a single crop region.
        
        Args:
            image: Source image
            name: Crop name (e.g., 'student_name', 'q2')
            box: (x1, y1, x2, y2) coordinates
            
        Returns:
            CropResult for the region
        """
        try:
            # Step 1: Crop the region
            cropped = self._crop(image, box)
            
            if cropped is None or cropped.size == 0:
                return CropResult(
                    name=name,
                    image=None,
                    original_box=box,
                    success=False,
                    error_message="Failed to crop region"
                )
            
            # Step 2: Enhance for better readability
            if self.config.enhance_sharpen or self.config.enhance_contrast:
                cropped = self._enhance(cropped)
            
            # Step 3: Resize for UI consistency
            resized = self._resize(cropped)
            
            return CropResult(
                name=name,
                image=resized,
                original_box=box,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error extracting crop '{name}': {e}")
            return CropResult(
                name=name,
                image=None,
                original_box=box,
                success=False,
                error_message=str(e)
            )
    
    def _crop(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Crop a region from the image with bounds checking.
        
        Args:
            image: Source image
            box: (x1, y1, x2, y2) coordinates
            
        Returns:
            Cropped image or None if invalid
        """
        x1, y1, x2, y2 = box
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        
        # Handle invalid boxes
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid crop box: {box}")
            return None
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            logger.warning(f"Empty crop result for box: {box}")
            return None
        
        logger.debug(f"Cropped '{box}': {roi.shape}")
        return roi
    
    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance crop image for better readability.
        
        Applies:
        - Grayscale conversion (if color)
        - Sharpening (optional)
        - Contrast normalization (optional)
        
        Args:
            image: Input crop image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        result = gray
        
        # Apply sharpening
        if self.config.enhance_sharpen:
            kernel_size = self.config.sharpen_kernel
            if kernel_size % 2 == 0:
                kernel_size += 1  # Must be odd
            
            # Create sharpening kernel
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ], dtype=np.float32) / 9
            
            result = cv2.filter2D(result, -1, kernel)
        
        # Apply contrast normalization
        if self.config.enhance_contrast:
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        return result
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize crop to target width while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        target_width = self.config.target_width
        current_height, current_width = image.shape[:2]
        
        if current_width == target_width:
            return image
        
        # Calculate new height maintaining aspect ratio
        scale = target_width / current_width
        new_height = int(current_height * scale)
        
        # Resize
        resized = cv2.resize(
            image,
            (target_width, new_height),
            interpolation=cv2.INTER_LANCZOS4
        )
        
        logger.debug(f"Resized: {current_width}x{current_height} -> {target_width}x{new_height}")
        return resized
    
    def _save_debug_crops(self, collection: CropCollection) -> None:
        """
        Save debug crop images.
        
        Args:
            collection: CropCollection to save
        """
        if not self.config.debug_output_dir:
            return
        
        try:
            self.config.debug_output_dir.mkdir(parents=True, exist_ok=True)
            
            for name in self.crop_boxes.keys():
                crop = collection.get(name)
                if crop is not None and crop.size > 0:
                    filename = f"{name}_{self._debug_counter:03d}.png"
                    filepath = self.config.debug_output_dir / filename
                    cv2.imwrite(str(filepath), crop)
            
            self._debug_counter += 1
            logger.debug(f"Saved debug crops")
            
        except Exception as e:
            logger.warning(f"Failed to save debug crops: {e}")
    
    def get_crop_boxes(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Get current crop box coordinates."""
        return self.crop_boxes.copy()
    
    def set_crop_boxes(self, boxes: Dict[str, Tuple[int, int, int, int]]) -> None:
        """
        Set new crop box coordinates.
        
        Args:
            boxes: Dictionary of crop name -> (x1, y1, x2, y2)
        """
        self.crop_boxes = boxes
        logger.info(f"Crop boxes updated: {len(boxes)} regions")
    
    def extract_single(
        self,
        image: np.ndarray,
        name: str
    ) -> Optional[np.ndarray]:
        """
        Extract a single named crop.
        
        Args:
            image: Source image
            name: Crop name (e.g., 'student_name', 'q2')
            
        Returns:
            Cropped image or None
        """
        if name not in self.crop_boxes:
            logger.warning(f"Unknown crop name: {name}")
            return None
        
        result = self._extract_single(image, name, self.crop_boxes[name])
        return result.image if result.success else None


# ============================================================================
# Factory Functions
# ============================================================================

def create_crop_engine(
    target_width: int = 400,
    enhance_sharpen: bool = True,
    enhance_contrast: bool = False,
    debug_mode: bool = False,
    debug_output_dir: Optional[Path] = None,
    crop_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None
) -> CropEngine:
    """
    Factory function to create a Crop Engine with common settings.
    
    Args:
        target_width: Target width for normalized crops
        enhance_sharpen: Whether to apply sharpening
        enhance_contrast: Whether to apply contrast normalization
        debug_mode: Whether to save debug crops
        debug_output_dir: Directory for debug images
        crop_boxes: Custom crop coordinates
        
    Returns:
        Configured CropEngine instance
    """
    config = CropConfig(
        target_width=target_width,
        enhance_sharpen=enhance_sharpen,
        enhance_contrast=enhance_contrast,
        debug_mode=debug_mode,
        debug_output_dir=debug_output_dir
    )
    return CropEngine(config, crop_boxes)


def extract_all_crops(image: np.ndarray) -> CropCollection:
    """
    Convenience function to extract all crops with default settings.
    
    Args:
        image: Preprocessed image
        
    Returns:
        CropCollection
    """
    engine = create_crop_engine()
    return engine.extract_all(image)