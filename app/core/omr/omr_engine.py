"""
OMR (Optical Mark Recognition) Engine for the Vision-Accelerated Exam Data Entry System.

This module implements Phase B: Feature Extraction - OMR processing using pixel density
analysis rather than character recognition.

Core Concept:
- Each question has 4 answer boxes (A, B, C, D)
- Extract each box region and compute pixel density (black pixels)
- Select the option with the highest density
- Assign confidence score and detect ambiguity

Author: Vision System Team
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OMRConfig:
    """
    Configuration dataclass for OMR engine parameters.
    
    Attributes:
        questions_count: Number of questions in the exam (default: 30)
        options_per_question: Number of options per question (default: 4)
        min_mark_threshold: Minimum pixel density to consider a mark (default: 0.15)
        ambiguity_threshold: Threshold for marking as ambiguous (default: 0.10)
        empty_threshold: Threshold for marking as empty/unanswered (default: 0.02)
        debug_mode: Whether to save debug visualizations
        debug_output_dir: Directory for debug images
    """
    questions_count: int = 30
    options_per_question: int = 4
    min_mark_threshold: float = 0.15
    ambiguity_threshold: float = 0.10
    empty_threshold: float = 0.02
    debug_mode: bool = False
    debug_output_dir: Optional[Path] = None


# Default template coordinates for a standard exam sheet
# These coordinates assume a normalized image height of 2000px
# Format: {question_number: {option: (x1, y1, x2, y2)}}
DEFAULT_TEMPLATE: Dict[int, Dict[str, Tuple[int, int, int, int]]] = {
    # Questions 1-10 (first row)
    1: {"A": (50, 200, 90, 230), "B": (100, 200, 140, 230), "C": (150, 200, 190, 230), "D": (200, 200, 240, 230)},
    2: {"A": (50, 240, 90, 270), "B": (100, 240, 140, 270), "C": (150, 240, 190, 270), "D": (200, 240, 240, 270)},
    3: {"A": (50, 280, 90, 310), "B": (100, 280, 140, 310), "C": (150, 280, 190, 310), "D": (200, 280, 240, 310)},
    4: {"A": (50, 320, 90, 350), "B": (100, 320, 140, 350), "C": (150, 320, 190, 350), "D": (200, 320, 240, 350)},
    5: {"A": (50, 360, 90, 390), "B": (100, 360, 140, 390), "C": (150, 360, 190, 390), "D": (200, 360, 240, 390)},
    6: {"A": (50, 400, 90, 430), "B": (100, 400, 140, 430), "C": (150, 400, 190, 430), "D": (200, 400, 240, 430)},
    7: {"A": (50, 440, 90, 470), "B": (100, 440, 140, 470), "C": (150, 440, 190, 470), "D": (200, 440, 240, 470)},
    8: {"A": (50, 480, 90, 510), "B": (100, 480, 140, 510), "C": (150, 480, 190, 510), "D": (200, 480, 240, 510)},
    9: {"A": (50, 520, 90, 550), "B": (100, 520, 140, 550), "C": (150, 520, 190, 550), "D": (200, 520, 240, 550)},
    10: {"A": (50, 560, 90, 590), "B": (100, 560, 140, 590), "C": (150, 560, 190, 590), "D": (200, 560, 240, 590)},
    # Questions 11-20 (second row)
    11: {"A": (300, 200, 340, 230), "B": (350, 200, 390, 230), "C": (400, 200, 440, 230), "D": (450, 200, 490, 230)},
    12: {"A": (300, 240, 340, 270), "B": (350, 240, 390, 270), "C": (400, 240, 440, 270), "D": (450, 240, 490, 270)},
    13: {"A": (300, 280, 340, 310), "B": (350, 280, 390, 310), "C": (400, 280, 440, 310), "D": (450, 280, 490, 310)},
    14: {"A": (300, 320, 340, 350), "B": (350, 320, 390, 350), "C": (400, 320, 440, 350), "D": (450, 320, 490, 350)},
    15: {"A": (300, 360, 340, 390), "B": (350, 360, 390, 390), "C": (400, 360, 440, 390), "D": (450, 360, 490, 390)},
    16: {"A": (300, 400, 340, 430), "B": (350, 400, 390, 430), "C": (400, 400, 440, 430), "D": (450, 400, 490, 430)},
    17: {"A": (300, 440, 340, 470), "B": (350, 440, 390, 470), "C": (400, 440, 440, 470), "D": (450, 440, 490, 470)},
    18: {"A": (300, 480, 340, 510), "B": (350, 480, 390, 510), "C": (400, 480, 440, 510), "D": (450, 480, 490, 510)},
    19: {"A": (300, 520, 340, 550), "B": (350, 520, 390, 550), "C": (400, 520, 440, 550), "D": (450, 520, 490, 550)},
    20: {"A": (300, 560, 340, 590), "B": (350, 560, 390, 590), "C": (400, 560, 440, 590), "D": (450, 560, 490, 590)},
    # Questions 21-30 (third row)
    21: {"A": (550, 200, 590, 230), "B": (600, 200, 640, 230), "C": (650, 200, 690, 230), "D": (700, 200, 740, 230)},
    22: {"A": (550, 240, 590, 270), "B": (600, 240, 640, 270), "C": (650, 240, 690, 270), "D": (700, 240, 740, 270)},
    23: {"A": (550, 280, 590, 310), "B": (600, 280, 640, 310), "C": (650, 280, 690, 310), "D": (700, 280, 740, 310)},
    24: {"A": (550, 320, 590, 350), "B": (600, 320, 640, 350), "C": (650, 320, 690, 350), "D": (700, 320, 740, 350)},
    25: {"A": (550, 360, 590, 390), "B": (600, 360, 640, 390), "C": (650, 360, 690, 390), "D": (700, 360, 740, 390)},
    26: {"A": (550, 400, 590, 430), "B": (600, 400, 640, 430), "C": (650, 400, 690, 430), "D": (700, 400, 740, 430)},
    27: {"A": (550, 440, 590, 470), "B": (600, 440, 640, 470), "C": (650, 440, 690, 470), "D": (700, 440, 740, 470)},
    28: {"A": (550, 480, 590, 510), "B": (600, 480, 640, 510), "C": (650, 480, 690, 510), "D": (700, 480, 740, 510)},
    29: {"A": (550, 520, 590, 550), "B": (600, 520, 640, 550), "C": (650, 520, 690, 550), "D": (700, 520, 740, 550)},
    30: {"A": (550, 560, 590, 590), "B": (600, 560, 640, 590), "C": (650, 560, 690, 590), "D": (700, 560, 740, 590)},
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OMRQuestionResult:
    """Result for a single OMR question."""
    question_number: int
    selected_answer: Optional[str]  # 'A', 'B', 'C', 'D' or None
    confidence: float  # 0.0 to 1.0
    flag: str  # 'OK', 'AMBIGUOUS', 'EMPTY'
    densities: Dict[str, float] = field(default_factory=dict)  # {'A': 0.2, 'B': 0.8, ...}
    max_density: float = 0.0
    second_max_density: float = 0.0


@dataclass
class OMRResult:
    """Complete OMR extraction result."""
    answers: List[Optional[str]]  # List of 30 answers
    confidences: List[float]  # List of 30 confidence scores
    flags: List[str]  # List of 30 flags
    question_results: List[OMRQuestionResult] = field(default_factory=list)
    processing_time: float = 0.0
    total_questions: int = 0
    ambiguous_count: int = 0
    empty_count: int = 0
    ok_count: int = 0


# ============================================================================
# OMR Engine
# ============================================================================

class OMREngine:
    """
    Production-grade OMR (Optical Mark Recognition) engine.
    
    Uses pixel density analysis to extract MCQ answers from preprocessed
    exam sheet images. Does NOT use OCR - instead counts black pixels
    in predefined coordinate boxes to determine selected answers.
    
    Example:
        >>> config = OMRConfig(questions_count=30, debug_mode=True)
        >>> engine = OMREngine(config)
        >>> result = engine.extract(preprocessed_image)
        >>> print(result.answers)  # ['A', 'C', 'B', ...]
    """
    
    def __init__(
        self,
        config: Optional[OMRConfig] = None,
        template: Optional[Dict[int, Dict[str, Tuple[int, int, int, int]]]] = None
    ):
        """
        Initialize the OMR engine.
        
        Args:
            config: OMRConfig instance. If None, uses defaults.
            template: Template coordinates. If None, uses DEFAULT_TEMPLATE.
        """
        self.config = config or OMRConfig()
        self.template = template or DEFAULT_TEMPLATE
        self._debug_counter: int = 0
        
        # Validate template matches questions count
        self._validate_template()
        
        logger.info(f"OMR Engine initialized: {self.config.questions_count} questions")
    
    def _validate_template(self) -> None:
        """Validate that template has all required questions and options."""
        required_options = {'A', 'B', 'C', 'D'}
        
        for q in range(1, self.config.questions_count + 1):
            if q not in self.template:
                raise ValueError(f"Question {q} missing from template")
            
            template_options = set(self.template[q].keys())
            if not template_options.issuperset(required_options):
                missing = required_options - template_options
                raise ValueError(f"Question {q} missing options: {missing}")
    
    def extract(self, image: np.ndarray) -> OMRResult:
        """
        Extract MCQ answers from a preprocessed exam sheet image.
        
        Args:
            image: Preprocessed binary or grayscale image (numpy array)
                   Should be the output of the preprocessing phase
            
        Returns:
            OMRResult containing:
            - answers: List of 30 selected answers (None for empty)
            - confidences: List of 30 confidence scores (0.0 to 1.0)
            - flags: List of 30 status flags ('OK', 'AMBIGUOUS', 'EMPTY')
            - question_results: Detailed results for each question
            
        Raises:
            ValueError: If input image is None or empty
        """
        start_time = time.time()
        
        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        logger.info(f"Starting OMR extraction for {self.config.questions_count} questions")
        
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Process each question
        question_results: List[OMRQuestionResult] = []
        
        for q_num in range(1, self.config.questions_count + 1):
            result = self._process_question(gray, q_num)
            question_results.append(result)
        
        # Extract lists for compatibility
        answers = [r.selected_answer for r in question_results]
        confidences = [r.confidence for r in question_results]
        flags = [r.flag for r in question_results]
        
        # Calculate statistics
        ok_count = sum(1 for f in flags if f == "OK")
        ambiguous_count = sum(1 for f in flags if f == "AMBIGUOUS")
        empty_count = sum(1 for f in flags if f == "EMPTY")
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"OMR extraction complete: {ok_count} OK, {ambiguous_count} ambiguous, "
            f"{empty_count} empty in {processing_time:.3f}s"
        )
        
        # Save debug visualization if enabled
        if self.config.debug_mode:
            self._save_debug_visualization(image, question_results)
        
        return OMRResult(
            answers=answers,
            confidences=confidences,
            flags=flags,
            question_results=question_results,
            processing_time=processing_time,
            total_questions=self.config.questions_count,
            ambiguous_count=ambiguous_count,
            empty_count=empty_count,
            ok_count=ok_count
        )
    
    def _process_question(self, gray: np.ndarray, question_num: int) -> OMRQuestionResult:
        """
        Process a single question to extract the answer.
        
        Args:
            gray: Grayscale image
            question_num: Question number (1-30)
            
        Returns:
            OMRQuestionResult for the question
        """
        # Get template coordinates for this question
        boxes = self.template[question_num]
        
        # Compute density for each option
        densities: Dict[str, float] = {}
        
        for option in ['A', 'B', 'C', 'D']:
            box = boxes[option]
            density = self._compute_density(gray, box)
            densities[option] = density
        
        # Select answer based on densities
        return self._select_answer(question_num, densities)
    
    def _compute_density(
        self,
        gray: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute pixel density (ratio of dark pixels) in a box region.
        
        Args:
            gray: Grayscale image
            box: (x1, y1, x2, y2) coordinates
            
        Returns:
            Pixel density as a ratio (0.0 to 1.0)
        """
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image bounds
        h, w = gray.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        
        # Handle invalid boxes
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Extract ROI
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0
        
        # Count dark pixels (values below threshold)
        # For binary images: dark = 0, light = 255
        # For grayscale: we use Otsu's threshold or a fixed value
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        total_pixels = binary.size
        dark_pixels = np.sum(binary == 0)  # Count black pixels
        
        density = dark_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return density
    
    def _select_answer(
        self,
        question_num: int,
        densities: Dict[str, float]
    ) -> OMRQuestionResult:
        """
        Select the answer based on pixel densities.
        
        Logic:
        - Select option with max density
        - If max < min_mark_threshold → mark as EMPTY
        - If difference between top two < ambiguity_threshold → mark as AMBIGUOUS
        - Confidence = max_density - second_max_density
        
        Args:
            question_num: Question number
            densities: Dictionary of option -> density
            
        Returns:
            OMRQuestionResult with selected answer, confidence, and flag
        """
        # Sort options by density (descending)
        sorted_options = sorted(densities.items(), key=lambda x: x[1], reverse=True)
        
        max_option, max_density = sorted_options[0]
        second_option, second_max_density = sorted_options[1] if len(sorted_options) > 1 else (None, 0.0)
        
        # Determine flag based on density thresholds
        if max_density < self.config.empty_threshold:
            # No significant mark - empty
            flag = "EMPTY"
            selected_answer = None
            confidence = 0.0
        elif max_density < self.config.min_mark_threshold:
            # Mark present but below threshold - treat as empty
            flag = "EMPTY"
            selected_answer = None
            confidence = 0.0
        else:
            # Significant mark present
            # Check for ambiguity
            density_difference = max_density - second_max_density
            
            if density_difference < self.config.ambiguity_threshold:
                # Too close to second option - ambiguous
                flag = "AMBIGUOUS"
                selected_answer = max_option
                confidence = density_difference / self.config.ambiguity_threshold
                confidence = min(confidence, 1.0)  # Normalize
            else:
                # Clear answer
                flag = "OK"
                selected_answer = max_option
                confidence = density_difference
                # Normalize confidence (cap at 1.0)
                confidence = min(confidence, 1.0)
        
        return OMRQuestionResult(
            question_number=question_num,
            selected_answer=selected_answer,
            confidence=confidence,
            flag=flag,
            densities=densities,
            max_density=max_density,
            second_max_density=second_max_density
        )
    
    def _save_debug_visualization(
        self,
        image: np.ndarray,
        results: List[OMRQuestionResult]
    ) -> None:
        """
        Save debug visualization with bounding boxes and results.
        
        Args:
            image: Original image
            results: List of question results
        """
        if not self.config.debug_output_dir:
            return
        
        try:
            self.config.debug_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a copy for drawing
            if len(image.shape) == 2:
                vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                vis = image.copy()
            
            # Draw boxes for each question
            for result in results:
                q_num = result.question_number
                boxes = self.template[q_num]
                
                for option, box in boxes.items():
                    x1, y1, x2, y2 = box
                    
                    # Color based on result
                    if result.selected_answer == option:
                        if result.flag == "OK":
                            color = (0, 255, 0)  # Green
                        elif result.flag == "AMBIGUOUS":
                            color = (0, 165, 255)  # Orange
                        else:
                            color = (128, 128, 128)  # Gray
                    else:
                        color = (200, 200, 200)  # Light gray
                    
                    # Draw rectangle
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                    
                    # Add label
                    label = f"{option}:{result.densities.get(option, 0):.2f}"
                    cv2.putText(vis, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Save
            filename = f"omr_debug_{self._debug_counter:03d}.png"
            filepath = self.config.debug_output_dir / filename
            cv2.imwrite(str(filepath), vis)
            self._debug_counter += 1
            
            logger.debug(f"Saved OMR debug visualization: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug visualization: {e}")
    
    def get_template(self) -> Dict[int, Dict[str, Tuple[int, int, int, int]]]:
        """Get the current template coordinates."""
        return self.template.copy()
    
    def set_template(
        self,
        template: Dict[int, Dict[str, Tuple[int, int, int, int]]]
    ) -> None:
        """
        Set new template coordinates.
        
        Args:
            template: New template coordinates
        """
        self.template = template
        self._validate_template()
        logger.info("Template updated")


# ============================================================================
# Factory Functions
# ============================================================================

def create_omr_engine(
    questions_count: int = 30,
    min_mark_threshold: float = 0.15,
    ambiguity_threshold: float = 0.10,
    debug_mode: bool = False,
    debug_output_dir: Optional[Path] = None,
    template: Optional[Dict[int, Dict[str, Tuple[int, int, int, int]]]] = None
) -> OMREngine:
    """
    Factory function to create an OMR engine with common settings.
    
    Args:
        questions_count: Number of questions
        min_mark_threshold: Minimum density to consider a mark
        ambiguity_threshold: Threshold for ambiguous detection
        debug_mode: Whether to save debug visualizations
        debug_output_dir: Directory for debug images
        template: Optional custom template
        
    Returns:
        Configured OMREngine instance
    """
    config = OMRConfig(
        questions_count=questions_count,
        min_mark_threshold=min_mark_threshold,
        ambiguity_threshold=ambiguity_threshold,
        debug_mode=debug_mode,
        debug_output_dir=debug_output_dir
    )
    return OMREngine(config, template)


def extract_omr(image: np.ndarray, questions_count: int = 30) -> OMRResult:
    """
    Convenience function to extract OMR answers with default settings.
    
    Args:
        image: Preprocessed image
        questions_count: Number of questions
        
    Returns:
        OMRResult
    """
    engine = create_omr_engine(questions_count=questions_count)
    return engine.extract(image)