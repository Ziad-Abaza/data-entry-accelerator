"""
Status Panel widget for the Turbo Data Entry Dashboard.

Displays:
- Current student ID
- OCR confidence
- OMR status (GREEN/YELLOW/RED)
- Duplicate ID warnings
- Queue status
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPalette, QColor
from typing import Optional


class StatusPanel(QWidget):
    """
    Status panel displaying current processing state and indicators.
    
    Signals:
        status_changed: Emitted when status changes
    """
    
    # Status colors
    COLOR_GREEN = "#28a745"
    COLOR_YELLOW = "#ffc107"
    COLOR_RED = "#dc3545"
    COLOR_GRAY = "#6c757d"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        
        # Create status groups
        # Queue status
        self._queue_label = self._create_label("Queue: 0/0")
        main_layout.addWidget(self._queue_label)
        
        main_layout.addWidget(self._create_separator())
        
        # Student ID
        self._id_label = self._create_label("ID: --")
        main_layout.addWidget(self._id_label)
        
        main_layout.addWidget(self._create_separator())
        
        # OCR Confidence
        self._ocr_label = self._create_label("OCR: --")
        main_layout.addWidget(self._ocr_label)
        
        main_layout.addWidget(self._create_separator())
        
        # OMR Status indicator
        self._omr_indicator = QFrame()
        self._omr_indicator.setFixedSize(20, 20)
        self._omr_indicator.setStyleSheet(f"background-color: {self.COLOR_GRAY}; border-radius: 10px;")
        main_layout.addWidget(QLabel("OMR:"))
        main_layout.addWidget(self._omr_indicator)
        
        main_layout.addWidget(self._create_separator())
        
        # Current field
        self._field_label = self._create_label("Field: --")
        main_layout.addWidget(self._field_label)
        
        main_layout.addWidget(self._create_separator())
        
        # Question progress
        self._progress_label = self._create_label("Q: 0/30")
        main_layout.addWidget(self._progress_label)
        
        # Stretch to fill
        main_layout.addStretch()
        
        # Set background
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }
            QLabel {
                color: #212529;
                font-weight: bold;
            }
        """)
    
    def _create_label(self, text: str) -> QLabel:
        """Create a styled label."""
        label = QLabel(text)
        label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        return label
    
    def _create_separator(self) -> QFrame:
        """Create a vertical separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #dee2e6;")
        return sep
    
    def update_queue(self, pending: int, processed: int) -> None:
        """Update queue status."""
        self._queue_label.setText(f"Queue: {pending}/{processed}")
    
    def update_student_id(self, academic_id: str) -> None:
        """Update current student ID."""
        self._id_label.setText(f"ID: {academic_id or '--'}")
    
    def update_ocr_confidence(self, confidence: float) -> None:
        """Update OCR confidence display."""
        if confidence > 0:
            self._ocr_label.setText(f"OCR: {confidence:.0%}")
        else:
            self._ocr_label.setText("OCR: --")
    
    def update_omr_status(self, status: str) -> None:
        """
        Update OMR status indicator.
        
        Args:
            status: 'GREEN', 'YELLOW', 'RED', or 'GRAY'
        """
        color_map = {
            "GREEN": self.COLOR_GREEN,
            "YELLOW": self.COLOR_YELLOW,
            "RED": self.COLOR_RED,
            "GRAY": self.COLOR_GRAY,
        }
        color = color_map.get(status.upper(), self.COLOR_GRAY)
        self._omr_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 10px;"
        )
    
    def update_current_field(self, field_name: str) -> None:
        """Update current field display."""
        self._field_label.setText(f"Field: {field_name}")
    
    def update_progress(self, current: int, total: int) -> None:
        """Update question progress."""
        self._progress_label.setText(f"Q: {current}/{total}")
    
    def show_duplicate_warning(self, show: bool) -> None:
        """Show/hide duplicate ID warning."""
        if show:
            self._id_label.setStyleSheet("color: #dc3545;")
            self._id_label.setText("ID: DUPLICATE!")
        else:
            self._id_label.setStyleSheet("")
    
    def clear(self) -> None:
        """Reset all displays to default."""
        self._queue_label.setText("Queue: 0/0")
        self._id_label.setText("ID: --")
        self._id_label.setStyleSheet("")
        self._ocr_label.setText("OCR: --")
        self._omr_indicator.setStyleSheet(f"background-color: {self.COLOR_GRAY}; border-radius: 10px;")
        self._field_label.setText("Field: --")
        self._progress_label.setText("Q: 0/30")