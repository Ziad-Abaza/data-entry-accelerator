"""
Image Viewer widget for the Turbo Data Entry Dashboard.

Displays:
- Cropped image snippets from CropEngine
- Zoom in/out support
- Auto-update on field change
- High-contrast display for readability
"""

import numpy as np
import cv2
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from typing import Optional


class ImageViewer(QWidget):
    """
    High-performance image viewer for displaying cropped exam sections.
    
    Features:
    - Fast numpy to Qt image conversion
    - Zoom controls
    - Field label display
    - Optional overlay for annotations
    
    Signals:
        zoom_changed: Emitted when zoom level changes
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._zoom_level: float = 1.0
        self._min_zoom: float = 0.5
        self._max_zoom: float = 3.0
        self._current_field: str = ""
        self._current_image: Optional[np.ndarray] = None
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Field label
        self._field_label = QLabel("No Image")
        self._field_label.setAlignment(Qt.AlignCenter)
        self._field_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._field_label.setStyleSheet("""
            QLabel {
                background-color: #343a40;
                color: white;
                padding: 8px;
                border-radius: 4px 4px 0 0;
            }
        """)
        layout.addWidget(self._field_label)
        
        # Scroll area for zooming
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(False)
        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #212529;
                border: none;
            }
            QScrollBar:vertical {
                background: #343a40;
            }
            QScrollBar::handle:vertical {
                background: #6c757d;
            }
        """)
        layout.addWidget(self._scroll_area)
        
        # Image label
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(400, 200)
        self._scroll_area.setWidget(self._image_label)
        
        # Zoom indicator
        self._zoom_label = QLabel("100%")
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setStyleSheet("""
            QLabel {
                background-color: #343a40;
                color: #adb5bd;
                padding: 4px;
                border-radius: 0 0 4px 4px;
            }
        """)
        layout.addWidget(self._zoom_label)
        
        # Set minimum size
        self.setMinimumWidth(450)
        self.setMinimumHeight(350)
    
    def set_image(self, image: Optional[np.ndarray], field_name: str = "") -> None:
        """
        Set the displayed image.
        
        Args:
            image: numpy array (BGR or grayscale) or None
            field_name: Name of the field being displayed
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"set_image called: field={field_name}, image={'None' if image is None else f'shape={image.shape}'}")
        
        self._current_field = field_name
        self._current_image = image
        
        # Update field label
        self._field_label.setText(field_name or "No Image")
        
        if image is None or image.size == 0:
            logger.debug("Image is None or empty, showing placeholder")
            self._image_label.setText("No Image")
            self._image_label.setPixmap(QPixmap())
            return
        
        # Convert numpy array to QPixmap
        pixmap = self._numpy_to_pixmap(image)
        
        if pixmap:
            logger.debug(f"Got pixmap: {pixmap.width()}x{pixmap.height()}")
            # Apply zoom
            scaled_size = QSize(
                int(pixmap.width() * self._zoom_level),
                int(pixmap.height() * self._zoom_level)
            )
            scaled_pixmap = pixmap.scaled(
                scaled_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self._image_label.setPixmap(scaled_pixmap)
            self._image_label.resize(scaled_size)
            logger.debug(f"Set pixmap to label: {scaled_size}")
        else:
            logger.warning("Pixmap conversion failed!")
            self._image_label.setText("Conversion Error")
    
    def _numpy_to_pixmap(self, image: np.ndarray) -> Optional[QPixmap]:
        """
        Convert numpy array to QPixmap.
        
        Args:
            image: numpy array (BGR or grayscale)
            
        Returns:
            QPixmap or None
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.debug(f"Converting image: shape={image.shape}, dtype={image.dtype}")
            
            # Handle grayscale
            if len(image.shape) == 2:
                h, w = image.shape
                bytes_per_line = w
                # Use .copy() to ensure data persists after numpy array is freed
                q_image = QImage(image.copy().data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            else:
                # Handle BGR (OpenCV default)
                h, w, ch = image.shape
                bytes_per_line = w * ch
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Use .copy() to ensure data persists
                q_image = QImage(rgb_image.copy().data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            logger.debug(f"QImage created: {q_image.width()}x{q_image.height()}")
            
            pixmap = QPixmap.fromImage(q_image)
            logger.debug(f"QPixmap created: {pixmap.width()}x{pixmap.height()}")
            return pixmap
            
        except Exception as e:
            print(f"Image conversion error: {e}")
            return None
    
    def zoom_in(self) -> None:
        """Increase zoom level."""
        if self._zoom_level < self._max_zoom:
            self._zoom_level = min(self._zoom_level + 0.25, self._max_zoom)
            self._update_zoom_display()
            self._refresh_image()
    
    def zoom_out(self) -> None:
        """Decrease zoom level."""
        if self._zoom_level > self._min_zoom:
            self._zoom_level = max(self._zoom_level - 0.25, self._min_zoom)
            self._update_zoom_display()
            self._refresh_image()
    
    def reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self._zoom_level = 1.0
        self._update_zoom_display()
        self._refresh_image()
    
    def _update_zoom_display(self) -> None:
        """Update zoom level display."""
        self._zoom_label.setText(f"{int(self._zoom_level * 100)}%")
    
    def _refresh_image(self) -> None:
        """Refresh the current image with new zoom."""
        if self._current_image is not None:
            self.set_image(self._current_image, self._current_field)
    
    def get_zoom_level(self) -> float:
        """Get current zoom level."""
        return self._zoom_level
    
    def clear(self) -> None:
        """Clear the viewer."""
        self.set_image(None, "No Image")
        self._zoom_label.setText("100%")
        self._zoom_level = 1.0
    
    def keyPressEvent(self, event):
        """Handle keyboard events for zoom."""
        from PySide6.QtCore import Qt
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_in()
        elif event.key() == Qt.Key_Minus:
            self.zoom_out()
        elif event.key() == Qt.Key_0:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)