"""
Main Window for the Turbo Data Entry Dashboard.

Integrates:
- Status Panel (queue, ID, OCR, OMR status)
- Image Viewer (cropped snippets)
- Data Table (real-time spreadsheet)
- Keyboard controller (A/B/C/D navigation)
- Pipeline integration
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QFileDialog,
    QStatusBar, QMenuBar, QMenu, QToolBar, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QAction, QKeyEvent, QFont

from app.ui.widgets.status_panel import StatusPanel
from app.ui.widgets.image_viewer import ImageViewer
from app.ui.widgets.data_table import DataTable

# Import pipeline components
from app.services import PipelineService, ExportService
from app.core.preprocessing import ImagePreprocessor, PreprocessingConfig
from app.core.omr import OMREngine, OMRConfig
from app.core.ocr import OCREngine, OCRConfig
from app.core.cropping import CropEngine, CropConfig
from app.models import ExamRecord, RecordStatus
from config import get_config


# Field order for turbo input workflow
FIELD_ORDER = [
    ("student_name", "Student Name"),
    ("academic_id", "Academic ID"),
    ("q2", "Question 2"),
    ("q3", "Question 3"),
    ("q4", "Question 4"),
]


@dataclass
class ProcessingTask:
    """Represents a single image processing task."""
    image_path: Path
    status: str = "pending"  # pending, processing, completed, failed
    record: Optional[ExamRecord] = None
    error: Optional[str] = None


class ProcessingWorker(QThread):
    """
    Background worker for processing images without blocking UI.
    
    Signals:
        finished: Processing completed for one image
        progress: Progress update (current, total)
        error: Error occurred
    """
    
    finished = Signal(object)  # ProcessingTask
    progress = Signal(int, int)  # current, total
    error = Signal(str, str)  # image_path, error_message
    
    def __init__(self, tasks: List[ProcessingTask], parent=None):
        super().__init__(parent)
        self._tasks = tasks
        self._running = True
        
        # Initialize pipeline components
        self._preprocessor = ImagePreprocessor(PreprocessingConfig(master_height=2000))
        self._omr_engine = OMREngine(OMRConfig())
        self._ocr_engine = OCREngine(OCRConfig())
        self._crop_engine = CropEngine(CropConfig(target_width=400))
    
    def run(self) -> None:
        """Process all tasks in background."""
        for i, task in enumerate(self._tasks):
            if not self._running:
                break
            
            self.progress.emit(i + 1, len(self._tasks))
            
            try:
                # Load image
                import cv2
                image = cv2.imread(str(task.image_path))
                
                if image is None:
                    raise ValueError(f"Failed to load image: {task.image_path}")
                
                # Run pipeline
                # 1. Preprocess
                processed = self._preprocessor.process(image)
                
                # 2. OMR extraction
                omr_result = self._omr_engine.extract(processed)
                
                # 3. OCR extraction
                ocr_result = self._ocr_engine.extract_academic_id(processed)
                
                # 4. Crop extraction
                crops = self._crop_engine.extract_all(processed)
                
                # Create exam record
                record = ExamRecord(
                    student_name="",  # Will be filled from UI
                    academic_id=ocr_result.text,
                    mcq_answers=omr_result.question_results,
                    source_image_path=str(task.image_path),
                    id_confidence=ocr_result.confidence,
                    crops=crops.to_dict()
                )
                
                # Determine status based on confidence
                if ocr_result.valid and omr_result.ok_count >= 25:
                    record.status = RecordStatus.COMPLETED
                    record.overall_confidence = min(ocr_result.confidence, 0.8)
                elif ocr_result.valid:
                    record.status = RecordStatus.REVIEW_REQUIRED
                    record.overall_confidence = min(ocr_result.confidence, 0.5)
                else:
                    record.status = RecordStatus.REVIEW_REQUIRED
                    record.overall_confidence = 0.3
                
                task.record = record
                task.status = "completed"
                self.finished.emit(task)
                
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                self.error.emit(str(task.image_path), str(e))
                self.finished.emit(task)
    
    def stop(self) -> None:
        """Stop processing."""
        self._running = False


class MainWindow(QMainWindow):
    """
    Main window for the Turbo Data Entry Dashboard.
    
    Features:
    - Real-time image processing in background thread
    - Keyboard-driven MCQ input (A/B/C/D)
    - Auto-advance between fields
    - Live data table updates
    - Status indicators (GREEN/YELLOW/RED)
    """
    
    def __init__(self):
        super().__init__()
        
        # Configuration
        self._config = get_config()
        
        # Pipeline services
        self._pipeline = PipelineService()
        self._export_service = ExportService()
        
        # Processing state
        self._tasks: List[ProcessingTask] = []
        self._current_task_index: int = -1
        self._current_record: Optional[ExamRecord] = None
        self._worker: Optional[ProcessingWorker] = None
        
        # UI state
        self._current_field_index: int = 0
        self._session_ids: set = set()
        
        # Initialize UI
        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        
        # Show welcome state
        self._update_ui_for_state("empty")
    
    def _init_ui(self) -> None:
        """Initialize the main UI layout."""
        self.setWindowTitle("Vision-Accelerated Exam Data Entry - Turbo Dashboard")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Status panel (TOP)
        self._status_panel = StatusPanel()
        main_layout.addWidget(self._status_panel)
        
        # Center area (Image Viewer + Controls)
        center_layout = QHBoxLayout()
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(10)
        
        # Image viewer (LEFT)
        self._image_viewer = ImageViewer()
        center_layout.addWidget(self._image_viewer, 2)
        
        # Control panel (RIGHT)
        control_panel = self._create_control_panel()
        center_layout.addWidget(control_panel, 1)
        
        main_layout.addLayout(center_layout)
        
        # Data table (BOTTOM)
        self._data_table = DataTable()
        main_layout.addWidget(self._data_table, 1)
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready - Load images to begin")
    
    def _create_control_panel(self) -> QWidget:
        """Create the right-side control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Controls")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # MCQ buttons
        mcq_label = QLabel("MCQ Input (A/B/C/D)")
        mcq_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(mcq_label)
        
        mcq_layout = QHBoxLayout()
        for opt in ["A", "B", "C", "D"]:
            btn = QPushButton(opt)
            btn.setFixedSize(50, 50)
            btn.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #007bff;
                    color: white;
                    border-radius: 8px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
                QPushButton:pressed {
                    background-color: #004085;
                }
            """)
            btn.clicked.connect(lambda checked, o=opt: self._on_mcq_select(o))
            mcq_layout.addWidget(btn)
        
        layout.addLayout(mcq_layout)
        
        # Navigation buttons
        nav_label = QLabel("Navigation")
        nav_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(nav_label)
        
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("← Prev")
        prev_btn.clicked.connect(self._on_prev_field)
        nav_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("Next →")
        next_btn.clicked.connect(self._on_next_field)
        nav_layout.addWidget(next_btn)
        
        layout.addLayout(nav_layout)
        
        # Current answer display
        answer_label = QLabel("Current Answer:")
        answer_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(answer_label)
        
        self._current_answer_label = QLabel("-")
        self._current_answer_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self._current_answer_label.setAlignment(Qt.AlignCenter)
        self._current_answer_label.setStyleSheet("""
            QLabel {
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self._current_answer_label)
        
        # Zoom controls
        zoom_label = QLabel("Zoom")
        zoom_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(zoom_label)
        
        zoom_layout = QHBoxLayout()
        
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedSize(30, 30)
        zoom_out_btn.clicked.connect(self._image_viewer.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        
        zoom_reset_btn = QPushButton("100%")
        zoom_reset_btn.setFixedSize(50, 30)
        zoom_reset_btn.clicked.connect(self._image_viewer.reset_zoom)
        zoom_layout.addWidget(zoom_reset_btn)
        
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(30, 30)
        zoom_in_btn.clicked.connect(self._image_viewer.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        
        layout.addLayout(zoom_layout)
        
        layout.addStretch()
        
        return panel
    
    def _init_menu(self) -> None:
        """Initialize the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Images...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_images)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction("Open &Folder...", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export to Excel...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export_excel)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Process menu
        process_menu = menubar.addMenu("&Process")
        
        process_action = QAction("&Start Processing", self)
        process_action.setShortcut("F5")
        process_action.triggered.connect(self._on_start_processing)
        process_menu.addAction(process_action)
        
        clear_action = QAction("&Clear Session", self)
        clear_action.triggered.connect(self._on_clear_session)
        process_menu.addAction(clear_action)
    
    def _init_toolbar(self) -> None:
        """Initialize the toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Open button
        open_btn = QPushButton("Open Images")
        open_btn.clicked.connect(self._on_open_images)
        toolbar.addWidget(open_btn)
        
        # Process button
        self._process_btn = QPushButton("Process")
        self._process_btn.setEnabled(False)
        self._process_btn.clicked.connect(self._on_start_processing)
        toolbar.addWidget(self._process_btn)
        
        toolbar.addSeparator()
        
        # Export button
        self._export_btn = QPushButton("Export")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_excel)
        toolbar.addWidget(self._export_btn)
    
    def _update_ui_for_state(self, state: str) -> None:
        """Update UI based on current state."""
        if state == "empty":
            self._image_viewer.clear()
            self._status_panel.clear()
            self._process_btn.setEnabled(False)
            self._export_btn.setEnabled(False)
        elif state == "ready":
            self._process_btn.setEnabled(True)
        elif state == "processing":
            self._process_btn.setEnabled(False)
        elif state == "record_ready":
            self._export_btn.setEnabled(True)
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def _on_open_images(self) -> None:
        """Handle open images action."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Exam Sheet Images",
            str(self._config.directories.input_dir),
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if files:
            self._load_images([Path(f) for f in files])
    
    def _on_open_folder(self) -> None:
        """Handle open folder action."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Exam Sheets",
            str(self._config.directories.input_dir)
        )
        
        if directory:
            from app.utils import get_image_files
            image_paths = get_image_files(Path(directory))
            self._load_images(image_paths)
    
    def _load_images(self, paths: List[Path]) -> None:
        """Load images into the processing queue."""
        self._tasks = [ProcessingTask(path=path) for path in paths]
        self._current_task_index = 0
        
        self._status_panel.update_queue(len(self._tasks), 0)
        self._status_bar.showMessage(f"Loaded {len(self._tasks)} images")
        self._update_ui_for_state("ready")
        
        # Auto-start processing
        self._on_start_processing()
    
    def _on_start_processing(self) -> None:
        """Start processing images in background."""
        if not self._tasks:
            return
        
        self._update_ui_for_state("processing")
        self._status_bar.showMessage("Processing images...")
        
        # Create worker
        self._worker = ProcessingWorker(self._tasks)
        self._worker.finished.connect(self._on_task_finished)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_task_error)
        self._worker.start()
    
    def _on_task_finished(self, task: ProcessingTask) -> None:
        """Handle task completion."""
        # Update queue display
        completed = sum(1 for t in self._tasks if t.status == "completed")
        self._status_panel.update_queue(
            len(self._tasks) - completed,
            completed
        )
        
        if task.status == "completed" and task.record:
            # Load the first completed record
            if self._current_task_index < 0:
                self._current_task_index = self._tasks.index(task)
                self._load_current_record()
    
    def _on_progress(self, current: int, total: int) -> None:
        """Handle progress update."""
        self._status_bar.showMessage(f"Processing: {current}/{total}")
    
    def _on_task_error(self, image_path: str, error: str) -> None:
        """Handle task error."""
        self._status_bar.showMessage(f"Error processing {image_path}: {error}")
    
    def _load_current_record(self) -> None:
        """Load the current record for display."""
        if 0 <= self._current_task_index < len(self._tasks):
            task = self._tasks[self._current_task_index]
            if task.status == "completed" and task.record:
                self._current_record = task.record
                self._current_field_index = 0
                self._display_current_field()
                self._update_ui_for_state("record_ready")
    
    def _display_current_field(self) -> None:
        """Display the current field in the image viewer."""
        if not self._current_record:
            return
        
        field_name, field_label = FIELD_ORDER[self._current_field_index]
        
        # Get crop image
        crops = self._current_record.crops
        if field_name in crops:
            crop_image = crops[field_name]
            if hasattr(crop_image, 'get'):
                crop_image = crop_image.get(field_name)
        else:
            crop_image = None
        
        # Update viewer
        self._image_viewer.set_image(crop_image, field_label)
        
        # Update status panel
        self._status_panel.update_current_field(field_label)
        self._status_panel.update_student_id(self._current_record.academic_id)
        self._status_panel.update_ocr_confidence(self._current_record.id_confidence)
        
        # Update progress
        self._status_panel.update_progress(
            self._current_field_index + 1,
            len(FIELD_ORDER)
        )
        
        # Determine OMR status
        if self._current_record.overall_confidence >= 0.7:
            self._status_panel.update_omr_status("GREEN")
        elif self._current_record.overall_confidence >= 0.4:
            self._status_panel.update_omr_status("YELLOW")
        else:
            self._status_panel.update_omr_status("RED")
        
        # Update current answer display
        if field_name.startswith("q"):
            # MCQ field - show answer
            q_num = int(field_name[1])
            answer = self._current_record.get_mcq_answer(q_num)
            self._current_answer_label.setText(answer or "-")
        else:
            # Text field
            self._current_answer_label.setText("-")
    
    def _on_mcq_select(self, option: str) -> None:
        """Handle MCQ selection (A/B/C/D)."""
        if not self._current_record:
            return
        
        # Save answer
        q_num = self._current_field_index + 1
        self._current_record.set_mcq_answer(q_num, option)
        
        # Update display
        self._current_answer_label.setText(option)
        
        # Auto-advance
        if self._current_field_index < len(FIELD_ORDER) - 1:
            self._on_next_field()
    
    def _on_prev_field(self) -> None:
        """Navigate to previous field."""
        if self._current_field_index > 0:
            self._current_field_index -= 1
            self._display_current_field()
    
    def _on_next_field(self) -> None:
        """Navigate to next field."""
        if self._current_field_index < len(FIELD_ORDER) - 1:
            self._current_field_index += 1
            self._display_current_field()
        else:
            # All fields done - finalize record
            self._finalize_record()
    
    def _finalize_record(self) -> None:
        """Finalize the current record and add to table."""
        if not self._current_record:
            return
        
        # Check for duplicate ID
        academic_id = self._current_record.academic_id
        if academic_id in self._session_ids:
            self._status_panel.show_duplicate_warning(True)
            QMessageBox.warning(
                self,
                "Duplicate ID",
                f"Academic ID {academic_id} already exists in this session!"
            )
        
        self._session_ids.add(academic_id)
        
        # Add to data table
        row_data = self._current_record.to_dict()
        self._data_table.add_row(row_data)
        
        # Move to next record
        self._current_task_index += 1
        if self._current_task_index < len(self._tasks):
            # Find next completed task
            while self._current_task_index < len(self._tasks):
                task = self._tasks[self._current_task_index]
                if task.status == "completed":
                    self._load_current_record()
                    break
                self._current_task_index += 1
        else:
            # No more records
            self._status_bar.showMessage("All records processed!")
            self._update_ui_for_state("ready")
    
    def _on_export_excel(self) -> None:
        """Export data to Excel."""
        if self._data_table.row_count() == 0:
            QMessageBox.information(self, "Info", "No data to export")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to Excel",
            str(self._config.directories.output_dir / "exam_data.xlsx"),
            "Excel Files (*.xlsx)"
        )
        
        if path:
            df = self._data_table.to_dataframe()
            success = self._export_service.to_excel(df, Path(path))
            if success:
                QMessageBox.information(self, "Success", f"Exported to {path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to export")
    
    def _on_clear_session(self) -> None:
        """Clear the current session."""
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Clear all processed records?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self._tasks.clear()
            self._current_task_index = -1
            self._current_record = None
            self._session_ids.clear()
            self._data_table.clear()
            self._image_viewer.clear()
            self._status_panel.clear()
            self._update_ui_for_state("empty")
            self._status_bar.showMessage("Session cleared")
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard events for fast input."""
        key = event.key()
        
        # MCQ quick keys
        if key == Qt.Key_A:
            self._on_mcq_select("A")
        elif key == Qt.Key_B:
            self._on_mcq_select("B")
        elif key == Qt.Key_C:
            self._on_mcq_select("C")
        elif key == Qt.Key_D:
            self._on_mcq_select("D")
        # Navigation
        elif key == Qt.Key_Left:
            self._on_prev_field()
        elif key == Qt.Key_Right:
            self._on_next_field()
        # Zoom
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self._image_viewer.zoom_in()
        elif key == Qt.Key_Minus:
            self._image_viewer.zoom_out()
        elif key == Qt.Key_0:
            self._image_viewer.reset_zoom()
        else:
            super().keyPressEvent(event)


def run_dashboard() -> int:
    """Run the dashboard application."""
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Vision Exam Data Entry")
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_dashboard())