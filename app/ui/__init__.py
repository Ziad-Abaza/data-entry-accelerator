"""
UI module for the Vision-Accelerated Exam Data Entry System.

This module contains the PySide6-based graphical user interface components
including the main window, widgets, and event handlers.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QStatusBar, QMenuBar, QMenu, QToolBar, QProgressBar,
    QScrollArea, QGroupBox, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QAction, QKeyEvent, QPixmap, QImage

from app.models import ExamRecord, RecordStatus, ProcessingResult
from app.services import PipelineService, ExportService, QueueService
from config import get_config

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for the Exam Data Entry System.
    
    Provides the primary UI for:
    - Image queue management
    - Visual verification (turbo-input view)
    - Data table display
    - Export functionality
    """
    
    def __init__(self):
        super().__init__()
        
        self.config = get_config().ui
        self.pipeline_service = PipelineService()
        self.export_service = ExportService()
        self.queue_service = QueueService()
        
        self._current_record: Optional[ExamRecord] = None
        self._current_index: int = 0
        self._records: List[ExamRecord] = []
        
        self._init_ui()
        self._connect_signals()
    
    def _init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle(self.config.window_title)
        self.setGeometry(100, 100, self.config.window_width, self.config.window_height)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add menu bar
        self._create_menu_bar()
        
        # Add toolbar
        self._create_toolbar()
        
        # Scanning pane (queue management)
        self._scanning_pane = self._create_scanning_pane()
        main_layout.addWidget(self._scanning_pane, 1)
        
        # Turbo-input view (visual focus)
        self._turbo_view = self._create_turbo_view()
        main_layout.addWidget(self._turbo_view, 2)
        
        # Data table
        self._data_table = self._create_data_table()
        main_layout.addWidget(self._data_table, 1)
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)
    
    def _create_menu_bar(self) -> None:
        """Create the application menu bar."""
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
        
        export_excel_action = QAction("Export to &Excel...", self)
        export_excel_action.setShortcut("Ctrl+E")
        export_excel_action.triggered.connect(self._on_export_excel)
        file_menu.addAction(export_excel_action)
        
        export_csv_action = QAction("Export to &CSV...", self)
        export_csv_action.setShortcut("Ctrl+Shift+E")
        export_csv_action.triggered.connect(self._on_export_csv)
        file_menu.addAction(export_csv_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Process menu
        process_menu = menubar.addMenu("&Process")
        
        process_action = QAction("&Process Queue", self)
        process_action.setShortcut("F5")
        process_action.triggered.connect(self._on_process_queue)
        process_menu.addAction(process_action)
        
        clear_action = QAction("&Clear Session", self)
        clear_action.triggered.connect(self._on_clear_session)
        process_menu.addAction(clear_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self) -> None:
        """Create the application toolbar."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Open button
        self._btn_open = QPushButton("Open Images")
        self._btn_open.clicked.connect(self._on_open_images)
        toolbar.addWidget(self._btn_open)
        
        # Process button
        self._btn_process = QPushButton("Process")
        self._btn_process.clicked.connect(self._on_process_queue)
        self._btn_process.setEnabled(False)
        toolbar.addWidget(self._btn_process)
        
        toolbar.addSeparator()
        
        # Previous button
        self._btn_prev = QPushButton("Previous")
        self._btn_prev.clicked.connect(self._on_previous)
        self._btn_prev.setEnabled(False)
        toolbar.addWidget(self._btn_prev)
        
        # Next button
        self._btn_next = QPushButton("Next")
        self._btn_next.clicked.connect(self._on_next)
        self._btn_next.setEnabled(False)
        toolbar.addWidget(self._btn_next)
        
        toolbar.addSeparator()
        
        # Export button
        self._btn_export = QPushButton("Export")
        self._btn_export.clicked.connect(self._on_export_excel)
        self._btn_export.setEnabled(False)
        toolbar.addWidget(self._btn_export)
    
    def _create_scanning_pane(self) -> QWidget:
        """Create the scanning pane (queue management)."""
        group = QGroupBox("Processing Queue")
        layout = QVBoxLayout()
        
        # Queue info label
        self._queue_label = QLabel("No images loaded")
        self._queue_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._queue_label)
        
        group.setLayout(layout)
        return group
    
    def _create_turbo_view(self) -> QWidget:
        """Create the turbo-input view (visual focus)."""
        group = QGroupBox("Turbo-Input View")
        layout = QHBoxLayout()
        
        # Left: Current field image
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Current Field:"))
        self._field_image_label = QLabel()
        self._field_image_label.setMinimumSize(300, 200)
        self._field_image_label.setFrameStyle(QFrame.StyledPanel)
        self._field_image_label.setAlignment(Qt.AlignCenter)
        self._field_image_label.setText("No image")
        left_layout.addWidget(self._field_image_label)
        
        # Right: Record info
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Record Info:"))
        
        self._record_info_label = QLabel("No record selected")
        self._record_info_label.setFrameStyle(QFrame.StyledPanel)
        right_layout.addWidget(self._record_info_label)
        
        # MCQ buttons
        mcq_layout = QHBoxLayout()
        for opt in ["A", "B", "C", "D"]:
            btn = QPushButton(opt)
            btn.setFixedSize(50, 50)
            btn.clicked.connect(lambda checked, o=opt: self._on_mcq_select(o))
            mcq_layout.addWidget(btn)
        self._mcq_buttons = mcq_layout
        right_layout.addLayout(mcq_layout)
        
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_data_table(self) -> QTableWidget:
        """Create the data table widget."""
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["ID", "Student Name", "Academic ID", "Status", "Confidence", "Q1"])
        table.setMinimumHeight(150)
        return table
    
    def _connect_signals(self) -> None:
        """Connect internal signals and slots."""
        pass
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    @Slot()
    def _on_open_images(self) -> None:
        """Handle open images action."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Exam Sheet Images",
            str(get_config().directories.input_dir),
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if files:
            self._load_images([Path(f) for f in files])
    
    @Slot()
    def _on_open_folder(self) -> None:
        """Handle open folder action."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Exam Sheets",
            str(get_config().directories.input_dir)
        )
        
        if directory:
            self._load_images([Path(directory)])
    
    def _load_images(self, paths: List[Path]) -> None:
        """Load images into the queue."""
        # Handle both files and directories
        image_paths = []
        for path in paths:
            if path.is_dir():
                from app.utils import get_image_files
                image_paths.extend(get_image_files(path))
            elif path.is_file():
                image_paths.append(path)
        
        if image_paths:
            self.queue_service.add_batch(image_paths)
            self._update_queue_display()
            self._btn_process.setEnabled(True)
            self._status_bar.showMessage(f"Loaded {len(image_paths)} images")
    
    @Slot()
    def _on_process_queue(self) -> None:
        """Process the image queue."""
        pending = self.queue_service.get_pending()
        if not pending:
            QMessageBox.information(self, "Info", "No images in queue")
            return
        
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(len(pending))
        self._progress_bar.setValue(0)
        
        self._records.clear()
        
        for i, path in enumerate(pending):
            result = self.pipeline_service.process_image(path)
            
            if result.success and result.record:
                self._records.append(result.record)
                self.queue_service.mark_processed(path)
            
            self._progress_bar.setValue(i + 1)
        
        self._progress_bar.setVisible(False)
        self._update_data_table()
        self._update_queue_display()
        
        if self._records:
            self._current_index = 0
            self._display_record(self._records[0])
            self._btn_export.setEnabled(True)
        
        stats = self.pipeline_service.get_stats()
        self._status_bar.showMessage(
            f"Processed: {stats.successful}/{stats.total_processed} "
            f"({stats.success_rate:.1f}% success)"
        )
    
    @Slot()
    def _on_previous(self) -> None:
        """Navigate to previous record."""
        if self._current_index > 0:
            self._current_index -= 1
            self._display_record(self._records[self._current_index])
    
    @Slot()
    def _on_next(self) -> None:
        """Navigate to next record."""
        if self._current_index < len(self._records) - 1:
            self._current_index += 1
            self._display_record(self._records[self._current_index])
    
    @Slot()
    def _on_mcq_select(self, option: str) -> None:
        """Handle MCQ option selection."""
        if self._current_record and self._current_index < 30:
            self._current_record.set_mcq_answer(self._current_index + 1, option)
            self._update_record_display()
            
            # Auto-advance if enabled
            if self.config.auto_advance and self._current_index < 29:
                self._on_next()
    
    @Slot()
    def _on_export_excel(self) -> None:
        """Export to Excel."""
        if not self._records:
            QMessageBox.information(self, "Info", "No records to export")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to Excel",
            str(get_config().directories.output_dir / "exam_data.xlsx"),
            "Excel Files (*.xlsx)"
        )
        
        if path:
            success = self.export_service.to_excel(self._records, Path(path))
            if success:
                QMessageBox.information(self, "Success", f"Exported to {path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to export")
    
    @Slot()
    def _on_export_csv(self) -> None:
        """Export to CSV."""
        if not self._records:
            QMessageBox.information(self, "Info", "No records to export")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to CSV",
            str(get_config().directories.output_dir / "exam_data.csv"),
            "CSV Files (*.csv)"
        )
        
        if path:
            success = self.export_service.to_csv(self._records, Path(path))
            if success:
                QMessageBox.information(self, "Success", f"Exported to {path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to export")
    
    @Slot()
    def _on_clear_session(self) -> None:
        """Clear the current session."""
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Clear all processed records?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.pipeline_service.clear_session()
            self.queue_service.clear()
            self._records.clear()
            self._current_record = None
            self._current_index = 0
            self._update_queue_display()
            self._update_data_table()
            self._clear_turbo_view()
            self._btn_export.setEnabled(False)
            self._status_bar.showMessage("Session cleared")
    
    @Slot()
    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About",
            "Vision-Accelerated Exam Data Entry System\n\n"
            "Version 1.0.0\n\n"
            "A computer vision system for automated exam data extraction."
        )
    
    # =========================================================================
    # UI Update Methods
    # =========================================================================
    
    def _update_queue_display(self) -> None:
        """Update the queue display."""
        pending = self.queue_service.pending_count
        processed = self.queue_service.processed_count
        failed = self.queue_service.failed_count
        
        self._queue_label.setText(
            f"Pending: {pending} | Processed: {processed} | Failed: {failed}"
        )
        
        self._btn_prev.setEnabled(len(self._records) > 0 and self._current_index > 0)
        self._btn_next.setEnabled(
            len(self._records) > 0 and self._current_index < len(self._records) - 1
        )
    
    def _update_data_table(self) -> None:
        """Update the data table with current records."""
        self._data_table.setRowCount(len(self._records))
        
        for i, record in enumerate(self._records):
            self._data_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._data_table.setItem(i, 1, QTableWidgetItem(record.student_name))
            self._data_table.setItem(i, 2, QTableWidgetItem(record.academic_id))
            self._data_table.setItem(i, 3, QTableWidgetItem(record.status.value))
            self._data_table.setItem(
                i, 4, QTableWidgetItem(f"{record.overall_confidence:.0%}")
            )
            self._data_table.setItem(
                i, 5, QTableWidgetItem(record.get_mcq_answer(1) or "-")
            )
    
    def _display_record(self, record: ExamRecord) -> None:
        """Display a record in the turbo view."""
        self._current_record = record
        self._update_record_display()
    
    def _update_record_display(self) -> None:
        """Update the turbo view with current record."""
        if not self._current_record:
            return
        
        record = self._current_record
        
        # Update record info
        self._record_info_label.setText(
            f"Name: {record.student_name or 'N/A'}\n"
            f"ID: {record.academic_id or 'N/A'}\n"
            f"Question: {self._current_index + 1}/30\n"
            f"Answer: {record.get_mcq_answer(self._current_index + 1) or '-'}\n"
            f"Confidence: {record.overall_confidence:.0%}"
        )
        
        # Update field image (placeholder)
        self._field_image_label.setText(f"Q{self._current_index + 1} Image")
    
    def _clear_turbo_view(self) -> None:
        """Clear the turbo view."""
        self._field_image_label.setText("No image")
        self._record_info_label.setText("No record selected")
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard events for single-key navigation."""
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
        elif key == Qt.Key_Left or key == Qt.Key_P:
            self._on_previous()
        elif key == Qt.Key_Right or key == Qt.Key_N:
            self._on_next()
        else:
            super().keyPressEvent(event)


def run_ui() -> int:
    """
    Run the application UI.

    Returns:
        Application exit code
    """
    from app.ui.main_window import MainWindow
    
    app = QApplication(sys.argv)
    app.setApplicationName("Vision Exam Data Entry")
    
    window = MainWindow()
    window.show()
    
    return app.exec()