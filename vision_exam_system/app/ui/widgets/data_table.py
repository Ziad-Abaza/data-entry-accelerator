"""
Data Table widget for the Turbo Data Entry Dashboard.

Displays:
- Real-time Pandas-backed table
- Instant row updates
- Columns: ID, Q1-Q30, Status
- Color-coded cells for status
"""

import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, 
    QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor
from typing import Optional, List, Dict, Any


class DataTable(QWidget):
    """
    Real-time data table backed by Pandas DataFrame.
    
    Features:
    - Instant updates without UI lag
    - Color-coded status cells
    - Scroll to latest row
    - Editable cells for manual correction
    
    Signals:
        cell_edited: Emitted when a cell is edited (row, col, value)
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._df: pd.DataFrame = pd.DataFrame()
        self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Table widget
        self._table = QTableWidget()
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                gridline-color: #dee2e6;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #e9ecef;
                color: #212529;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #dee2e6;
                font-weight: bold;
                color: #495057;
            }
        """)
        
        # Configure table
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # Setup columns
        self._setup_columns()
        
        # Header behavior
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        layout.addWidget(self._table)
        
        # Set minimum height
        self.setMinimumHeight(150)
    
    def _setup_columns(self) -> None:
        """Setup table columns."""
        columns = ["#", "ID", "Name", "Status", "Confidence"]
        # Add Q1-Q30 columns
        columns.extend([f"Q{i}" for i in range(1, 31)])
        
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        
        # Store column indices for quick access
        self._col_indices = {col: i for i, col in enumerate(columns)}
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the DataFrame and update the table.
        
        Args:
            df: Pandas DataFrame with exam data
        """
        self._df = df.copy()
        self._refresh_table()
    
    def _refresh_table(self) -> None:
        """Refresh the table from DataFrame."""
        self._table.setRowCount(len(self._df))
        
        for row_idx, row in self._df.iterrows():
            # Row number
            self._set_cell(row_idx, 0, str(row_idx + 1))
            
            # ID
            self._set_cell(row_idx, 1, str(row.get("academic_id", "")))
            
            # Name
            self._set_cell(row_idx, 2, str(row.get("student_name", "")))
            
            # Status
            status = str(row.get("status", ""))
            self._set_cell(row_idx, 3, status, status_color(status))
            
            # Confidence
            conf = row.get("overall_confidence", 0)
            self._set_cell(row_idx, 4, f"{conf:.0%}" if conf else "--")
            
            # MCQ answers (Q1-Q30)
            for q_idx in range(30):
                answer = row.get(f"Q{q_idx+1}", "")
                self._set_cell(row_idx, 5 + q_idx, str(answer) if answer else "-")
        
        # Scroll to bottom
        if len(self._df) > 0:
            self._table.scrollToBottom()
    
    def _set_cell(
        self, 
        row: int, 
        col: int, 
        text: str, 
        bg_color: Optional[QColor] = None
    ) -> None:
        """Set a table cell value with optional background color."""
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        
        if bg_color:
            item.setBackground(bg_color)
        
        self._table.setItem(row, col, item)
    
    def add_row(self, data: Dict[str, Any]) -> None:
        """
        Add a new row to the table.
        
        Args:
            data: Dictionary with row data
        """
        # Add to DataFrame
        new_row = pd.DataFrame([data])
        self._df = pd.concat([self._df, new_row], ignore_index=True)
        
        # Refresh table
        row_idx = len(self._df) - 1
        self._table.insertRow(row_idx)
        self._populate_row(row_idx, data)
        
        # Scroll to new row
        self._table.scrollToItem(self._table.item(row_idx, 0))
    
    def _populate_row(self, row_idx: int, data: Dict[str, Any]) -> None:
        """Populate a row with data."""
        # Row number
        self._set_cell(row_idx, 0, str(row_idx + 1))
        
        # ID
        self._set_cell(row_idx, 1, str(data.get("academic_id", "")))
        
        # Name
        self._set_cell(row_idx, 2, str(data.get("student_name", "")))
        
        # Status
        status = str(data.get("status", ""))
        self._set_cell(row_idx, 3, status, status_color(status))
        
        # Confidence
        conf = data.get("overall_confidence", 0)
        self._set_cell(row_idx, 4, f"{conf:.0%}" if conf else "--")
        
        # MCQ answers
        for q_idx in range(30):
            answer = data.get(f"Q{q_idx+1}", "")
            self._set_cell(row_idx, 5 + q_idx, str(answer) if answer else "-")
    
    def update_row(self, row_idx: int, data: Dict[str, Any]) -> None:
        """
        Update an existing row.
        
        Args:
            row_idx: Row index to update
            data: New row data
        """
        if 0 <= row_idx < len(self._df):
            for key, value in data.items():
                if key in self._df.columns:
                    self._df.at[row_idx, key] = value
            
            self._populate_row(row_idx, data)
    
    def get_row_data(self, row_idx: int) -> Optional[Dict[str, Any]]:
        """Get data for a specific row."""
        if 0 <= row_idx < len(self._df):
            return self._df.iloc[row_idx].to_dict()
        return None
    
    def get_selected_row(self) -> int:
        """Get the currently selected row index."""
        selected = self._table.selectedItems()
        if selected:
            return selected[0].row()
        return -1
    
    def clear(self) -> None:
        """Clear all rows."""
        self._df = pd.DataFrame()
        self._table.setRowCount(0)
    
    def row_count(self) -> int:
        """Get the number of rows."""
        return len(self._df)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get the underlying DataFrame."""
        return self._df.copy()


def status_color(status: str) -> Optional[QColor]:
    """Get background color for status."""
    status_lower = status.lower()
    if "completed" in status_lower or "ok" in status_lower:
        return QColor("#d4edda")  # Green
    elif "review" in status_lower or "ambiguous" in status_lower:
        return QColor("#fff3cd")  # Yellow
    elif "failed" in status_lower or "error" in status_lower:
        return QColor("#f8d7da")  # Red
    return None