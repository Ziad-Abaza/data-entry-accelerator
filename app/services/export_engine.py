"""
Export Engine - Excel/CSV Generator

Provides fault-tolerant data export with validation and formatting.
Always generates both .xlsx (primary) and .csv (fallback).

Design Principles:
- Validate before export
- Auto-format columns
- Thread-safe operations
- Never lose data on export failure
"""

import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExportValidationError(Exception):
    """Raised when export validation fails."""
    pass


class ExportEngine:
    """
    Handles data export to Excel and CSV formats.
    
    Responsibilities:
    - Convert structured data to DataFrame
    - Validate data before export
    - Excel export with formatting
    - CSV backup generation
    - Thread-safe buffer management
    
    Thread-safe for concurrent access.
    """
    
    # Required columns for valid export
    REQUIRED_COLUMNS = ['Academic ID']
    
    # Column schema for exam data
    COLUMN_SCHEMA = {
        'ID': 'int64',
        'Student Name': 'str',
        'Academic ID': 'str',
        'Q1_1': 'str', 'Q1_2': 'str', 'Q1_3': 'str', 'Q1_4': 'str', 'Q1_5': 'str',
        'Q1_6': 'str', 'Q1_7': 'str', 'Q1_8': 'str', 'Q1_9': 'str', 'Q1_10': 'str',
        'Q1_11': 'str', 'Q1_12': 'str', 'Q1_13': 'str', 'Q1_14': 'str', 'Q1_15': 'str',
        'Q1_16': 'str', 'Q1_17': 'str', 'Q1_18': 'str', 'Q1_19': 'str', 'Q1_20': 'str',
        'Q1_21': 'str', 'Q1_22': 'str', 'Q1_23': 'str', 'Q1_24': 'str', 'Q1_25': 'str',
        'Q1_26': 'str', 'Q1_27': 'str', 'Q1_28': 'str', 'Q1_29': 'str', 'Q1_30': 'str',
        'Status': 'str',
        'Confidence': 'float64',
        'Source': 'str'
    }
    
    def __init__(self, export_path: str = "data/exports"):
        """
        Initialize the export engine.
        
        Args:
            export_path: Path to export directory
        """
        self._export_path = Path(export_path)
        self._export_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Buffer for rows
        self._buffer: List[Dict[str, Any]] = []
        
        # Export history
        self._last_export_path: Optional[Path] = None
        self._last_export_time: Optional[datetime] = None
        
        logger.info(f"ExportEngine initialized: {self._export_path}")
    
    # =========================================================================
    # Buffer Management
    # =========================================================================
    
    def buffer(self, row: Dict[str, Any]) -> bool:
        """
        Add a row to the export buffer.
        
        Args:
            row: Row data to buffer
            
        Returns:
            True if buffered successfully
        """
        with self._lock:
            try:
                # Validate row before buffering
                self._validate_row(row)
                self._buffer.append(row.copy())
                logger.debug(f"Buffered row: {row.get('Academic ID', 'unknown')}")
                return True
            except ExportValidationError as e:
                logger.warning(f"Row validation failed: {e}")
                return False
    
    def buffer_batch(self, rows: List[Dict[str, Any]]) -> int:
        """
        Add multiple rows to the buffer.
        
        Args:
            rows: List of row data
            
        Returns:
            Number of rows successfully buffered
        """
        with self._lock:
            count = 0
            for row in rows:
                if self.buffer(row):
                    count += 1
            return count
    
    def clear_buffer(self) -> int:
        """
        Clear the export buffer.
        
        Returns:
            Number of rows cleared
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            logger.info(f"Cleared {count} buffered rows")
            return count
    
    def get_buffer_count(self) -> int:
        """Get number of rows in buffer."""
        with self._lock:
            return len(self._buffer)
    
    # =========================================================================
    # Data Validation
    # =========================================================================
    
    def _validate_row(self, row: Dict[str, Any]) -> None:
        """
        Validate a single row before export.
        
        Args:
            row: Row data to validate
            
        Raises:
            ExportValidationError: If validation fails
        """
        # Check required Academic ID
        academic_id = row.get('Academic ID', '').strip()
        if not academic_id:
            raise ExportValidationError("Missing Academic ID")
        
        # Check for null/empty rows
        if not any(row.values()):
            raise ExportValidationError("Empty row detected")
        
        # Validate MCQ answers (should be A/B/C/D or empty)
        for i in range(1, 31):
            key = f'Q1_{i}'
            if key in row:
                answer = str(row[key]).strip().upper()
                if answer and answer not in ['A', 'B', 'C', 'D']:
                    logger.warning(f"Invalid MCQ answer at {key}: {answer}")
    
    def validate_buffer(self) -> Dict[str, Any]:
        """
        Validate all buffered rows.
        
        Returns:
            Validation result dict with valid_count, errors, warnings
        """
        with self._lock:
            valid_count = 0
            errors = []
            warnings = []
            
            for i, row in enumerate(self._buffer):
                try:
                    self._validate_row(row)
                    valid_count += 1
                except ExportValidationError as e:
                    errors.append(f"Row {i+1}: {e}")
            
            # Check for incomplete MCQ sets
            for i, row in enumerate(self._buffer):
                mcq_count = sum(1 for k in row.keys() if k.startswith('Q1_') and row.get(k))
                if mcq_count < 30:
                    warnings.append(f"Row {i+1}: Only {mcq_count}/30 MCQ answered")
            
            return {
                'valid_count': valid_count,
                'total_count': len(self._buffer),
                'errors': errors,
                'warnings': warnings,
                'is_valid': len(errors) == 0
            }
    
    # =========================================================================
    # DataFrame Conversion
    # =========================================================================
    
    def _rows_to_dataframe(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert rows to DataFrame with proper schema.
        
        Args:
            rows: List of row data
            
        Returns:
            pandas DataFrame
        """
        if not rows:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure all schema columns exist
        for col, dtype in self.COLUMN_SCHEMA.items():
            if col not in df.columns:
                if dtype == 'float64':
                    df[col] = np.nan
                else:
                    df[col] = ''
        
        # Reorder columns according to schema
        column_order = list(self.COLUMN_SCHEMA.keys())
        existing_cols = [c for c in column_order if c in df.columns]
        extra_cols = [c for c in df.columns if c not in column_order]
        df = df[existing_cols + extra_cols]
        
        # Apply dtypes
        for col, dtype in self.COLUMN_SCHEMA.items():
            if col in df.columns:
                try:
                    if dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype == 'int64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    else:
                        df[col] = df[col].astype(str)
                except Exception as e:
                    logger.warning(f"Failed to convert column {col}: {e}")
        
        return df
    
    # =========================================================================
    # Export Operations
    # =========================================================================
    
    def export_to_excel(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Export data to Excel file.
        
        Args:
            data: Row data (uses buffer if None)
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to exported file or None on failure
        """
        with self._lock:
            # Use buffer if no data provided
            rows = data if data is not None else self._buffer
            
            if not rows:
                logger.warning("No data to export")
                return None
            
            # Generate filename
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"exam_results_{timestamp}.xlsx"
            
            output_path = self._export_path / filename
            
            try:
                # Convert to DataFrame
                df = self._rows_to_dataframe(rows)
                
                # Export to Excel with formatting
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Exam Results')
                    
                    # Get workbook and worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['Exam Results']
                    
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                
                self._last_export_path = output_path
                self._last_export_time = datetime.now()
                
                logger.info(f"Exported to Excel: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Excel export failed: {e}", exc_info=True)
                return None
    
    def export_to_csv(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Export data to CSV file.
        
        Args:
            data: Row data (uses buffer if None)
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to exported file or None on failure
        """
        with self._lock:
            # Use buffer if no data provided
            rows = data if data is not None else self._buffer
            
            if not rows:
                logger.warning("No data to export")
                return None
            
            # Generate filename
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"exam_results_{timestamp}.csv"
            
            output_path = self._export_path / filename
            
            try:
                # Convert to DataFrame
                df = self._rows_to_dataframe(rows)
                
                # Export to CSV
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                logger.info(f"Exported to CSV: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"CSV export failed: {e}", exc_info=True)
                return None
    
    def export_all(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Export to both Excel and CSV.
        
        Args:
            data: Row data (uses buffer if None)
            filename: Base filename (without extension)
            
        Returns:
            Dict with 'excel' and 'csv' paths
        """
        # Generate base filename
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"exam_results_{timestamp}"
        
        excel_path = self.export_to_excel(data, f"{filename}.xlsx")
        csv_path = self.export_to_csv(data, f"{filename}.csv")
        
        return {
            'excel': excel_path,
            'csv': csv_path
        }
    
    # =========================================================================
    # Export History
    # =========================================================================
    
    def get_last_export(self) -> Optional[Dict[str, Any]]:
        """Get information about the last export."""
        with self._lock:
            if not self._last_export_path:
                return None
            
            return {
                'path': str(self._last_export_path),
                'time': self._last_export_time.isoformat() if self._last_export_time else None,
                'exists': self._last_export_path.exists()
            }
    
    def get_export_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get export history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of export info dicts
        """
        with self._lock:
            history = []
            
            # Get Excel files
            for f in sorted(self._export_path.glob("exam_results_*.xlsx"), reverse=True)[:limit]:
                history.append({
                    'filename': f.name,
                    'path': str(f),
                    'size_bytes': f.stat().st_size,
                    'created': datetime.fromtimestamp(f.stat().st_ctime).isoformat()
                })
            
            return history
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_buffer_dataframe(self) -> pd.DataFrame:
        """
        Get current buffer as DataFrame (for preview).
        
        Returns:
            DataFrame of buffered data
        """
        with self._lock:
            return self._rows_to_dataframe(self._buffer)
    
    def export_to_json(self, path: str) -> bool:
        """
        Export buffer to JSON (for debugging).
        
        Args:
            path: Output file path
            
        Returns:
            True if successful
        """
        with self._lock:
            try:
                import json
                output_path = Path(path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self._buffer, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Exported to JSON: {output_path}")
                return True
            except Exception as e:
                logger.error(f"JSON export failed: {e}")
                return False