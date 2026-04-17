# Vision-Accelerated Exam Data Entry System

A production-grade Python application that processes exam sheet images into structured data using OpenCV, OCR, and OMR techniques.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Dashboard](#using-the-dashboard)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This system automates exam data entry by:
- **Preprocessing**: Auto-rotation, deskewing, noise removal
- **OMR**: Optical Mark Recognition for multiple-choice questions (30 questions, 4 options each)
- **OCR**: Optical Character Recognition for academic IDs
- **Cropping**: Extracts display regions for UI verification
- **Export**: Excel/CSV output with validation

**Target**: Process exam sheets in ~300ms per image (excluding OCR latency)

---

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- Windows 10/11
- ~2GB disk space
- Tesseract OCR (optional, for text recognition)

### Step 1: Navigate to Project

```powershell
cd "D:\coding\projects\AI\OCR\Data Entry Accelerator"
```

### Step 2: Create Virtual Environment (if not exists)

```powershell
python -m venv .venv
```

### Step 3: Activate Virtual Environment

```powershell
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Required packages:**
| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.8.1.78 | Image processing |
| numpy | 1.24.3 | Array operations |
| pandas | 2.0.3 | Data handling |
| PySide6 | 6.6.1 | GUI framework |
| openpyxl | 3.1.2 | Excel export |
| pytesseract | 0.3.10 | OCR engine |
| Pillow | 10.1.0 | Image handling |

### Step 5: Install Tesseract OCR (Optional)

For OCR functionality:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Add to PATH or configure in `config/__init__.py`

---

## 🚀 Running the Application

### GUI Mode (Recommended)

```powershell
.venv\Scripts\python.exe main.py --mode gui
```

The application window will open with:
- Status panel at top
- Image viewer (center-left)
- Control panel (center-right)
- Data table at bottom

### CLI Mode

```powershell
.venv\Scripts\python.exe main.py --mode cli --input "path\to\images" --output "output.xlsx"
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Application mode: `gui` or `cli` | `gui` |
| `--input` | Input directory or file path | - |
| `--output` | Output file path for export | - |
| `--format` | Export format: `excel`, `csv`, `json` | `excel` |
| `--log-level` | Logging level: `debug`, `info`, `warning`, `error` | `info` |

---

## 📖 Using the Dashboard

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [Menu Bar]  File | Process | Help                           │
├─────────────────────────────────────────────────────────────┤
│ [Toolbar]  Open Images | Process | Export                   │
├─────────────────────────────────────────────────────────────┤
│ STATUS PANEL                                                │
│ Queue: 5/10 | ID: 123456 | OCR: 85% | OMR: ● | Q: 5/30      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │                     │  │  Controls                    │  │
│  │   IMAGE VIEWER      │  │  [A] [B] [C] [D]            │  │
│  │                     │  │  ← Prev | Next →            │  │
│  │   (Cropped region)  │  │  Current: B                 │  │
│  │                     │  │  Zoom: [-] [100%] [+]       │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ DATA TABLE                                                  │
│ # | ID | Name | Status | Q1 | Q2 | ... | Q30               │
│ 1 | 123 | John | OK    | A | B | ... | C                   │
└─────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` / `B` / `C` / `D` | Select MCQ answer for current question |
| `←` / `→` | Navigate between fields |
| `+` / `-` | Zoom image in/out |
| `0` | Reset zoom to 100% |
| `Ctrl+O` | Open images |
| `Ctrl+E` | Export to Excel |
| `F5` | Start processing |
| `Ctrl+Q` | Exit application |

### Workflow

1. **Load Images**: Click "Open Images" or press `Ctrl+O`
2. **Process**: Images are automatically processed in background thread
3. **Verify**: Review each student's data in the image viewer
4. **Input**: Use A/B/C/D keys to correct MCQ answers if needed
5. **Navigate**: Arrow keys to move between questions/fields
6. **Export**: Click "Export" or press `Ctrl+E` to save results

### Status Indicators

| Color | Meaning |
|-------|---------|
| 🟢 Green | High confidence (>70%) - data looks good |
| 🟡 Yellow | Medium confidence (40-70%) - needs review |
| 🔴 Red | Low confidence (<40%) - likely errors |
| ⚪ Gray | No data / not processed |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI Layer (PySide6)                      │
│  MainWindow → StatusPanel | ImageViewer | DataTable            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator (Brain)                         │
│  - Coordinates pipeline execution                               │
│  - Manages UI state & field navigation                          │
│  - Validates data (duplicates, incomplete MCQ)                  │
│  - Produces final output rows                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Preprocessor │    │ OMR Engine  │    │ OCR Engine   │
│ (OpenCV)      │    │ (Density)   │    │ (Tesseract)  │
└───────────────┘    └──────────────┘    └──────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Persistence Layer                               │
│  SessionManager (auto-save) | ExportEngine (Excel/CSV)         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **Orchestrator** | `app/services/orchestrator.py` | Central coordinator, UI binding, validation |
| **SessionManager** | `app/services/session_manager.py` | Auto-save rows, crash recovery |
| **ExportEngine** | `app/services/export_engine.py` | Excel/CSV generation, validation |
| **OMR Engine** | `app/core/omr/omr_engine.py` | Multiple-choice answer extraction |
| **OCR Engine** | `app/core/ocr/ocr_engine.py` | Academic ID text recognition |
| **Crop Engine** | `app/core/cropping/crop_engine.py` | Region extraction for UI |

### Design Principles

- **UI = dumb renderer** — Only displays data, no business logic
- **Orchestrator = system brain** — Central coordinator for all operations
- **CV modules = stateless tools** — Pure functions for image processing
- **SessionManager = memory** — Persists intermediate state
- **ExportEngine = final output** — Generates structured datasets

---

## 📁 Project Structure

```
Data Entry Accelerator/
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Test configuration
├── README.md                    # This file
│
├── config/
│   └── __init__.py              # Configuration dataclasses
│
├── app/
│   ├── __init__.py
│   │
│   ├── core/                    # Computer Vision modules
│   │   ├── __init__.py          # CVProcessor wrapper
│   │   ├── preprocessing/       # Image preprocessing
│   │   │   ├── __init__.py
│   │   │   └── preprocessor.py  # 6-step pipeline
│   │   ├── omr/                 # Optical Mark Recognition
│   │   │   ├── __init__.py
│   │   │   └── omr_engine.py    # Pixel density analysis
│   │   ├── ocr/                 # Optical Character Recognition
│   │   │   ├── __init__.py
│   │   │   └── ocr_engine.py    # Tesseract wrapper
│   │   └── cropping/            # Region extraction
│   │       ├── __init__.py
│   │       └── crop_engine.py   # 5 crop regions
│   │
│   ├── services/                # Business logic
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Central controller
│   │   ├── session_manager.py   # Auto-save & recovery
│   │   └── export_engine.py     # Excel/CSV export
│   │
│   ├── models/                  # Data classes
│   │   └── __init__.py          # ExamRecord, ProcessingResult, etc.
│   │
│   ├── ui/                      # PySide6 GUI
│   │   ├── __init__.py
│   │   ├── main_window.py       # Main application window
│   │   └── widgets/
│   │       ├── status_panel.py  # Status display
│   │       ├── image_viewer.py  # Image display with zoom
│   │       └── data_table.py    # Real-time data table
│   │
│   └── utils/                   # Utilities
│       └── __init__.py          # Logging, file helpers
│
├── data/
│   ├── sessions/                # Session JSON files (auto-saved)
│   └── exports/                 # Excel/CSV output files
│
└── tests/                       # Unit tests
    ├── test_preprocessing.py
    ├── test_omr.py
    └── test_ocr.py
```

---

## 🔍 Troubleshooting

### ModuleNotFoundError: No module named 'cv2'

```powershell
.venv\Scripts\activate
pip install opencv-python
```

### ModuleNotFoundError: No module named 'PySide6'

```powershell
.venv\Scripts\activate
pip install PySide6
```

### Tesseract Not Found

1. Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Update `config/__init__.py`:
   ```python
   OCRConfig(tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
   ```

### Permission Error on Export

Ensure directories exist:
```powershell
New-Item -ItemType Directory -Path "data\sessions" -Force
New-Item -ItemType Directory -Path "data\exports" -Force
```

### Application Won't Start

Check Python version:
```powershell
python --version  # Should be 3.9+
```

Verify virtual environment:
```powershell
.venv\Scripts\activate  # Should show (.venv) in prompt
```

### Import Errors

If you see `NameError` errors, ensure all imports are correct:
```powershell
.venv\Scripts\python.exe -c "from app.core import CVProcessor; print('OK')"
```

---

## 📊 Output Format

Exported Excel/CSV files contain:

| Column | Type | Description |
|--------|------|-------------|
| ID | int | Row number |
| Student Name | str | Student name (from UI input) |
| Academic ID | str | Student ID (from OCR) |
| Q1_1 to Q1_30 | str | MCQ answers (A/B/C/D) |
| Status | str | COMPLETED or REVIEW_REQUIRED |
| Confidence | float | Overall confidence score (0-1) |
| Source | str | Source image path |

### Example Output

| ID | Academic ID | Q1_1 | Q1_2 | Q1_3 | ... | Q1_30 | Status | Confidence |
|----|-------------|------|------|------|-----|-------|--------|------------|
| 1 | 12345678 | A | B | C | ... | D | COMPLETED | 0.85 |
| 2 | 23456789 | A | A | B | ... | A | REVIEW_REQUIRED | 0.45 |

---

## ⚡ Performance

- **Pipeline execution**: ~300ms per image (excluding OCR)
- **UI response**: <20ms for queries
- **Auto-save**: <50ms per row
- **Session recovery**: Instant resume from last saved state

---

## 📝 License

MIT License - See project root for details.