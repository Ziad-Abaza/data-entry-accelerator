---
noteId: "251180303a7311f194315fa64def32b1"
tags: []

---

# Vision-Accelerated Exam Data Entry System

A production-grade Python application that processes exam sheet images into structured data using OpenCV, OCR, and OMR techniques.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Using the Dashboard](#using-the-dashboard)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- Windows 10/11 (primary target)
- Tesseract OCR (optional, for OCR functionality)

### Step 1: Clone or Navigate to Project

```powershell
cd ".\vision_exam_system"
```

### Step 2: Create Virtual Environment

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

For OCR functionality, install Tesseract:

1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and note the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`)
3. Update `config/__init__.py` if using non-default path

---

## 🚀 Running the Application

### GUI Mode (Recommended)

```powershell
python main.py --mode gui
```

Or simply:

```powershell
python main.py
```

### CLI Mode

```powershell
python main.py --mode cli --input "path\to\images" --output "output.xlsx"
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
| `A` / `B` / `C` / `D` | Select MCQ answer |
| `←` / `→` | Navigate fields |
| `+` / `-` | Zoom in/out |
| `0` | Reset zoom |
| `Ctrl+O` | Open images |
| `Ctrl+E` | Export to Excel |
| `F5` | Start processing |
| `Ctrl+Q` | Exit |

### Workflow

1. **Load Images**: Click "Open Images" or use `Ctrl+O`
2. **Process**: Images are automatically processed in background
3. **Verify**: Review each student's data in the image viewer
4. **Input**: Use A/B/C/D keys to correct MCQ answers
5. **Navigate**: Arrow keys to move between fields
6. **Export**: Click "Export" or `Ctrl+E` to save results

### Status Indicators

| Color | Meaning |
|-------|---------|
| 🟢 Green | High confidence (>70%) |
| 🟡 Yellow | Medium confidence (40-70%) |
| 🔴 Red | Low confidence (<40%) |
| ⚪ Gray | No data |

---

## 📁 Project Structure

```
vision_exam_system/
├── main.py                    # Entry point
├── config/
│   └── __init__.py            # Configuration
├── app/
│   ├── core/
│   │   ├── preprocessing/     # Image preprocessing
│   │   ├── omr/               # OMR engine
│   │   ├── ocr/               # OCR engine
│   │   └── cropping/          # Crop extraction
│   ├── services/
│   │   ├── orchestrator.py    # Central controller
│   │   ├── session_manager.py # Session persistence
│   │   └── export_engine.py   # Excel/CSV export
│   ├── models/                # Data classes
│   ├── ui/
│   │   ├── main_window.py     # Main window
│   │   └── widgets/           # UI components
│   └── utils/                 # Utilities
├── data/
│   ├── sessions/              # Session files
│   └── exports/               # Export files
├── requirements.txt           # Dependencies
└── README.md                  # This file
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
   OCRConfig(tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
   ```

### Permission Error on Export

Ensure the `data/exports` directory exists and is writable:
```powershell
New-Item -ItemType Directory -Path "data\exports" -Force
```

### Application Won't Start

Check Python version:
```powershell
python --version  # Should be 3.9+
```

Verify virtual environment is activated:
```powershell
.venv\Scripts\activate  # Should show (.venv) in prompt
```

---

## 📊 Output Format

Exported Excel/CSV files contain:

| Column | Type | Description |
|--------|------|-------------|
| ID | int | Row number |
| Student Name | str | Student name |
| Academic ID | str | Student ID |
| Q1_1 to Q1_30 | str | MCQ answers (A/B/C/D) |
| Status | str | COMPLETED or REVIEW_REQUIRED |
| Confidence | float | Overall confidence score |
| Source | str | Source image path |

---

## 🔐 Design Principles

- **UI = dumb renderer** — UI only displays data, no business logic
- **Orchestrator = system brain** — Central coordinator for all operations
- **CV modules = stateless tools** — Pure functions for image processing
- **SessionManager = memory** — Persists intermediate state
- **ExportEngine = final output** — Generates structured datasets