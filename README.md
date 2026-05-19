# Parallel Smart Tracker

A desktop application for automated cell tracking in microscopy videos, combining Level Set segmentation algorithms with parallel computing for fast, accurate analysis.

Built with **PySide6** · **scikit-image** · **OpenCV** · **PyInstaller**

---

## What it does

Biologists manually tracing cells in time-lapse microscopy videos can spend hours on just a few minutes of footage. Smart Tracker automates this:

1. Loads a folder of `.tif` frames (or any standard image format)
2. Preprocesses each frame (median filter → Top-Hat → CLAHE)
3. Segments cells using a Level Set algorithm (Chan-Vese by default)
4. Uses each frame's result as the seed for the next — so the contour "follows" the cell through time
5. Runs all frames in parallel across every available CPU core

The result is a clean binary mask per frame, showing exactly where each cell is.

---

## Algorithms

### Chan-Vese (default)
Region-based Level Set that minimises the difference between pixel intensity and the mean inside/outside the contour. No sharp edges required — ideal for blurry or low-contrast cells.

### GAC (Geodesic Active Contours)
Edge-based Level Set that follows intensity gradients. Faster but needs cleaner borders.

### LBF (Local Binary Fitting)
Locally adaptive region model. Handles cells with uneven internal intensity (e.g. visible nucleus) but is the slowest of the three.

| | GAC | Chan-Vese | LBF |
|---|---|---|---|
| Type | Edge-based | Region global | Region local |
| Works on blurry cells | ✗ | ✓ | ✓ |
| Speed (100 frames) | ~12 s | ~24 s | ~82 s |
| Accuracy | Low | High | High |

**Chan-Vese + CLAHE preprocessing is the recommended configuration** for typical fluorescence microscopy data.

---

## Features

| Tab | Description |
|-----|-------------|
| **Tracking** | Frame-by-frame live preview with overlay. Configurable iterations, smoothing, and lambda parameters. Serial or parallel mode. |
| **Confronto** | Side-by-side comparison of GAC vs LBF on the same dataset, with independent parameter panels and a speedup chart. |
| **Speedup** | Benchmark serial vs parallel processing and plot the wall-clock time per core count. |
| **Info** | Algorithm documentation and usage guide. |

The sidebar is collapsible — click the arrow to reclaim screen space. All parameter panels are also collapsible.

---

## Installation (local development)

```bash
# 1. Clone
git clone https://github.com/massimomarrone/parallel-smart-tracker.git
cd parallel-smart-tracker

# 2. Create a clean virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install PySide6 numpy tifffile scipy scikit-image opencv-python
```

```bash
# Run
python smart_tracker_pyside6.py
```

---

## Building a standalone app

### macOS (.app)
```bash
pip install pyinstaller
pyinstaller smart_tracker.spec --clean
# Output: dist/Smart Tracker.app
```

### Windows (.exe)
```bash
pip install pyinstaller
pyinstaller smart_tracker_win.spec --clean
# Output: dist/SmartTracker/SmartTracker.exe
```

> **Tip:** always build inside a clean virtual environment (not Anaconda) to keep the bundle small (~300 MB vs 1+ GB).

---

## CI/CD — GitHub Actions

Every push to `main` automatically builds both platforms in parallel:

- `build-macos` → `Smart_Tracker_macOS.zip` (contains `Smart Tracker.app`)
- `build-windows` → `SmartTracker/` folder (portable, no installer needed)

Download the artifacts from the **Actions** tab of this repository after each build completes.

> The virtual environment is **not** committed to the repo. GitHub Actions installs all dependencies fresh on each runner via `pip install`.

---

## Project structure

```
.
├── smart_tracker_pyside6.py      # Main application (PySide6 UI + algorithms)
├── smart_tracker.spec            # PyInstaller spec — macOS
├── smart_tracker_win.spec        # PyInstaller spec — Windows
└── .github/
    └── workflows/
        └── build.yml             # CI build for macOS + Windows
```

---

## Tech stack

- **PySide6** — Qt6 UI framework
- **scikit-image** — Chan-Vese and GAC Level Set implementations
- **OpenCV** — image preprocessing (median filter, Top-Hat morphology, CLAHE)
- **SciPy** — signal and ndimage utilities
- **tifffile** — multi-frame TIFF loading
- **concurrent.futures** — `ProcessPoolExecutor` for parallel frame processing
- **PyInstaller** — standalone packaging for macOS and Windows

---

## Author

**Massimo Marrone** — Calcolo Scientifico project
