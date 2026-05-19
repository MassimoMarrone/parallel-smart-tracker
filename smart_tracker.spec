# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Smart Tracker — macOS .app bundle

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Hidden imports needed for scikit-image, scipy, opencv, and PySide6 Charts
hidden_imports = (
    collect_submodules("skimage")
    + collect_submodules("scipy")
    + collect_submodules("scipy.ndimage")
    + collect_submodules("scipy.signal")
    + [
        "PySide6.QtCharts",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "cv2",
        "tifffile",
        "numpy",
        "concurrent.futures",
    ]
)

# Data files for scikit-image (its own data assets)
datas = collect_data_files("skimage")

a = Analysis(
    ["smart_tracker_pyside6.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "IPython",
        "jupyter",
        "notebook",
        "test",
        "tests",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SmartTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # no terminal window
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SmartTracker",
)

app = BUNDLE(
    coll,
    name="Smart Tracker.app",
    icon=None,          # add a .icns file path here if you have one
    bundle_identifier="com.massimomarrone.smarttracker",
    info_plist={
        "CFBundleDisplayName": "Smart Tracker",
        "CFBundleShortVersionString": "2.0.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
