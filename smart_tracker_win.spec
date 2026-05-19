# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Smart Tracker — Windows .exe

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

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
    icon=None,      # add a .ico file path here if you have one
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
