# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe: one self-contained program, ``dist/seamsplit`` (``seamsplit.exe`` on
Windows), with Python, numpy, the GUI's pages and PyAV's copy of FFmpeg inside. Nothing
else has to be installed on the machine that runs it.

Build (from the repository root, in an environment with the package and PyInstaller):

    pip install . pyinstaller
    pyinstaller packaging/seamsplit.spec --noconfirm

The program is built for the system it is built on; there is no cross-compiling.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files("seamsplit")  # static/index.html, audition.js, audition_page.html
binaries = collect_dynamic_libs("av")  # the FFmpeg libraries PyAV carries (av.libs / *.dylibs)

a = Analysis(
    ["seamsplit_entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "unittest",
        "pydoc",
        "pytest",
        "black",
        "mypy",
        "flake8",
        "IPython",
        "matplotlib",
        "scipy",
        "pandas",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="seamsplit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX makes antivirus programs suspicious and saves little next to FFmpeg
    runtime_tmpdir=None,
    console=True,  # a command-line program; "seamsplit gui" opens the browser page itself
)
