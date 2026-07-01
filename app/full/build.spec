# build.spec — PyInstaller configuration for AutoBIDSify Full (Tkinter)
#
# Build (from the full/ directory):  pyinstaller build.spec
# Produces dist/AutoBIDSify-Full/ (onedir).
#
# GUI is pure Tkinter. The full autobidsify package is pip-installed (not
# vendored), so we collect it and its LLM client deps. No pywebview/pythonnet.

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_all

block_cipher = None

datas = []
hidden = []
binaries = []

# Collect the full autobidsify package (all stages + planner + llm).
_a_datas, _a_bin, _a_hidden = collect_all("autobidsify")
datas += _a_datas; binaries += _a_bin; hidden += _a_hidden

# Scientific libs with dynamic data/submodules.
for pkg in ["bids_validator", "bidsschematools", "snirf", "nibabel"]:
    datas += collect_data_files(pkg)
for pkg in ["scipy", "h5py", "nibabel", "snirf", "bjdata",
            "bids_validator", "bidsschematools", "yaml", "numpy",
            "openai", "requests"]:
    hidden += collect_submodules(pkg)

excludes = [
    "matplotlib", "pytest", "_pytest",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy",
    "webview", "pywebview", "clr", "pythonnet", "clr_loader",
    "IPython", "jupyter", "notebook", "pandas", "PIL", "Pillow",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AutoBIDSify-Full",
    debug=False, strip=False, upx=False, console=False,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="AutoBIDSify-Full",
)
