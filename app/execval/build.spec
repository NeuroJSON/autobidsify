# build.spec — PyInstaller configuration for AutoBIDSify ExecVal (Tkinter)
#
# Build (from the app/execval/ directory):  pyinstaller build.spec
# Produces dist/AutoBIDSify/ (onedir).
#
# GUI is pure Tkinter. ExecVal only runs execute + validate, so it imports
# autobidsify directly from the same repository (installed with `pip install
# -e .` at the repo root). The execute/validate import chain does not touch the
# LLM clients (openai/dashscope) or the planner/document stages, so those are
# excluded here to keep the package small. No pywebview/pythonnet.

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

datas = []
hidden = []
binaries = []

# Only collect the submodules ExecVal's execute/validate path actually imports.
# We deliberately do NOT collect_all("autobidsify"), because that would sweep in
# planner/llm and force the openai/dashscope clients into the bundle. Listing the
# needed modules explicitly keeps the LLM stack out.
hidden += [
    "autobidsify",
    "autobidsify.utils",
    "autobidsify.constants",
    "autobidsify.anonymize",
    "autobidsify.converters",
    "autobidsify.converters.executor",
    "autobidsify.converters.validators",
    "autobidsify.converters.mri_convert",
    "autobidsify.converters.jnifti_converter",
    "autobidsify.converters.nirs_convert",
    "autobidsify.converters.eeg_convert",
]

# Scientific libs with dynamic data/submodules (needed by the execute path).
for pkg in ["bids_validator", "bidsschematools", "snirf", "nibabel"]:
    datas += collect_data_files(pkg)
for pkg in ["scipy", "h5py", "nibabel", "snirf", "bjdata",
            "bids_validator", "bidsschematools", "yaml", "numpy"]:
    hidden += collect_submodules(pkg)

excludes = [
    "matplotlib", "pytest", "_pytest",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy",
    "webview", "pywebview", "clr", "pythonnet", "clr_loader",
    "IPython", "jupyter", "notebook", "pandas", "PIL", "Pillow",
    # ExecVal never runs the LLM-driven stages, so keep the clients and the
    # planning/document modules out of the bundle.
    "openai", "dashscope",
    "autobidsify.llm", "autobidsify.planner",
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
    name="AutoBIDSify",
    debug=False, strip=False, upx=False, console=False,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, name="AutoBIDSify",
)
