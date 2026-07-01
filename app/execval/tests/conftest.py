"""
conftest.py
===========

Shared pytest fixtures for the ExecVal test suite.

Makes the execval package importable (so `import bundle`, `import worker`
resolve) and provides helpers to build fake plan-bundle files on disk in a
temporary directory.
"""

import sys
from pathlib import Path

import pytest

# Make execval/ importable so tests can `import bundle` / `import worker`.
EXECVAL_DIR = Path(__file__).resolve().parent.parent
if str(EXECVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EXECVAL_DIR))


# Canonical plan-bundle filenames (kept in sync with bundle.py).
REQUIRED = [
    "ingest_info.json",
    "BIDSPlan.yaml",
    "dataset_description.json",
    "README.md",
    "participants.tsv",
]
OPTIONAL = [
    "mat_mapping.json",
    "headers_normalized.json",
    "voxel_final_plan.json",
]


def _write(path: Path, text: str = "{}") -> Path:
    """Create a file with some content, making parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def make_bundle_dir(tmp_path):
    """
    Return a factory that creates a folder populated with the requested
    plan-bundle files.

    Usage:
        d = make_bundle_dir(REQUIRED)              # only required files
        d = make_bundle_dir(REQUIRED + ["mat_mapping.json"])
    """
    def _factory(filenames, subdir="bundle"):
        root = tmp_path / subdir
        for name in filenames:
            _write(root / name)
        return root
    return _factory