"""
bundle.py
=========

Resolve an AutoBIDSify "plan bundle" from arbitrary user input.

The web pipeline (ingest -> ... -> plan) produces a fixed set of files with
fixed names. The user may hand them to the desktop app in any shape:

  * a single folder containing the files (possibly nested),
  * a .zip archive,
  * several loose files dragged in together,
  * any mix of the above.

This module scans whatever was provided and locates each target file *by name*
(case-insensitive). It never relies on directory layout. The result tells the
UI which files were found, where they are, and whether the required set is
complete.

Required files (execute / BIDS compliance cannot proceed without them):
    BIDSPlan.yaml, mat_mapping.json, dataset_description.json,
    README.md, participants.tsv

Optional files (consumed as aux_inputs when present):
    headers_normalized.json  -> aux_inputs["normalized_headers"]  (NIRS)
    voxel_final_plan.json     -> aux_inputs["final_mapping_plan"]   (MRI)
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


REQUIRED_FILES: List[str] = [
    "ingest_info.json",
    "BIDSPlan.yaml",
    "dataset_description.json",
    "README.md",
    "participants.tsv",
]

OPTIONAL_FILES: List[str] = [
    # mat_mapping.json exists only when the dataset has fNIRS data stored as
    # .mat (Homer3/MATLAB). EEG, MRI, and SNIRF-based fNIRS datasets do not
    # produce it, so it must be optional.
    "mat_mapping.json",
    "headers_normalized.json",
    "voxel_final_plan.json",
]

ALL_TARGETS: List[str] = REQUIRED_FILES + OPTIONAL_FILES
_TARGET_LOOKUP: Dict[str, str] = {name.lower(): name for name in ALL_TARGETS}


@dataclass
class BundleResult:
    """Outcome of resolving a plan bundle."""

    found: Dict[str, Path] = field(default_factory=dict)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when every required file was located."""
        return not self.missing_required

    def status_of(self, name: str) -> str:
        """Return 'found', 'missing', or 'optional-absent' for a target name."""
        if name in self.found:
            return "found"
        if name in REQUIRED_FILES:
            return "missing"
        return "optional-absent"


class BundleResolver:
    """
    Collect plan-bundle files from arbitrary input paths.

    Any .zip among the inputs is extracted into a private temporary directory.
    Call cleanup() (or use as a context manager) when done to remove it.
    """

    def __init__(self) -> None:
        self._tmp_dir: Optional[Path] = None

    def __enter__(self) -> "BundleResolver":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary extraction directory, if one was created."""
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = None

    def resolve(self, input_paths: List[Path]) -> BundleResult:
        """Locate target files among the given files/folders/zips."""
        candidates: List[Path] = []
        for raw in input_paths:
            p = Path(raw)
            if not p.exists():
                continue
            if p.is_dir():
                candidates.extend(self._walk_dir(p))
            elif p.suffix.lower() == ".zip":
                candidates.extend(self._extract_zip(p))
            else:
                candidates.append(p)

        best: Dict[str, Path] = {}
        for f in candidates:
            canonical = _TARGET_LOOKUP.get(f.name.lower())
            if canonical is None:
                continue
            prev = best.get(canonical)
            if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
                best[canonical] = f.resolve()

        result = BundleResult(found=best)
        result.missing_required = [n for n in REQUIRED_FILES if n not in best]
        result.missing_optional = [n for n in OPTIONAL_FILES if n not in best]
        return result

    def _walk_dir(self, folder: Path) -> List[Path]:
        return [p for p in folder.rglob("*") if p.is_file()]

    def _extract_zip(self, archive: Path) -> List[Path]:
        if self._tmp_dir is None:
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="autobidsify_bundle_"))
        dest = self._tmp_dir / archive.stem
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile:
            return []
        return [p for p in dest.rglob("*") if p.is_file()]


def resolve_bundle(input_paths: List[Path]) -> BundleResult:
    """One-shot resolution helper (does not clean up extracted zips)."""
    return BundleResolver().resolve(input_paths)