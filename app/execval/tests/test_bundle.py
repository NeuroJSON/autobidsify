"""
test_bundle.py
==============

Unit tests for bundle.py — plan-bundle file resolution.

These tests are pure logic: no GUI, no real datasets, no autobidsify import.
"""

import time
import zipfile
from pathlib import Path

import bundle
from conftest import REQUIRED, OPTIONAL


def test_all_required_found(make_bundle_dir):
    """A folder with every required file resolves as complete."""
    d = make_bundle_dir(REQUIRED)
    result = bundle.resolve_bundle([d])
    assert result.is_complete
    assert not result.missing_required
    for name in REQUIRED:
        assert name in result.found


def test_missing_one_required_is_incomplete(make_bundle_dir):
    """Dropping one required file makes the bundle incomplete."""
    partial = [f for f in REQUIRED if f != "BIDSPlan.yaml"]
    d = make_bundle_dir(partial)
    result = bundle.resolve_bundle([d])
    assert not result.is_complete
    assert "BIDSPlan.yaml" in result.missing_required


def test_mat_mapping_is_optional(make_bundle_dir):
    """Without mat_mapping.json the bundle is still complete (it is optional)."""
    d = make_bundle_dir(REQUIRED)  # note: no mat_mapping.json
    result = bundle.resolve_bundle([d])
    assert result.is_complete
    assert "mat_mapping.json" in result.missing_optional


def test_optional_files_detected(make_bundle_dir):
    """Optional files, when present, are reported in `found`."""
    d = make_bundle_dir(REQUIRED + ["mat_mapping.json"])
    result = bundle.resolve_bundle([d])
    assert result.is_complete
    assert "mat_mapping.json" in result.found
    assert "headers_normalized.json" in result.missing_optional


def test_case_insensitive_match(tmp_path):
    """File names match regardless of case (e.g. readme.md)."""
    root = tmp_path / "b"
    root.mkdir()
    (root / "ingest_info.json").write_text("{}")
    (root / "BIDSPlan.yaml").write_text("{}")
    (root / "dataset_description.json").write_text("{}")
    (root / "readme.md").write_text("x")          # lowercase
    (root / "participants.tsv").write_text("x")
    result = bundle.resolve_bundle([root])
    assert "README.md" in result.found
    assert result.is_complete


def test_loose_files_mixed(tmp_path):
    """Individual file paths (not a folder) are resolved too."""
    files = []
    for name in REQUIRED:
        p = tmp_path / name
        p.write_text("{}")
        files.append(p)
    result = bundle.resolve_bundle(files)
    assert result.is_complete


def test_zip_is_extracted_and_resolved(tmp_path):
    """A .zip containing the bundle files is extracted and recognized."""
    archive = tmp_path / "plan.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in REQUIRED:
            zf.writestr(name, "{}")
    with bundle.BundleResolver() as r:
        result = r.resolve([archive])
        assert result.is_complete


def test_newest_wins_on_duplicate(tmp_path):
    """When the same filename appears twice, the newest file is chosen."""
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir(); new_dir.mkdir()
    old = old_dir / "BIDSPlan.yaml"
    new = new_dir / "BIDSPlan.yaml"
    old.write_text("old")
    time.sleep(0.01)
    new.write_text("new")
    # make sure new has a later mtime
    import os
    os.utime(new, (time.time() + 10, time.time() + 10))
    result = bundle.resolve_bundle([old_dir, new_dir])
    assert result.found["BIDSPlan.yaml"] == new.resolve()


def test_status_of(make_bundle_dir):
    """status_of reports found / missing / optional-absent correctly."""
    d = make_bundle_dir(REQUIRED)
    result = bundle.resolve_bundle([d])
    assert result.status_of("BIDSPlan.yaml") == "found"
    assert result.status_of("mat_mapping.json") == "optional-absent"

    partial = [f for f in REQUIRED if f != "README.md"]
    d2 = make_bundle_dir(partial, subdir="b2")
    result2 = bundle.resolve_bundle([d2])
    assert result2.status_of("README.md") == "missing"