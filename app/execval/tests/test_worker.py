"""
test_worker.py
==============

Unit tests for worker.py helper logic that does NOT require a real dataset or
the heavy autobidsify dependencies:

  - _strip_ansi      : remove terminal color codes
  - _stage_bundle    : copy files to the layout execute expects
  - _build_aux_inputs: read optional JSON into aux_inputs

The actual execute/validate run is intentionally not tested here (it needs a
real dataset and the full scientific stack, which is too heavy for CI).
"""

import json
from pathlib import Path

import worker


# --- _strip_ansi ------------------------------------------------------------

def test_strip_ansi_removes_color_codes():
    colored = "\x1b[92m[INFO] checked 219 files\x1b[0m"
    assert worker._strip_ansi(colored) == "[INFO] checked 219 files"


def test_strip_ansi_leaves_plain_text():
    plain = "[WORKER] Done."
    assert worker._strip_ansi(plain) == plain


def test_strip_ansi_handles_warning_color():
    s = "\x1b[93m[WARNING] something\x1b[0m"
    assert worker._strip_ansi(s) == "[WARNING] something"


# --- _stage_bundle ----------------------------------------------------------

def test_stage_bundle_places_files_correctly(tmp_path):
    """Staging files go to _staging/; trio files go to output root."""
    # create source bundle files
    src = tmp_path / "src"
    src.mkdir()
    names = [
        "BIDSPlan.yaml", "mat_mapping.json",       # -> _staging/
        "dataset_description.json", "README.md", "participants.tsv",  # -> root
    ]
    found = {}
    for n in names:
        p = src / n
        p.write_text("{}")
        found[n] = str(p)

    output_dir = tmp_path / "out"
    logs = []
    worker._stage_bundle(found, output_dir, logs.append)

    # staging files
    assert (output_dir / "_staging" / "BIDSPlan.yaml").exists()
    assert (output_dir / "_staging" / "mat_mapping.json").exists()
    # trio files at root
    assert (output_dir / "dataset_description.json").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "participants.tsv").exists()


def test_stage_bundle_skips_missing_source(tmp_path):
    """A listed file that doesn't exist on disk is skipped without crashing."""
    found = {"BIDSPlan.yaml": str(tmp_path / "does_not_exist.yaml")}
    output_dir = tmp_path / "out"
    worker._stage_bundle(found, output_dir, lambda _l: None)
    assert not (output_dir / "_staging" / "BIDSPlan.yaml").exists()


# --- _build_aux_inputs ------------------------------------------------------

def test_build_aux_inputs_reads_optional_json(tmp_path):
    """Optional JSON files become the expected aux_inputs keys."""
    staging = tmp_path / "_staging"
    staging.mkdir()
    (staging / "headers_normalized.json").write_text(json.dumps({"a": 1}))
    (staging / "voxel_final_plan.json").write_text(json.dumps({"b": 2}))

    aux = worker._build_aux_inputs(staging)
    assert aux["normalized_headers"] == {"a": 1}
    assert aux["final_mapping_plan"] == {"b": 2}


def test_build_aux_inputs_empty_when_absent(tmp_path):
    """No optional files -> empty aux_inputs."""
    staging = tmp_path / "_staging"
    staging.mkdir()
    aux = worker._build_aux_inputs(staging)
    assert aux == {}
