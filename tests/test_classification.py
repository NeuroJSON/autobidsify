# tests/test_classification.py
# Unit tests for autobidsify/stages/classification.py
# Updated: EEG support added (_detect_extension, classify_and_stage).
# All tests are pure Python — no LLM, no real disk I/O except tmp_path.

import json
import pytest
from pathlib import Path

from autobidsify.stages.classification import (
    _detect_extension,
    classify_and_stage,
)


# ============================================================================
# _detect_extension
# ============================================================================

class TestDetectExtension:

    # ── compound .nii.gz ─────────────────────────────────────────────────

    def test_nii_gz_flat(self):
        assert _detect_extension("scan.nii.gz") == ".nii.gz"

    def test_nii_gz_in_path(self):
        assert _detect_extension("sub-01/anat/sub-01_T1w.nii.gz") == ".nii.gz"

    def test_nii_gz_uppercase(self):
        # Case-insensitive
        assert _detect_extension("SCAN.NII.GZ") == ".nii.gz"

    # ── single extensions ────────────────────────────────────────────────

    @pytest.mark.parametrize("relpath, expected", [
        ("VHMCT1mm-Hip (134).dcm",          ".dcm"),
        ("sub-01/anat/scan.nii",             ".nii"),
        ("data/signal.mat",                  ".mat"),
        ("sub-01/nirs/scan.snirf",           ".snirf"),
        ("sub-01/nirs/scan.nirs",            ".nirs"),
        ("sub-01/anat/scan.jnii",            ".jnii"),
        ("sub-01/anat/scan.bnii",            ".bnii"),
        ("Subject00_1.edf",                  ".edf"),
        ("sub-01/eeg/scan.vhdr",             ".vhdr"),
        ("sub-01/eeg/scan.set",              ".set"),
        ("sub-01/eeg/scan.bdf",              ".bdf"),
        ("subject-info.csv",                 ".csv"),
        ("README.txt",                       ".txt"),
    ])
    def test_single_extension(self, relpath, expected):
        assert _detect_extension(relpath) == expected


# ============================================================================
# classify_and_stage
# ============================================================================

class TestClassifyAndStage:

    def _make_bundle(self, tmp_path: Path, files: list[str]) -> dict:
        """Create a minimal evidence bundle with real placeholder files."""
        data_root = tmp_path / "data"
        data_root.mkdir()
        for relpath in files:
            fp = data_root / relpath
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"")
        return {"root": str(data_root), "all_files": files}

    # ── MRI files go to mri_files ────────────────────────────────────────

    def test_dcm_classified_as_mri(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["VHMCT1mm-Hip.dcm"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "VHMCT1mm-Hip.dcm" in plan["mri_files"]
        assert len(plan["nirs_files"]) == 0
        assert len(plan["eeg_files"]) == 0

    def test_nii_gz_classified_as_mri(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["sub-01/anat/scan.nii.gz"])
        plan = classify_and_stage(bundle, tmp_path)
        assert plan["mri_files"] == ["sub-01/anat/scan.nii.gz"]

    def test_jnii_classified_as_mri(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["scan.jnii"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "scan.jnii" in plan["mri_files"]

    # ── NIRS files go to nirs_files ──────────────────────────────────────

    def test_snirf_classified_as_nirs(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["sub-01/nirs/scan.snirf"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "sub-01/nirs/scan.snirf" in plan["nirs_files"]

    def test_mat_classified_as_nirs(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["S01.mat"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "S01.mat" in plan["nirs_files"]
        assert "S01.mat" not in plan["mri_files"]

    def test_nirs_ext_classified_as_nirs(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["sub-01.nirs"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "sub-01.nirs" in plan["nirs_files"]

    # ── EEG files go to eeg_files ────────────────────────────────────────

    def test_edf_classified_as_eeg(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["Subject00_1.edf"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "Subject00_1.edf" in plan["eeg_files"]

    def test_vhdr_classified_as_eeg(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["sub-01_eeg.vhdr"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "sub-01_eeg.vhdr" in plan["eeg_files"]

    def test_set_classified_as_eeg(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["scan.set"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "scan.set" in plan["eeg_files"]

    def test_bdf_classified_as_eeg(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["scan.bdf"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "scan.bdf" in plan["eeg_files"]

    def test_eeg_aux_classified_as_eeg(self, tmp_path):
        # .vmrk, .eeg, .fdt are BrainVision/EEGLAB aux files → eeg pool
        for ext in [".vmrk", ".eeg", ".fdt"]:
            sub_dir = tmp_path / f"run_{ext.strip('.')}"
            sub_dir.mkdir()
            bundle = self._make_bundle(sub_dir, [f"scan{ext}"])
            plan = classify_and_stage(bundle, sub_dir)
            assert f"scan{ext}" in plan["eeg_files"], f"{ext} not in eeg_files"

    # ── Unknown / auxiliary files ────────────────────────────────────────

    def test_csv_goes_to_unknown(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["subject-info.csv"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "subject-info.csv" in plan["unknown_files"]

    def test_pdf_goes_to_unknown(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["paper.pdf"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "paper.pdf" in plan["unknown_files"]

    def test_txt_goes_to_unknown(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["README.txt"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "README.txt" in plan["unknown_files"]

    # ── Strict separation (no cross-pool contamination) ──────────────────

    def test_mat_never_in_mri(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["S01.mat", "scan.dcm"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "S01.mat" not in plan["mri_files"]
        assert "scan.dcm" not in plan["nirs_files"]

    def test_edf_never_in_mri_or_nirs(self, tmp_path):
        bundle = self._make_bundle(tmp_path,
                                   ["Subject00_1.edf", "scan.dcm", "S01.snirf"])
        plan = classify_and_stage(bundle, tmp_path)
        assert "Subject00_1.edf" not in plan["mri_files"]
        assert "Subject00_1.edf" not in plan["nirs_files"]

    # ── Mixed dataset ────────────────────────────────────────────────────

    def test_mixed_dataset_counts(self, tmp_path):
        files = [
            "scan.dcm",            # mri
            "scan.nii.gz",         # mri
            "sub-01.snirf",        # nirs
            "S01.mat",             # nirs
            "Subject00_1.edf",     # eeg
            "Subject00_2.edf",     # eeg
            "README.txt",          # unknown
            "subject-info.csv",    # unknown
        ]
        bundle = self._make_bundle(tmp_path, files)
        plan = classify_and_stage(bundle, tmp_path)
        assert plan["counts"]["mri_files"]   == 2
        assert plan["counts"]["nirs_files"]  == 2
        assert plan["counts"]["eeg_files"]   == 2
        assert plan["counts"]["unknown_files"] == 2
        assert plan["counts"]["all_files"]   == 8

    # ── Plan written to disk ─────────────────────────────────────────────

    def test_plan_written_to_staging(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["scan.dcm"])
        classify_and_stage(bundle, tmp_path)
        plan_path = tmp_path / "_staging" / "classification_plan.json"
        assert plan_path.exists()
        with open(plan_path) as f:
            saved = json.load(f)
        assert "mri_files" in saved
        assert "nirs_files" in saved
        assert "eeg_files" in saved

    # ── Pool directories created ─────────────────────────────────────────

    def test_pool_directories_created(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["scan.dcm", "S01.snirf",
                                               "Subject00_1.edf", "README.txt"])
        classify_and_stage(bundle, tmp_path)
        assert (tmp_path / "_staging" / "mri_pool").exists()
        assert (tmp_path / "_staging" / "nirs_pool").exists()
        assert (tmp_path / "_staging" / "eeg_pool").exists()
        assert (tmp_path / "_staging" / "unknown").exists()

    # ── Files physically staged ──────────────────────────────────────────

    def test_dcm_file_staged_to_mri_pool(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["scan.dcm"])
        classify_and_stage(bundle, tmp_path)
        assert (tmp_path / "_staging" / "mri_pool" / "scan.dcm").exists()

    def test_edf_file_staged_to_eeg_pool(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["Subject00_1.edf"])
        classify_and_stage(bundle, tmp_path)
        assert (tmp_path / "_staging" / "eeg_pool" / "Subject00_1.edf").exists()

    def test_snirf_file_staged_to_nirs_pool(self, tmp_path):
        bundle = self._make_bundle(tmp_path, ["sub-01.snirf"])
        classify_and_stage(bundle, tmp_path)
        assert (tmp_path / "_staging" / "nirs_pool" / "sub-01.snirf").exists()

    # ── Empty file list ──────────────────────────────────────────────────

    def test_empty_file_list(self, tmp_path):
        bundle = self._make_bundle(tmp_path, [])
        plan = classify_and_stage(bundle, tmp_path)
        assert plan["counts"]["all_files"] == 0
        assert plan["mri_files"]   == []
        assert plan["nirs_files"]  == []
        assert plan["eeg_files"]   == []
        assert plan["unknown_files"] == []
