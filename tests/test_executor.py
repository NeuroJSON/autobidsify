# tests/test_executor.py
# Unit tests for autobidsify/converters/executor.py
# Tests cover ONLY pure-Python logic — no file I/O, no LLM, no dcm2niix.
# Updated: EEG support, dual glob/regex match_pattern, flat-file glob matching.

import pytest
from autobidsify.converters.executor import (
    _match_glob_pattern,
    analyze_filepath_universal,
    infer_scan_type_from_filepath,
    infer_subdirectory_from_suffix,
    categorize_scan_type,
    _normalize_filename,
    _select_preferred_file,
)


# ============================================================================
# _match_glob_pattern  — updated: dual glob+regex mode, flat-file support
# ============================================================================

class TestMatchGlobPattern:

    # ── **/*.ext — any depth including root (flat) ────────────────────────

    def test_double_star_edf_deep(self):
        assert _match_glob_pattern("sub-01/eeg/Subject00_1.edf", "**/*.edf")

    def test_double_star_edf_flat(self):
        # Flat datasets: no subdirectory — ** must match zero dirs
        assert _match_glob_pattern("Subject00_1.edf", "**/*.edf")

    def test_double_star_nii_gz_deep(self):
        assert _match_glob_pattern("sub-01/anat/scan.nii.gz", "**/*.nii.gz")

    def test_double_star_nii_gz_flat(self):
        assert _match_glob_pattern("scan.nii.gz", "**/*.nii.gz")

    def test_double_star_wrong_ext(self):
        assert not _match_glob_pattern("sub-01/anat/scan.dcm", "**/*.nii.gz")

    # ── **/*suffix — suffix match at any depth (EEG _1/_2 pattern) ───────

    def test_double_star_suffix_flat_1(self):
        assert _match_glob_pattern("Subject00_1.edf", "**/*_1.edf")

    def test_double_star_suffix_flat_2(self):
        assert _match_glob_pattern("Subject35_2.edf", "**/*_2.edf")

    def test_double_star_suffix_no_cross_match(self):
        # _1.edf pattern must NOT match _2.edf
        assert not _match_glob_pattern("Subject00_2.edf", "**/*_1.edf")

    def test_double_star_suffix_deep_path(self):
        assert _match_glob_pattern("data/sub01/Subject00_1.edf", "**/*_1.edf")

    # ── **/TOKEN/** — match directory component ───────────────────────────

    def test_token_dir_match(self):
        assert _match_glob_pattern("Newark_sub41006/anat/BRIK/scan.nii.gz",
                                   "**/BRIK/**")

    def test_token_dir_no_match(self):
        assert not _match_glob_pattern("Newark_sub41006/anat/NIfTI/scan.nii.gz",
                                       "**/BRIK/**")

    # ── *token* — substring in path ──────────────────────────────────────

    def test_star_token_star_match(self):
        assert _match_glob_pattern("VHMCT1mm-Hip (134).dcm", "*VHM*")

    def test_star_token_star_no_match(self):
        assert not _match_glob_pattern("VHFCT1mm-Hip (45).dcm", "*VHM*")

    # ── *.ext — extension on filename ────────────────────────────────────

    def test_simple_extension_match(self):
        assert _match_glob_pattern("data/scan.dcm", "*.dcm")

    def test_simple_extension_no_match(self):
        assert not _match_glob_pattern("data/scan.nii.gz", "*.dcm")

    def test_token_dir_case_sensitive(self):
        # _match_glob_pattern is case-sensitive by design
        assert     _match_glob_pattern("sub-01/anat/BRIK/scan.nii.gz", "**/BRIK/**")
        assert not _match_glob_pattern("sub-01/anat/brik/scan.nii.gz", "**/BRIK/**")

    # ── glob star matches token in filename ───────────────────────────────

    def test_glob_star_token_in_filename(self):
        assert     _match_glob_pattern("Subject00_T1w.nii.gz", "*T1w*")
        assert not _match_glob_pattern("Subject00_T2w.nii.gz", "*T1w*")

    def test_glob_dot_star_not_regex(self):
        # ".*" in glob means: file starting with a dot followed by anything
        # It does NOT mean regex "match all" — that is pure-glob behavior
        assert not _match_glob_pattern("Subject08_2.edf", ".*")
        # Use **/*.edf to match any edf
        assert _match_glob_pattern("Subject08_2.edf", "**/*.edf")

    # ── prefix ────────────────────────────────────────────────────────────

    def test_prefix_match(self):
        assert _match_glob_pattern("VHM_scan.dcm", "VHM*")

    def test_prefix_no_match(self):
        assert not _match_glob_pattern("VHF_scan.dcm", "VHM*")

    # ── no wildcard = fnmatch on full path or filename ────────────────────

    def test_no_wildcard_matches_filename_exactly(self):
        # Without wildcards, fnmatch tries exact match on full path then filename
        assert not _match_glob_pattern("some/path/sub-01/T1w.nii.gz", "sub-01")
        # Use ** to match across path components
        assert _match_glob_pattern("some/path/sub-01/T1w.nii.gz", "**/sub-01/**")

    def test_no_wildcard_no_match_different_sub(self):
        assert not _match_glob_pattern("some/path/sub-02/T1w.nii.gz", "**/sub-01/**")


# ============================================================================
# _normalize_filename
# ============================================================================

class TestNormalizeFilename:

    def test_strips_sequence_parens(self):
        assert _normalize_filename("VHFCT1mm-Hip (134).dcm") == "vhfct1mm-hip"

    def test_strips_trailing_numeric_suffix(self):
        result = _normalize_filename("scan_001.dcm")
        assert "001" not in result

    def test_preserves_base_name(self):
        result = _normalize_filename("scan_mprage_anonymized.nii.gz")
        assert result == "scan_mprage_anonymized"

    def test_lowercase_output(self):
        result = _normalize_filename("ScanMprage.nii.gz")
        assert result == result.lower()

    def test_no_path_separator_in_result(self):
        result = _normalize_filename("sub-01/anat/scan.nii.gz")
        assert "/" not in result


# ============================================================================
# _select_preferred_file
# ============================================================================

class TestSelectPreferredFile:

    def test_single_file_returned(self):
        assert _select_preferred_file(["sub/NIfTI/scan.nii.gz"]) == \
               "sub/NIfTI/scan.nii.gz"

    def test_nifti_preferred_over_brik(self):
        files = ["sub/BRIK/scan.nii.gz", "sub/NIfTI/scan.nii.gz"]
        result = _select_preferred_file(files)
        assert "NIfTI" in result
        assert "BRIK" not in result

    def test_non_brik_preferred(self):
        files = ["sub/BRIK/scan.nii.gz", "sub/other/scan.nii.gz"]
        assert "BRIK" not in _select_preferred_file(files)

    def test_empty_list_returns_none(self):
        assert _select_preferred_file([]) is None

    def test_shorter_path_preferred(self):
        files = ["sub/anat/extra/deep/scan.nii.gz", "sub/anat/scan.nii.gz"]
        assert _select_preferred_file(files) == "sub/anat/scan.nii.gz"


# ============================================================================
# infer_scan_type_from_filepath
# ============================================================================

class TestInferScanType:

    def test_t1w_from_path_keyword(self):
        result = infer_scan_type_from_filepath("sub-01/anat/scan.nii.gz", [])
        assert result["subdirectory"] == "anat"

    def test_bold_from_func_keyword(self):
        result = infer_scan_type_from_filepath("sub-01/func/scan_rest.nii.gz", [])
        assert result["subdirectory"] == "func"

    def test_nirs_from_snirf_extension(self):
        result = infer_scan_type_from_filepath("sub-01/nirs/scan.snirf", [])
        assert result["subdirectory"] == "nirs"
        assert result["suffix"] == "nirs"

    def test_eeg_from_edf_extension(self):
        result = infer_scan_type_from_filepath("Subject00_1.edf", [])
        assert result["subdirectory"] == "eeg"
        assert result["suffix"] == "eeg"

    def test_eeg_from_vhdr_extension(self):
        result = infer_scan_type_from_filepath("sub-01/eeg/scan.vhdr", [])
        assert result["subdirectory"] == "eeg"

    def test_eeg_from_bdf_extension(self):
        result = infer_scan_type_from_filepath("sub-01/eeg/scan.bdf", [])
        assert result["subdirectory"] == "eeg"

    def test_llm_rule_takes_priority_over_heuristic(self):
        rules = [{"match_pattern": ".*anonymized.*",
                  "bids_template": "sub-X_T1w.nii.gz"}]
        result = infer_scan_type_from_filepath("scan_mprage_anonymized.nii.gz", rules)
        assert "T1w" in result["suffix"]

    def test_eeg_llm_rule_task_rest(self):
        rules = [{"match_pattern": "*_1.edf",
                  "bids_template": "sub-X_task-rest_eeg.edf"}]
        result = infer_scan_type_from_filepath("Subject00_1.edf", rules)
        assert "task-rest" in result["suffix"]
        assert result["subdirectory"] == "eeg"

    def test_eeg_llm_rule_task_arithmetic(self):
        rules = [{"match_pattern": "*_2.edf",
                  "bids_template": "sub-X_task-arithmetic_eeg.edf"}]
        result = infer_scan_type_from_filepath("Subject00_2.edf", rules)
        assert "task-arithmetic" in result["suffix"]
        assert result["subdirectory"] == "eeg"

    def test_placeholder_x_stripped_from_template(self):
        rules = [{"match_pattern": ".*",
                  "bids_template": "sub-X_ses-X_T1w.nii.gz"}]
        result = infer_scan_type_from_filepath("scan.nii.gz", rules)
        assert "ses-X" not in result["suffix"]

    def test_no_spurious_ses_injection(self):
        result = infer_scan_type_from_filepath("sub-01/anat/scan_T1w.nii.gz", [])
        assert "ses-X" not in result["suffix"]


# ============================================================================
# infer_subdirectory_from_suffix  — updated: eeg added
# ============================================================================

class TestInferSubdirectory:

    @pytest.mark.parametrize("suffix, expected", [
        ("T1w",                  "anat"),
        ("T2w",                  "anat"),
        ("task-rest_bold",       "func"),
        ("bold",                 "func"),
        ("nirs",                 "nirs"),
        ("task-motor_nirs",      "nirs"),
        ("eeg",                  "eeg"),       # NEW
        ("task-rest_eeg",        "eeg"),       # NEW
        ("task-arithmetic_eeg",  "eeg"),       # NEW
        ("dwi",                  "dwi"),
        ("unknown_scan",         "anat"),      # fallback
    ])
    def test_subdirectory_inference(self, suffix, expected):
        assert infer_subdirectory_from_suffix(suffix) == expected


# ============================================================================
# categorize_scan_type
# ============================================================================

class TestCategorizeScanType:

    @pytest.mark.parametrize("suffix, expected", [
        ("T1w",            "anatomical"),
        ("T2w",            "anatomical"),
        ("task-rest_bold", "functional"),
        ("nirs",           "functional"),
        ("dwi",            "diffusion"),
        ("unknown",        "unknown"),
    ])
    def test_categorization(self, suffix, expected):
        assert categorize_scan_type(suffix) == expected


# ============================================================================
# analyze_filepath_universal  — updated: EEG extension preservation
# ============================================================================

class TestAnalyzeFilepathUniversal:

    @pytest.fixture
    def assignment_rules(self):
        return [
            {"subject": "1", "original": "VHM", "match": ["*VHM*"]},
            {"subject": "2", "original": "VHF", "match": ["*VHF*"]},
        ]

    def test_assigns_subject_by_match_pattern(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHMCT1mm-Hip (134).dcm", assignment_rules, [], modality="mri"
        )
        assert result["subject_id"] == "1"

    def test_assigns_vhf_correctly(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHFCT1mm-Head (120).dcm", assignment_rules, [], modality="mri"
        )
        assert result["subject_id"] == "2"

    def test_no_sub_prefix_in_subject_id(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHMCT1mm-Hip.dcm", assignment_rules, [], modality="mri"
        )
        assert not result["subject_id"].startswith("sub-")

    def test_mri_bids_filename_nii_gz(self, assignment_rules):
        result = analyze_filepath_universal(
            "VHMCT1mm-Hip.dcm", assignment_rules, [], modality="mri"
        )
        assert result["bids_filename"].endswith(".nii.gz")

    def test_nirs_bids_filename_snirf(self):
        rules = [{"subject": "1", "original": "sub-01", "match": ["*sub-01*"]}]
        result = analyze_filepath_universal(
            "sub-01_task-rest_nirs.snirf", rules, [], modality="nirs"
        )
        assert result["bids_filename"].endswith(".snirf")

    def test_eeg_edf_extension_preserved(self):
        # EEG files must keep their original extension (.edf)
        rules = [{"subject": "00", "original": "Subject00", "match": ["*Subject00*"]}]
        result = analyze_filepath_universal(
            "Subject00_1.edf", rules, [], modality="eeg"
        )
        assert result["bids_filename"].endswith(".edf")

    def test_eeg_vhdr_extension_preserved(self):
        rules = [{"subject": "01", "original": "sub-01", "match": ["*sub-01*"]}]
        result = analyze_filepath_universal(
            "sub-01_eeg.vhdr", rules, [], modality="eeg"
        )
        assert result["bids_filename"].endswith(".vhdr")

    def test_eeg_subdirectory_is_eeg(self):
        rules = [{"subject": "00", "original": "Subject00", "match": ["*Subject00*"]}]
        result = analyze_filepath_universal(
            "Subject00_1.edf", rules, [], modality="eeg"
        )
        assert result["subdirectory"] == "eeg"

    def test_eeg_never_put_in_anat(self):
        rules = [{"subject": "00", "original": "Subject00", "match": ["*Subject00*"]}]
        result = analyze_filepath_universal(
            "Subject00_1.edf", rules, [], modality="eeg"
        )
        assert result["subdirectory"] != "anat"

    def test_eeg_task_entity_in_bids_filename(self):
        rules = [{"subject": "00", "original": "Subject00", "match": ["*Subject00*"]}]
        fn_rules = [{"match_pattern": "*_1.edf",
                     "bids_template": "sub-X_task-rest_eeg.edf"}]
        result = analyze_filepath_universal(
            "Subject00_1.edf", rules, fn_rules, modality="eeg"
        )
        assert "task-rest" in result["bids_filename"]

    def test_fallback_to_original_field(self):
        rules = [{"subject": "3", "original": "BZZ021", "match": []}]
        result = analyze_filepath_universal(
            "BZZ021_scan.nii.gz", rules, [], modality="mri"
        )
        assert result["subject_id"] == "3"

    def test_unknown_when_no_match(self):
        result = analyze_filepath_universal(
            "completely_unmatched_file.nii.gz", [], [], modality="mri"
        )
        assert result["subject_id"] == "unknown"

    def test_bids_standard_sub_extracted(self):
        result = analyze_filepath_universal(
            "sub-07/anat/sub-07_T1w.nii.gz", [], [], modality="mri"
        )
        assert result["subject_id"] == "07"
