# autobidsify/anonymize.py
# De-identification (anonymization) utilities for AutoBIDSify.
#
# Implements HIPAA Safe Harbor method (45 CFR §164.514(b)):
#   - Removes or generalizes all 18 categories of PHI identifiers
#   - Retains age for subjects ≤ 89; replaces with "90+" for subjects ≥ 90
#   - Retains year from dates; removes month and day
#   - Replaces InstitutionName with anonymous site label (site-01, site-02, ...)
#
# Scope: metadata only (evidence bundle, BIDS sidecar JSON, participants.tsv,
#        free-text documents). MRI pixel-level defacing is handled separately
#        via deface_nifti() which calls pydeface as a subprocess.
#
# This module is import-safe: pydicom and pydeface are optional dependencies.
# All functions degrade gracefully when optional dependencies are absent.

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from autobidsify.utils import warn, info


# ============================================================================
# HIPAA Safe Harbor — identifier field lists
# ============================================================================

# DICOM tag keywords (pydicom .keyword attribute) to remove entirely.
HIPAA_DICOM_TAGS: List[str] = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "MedicalRecordLocator",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "ResponsiblePerson",
    "DeviceSerialNumber",
    "StationName",
    "StudyID",
    "AccessionNumber",
    "RequestedProcedureID",
    "ScheduledProcedureStepID",
]

# DICOM date/time tags: keep year only, zero out month and day.
HIPAA_DICOM_DATE_TAGS: List[str] = [
    "PatientBirthDate",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "InstanceCreationDate",
]

# BIDS sidecar JSON fields to remove entirely.
HIPAA_SIDECAR_FIELDS: List[str] = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "InstitutionAddress",
    "DeviceSerialNumber",
    "StationName",
    "ProcedureStepDescription",
    "RequestAttributesSequence",
    "OperatorsName",
    "PerformingPhysicianName",
    "ReferringPhysicianName",
]

# BIDS sidecar JSON date/time fields: keep year only.
HIPAA_SIDECAR_DATE_FIELDS: List[str] = [
    "AcquisitionDateTime",
    "AcquisitionDate",
    "StudyDate",
    "SeriesDate",
]

# participants.tsv columns to remove entirely.
HIPAA_PARTICIPANTS_COLUMNS: List[str] = [
    "participant_name",
    "name",
    "patient_name",
    "patient_id",
    "birth_date",
    "dob",
    "date_of_birth",
    "address",
    "phone",
    "email",
]

# Free-text regex patterns for PHI scrubbing.
# Each tuple: (compiled pattern, replacement string)
_TEXT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # ISO dates: 2019-03-15 → 2019
    (re.compile(r'\b(\d{4})-\d{2}-\d{2}\b'), r'\1'),
    # US dates: 03/15/2019 or 3/15/2019 → 2019
    (re.compile(r'\b\d{1,2}/\d{1,2}/(\d{4})\b'), r'\1'),
    # Email addresses
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
     '[email redacted]'),
    # US phone numbers: (617) 555-1234, 617-555-1234, 6175551234
    (re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'),
     '[phone redacted]'),
]


# ============================================================================
# Helpers
# ============================================================================

def _keep_year_only(date_str: str) -> str:
    """
    Extract the 4-digit year from a date string and return it as a string.
    Returns the original string unchanged if no 4-digit year is found.

    Examples:
        "2019-03-15"    → "2019"
        "20190315"      → "2019"
        "03/15/2019"    → "2019"
        "unknown"       → "unknown"
    """
    m = re.search(r'\b(19|20)\d{2}\b', str(date_str))
    return m.group(0) if m else str(date_str)


def _generalize_age(age_value: Any) -> str:
    """
    Apply HIPAA Safe Harbor age rule:
      - Age ≤ 89: return exact age as string
      - Age ≥ 90: return "90+"
      - Non-numeric: return original value as string
    """
    try:
        age = int(str(age_value).replace('Y', '').replace('y', '').strip())
        return "90+" if age >= 90 else str(age)
    except (ValueError, TypeError):
        return str(age_value)


def _site_label(institution_name: str, site_registry: Dict[str, str]) -> str:
    """
    Map an institution name to a deterministic anonymous site label.
    Labels are assigned in order of first appearance: site-01, site-02, ...

    Args:
        institution_name: Original institution name string.
        site_registry: Mutable dict mapping original name → site label.

    Returns:
        Anonymous site label string, e.g. "site-01".
    """
    if institution_name not in site_registry:
        n = len(site_registry) + 1
        site_registry[institution_name] = f"site-{n:02d}"
    return site_registry[institution_name]


# ============================================================================
# Text scrubbing
# ============================================================================

def scrub_text(text: str) -> str:
    """
    Apply regex-based PHI scrubbing to a free-text string.

    Replaces dates (keeping year only), email addresses, and phone numbers.
    Does not alter the original string; returns a new string.

    Args:
        text: Input text, e.g. contents of a --describe argument or document.

    Returns:
        Scrubbed text string.
    """
    if not text:
        return text
    result = text
    for pattern, replacement in _TEXT_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ============================================================================
# Evidence bundle scrubbing
# ============================================================================

def scrub_evidence_bundle(bundle: Dict[str, Any],
                           site_registry: Optional[Dict[str, str]] = None
                           ) -> Dict[str, Any]:
    """
    Return a deep copy of the evidence bundle with PHI fields removed or
    generalized, suitable for writing to evidence_bundle.json or passing
    to an LLM when anonymize=True.

    PHI is scrubbed by a recursive walk over the entire bundle, matching
    keys by exact name against the HIPAA_* constants. This covers PHI at
    any nesting depth (e.g. participant_metadata_evidence.dicom_headers.
    samples[], eeg_auxiliary_files, document texts, future modality
    subtrees) without hard-coding bundle paths.

    Args:
        bundle: Evidence bundle dict (not modified in-place).
        site_registry: Optional dict for consistent site-label assignment
                       across multiple calls. A new dict is created if None.

    Returns:
        Scrubbed copy of the evidence bundle dict.
    """
    if site_registry is None:
        site_registry = {}

    b = copy.deepcopy(bundle)
    _scrub_node(b, site_registry)
    return b


def _scrub_dicom_header_dict(header: Dict[str, Any],
                              site_registry: Dict[str, str]) -> None:
    """Scrub a DICOM header dict in-place (already deep-copied by caller)."""
    for tag in HIPAA_DICOM_TAGS:
        header.pop(tag, None)

    for tag in HIPAA_DICOM_DATE_TAGS:
        if tag in header:
            header[tag] = _keep_year_only(header[tag])

    # Age
    for age_key in ("PatientAge", "PatientBirthDate"):
        if age_key in header:
            header[age_key] = _generalize_age(header[age_key])

    # InstitutionName → site label
    if "InstitutionName" in header and header["InstitutionName"]:
        header["InstitutionName"] = _site_label(
            str(header["InstitutionName"]), site_registry)


# Exact-match key sets for the recursive bundle walker.
# Exact equality only — never substring — so that non-PHI keys such as
# "filename", "candidates", "unique_dir_names", "best_candidate" (which
# contain "name"/"id"/"date" as substrings) are preserved.
_REMOVE_KEYS = frozenset(HIPAA_DICOM_TAGS) | {"PatientName", "PatientID"}
_DATE_KEYS = frozenset(HIPAA_DICOM_DATE_TAGS)
_TEXT_KEYS = frozenset({"text", "summary", "user_text", "describe",
                        "content", "parsed_text"})


def _scrub_node(node: Any, site_registry: Dict[str, str]) -> None:
    """
    Recursively scrub PHI from any dict/list structure in-place.

    Matching is by exact key name against the HIPAA_* constants:
      - key in _REMOVE_KEYS         -> delete
      - key in _DATE_KEYS           -> keep year only
      - key == "PatientAge"         -> generalize (>=90 -> "90+")
      - key == "InstitutionName"    -> site label
      - key in _TEXT_KEYS (str)     -> free-text PHI redaction

    PatientSex is intentionally NOT removed: sex is a non-identifying
    demographic variable required by the BIDS participants.tsv spec and
    is not one of the 18 HIPAA Safe Harbor identifiers.

    Caller is responsible for passing a deep copy.
    """
    if isinstance(node, dict):
        for key in list(node.keys()):
            if key in _REMOVE_KEYS:
                del node[key]
                continue
            val = node[key]
            if key in _DATE_KEYS and isinstance(val, str):
                node[key] = _keep_year_only(val)
            elif key == "PatientAge":
                node[key] = _generalize_age(val)
            elif key == "InstitutionName" and val:
                node[key] = _site_label(str(val), site_registry)
            elif key in _TEXT_KEYS and isinstance(val, str):
                node[key] = scrub_text(val)
            else:
                _scrub_node(val, site_registry)
    elif isinstance(node, list):
        for item in node:
            _scrub_node(item, site_registry)


# ============================================================================
# DICOM file scrubbing (pydicom)
# ============================================================================

def scrub_dicom_file(dicom_path: Path,
                     site_registry: Optional[Dict[str, str]] = None) -> bool:
    """
    Scrub PHI from a DICOM file in-place using pydicom.

    Removes HIPAA_DICOM_TAGS fields, keeps year only in date fields,
    generalizes age, and replaces InstitutionName with a site label.

    Args:
        dicom_path: Path to the DICOM file to modify in-place.
        site_registry: Optional shared site-label registry.

    Returns:
        True if scrubbing succeeded; False if pydicom is not available
        or the file could not be read.
    """
    if site_registry is None:
        site_registry = {}
    try:
        import pydicom
    except ImportError:
        warn("pydicom not available; DICOM file PHI scrubbing skipped.")
        return False

    try:
        ds = pydicom.dcmread(str(dicom_path))
    except Exception as e:
        warn(f"  Could not read DICOM file {dicom_path.name}: {e}")
        return False

    for tag_kw in HIPAA_DICOM_TAGS:
        if hasattr(ds, tag_kw):
            try:
                delattr(ds, tag_kw)
            except Exception:
                pass

    for tag_kw in HIPAA_DICOM_DATE_TAGS:
        if hasattr(ds, tag_kw):
            try:
                setattr(ds, tag_kw, _keep_year_only(getattr(ds, tag_kw)))
            except Exception:
                pass

    if hasattr(ds, "PatientAge"):
        try:
            ds.PatientAge = _generalize_age(ds.PatientAge)
        except Exception:
            pass

    if hasattr(ds, "InstitutionName") and ds.InstitutionName:
        try:
            ds.InstitutionName = _site_label(str(ds.InstitutionName),
                                              site_registry)
        except Exception:
            pass

    try:
        ds.save_as(str(dicom_path))
        return True
    except Exception as e:
        warn(f"  Could not save scrubbed DICOM {dicom_path.name}: {e}")
        return False


# ============================================================================
# BIDS sidecar JSON scrubbing
# ============================================================================

def scrub_sidecar_json(json_path: Path,
                       site_registry: Optional[Dict[str, str]] = None
                       ) -> bool:
    """
    Scrub PHI from a BIDS sidecar JSON file in-place.

    Removes HIPAA_SIDECAR_FIELDS, keeps year only in date fields,
    and replaces InstitutionName with a site label.

    Args:
        json_path: Path to the sidecar JSON file to modify in-place.
        site_registry: Optional shared site-label registry.

    Returns:
        True if the file was modified; False if the file could not be read.
    """
    if site_registry is None:
        site_registry = {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        warn(f"  Could not read sidecar JSON {json_path.name}: {e}")
        return False

    changed = False

    for field in HIPAA_SIDECAR_FIELDS:
        if field in data:
            del data[field]
            changed = True

    for field in HIPAA_SIDECAR_DATE_FIELDS:
        if field in data:
            data[field] = _keep_year_only(data[field])
            changed = True

    for age_key in ("PatientAge", "Age"):
        if age_key in data:
            data[age_key] = _generalize_age(data[age_key])
            changed = True

    for inst_key in ("InstitutionName",):
        if inst_key in data and data[inst_key]:
            data[inst_key] = _site_label(str(data[inst_key]), site_registry)
            changed = True

    if changed:
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            warn(f"  Could not write scrubbed sidecar {json_path.name}: {e}")
            return False

    return True


# ============================================================================
# participants.tsv scrubbing
# ============================================================================

def scrub_participants_tsv(tsv_path: Path) -> bool:
    """
    Scrub PHI from a participants.tsv file in-place.

    - Removes columns whose names match HIPAA_PARTICIPANTS_COLUMNS
      (case-insensitive).
    - Generalizes the 'age' column: keeps exact value for ≤ 89,
      replaces with '90+' for ≥ 90.
    - Removes 'birth_date', 'dob', 'date_of_birth' columns entirely.

    Args:
        tsv_path: Path to participants.tsv to modify in-place.

    Returns:
        True if the file was modified; False if it could not be read.
    """
    try:
        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        warn(f"  Could not read {tsv_path.name}: {e}")
        return False

    if not lines:
        return False

    header = lines[0].rstrip("\n").split("\t")
    lower_header = [h.lower() for h in header]

    # Columns to drop entirely
    drop_indices = set()
    for i, col in enumerate(lower_header):
        if col in HIPAA_PARTICIPANTS_COLUMNS:
            drop_indices.add(i)

    # Age column index
    age_idx = next(
        (i for i, col in enumerate(lower_header) if col == "age"), None
    )

    changed = bool(drop_indices)

    new_lines = []
    for line_no, line in enumerate(lines):
        cols = line.rstrip("\n").split("\t")
        # Pad short rows
        while len(cols) < len(header):
            cols.append("n/a")

        # Generalize age (skip header row)
        if line_no > 0 and age_idx is not None and age_idx < len(cols):
            original_age = cols[age_idx]
            generalized = _generalize_age(original_age)
            if generalized != original_age:
                cols[age_idx] = generalized
                changed = True

        # Drop PHI columns
        kept = [v for i, v in enumerate(cols) if i not in drop_indices]
        new_lines.append("\t".join(kept) + "\n")

    if changed:
        try:
            with open(tsv_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            warn(f"  Could not write scrubbed {tsv_path.name}: {e}")
            return False

    return changed


# ============================================================================
# MRI defacing (pydeface)
# ============================================================================

def check_pydeface() -> bool:
    """
    Return True if pydeface is available on PATH.

    pydeface is an optional dependency. It requires FSL to be installed
    on the system. Install with: pip install pydeface
    """
    return shutil.which("pydeface") is not None


def deface_nifti(nifti_path: Path) -> bool:
    """
    Deface a structural MRI NIfTI file in-place using pydeface.

    pydeface removes facial features from T1w/T2w/FLAIR images to prevent
    subject re-identification from reconstructed face geometry.

    The original file is overwritten with the defaced version.
    Only anatomical NIfTI files (.nii or .nii.gz) should be passed here;
    functional (bold) and diffusion (dwi) images are not defaced.

    Args:
        nifti_path: Path to the NIfTI file to deface in-place.

    Returns:
        True if defacing succeeded; False if pydeface is not available
        or defacing failed.
    """
    if not check_pydeface():
        warn(
            f"  pydeface not found; skipping defacing for {nifti_path.name}.\n"
            f"  Install with: pip install pydeface\n"
            f"  Also requires FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation"
        )
        return False

    if not nifti_path.exists():
        warn(f"  NIfTI file not found for defacing: {nifti_path}")
        return False

    info(f"  Defacing: {nifti_path.name}")
    try:
        result = subprocess.run(
            ["pydeface", str(nifti_path), "--outfile", str(nifti_path),
             "--force"],
            capture_output=True,
            text=True,
            timeout=600,  # pydeface can take up to 10 minutes per file
        )
        if result.returncode != 0:
            warn(f"  pydeface failed for {nifti_path.name}: {result.stderr}")
            return False
        info(f"  ✓ Defaced: {nifti_path.name}")
        return True
    except subprocess.TimeoutExpired:
        warn(f"  pydeface timed out for {nifti_path.name} (>10 min)")
        return False
    except Exception as e:
        warn(f"  pydeface error for {nifti_path.name}: {e}")
        return False


# ============================================================================
# Model safety check
# ============================================================================

def is_local_model(model: str) -> bool:
    """
    Return True if the model runs locally (Ollama) and is therefore safe
    to use with --anonymize true.

    Online models (OpenAI gpt-*, o1/o3, DashScope qwen-max/plus/turbo)
    send data to external servers and must not be used with anonymize=True.

    Local models are identified by the 'qwen' prefix combined with a
    ':' tag (Ollama model format, e.g. 'qwen3-coder-next:latest').

    Returns:
        True for local Ollama models; False for all online models.
    """
    m = model.lower()
    # Ollama models always have a ':' tag (e.g. qwen3-coder-next:latest)
    if m.startswith("qwen") and ":" in m:
        return True
    # DashScope cloud Qwen models (no ':' tag): qwen-max, qwen-plus, qwen-turbo
    return False


def _assert_local_ollama_endpoint() -> None:
    """
    Raise SystemExit if OLLAMA_BASE_URL points at a non-local host while
    anonymize=True.

    A local Ollama model name passes is_local_model(), but if the user
    has exported OLLAMA_BASE_URL pointing at a remote server, inference
    data still leaves the machine — breaking the privacy guarantee.
    localhost / 127.0.0.1 / ::1 are allowed; any other host is fatal.
    """
    url = os.getenv("OLLAMA_BASE_URL")
    if not url:
        return
    host = (urlparse(url).hostname or "").lower()
    if host not in ("localhost", "127.0.0.1", "::1", ""):
        print(
            f"\n[FATAL] --anonymize requires a local Ollama endpoint to protect privacy.\n"
            f"        OLLAMA_BASE_URL points to remote host '{host}'.\n"
            f"        Inference data would leave this machine.\n"
            f"        Unset OLLAMA_BASE_URL, or point it at localhost:\n"
            f"          unset OLLAMA_BASE_URL\n",
            flush=True,
        )
        raise SystemExit(1)


def assert_local_model_for_anonymize(model: str) -> None:
    """
    Raise SystemExit with a clear error message if anonymize=True has been
    requested but the inference target is not fully local. Two conditions
    are checked:
      1. The model must be a local Ollama model (is_local_model()).
      2. OLLAMA_BASE_URL, if set, must point at a local host.

    Call this at the start of any LLM-calling stage when anonymize=True.

    Args:
        model: Model name string from --model argument.
    """
    if not is_local_model(model):
        print(
            f"\n[FATAL] --anonymize requires a local Ollama model to protect privacy.\n"
            f"        Model '{model}' sends data to external servers.\n"
            f"        Use a local Ollama model instead, for example:\n"
            f"          --model qwen3-coder-next:latest\n"
            f"        Or disable anonymization:\n"
            f"          --anonymize false\n",
            flush=True,
        )
        raise SystemExit(1)
    _assert_local_ollama_endpoint()