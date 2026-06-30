# converters/executor.py v10
# ENHANCED: Full support for MRI and fNIRS conversions
# MRI: DICOM→NIfTI, JNIfTI→NIfTI
# fNIRS: .mat/.nirs→SNIRF

from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import re
from collections import defaultdict
from autobidsify.utils import ensure_dir, write_json, write_yaml, copy_file, list_all_files, info, warn, read_json
from autobidsify.converters.mri_convert import run_dcm2niix_batch, check_dcm2niix_available
from autobidsify.converters.jnifti_converter import convert_jnifti_to_nifti, check_jnifti_support
from autobidsify.converters.nirs_convert import (
    write_snirf_from_normalized,
    generate_nirs_bids_sidecars,
    convert_mat_to_snirf,
    convert_nirs_to_snirf,
)
from autobidsify.converters.eeg_convert import generate_eeg_bids_sidecars
from autobidsify.anonymize import scrub_sidecar_json, scrub_participants_tsv, deface_nifti


def _sanitize_bids_label(label: str) -> str:
    """Remove all non-alphanumeric characters from a BIDS entity label."""
    return re.sub(r'[^a-zA-Z0-9]', '', label)


# ============================================================================
# ASCII tree
# ============================================================================

def _build_ascii_tree(root: Path, max_depth: int = 3) -> str:
    """Build ASCII tree visualization of a directory."""
    lines = [root.name + "/"]

    def walk(directory: Path, prefix: str = "", depth: int = 0):
        if depth >= max_depth:
            return
        try:
            entries = sorted(
                list(directory.iterdir()),
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except PermissionError:
            return
        entries = entries[: (15 if depth == 0 else 8)]
        for i, path in enumerate(entries):
            is_last  = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + path.name + ("/" if path.is_dir() else ""))
            if path.is_dir() and depth < max_depth - 1:
                walk(path, prefix + ("    " if is_last else "│   "), depth + 1)

    walk(root)
    return "\n".join(lines)


# ============================================================================
# Filename helpers
# ============================================================================

def _normalize_filename(filepath: str) -> str:
    """
    Normalize a filename by stripping extensions and trailing sequence numbers.

    Used to:
    - Identify DICOM series (same series = same normalized name).
    - Detect format duplicates (same content in different format directories).

    Examples:
        'VHFCT1mm-Hip (134).dcm'        → 'vhfct1mm-hip'
        'scan_mprage_anonymized.nii.gz'  → 'scan_mprage_anonymized'
        'scan_001.dcm'                   → 'scan'
    """
    name = filepath.split("/")[-1]
    while "." in name and len(name.split(".")[-1]) <= 6:
        name = name.rsplit(".", 1)[0]
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)   # strip trailing " (N)"
    name = re.sub(r"[_\-]\d+$", "", name)         # strip trailing _NNN or -NNN
    return name.strip().lower()


def _extract_acq_label(normalized_fname: str) -> str:
    """
    Extract a short, clean acq- label from a normalized DICOM filename.

    Strategy: split on digit boundaries, keep the last alphabetic token
    that is longer than 2 characters and is not a known scanner/format prefix.
    This isolates the body-part or scan-descriptor word.

    Examples:
        'vhfct1mmankle' → 'ankle'
        'vhfct1mmhead'  → 'head'
        'vhmct1mmhip'   → 'hip'
        'scanmprage'    → 'mprage'

    FIX: previously the entire normalized name (e.g. 'vhfct1mmankle') was used
    as the acq- label, producing names that were both non-descriptive and too
    long for some validators.
    """
    skip = {"vhf", "vhm", "ct", "mr", "mri", "mm", "scan", "the"}
    tokens = re.findall(r"[a-z]+", normalized_fname)
    meaningful = [t for t in tokens if len(t) > 2 and t not in skip]
    if meaningful:
        return meaningful[-1]          # last meaningful token = body part
    return normalized_fname[:20]       # fallback: cap at 20 chars


def _select_preferred_file(files: List[str]) -> str:
    """
    Select the best representative file from a set of format duplicates.

    Priority:
    1. Path contains 'nifti'  → explicit NIfTI format directory (preferred).
    2. Path does NOT contain 'brik' → exclude known duplicate-format directory.
    3. Shortest path depth → closest to root = most original copy.
    4. Alphabetical → deterministic tiebreak.
    """
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    def priority(f):
        parts = f.lower().split("/")
        return (
            0 if any("nifti" in p for p in parts) else 1,
            1 if any("brik"  in p for p in parts) else 0,
            len(parts),
            f,
        )

    return sorted(files, key=priority)[0]


# ============================================================================
# Glob pattern matching
# ============================================================================

def _match_glob_pattern(filepath: str, pattern: str) -> bool:
    """
    Universal glob-style pattern matcher for relative file paths.

    Implements full glob semantics using fnmatch:
    - '**'          matches zero or more path components (any depth)
    - '*'           matches anything within a single path component
    - '?'           matches any single character
    - '[seq]'       matches any character in seq

    Examples:
        '**/*.edf'          → any .edf at any depth, including root
        '**/*_1.edf'        → any file ending in _1.edf at any depth
        '*Subject*'         → filename contains Subject
        'data/**/*.nii.gz'  → .nii.gz under data/ at any depth
        '*.snirf'           → .snirf in root only
    """
    import fnmatch

    fp  = filepath.replace("\\", "/")
    pat = pattern.replace("\\", "/")

    # Normalise: collapse repeated slashes
    while "//" in fp:
        fp = fp.replace("//", "/")
    while "//" in pat:
        pat = pat.replace("//", "/")

    # If no '**' in pattern, use plain fnmatch on the full path and filename
    if "**" not in pat:
        filename = fp.split("/")[-1]
        # Try matching full path first, then filename only
        return fnmatch.fnmatch(fp, pat) or fnmatch.fnmatch(filename, pat)

    # Split pattern and path into parts for ** expansion
    pat_parts = pat.split("/")
    fp_parts  = fp.split("/")

    def _match_parts(pp: list, fp: list) -> bool:
        """
        Recursive matcher for path parts.
        pp = remaining pattern parts
        fp = remaining filepath parts
        """
        # Base cases
        if not pp and not fp:
            return True
        if not pp:
            return False
        if pp == ["**"]:
            # '**' at end matches everything remaining (including nothing)
            return True

        head = pp[0]

        if head == "**":
            # '**' can match zero parts (skip it) or one or more parts
            # Try matching zero parts: advance pattern only
            if _match_parts(pp[1:], fp):
                return True
            # Try matching one or more parts: advance filepath
            if fp and _match_parts(pp, fp[1:]):
                return True
            return False
        else:
            # Regular component: must match current filepath component
            if not fp:
                return False
            if fnmatch.fnmatch(fp[0], head):
                return _match_parts(pp[1:], fp[1:])
            return False

    return _match_parts(pat_parts, fp_parts)


# ============================================================================
# Scan-type inference
# ============================================================================

def infer_scan_type_from_filepath(filepath: str, filename_rules: List[Dict], modality: str = "mri") -> Dict[str, str]:
    """
    Infer BIDS scan-type suffix and subdirectory from a file path.

    Priority:
    1. LLM-generated filename_rules (match_pattern → bids_template).
    2. BIDS entities already embedded in the filename (ses-, task-, acq-, run-).
    3. Heuristic keyword detection in the path.
    4. Extension-based fallback.
    """
    path_lower = filepath.lower()
    filename   = filepath.split("/")[-1]
    fname_low  = filename.lower()

    # ------------------------------------------------------------------
    # Priority 1: LLM filename_rules
    # ------------------------------------------------------------------
    for rule in filename_rules:
        mp = rule.get("match_pattern", "").replace(r"\\", "\\")
        try:
            import fnmatch as _fnmatch
            _glob_ok  = _fnmatch.fnmatch(filename.lower(), mp.lower())
            try:
                _regex_ok = bool(re.search(mp, filename, re.IGNORECASE))
            except re.error:
                _regex_ok = False
            if not (_glob_ok or _regex_ok):
                continue
            template = rule.get("bids_template", "")
            m = re.search(r"sub-[^_]+_(.+?)\.(nii\.gz|snirf|nii|edf|vhdr|set|bdf)$", template)
            if not m:
                continue
            raw = m.group(1)
            # Remove placeholder entities (ses-X, task-X)
            raw = re.sub(r"ses-X_?",  "", raw)
            raw = re.sub(r"task-X_?", "", raw)
            raw = raw.strip("_")
            # Remove spurious ses- if no ses- directory exists in path
            if re.search(r"ses-[A-Za-z0-9]+", raw):
                if not re.search(r"/ses-[A-Za-z0-9]+/", filepath):
                    raw = re.sub(r"ses-[A-Za-z0-9]+_?", "", raw).strip("_")
            if raw:
                # Known BIDS suffix tokens and entity keys used to detect
                # token boundaries when merging multi-word entity values.
                _KNOWN_SUFFIXES = {
                    "nirs", "bold", "eeg", "dwi", "T1w", "T2w", "T1rho",
                    "T2star", "FLAIR", "FLASH", "PD", "PDT2", "angio",
                    "inplaneT1", "inplaneT2", "phase", "magnitude",
                }
                _KNOWN_ENTITY_KEYS = {
                    "ses", "task", "acq", "run", "dir", "echo",
                    "part", "chunk", "res", "space", "split", "trc",
                }

                def _merge_entity_values(s: str) -> str:
                    """
                    Merge underscore-separated words that belong to the same
                    entity value into a single alphanumeric token.

                    BIDS entity values must be purely alphanumeric — no
                    underscores or spaces are allowed inside them.

                    Strategy: scan tokens left-to-right; when we are inside
                    an entity (just saw "key-value"), keep absorbing following
                    tokens into that entity's value until we hit a new entity
                    (next token contains '-' with a known key) or a known
                    BIDS suffix token.

                    Examples:
                      "task-mental_arithmetic_nirs"
                        → tokens: ["task-mental", "arithmetic", "nirs"]
                        → "arithmetic" is absorbed into task value
                        → result: "task-mentalarithmetic_nirs"

                      "task-rest_run-1_bold"
                        → tokens: ["task-rest", "run-1", "bold"]
                        → "run-1" starts a new entity → stop absorbing
                        → result: "task-rest_run-1_bold"  (unchanged)

                      "acq-cthead_T1w"
                        → tokens: ["acq-cthead", "T1w"]
                        → "T1w" is a known suffix → stop absorbing
                        → result: "acq-cthead_T1w"  (unchanged)
                    """
                    tokens = s.split('_')
                    result = []
                    i = 0
                    while i < len(tokens):
                        tok = tokens[i]
                        if '-' in tok:
                            key, _, val = tok.partition('-')
                            # Absorb following tokens that are neither a new
                            # entity nor a known suffix.
                            while i + 1 < len(tokens):
                                nxt = tokens[i + 1]
                                nxt_low = nxt.lower()
                                # Stop if next token is a new entity key
                                is_new_entity = (
                                    '-' in nxt and
                                    nxt.split('-', 1)[0].lower() in _KNOWN_ENTITY_KEYS
                                )
                                # Stop if next token is a known BIDS suffix
                                is_suffix = (
                                    nxt in _KNOWN_SUFFIXES or
                                    nxt_low in {x.lower() for x in _KNOWN_SUFFIXES}
                                )
                                if is_new_entity or is_suffix:
                                    break
                                # Absorb: concatenate without separator
                                val += nxt
                                i += 1
                            # Strip non-alphanumeric from the merged value
                            val_clean = re.sub(r'[^a-zA-Z0-9]', '', val)
                            result.append(f"{key}-{val_clean}")
                        else:
                            result.append(tok)
                        i += 1
                    return '_'.join(result)

                def _sanitize_suffix(s: str) -> str:
                    """
                    Final sanitize pass: strip any remaining non-alphanumeric
                    characters from entity values (e.g. spaces, dots, slashes).
                    Non-entity tokens (T1w, bold, nirs…) are preserved as-is.
                    """
                    tokens = s.split('_')
                    result = []
                    for tok in tokens:
                        if '-' in tok:
                            key, _, val = tok.partition('-')
                            result.append(f"{key}-{re.sub(r'[^a-zA-Z0-9]', '', val)}")
                        else:
                            result.append(tok)
                    return '_'.join(result)

                raw = _merge_entity_values(raw)
                raw = _sanitize_suffix(raw)
                subdir = infer_subdirectory_from_suffix(raw)
                return {"suffix": raw, "subdirectory": subdir,
                        "category": categorize_scan_type(raw)}
        except Exception:
            continue

    # ------------------------------------------------------------------
    # Priority 2: Entities already in filename
    # ------------------------------------------------------------------
    entities: Dict[str, str] = {}
    for key, pattern in [("ses",  r"ses-([A-Za-z0-9]+)"),
                          ("task", r"task-([A-Za-z0-9]+)"),
                          ("acq",  r"acq-([A-Za-z0-9]+)"),
                          ("run",  r"run-([A-Za-z0-9]+)")]:
        m = re.search(pattern, filename)
        if m:
            entities[key] = m.group(1)

    # Infer task from filename keywords when no task- entity is present.
    # This handles datasets where files are named by task content rather than
    # BIDS convention (e.g. "2_finger_tapping.snirf", "3_walking.snirf").
    if "task" not in entities:
        fname_no_ext = fname_low.rsplit(".", 1)[0]
        if any(kw in fname_no_ext for kw in ("rest", "resting")):
            entities["task"] = "rest"
        elif any(kw in fname_no_ext for kw in ("finger", "tapping", "fingertap")):
            entities["task"] = "fingertapping"
        elif "walking" in fname_no_ext or "walk" in fname_no_ext:
            entities["task"] = "walking"
        elif modality in ("nirs", "eeg"):
            # task- is REQUIRED for nirs and eeg per BIDS spec.
            # No keyword matched — use placeholder and warn.
            entities["task"] = "unknown"
            warn(f"  ⚠ Cannot infer task label for '{filepath}'. "
                 f"Using task-unknown. Add task info to --describe to fix this.")
        elif modality == "mri" and (fname_low.endswith((".nii", ".nii.gz")) and
             any(k in fname_low for k in ("bold", "func"))):
            # func MRI also requires task-
            entities["task"] = "unknown"
            warn(f"  ⚠ Cannot infer task label for func MRI '{filepath}'. "
                 f"Using task-unknown.")
        # anat (T1w, T2w) and dwi: task- is OPTIONAL — no fallback needed.

    if fname_low.endswith(".snirf") or "nirs" in fname_low:
        modality_label, subdir = "nirs", "nirs"
    elif fname_low.endswith((".edf", ".vhdr", ".set", ".bdf")):
        modality_label, subdir = "eeg",  "eeg"
    elif any(k in fname_low for k in ("t1w", "t1")):
        modality_label, subdir = "T1w",  "anat"
    elif any(k in fname_low for k in ("t2w", "t2")):
        modality_label, subdir = "T2w",  "anat"
    elif any(k in fname_low for k in ("bold", "func")):
        modality_label, subdir = "bold", "func"
    elif "dwi" in fname_low:
        modality_label, subdir = "dwi",  "dwi"
    else:
        modality_label, subdir = None,   "anat"

    # override subdir when task entity is present
    # BIDS rule: task-* goes to func/ for MRI, but NOT for nirs or eeg.
    if subdir not in ("nirs", "eeg"):
        if "task" in entities or "func" in path_lower:
            subdir = "func"
            if not modality_label:
                modality_label = "bold"

    if entities or modality_label:
        parts = []
        for key in ("ses", "task", "acq", "run"):
            if key in entities:
                parts.append(f"{key}-{entities[key]}")
        if modality_label:
            parts.append(modality_label)
        if parts:
            clean_parts = []
            for p in parts:
                if "-" in p:
                    k, v = p.split("-", 1)
                    clean_parts.append(f"{k}-{re.sub(r'[^a-zA-Z0-9]', '', v)}")
                else:
                    clean_parts.append(p)
            suffix = "_".join(clean_parts)
            return {"suffix": suffix, "subdirectory": subdir,
                    "category": categorize_scan_type(suffix)}

    # ------------------------------------------------------------------
    # Priority 3: Heuristic path keywords
    # ------------------------------------------------------------------
    if any(kw in path_lower for kw in ("anat", "mprage", "t1w", "t1 ")):
        return {"suffix": "T1w",            "subdirectory": "anat", "category": "anatomical"}
    if any(kw in path_lower for kw in ("func", "bold")):
        m = re.search(r"task[_-]([A-Za-z0-9]+)", path_lower)
        suffix = f"task-{m.group(1)}_bold" if m else "task-rest_bold"
        return {"suffix": suffix, "subdirectory": "func", "category": "functional"}
    if "rest" in path_lower:
        return {"suffix": "task-rest_bold",  "subdirectory": "func", "category": "functional"}
    if any(kw in path_lower for kw in ("nirs", "fnirs", ".snirf")):
        return {"suffix": "nirs",            "subdirectory": "nirs", "category": "functional"}
    if "dwi" in path_lower:
        return {"suffix": "dwi",             "subdirectory": "dwi",  "category": "diffusion"}

    # ------------------------------------------------------------------
    # Priority 4: Extension fallback
    # ------------------------------------------------------------------
    if fname_low.endswith(".snirf"):
        return {"suffix": "nirs", "subdirectory": "nirs", "category": "functional"}
    if fname_low.endswith((".nii", ".nii.gz")):
        return {"suffix": "T1w", "subdirectory": "anat", "category": "anatomical"}
    if fname_low.endswith((".edf", ".vhdr", ".set", ".bdf")):
        return {"suffix": "eeg", "subdirectory": "eeg", "category": "functional"}

    return {"suffix": "unknown", "subdirectory": "anat", "category": "unknown"}


def infer_subdirectory_from_suffix(suffix: str) -> str:
    """Map a BIDS suffix string to its subdirectory name."""
    s = suffix.lower()
    if "t1w" in s or "t2w" in s:  return "anat"
    if "bold" in s:                return "func"
    if "nirs" in s:                return "nirs"
    if "eeg"  in s:                return "eeg"
    if "dwi"  in s:                return "dwi"
    return "anat"


def categorize_scan_type(suffix: str) -> str:
    """Return a broad category string for a BIDS suffix."""
    s = suffix.lower()
    if "t1w" in s or "t2w" in s:         return "anatomical"
    if "bold" in s or "nirs" in s:       return "functional"
    if "dwi"  in s:                      return "diffusion"
    return "unknown"


# ============================================================================
# Universal filepath analyzer
# ============================================================================

def analyze_filepath_universal(
    filepath: str,
    assignment_rules: List[Dict],
    filename_rules: List[Dict],
    modality: str = "mri",
) -> Dict[str, Any]:
    """
    Determine the BIDS subject ID and output filename for a source file.

    Subject assignment priority:
    1. 'match' glob patterns in assignment_rules.
    2. 'original' substring match.
    3. 'prefix' filename-prefix match.
    4. Standard BIDS sub-XX pattern already in the path.
    5. Fallback: 'unknown'.
    """
    filename   = filepath.split("/")[-1]
    path_parts = filepath.split("/")
    subject_id: Optional[str] = None

    for rule in assignment_rules:
        for pat in rule.get("match", []):
            if _match_glob_pattern(filepath, pat):
                subject_id = rule.get("subject")
                break
        if subject_id:
            break

    if not subject_id:
        for rule in assignment_rules:
            orig = rule.get("original")
            if orig and orig.lower() in filepath.lower():
                subject_id = rule.get("subject")
                break

    if not subject_id:
        for rule in assignment_rules:
            pfx = rule.get("prefix")
            if pfx and filename.lower().startswith(pfx.lower()):
                subject_id = rule.get("subject")
                break

    if not subject_id:
        for part in path_parts:
            m = re.search(r"sub[_-]?(\w+)", part, re.IGNORECASE)
            if m:
                subject_id = m.group(1)
                break

    if not subject_id:
        subject_id = "unknown"

    # Strip accidental 'sub-' prefix from the bare ID
    if subject_id.startswith("sub-"):
        subject_id = subject_id[4:]

    scan_info = infer_scan_type_from_filepath(filepath, filename_rules, modality=modality)
    if modality == "nirs":
        ext = ".snirf"
    elif modality == "eeg":
        # Preserve original EEG extension
        orig_ext = "." + filepath.rsplit(".", 1)[-1] if "." in filepath else ""
        ext = orig_ext if orig_ext in (".edf", ".vhdr", ".set", ".bdf") else ".edf"
    else:
        ext = ".nii.gz"
    
    clean_suffix = scan_info['suffix']  # already sanitized in infer_scan_type_from_filepath
    bids_filename = f"sub-{subject_id}_{clean_suffix}{ext}"

    return {
        "subject_id":       subject_id,
        "scan_type_suffix": scan_info["suffix"],
        "bids_filename":    bids_filename,
        "subdirectory":     scan_info["subdirectory"],
        "scan_category":    scan_info["category"],
        "original_filepath": filepath,
        "modality":         modality,
    }


# ============================================================================
# Main executor
# ============================================================================

def execute_bids_plan(
    input_root: Path,
    output_dir: Path,
    plan: Dict[str, Any],
    aux_inputs: Dict[str, Any],
    anonymize: bool = False,
    deface: bool = False,
) -> Dict[str, Any]:
    """
    Execute the BIDS conversion plan produced by the planner stage.

    Conversions performed:
      MRI  : DICOM → NIfTI (dcm2niix), JNIfTI → NIfTI
      fNIRS: .mat / .nirs → SNIRF
      Ready: .nii/.nii.gz (MRI) and .snirf (fNIRS) copied directly.

    Unprocessed files are copied verbatim to bids_compatible/derivatives/.
    """
    info("=== Executing BIDS Plan v10 ===")

    bids_root = Path(output_dir) / "bids_compatible"
    ensure_dir(bids_root)
    ensure_dir(bids_root / "derivatives")

    processed_sources: set = set()

    # Load per-file mat_mapping if generated by plan stage.
    # Keys are relative file paths (same as all_files_str entries).
    # Value is the mapping dict for that file.
    # Empty dict if mat_mapping.json absent or has no "files" key.
    _mat_file_mappings: Dict[str, Dict] = {}
    _mat_mapping_doc_path = Path(output_dir) / "_staging" / "mat_mapping.json"
    if _mat_mapping_doc_path.exists():
        try:
            _mat_mapping_doc = read_json(_mat_mapping_doc_path)
            _mat_file_mappings = _mat_mapping_doc.get("files", {})
            info(f"  Loaded mat_mapping.json: "
                 f"{len(_mat_file_mappings)} file(s) mapped")
        except Exception as e:
            warn(f"  Could not read mat_mapping.json: {e}")

    logs:     List[Dict] = []
    successes = failures = 0

    # ── Step 1: copy trio files ───────────────────────────────────────
    info("\n[1/5] Organizing trio files...")
    for trio_file in ("dataset_description.json", "README.md", "participants.tsv"):
        src = output_dir / trio_file
        if src.exists():
            shutil.copy2(src, bids_root / trio_file)
            info(f"  ✓ {trio_file}")

    # ── Step 2: process data files ────────────────────────────────────
    info("\n[2/5] Processing data files...")
    all_files_paths = list_all_files(input_root)
    all_files_str   = [
        str(p.relative_to(input_root)).replace("\\", "/") for p in all_files_paths
    ]
    path_str_to_path = {s: p for s, p in zip(all_files_str, all_files_paths)}
    info(f"Total files: {len(all_files_str)}")

    assignment_rules: List[Dict] = plan.get("assignment_rules", [])
    mappings:         List[Dict] = plan.get("mappings", [])
    info(f"  Assignment rules: {len(assignment_rules)} subjects")

    for mapping in mappings:
        modality      = mapping.get("modality")
        patterns      = mapping.get("match", [])
        filename_rules = mapping.get("filename_rules", [])
        format_ready  = mapping.get("format_ready", False)
        convert_to    = mapping.get("convert_to", "none")
        exclude_pats  = mapping.get("exclude", [])

        info(f"\n  Processing {modality} files...")
        info(f"    Format ready: {format_ready}, Convert to: {convert_to}")

        # Match files (with exclusion)
        matched_files: List[str] = []
        for fp in all_files_str:
            if exclude_pats and any(_match_glob_pattern(fp, ex) for ex in exclude_pats):
                continue
            if any(_match_glob_pattern(fp, pat) for pat in patterns):
                matched_files.append(fp)

        if not matched_files:
            warn(f"    ⚠ No files matched for {modality}")
            continue
        info(f"    ✓ Matched: {len(matched_files)} files")

        # ── fNIRS conversion (direct .mat/.nirs → SNIRF) ──────────────
        if modality == "nirs" and not format_ready:
            info(f"    → fNIRS conversion required ({convert_to})")
            if convert_to != "snirf":
                warn(f"    ⚠ Unknown conversion target: {convert_to}")
                failures += len(matched_files)
                continue

            normalized_path = output_dir / "_staging" / "nirs_headers_normalized.json"
            if normalized_path.exists():
                # CSV → SNIRF via normalized headers
                info("    → Using normalized headers for CSV→SNIRF conversion")
                try:
                    normalized  = read_json(normalized_path)
                    snirf_files = write_snirf_from_normalized(
                        normalized=normalized,
                        input_root=input_root,
                        output_dir=bids_root / "_temp_snirf",
                    )
                    if snirf_files:
                        info(f"    ✓ Generated {len(snirf_files)} SNIRF files from CSV")
                        for _sf in snirf_files:
                            generate_nirs_bids_sidecars(_sf, _sf.stem)
                        successes += len(snirf_files)
                    else:
                        warn("    ✗ CSV→SNIRF conversion produced no files")
                        failures += len(matched_files)
                except Exception as e:
                    warn(f"    ✗ CSV→SNIRF conversion failed: {e}")
                    failures += len(matched_files)
            else:
                # Direct .mat / .nirs → SNIRF
                info("    → Direct file conversion (.mat/.nirs → SNIRF)")
                converted_count = 0
                for fp_str in matched_files:
                    fp = path_str_to_path.get(fp_str)
                    if not fp:
                        continue
                    analysis = analyze_filepath_universal(fp_str, assignment_rules,
                                                             filename_rules, modality="nirs")
                    dst = (bids_root
                           / f"sub-{analysis['subject_id']}"
                           / analysis.get("subdirectory", "nirs")
                           / analysis["bids_filename"])
                    ensure_dir(dst.parent)
                    ext = fp.suffix.lower()

                    if ext == ".mat":
                        _mapping = _mat_file_mappings.get(fp_str)
                        n_blocks = int((_mapping or {}).get("n_blocks", 1))

                        # Validate actual block count directly from the file.
                        # Files in the same structural group share one LLM mapping
                        # but may have different block counts (e.g. S01 has 3 runs,
                        # S08 has 4 runs). The file is the authoritative source.
                        if n_blocks > 1:
                            try:
                                from scipy.io import loadmat as _loadmat
                                _mat_check = _loadmat(str(fp), squeeze_me=False)
                                _da = (_mapping or {}).get("data_assembly") or {}
                                _top_key = (
                                    (_da.get("var") or "").split(".")[0].split("[")[0]
                                )
                                if _top_key and _top_key in _mat_check:
                                    _arr = _mat_check[_top_key]
                                    if (hasattr(_arr, "dtype")
                                            and _arr.dtype == object
                                            and _arr.size > 1):
                                        n_blocks = int(_arr.size)
                            except Exception:
                                pass  # keep mapping value on failure

                        if n_blocks <= 1:
                            # Single block — existing behaviour unchanged
                            result = convert_mat_to_snirf(
                                fp, dst, quiet=False,
                                _mat_mapping=_mapping,
                            )
                            if result:
                                converted_count += 1
                                successes += 1
                                processed_sources.add(fp_str)
                                generate_nirs_bids_sidecars(result, result.stem)
                            else:
                                failures += 1

                        else:
                            # Multi-block: generate run-1 … run-N
                            # Insert run-<N> before the final suffix token.
                            # e.g. "sub-1_task-fingertapping_nirs.snirf"
                            #   -> "sub-1_task-fingertapping_run-1_nirs.snirf"
                            dst_name = dst.name
                            # split off the last "_suffix.ext" part
                            last_us = dst_name.rfind("_")
                            dst_stem   = dst_name[:last_us]        # "sub-1_task-fingertapping"
                            dst_suffix = dst_name[last_us + 1:]    # "nirs.snirf"

                            block_success = 0
                            for blk in range(n_blocks):
                                run_label = blk + 1
                                dst_block = dst.parent / \
                                    f"{dst_stem}_run-{run_label}_{dst_suffix}"
                                ensure_dir(dst_block.parent)
                                result = convert_mat_to_snirf(
                                    fp, dst_block, quiet=False,
                                    _mat_mapping=_mapping,
                                    _block_index=blk,
                                )
                                if result:
                                    block_success += 1
                                    info(f"      ✓ run-{run_label}: {dst_block.name}")
                                    generate_nirs_bids_sidecars(result, result.stem)
                                else:
                                    warn(f"      ✗ run-{run_label} failed")

                            if block_success > 0:
                                converted_count += block_success
                                successes += block_success
                                processed_sources.add(fp_str)
                            else:
                                failures += 1

                    elif ext == ".nirs":
                        result = convert_nirs_to_snirf(fp, dst, quiet=False)
                        if result:
                            converted_count += 1
                            successes += 1
                            processed_sources.add(fp_str)
                            generate_nirs_bids_sidecars(result, result.stem)
                        else:
                            failures += 1

                    else:
                        warn(f"      ⚠ Unknown fNIRS format: {ext}")
                        failures += 1

                if converted_count:
                    info(f"    ✓ Converted {converted_count} fNIRS files to SNIRF")
                else:
                    warn("    ✗ No fNIRS files were converted")
            continue

        info(f"    DEBUG filename_rules: {filename_rules}")

        # ── MRI / format-ready: analyze → group → convert / copy ──────
        file_analyses = [
            analyze_filepath_universal(f, assignment_rules, filename_rules,
                                       modality=modality)
            for f in matched_files
        ]

        file_groups: Dict[str, Dict] = {}
        for analysis in file_analyses:
            subj        = analysis["subject_id"]
            scan_suffix = analysis["scan_type_suffix"]
            fp_str      = analysis["original_filepath"]
            is_dicom    = fp_str.lower().endswith(".dcm")

            if is_dicom:
                # Group DICOM files by subject + scan_suffix + normalized filename base.
                # The normalized base separates different body-part series
                # (e.g. VHFCT1mm-Hip vs VHFCT1mm-Head) that share the same suffix (T1w).
                fname_base = _normalize_filename(fp_str)
                group_key  = f"{subj}_{scan_suffix}_{fname_base}"
            else:
                # For non-DICOM files (fNIRS SNIRF, NIfTI), include the
                # filename base in the group key so that multiple files
                # with the same subject+suffix but different names
                # (e.g. different tasks) are kept as separate scan groups.
                fname_base_nir = _normalize_filename(fp_str)
                group_key = f"{subj}_{scan_suffix}_{fname_base_nir}"

            if group_key not in file_groups:
                if is_dicom:
                    fname_base = _normalize_filename(fp_str)

                    # BIDS filename strategy for DICOM series:
                    #
                    # Case A: LLM already provided an acq- label in scan_suffix
                    #         (e.g. scan_suffix = "acq-ankle_T1w" from filename_rules)
                    #         → use scan_suffix directly, do NOT add another acq-
                    #         Result: sub-1_acq-ankle_T1w.nii.gz  ✓
                    #
                    # Case B: LLM gave a generic suffix (e.g. scan_suffix = "T1w")
                    #         with no acq- entity.
                    #         → derive a short acq- label from the filename base
                    #         Result: sub-1_acq-ankle_T1w.nii.gz  ✓
                    #
                    # Previously, executor always added acq-{full_fname_base} regardless,
                    # producing double acq- entities like:
                    #   sub-1_acq-vhfct1mmankle_acq-ankle_T1w.nii.gz  ✗
                    if "acq-" in scan_suffix:
                        # Case A: LLM already set acq-, trust it
                        bids_fname = f"sub-{subj}_{scan_suffix}.nii.gz"
                    else:
                        # Case B: derive a clean, short label from the body-part word
                        acq_label  = _extract_acq_label(fname_base)
                        bids_fname = f"sub-{subj}_acq-{acq_label}_{scan_suffix}.nii.gz"

                    subdir = analysis["subdirectory"]
                else:
                    bids_fname = analysis["bids_filename"]
                    subdir     = analysis["subdirectory"]

                file_groups[group_key] = {
                    "subject_id":    subj,
                    "scan_suffix":   scan_suffix,
                    "bids_filename": bids_fname,
                    "subdirectory":  subdir,
                    "files":         [],
                    "modality":      modality,
                }
            file_groups[group_key]["files"].append(fp_str)

        info(f"    Grouped into {len(file_groups)} scan groups")

        # Deduplicate within each group
        for gdata in file_groups.values():
            if len(gdata["files"]) <= 1:
                continue
            norm_groups: Dict[str, List[str]] = defaultdict(list)
            for f in gdata["files"]:
                norm_groups[_normalize_filename(f)].append(f)
            deduped: List[str] = []
            for norm_files in norm_groups.values():
                parent_dirs = {"/".join(f.split("/")[:-1]) for f in norm_files}
                if len(parent_dirs) > 1:
                    deduped.append(_select_preferred_file(norm_files))
                else:
                    deduped.extend(norm_files)
            gdata["files"] = deduped

        # Subject summary
        subj_groups: Dict[str, int] = {}
        for gd in file_groups.values():
            subj_groups[gd["subject_id"]] = subj_groups.get(gd["subject_id"], 0) + 1
        info(f"    Subjects: {len(subj_groups)}")
        for sid in sorted(subj_groups, key=lambda x: int(x) if x.isdigit() else 0)[:15]:
            info(f"      sub-{sid}: {subj_groups[sid]} scan(s)")

        # Convert / copy each group
        info(f"    Processing {len(file_groups)} scan groups...")
        for gdata in file_groups.values():
            try:
                fp_str   = gdata["files"][0]
                fp       = path_str_to_path.get(fp_str)
                if not fp:
                    failures += 1
                    continue

                subj         = gdata["subject_id"]
                bids_filename = gdata["bids_filename"]
                subdirectory  = gdata["subdirectory"]
                file_ext      = ".nii.gz" if fp.name.lower().endswith(".nii.gz") \
                                else fp.suffix.lower()

                dst = bids_root / f"sub-{subj}" / subdirectory / bids_filename
                ensure_dir(dst.parent)

                done = False

                # JNIfTI → NIfTI
                if file_ext in (".jnii", ".bnii"):
                    if check_jnifti_support():
                        info(f"      → Converting JNIfTI: {fp.name}")
                        if convert_jnifti_to_nifti(fp, dst, quiet=True):
                            successes += 1; done = True
                            processed_sources.add(fp_str)
                        else:
                            warn("      ✗ JNIfTI conversion failed"); failures += 1
                    else:
                        warn("      ⚠ JNIfTI support unavailable (install nibabel)")
                        failures += 1

                # DICOM → NIfTI
                elif file_ext == ".dcm":
                    if check_dcm2niix_available():
                        info(f"      → Converting DICOM batch: {fp.name}")
                        all_dicoms = [
                            path_str_to_path[f] for f in gdata["files"]
                            if path_str_to_path.get(f)
                            and path_str_to_path[f].suffix.lower() == ".dcm"
                        ]
                        if all_dicoms:
                            if run_dcm2niix_batch(all_dicoms, dst, quiet=True):
                                info(f"      ✓ Converted {len(all_dicoms)} DICOM files")
                                successes += 1; done = True
                                for f in gdata["files"]:
                                    processed_sources.add(f)
                                # Scrub sidecar JSON if anonymize=True
                                if anonymize:
                                    json_path = dst.with_suffix("").with_suffix(".json")
                                    if not json_path.exists():
                                        # dcm2niix may name it without .nii in stem
                                        json_path = dst.parent / (dst.name.replace(".nii.gz", ".json"))
                                    if json_path.exists():
                                        scrub_sidecar_json(json_path)
                                        info(f"      ✓ Sidecar scrubbed: {json_path.name}")
                                # Deface anat NIfTI if deface=True
                                if deface and subdirectory == "anat" and dst.exists():
                                    deface_nifti(dst)
                            else:
                                warn("      ✗ DICOM conversion failed"); failures += 1
                        else:
                            warn("      ⚠ No DICOM files in group"); failures += 1
                    else:
                        warn("      ⚠ dcm2niix unavailable (install dcm2niix)")
                        failures += 1

                # SNIRF — already BIDS-ready
                elif file_ext == ".snirf" and modality == "nirs":
                    info(f"      → Copying SNIRF: {fp.name}")
                    copy_file(fp, dst)
                    successes += 1; done = True
                    processed_sources.add(fp_str)
                    generate_nirs_bids_sidecars(dst, dst.stem)
                    if anonymize:
                        for json_path in dst.parent.glob(f"{dst.stem}*.json"):
                            scrub_sidecar_json(json_path)
                
                # EDF/BrainVision/EEGLAB — already BIDS-ready
                elif file_ext in (".edf", ".vhdr", ".set", ".bdf") and modality == "eeg":
                    info(f"      → Copying EEG: {fp.name}")
                    copy_file(fp, dst)
                    successes += 1; done = True
                    processed_sources.add(fp_str)
                    # Copy auxiliary files (.vmrk, .eeg, .fdt) with same stem
                    for aux_ext in (".vmrk", ".eeg", ".fdt"):
                        aux_src = fp.parent / (fp.stem + aux_ext)
                        if aux_src.exists():
                            aux_dst = dst.parent / (dst.stem + aux_ext)
                            copy_file(aux_src, aux_dst)
                            # Mark aux as processed so it doesn't go to derivatives
                            rel = str(aux_src.relative_to(
                                Path(planning_inputs.get("data_root", str(fp.parent)))
                            )) if False else None
                            try:
                                processed_sources.add(str(aux_src.relative_to(input_root)))
                            except ValueError:
                                processed_sources.add(str(aux_src))
                    # Load eeg_event_mapping if available
                    _eeg_map_path = Path(output_dir) / "_staging" / "eeg_event_mapping.json"
                    _eeg_mapping = None
                    if _eeg_map_path.exists():
                        try:
                            _eeg_mapping = read_json(_eeg_map_path)
                        except Exception:
                            pass
                    _eeg_aux_map_path = Path(output_dir) / "_staging" / "eeg_aux_mapping.json"
                    _eeg_aux_mapping = None
                    if _eeg_aux_map_path.exists():
                        try:
                            _eeg_aux_mapping = read_json(_eeg_aux_map_path)
                        except Exception:
                            pass
                    generate_eeg_bids_sidecars(
                        dst, dst.stem, _eeg_mapping,
                        input_root, _eeg_aux_mapping
                    )
                    if anonymize:
                        for json_path in dst.parent.glob(f"{dst.stem}*.json"):
                            scrub_sidecar_json(json_path)

                # NIfTI — already BIDS-ready
                # FIX: removed the undefined _write_nifti_sidecar_if_needed() call
                # that was here and would raise NameError at runtime.
                elif file_ext in (".nii", ".nii.gz") and modality == "mri":
                    copy_file(fp, dst)
                    successes += 1; done = True
                    processed_sources.add(fp_str)
                    if deface and subdirectory == "anat" and dst.exists():
                        deface_nifti(dst)

                else:
                    warn(f"      ⚠ Unsupported format for {modality}: {file_ext}")
                    failures += 1

                if done:
                    logs.append({
                        "source":      fp_str,
                        "destination": f"sub-{subj}/{subdirectory}/{bids_filename}",
                        "action":      "convert" if file_ext in
                                       (".dcm", ".jnii", ".bnii", ".mat", ".nirs")
                                       else "copy",
                        "modality":    modality,
                        "status":      "success",
                    })

            except Exception as e:
                warn(f"      ✗ Failed: {e}")
                failures += 1

    # ── Step 3: copy unprocessed files to derivatives/ ────────────────
    info("\n[3/5] Copying unprocessed files to derivatives/...")
    derivatives_root = bids_root / "derivatives"
    unprocessed = [f for f in all_files_str if f not in processed_sources]
    info(f"  Total: {len(all_files_str)}, processed: {len(processed_sources)}, "
         f"unprocessed: {len(unprocessed)}")
    copied_deriv = 0
    for fp_str in unprocessed:
        src = path_str_to_path.get(fp_str)
        if src and src.exists():
            try:
                copy_file(src, derivatives_root / fp_str)
                copied_deriv += 1
            except Exception as e:
                warn(f"  Could not copy to derivatives: {fp_str}: {e}")
    info(f"  ✓ Copied {copied_deriv} files to derivatives/")
    # Scrub participants.tsv if anonymize=True
    if anonymize:
        parts_path = bids_root / "participants.tsv"
        if parts_path.exists():
            scrub_participants_tsv(parts_path)
            info("  ✓ participants.tsv scrubbed (anonymize=True)")

    # ── Step 4: write logs and manifest ───────────────────────────────
    info("\n[4/5] Finalizing...")
    write_json(Path(output_dir) / "_staging" / "conversion_log.json", logs)
    manifest_files = sorted(
        str(p.relative_to(bids_root)).replace("\\", "/")
        for p in bids_root.rglob("*") if p.is_file()
    )
    write_yaml(Path(output_dir) / "_staging" / "BIDSManifest.yaml", {
        "total_files": len(manifest_files),
        "files":       manifest_files,
        "tree":        _build_ascii_tree(bids_root),
    })

    # ── Step 5: summary ───────────────────────────────────────────────
    subject_dirs = list(bids_root.glob("sub-*"))
    info(f"\n[5/5] Summary")
    info("━" * 60)
    info("✓ BIDS Dataset Created")
    info(f"Location:         {bids_root}")
    info(f"Files processed:  {successes}")
    info(f"Failed:           {failures}")
    info(f"Subjects:         {len(subject_dirs)}")

    if subject_dirs:
        info("\nSubject directories:")
        for sd in sorted(subject_dirs)[:15]:
            nii   = len(list(sd.rglob("*.nii.gz")))
            snirf = len(list(sd.rglob("*.snirf")))
            total = nii + snirf
            edf = len(list(sd.rglob("*.edf"))) + len(list(sd.rglob("*.vhdr"))) + \
                  len(list(sd.rglob("*.set"))) + len(list(sd.rglob("*.bdf")))
            total = nii + snirf + edf
            if nii and snirf:
                info(f"  {sd.name}: {total} files ({nii} NIfTI, {snirf} SNIRF)")
            elif nii and edf:
                info(f"  {sd.name}: {total} files ({nii} NIfTI, {edf} EEG)")
            elif nii:
                info(f"  {sd.name}: {nii} NIfTI file(s)")
            elif snirf:
                info(f"  {sd.name}: {snirf} SNIRF file(s)")
            elif edf:
                info(f"  {sd.name}: {edf} EEG file(s)")
        if len(subject_dirs) > 15:
            info(f"  ... and {len(subject_dirs) - 15} more")
    else:
        warn("⚠ WARNING: No subject directories created!")

    return {
        "total_mappings":        len(mappings),
        "successful_conversions": successes,
        "failed_conversions":    failures,
        "bids_root":             str(bids_root),
        "subject_count":         len(subject_dirs),
    }