# converters/planner.py

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
import re
from datetime import datetime
from collections import defaultdict
from autobidsify.utils import write_json, read_json, write_yaml, info, warn, fatal
from autobidsify.constants import SEVERITY_BLOCK
from autobidsify.llm import llm_nirs_draft, llm_nirs_normalize, llm_bids_plan
from autobidsify.llm import llm_map_mat_to_snirf, llm_map_eeg_events, llm_analyze_eeg_aux
from autobidsify.converters.nirs_convert import inspect_mat_structure, _structure_fingerprint
from autobidsify.converters.nirs_convert import inspect_mat_structure, _structure_fingerprint
from autobidsify.filename_tokenizer import (
    FilenameTokenizer, FilenamePatternAnalyzer, expand_naming_structure,
)
from autobidsify.anonymize import scrub_text, scrub_participants_tsv

HEADERS_DRAFT      = "nirs_headers_draft.json"
HEADERS_NORMALIZED = "nirs_headers_normalized.json"
BIDS_PLAN          = "BIDSPlan.yaml"

# Data file extensions — used for filtering in multiple places
_DATA_EXTS = {
    '.snirf', '.nirs', '.mat',
    '.dcm', '.nii', '.jnii', '.bnii', '.nii.gz',
    '.edf', '.vhdr', '.set', '.bdf',
}


# ============================================================================
# Helpers
# ============================================================================

def _parse_llm_json_response(response_text: str, step_name: str) -> Optional[Dict[str, Any]]:
    """Strip markdown fences and parse JSON from LLM response."""
    if not response_text or not response_text.strip():
        warn(f"{step_name}: LLM returned empty response")
        return None

    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text)
                return obj
            except Exception:
                pass
        warn(f"{step_name}: Failed to parse JSON: {e}")
        return None


def _is_data_file(path: str) -> bool:
    """Return True if path has a recognised neuroimaging extension."""
    low = path.lower()
    if low.endswith('.nii.gz'):
        return True
    ext = ('.' + low.rsplit('.', 1)[-1]) if '.' in low else ''
    return ext in _DATA_EXTS


def _extract_subjects_from_directory_structure(all_files: List[str]) -> Dict[str, Any]:
    """
    Detect subjects from top-level directory names.
    Supports: site_sub-NN, sub-NN, subject-NN, pure-numeric dirs.
    """
    patterns = [
        (r'([A-Za-z]+)_sub(\d+)', True,  2, 1, "site_prefixed"),
        (r'sub-(\w+)',             False, 1, None, "standard_bids"),
        (r'subject[_-]?(\d+)',    False, 1, None, "simple"),
        (r'^(\d{3,})$',           False, 1, None, "numeric_only"),
    ]

    subject_records: List[Dict] = []
    seen_ids: set = set()

    for filepath in all_files:
        parts = filepath.split('/')
        for part in parts[:2]:
            for regex, has_site, id_grp, site_grp, pname in patterns:
                m = re.match(regex, part, re.IGNORECASE)
                if m:
                    original_id = m.group(0)
                    if original_id in seen_ids:
                        break
                    seen_ids.add(original_id)
                    subject_records.append({
                        "original_id": original_id,
                        "numeric_id":  m.group(id_grp),
                        "site":        m.group(site_grp) if has_site and site_grp else None,
                        "pattern_name": pname,
                    })
                    break

    if not subject_records:
        return {"success": False, "method": "directory_structure"}

    subject_records.sort(
        key=lambda x: int(x["numeric_id"]) if x["numeric_id"].isdigit() else 0
    )
    return {
        "success":       True,
        "method":        "directory_structure",
        "subject_records": subject_records,
        "subject_count": len(subject_records),
        "has_site_info": any(r["site"] for r in subject_records),
    }


def _extract_subjects_from_flat_filenames(all_files: List[str]) -> Dict[str, Any]:
    """
    Detect subjects from data-file filename prefixes.
    Only processes files with recognised neuroimaging extensions.
    """
    identifier_to_files: Dict[str, List[str]] = defaultdict(list)

    for filepath in all_files:
        filename = filepath.split('/')[-1]
        low = filename.lower()

        # Extension check
        if low.endswith('.nii.gz'):
            ext = '.nii.gz'
        else:
            ext = ('.' + low.rsplit('.', 1)[-1]) if '.' in low else ''
        if ext not in _DATA_EXTS:
            continue

        # Strip extension(s)
        name_no_ext = filename
        if name_no_ext.lower().endswith('.nii.gz'):
            name_no_ext = name_no_ext[:-7]
        elif '.' in name_no_ext:
            name_no_ext = name_no_ext.rsplit('.', 1)[0]

        m = re.match(r'^([A-Za-z0-9\-]+)', name_no_ext)
        if m:
            identifier_to_files[m.group(1)].append(filepath)

    if not identifier_to_files:
        return {"success": False, "method": "flat_filename"}

    def _sort_key(ident: str) -> int:
        nums = re.findall(r'\d+', ident)
        return int(nums[-1]) if nums else 999999

    sorted_ids = sorted(identifier_to_files.keys(), key=_sort_key)
    subject_records = [
        {
            "original_id":  ident,
            "numeric_id":   str(i),
            "site":         None,
            "pattern_name": "filename_identifier",
            "file_count":   len(identifier_to_files[ident]),
        }
        for i, ident in enumerate(sorted_ids, 1)
    ]

    info(f"  Detected {len(subject_records)} unique identifiers:")
    for rec in subject_records[:10]:
        info(f"    '{rec['original_id']}': {rec['file_count']} file(s)")
    if len(subject_records) > 10:
        info(f"    ... and {len(subject_records) - 10} more")

    return {
        "success":         True,
        "method":          "flat_filename_identifiers",
        "subject_records": subject_records,
        "subject_count":   len(subject_records),
        "has_site_info":   False,
    }


def _collect_extra_columns(metadata: Dict[str, Any]) -> List[str]:
    """Return deduplicated extra column names from participant_metadata."""
    seen: set = set()
    cols: List[str] = []
    for meta in metadata.values():
        for col in meta.keys():
            if col not in seen and col != "participant_id":
                seen.add(col)
                cols.append(col)
    return cols


def _write_participants_from_plan(
    plan_yaml: Dict[str, Any],
    out_dir: Path,
    user_n_subjects: Optional[int],
) -> None:
    """
    Write participants.tsv from LLM assignment_rules.
    LLM rules are authoritative; warn if count < user expectation.
    """
    parts_path = out_dir / "participants.tsv"
    if parts_path.exists():
        parts_path.unlink()

    rules  = plan_yaml.get("assignment_rules", [])
    labels = plan_yaml.get("subjects", {}).get("labels", [])

    seen:    set       = set()
    ordered: List[str] = []
    for rule in rules:
        sid = str(rule.get("subject", ""))
        if sid and sid not in seen:
            seen.add(sid)
            ordered.append(sid)

    if not ordered:
        ordered = [str(lbl) for lbl in labels]

    if user_n_subjects and len(ordered) < user_n_subjects:
        warn(f"  ⚠ participants.tsv has {len(ordered)} subjects "
             f"but user specified {user_n_subjects}. "
             f"LLM assignment_rules may be incomplete — check BIDSPlan.yaml.")

    metadata      = plan_yaml.get("participant_metadata", {})
    extra_columns = _collect_extra_columns(metadata)
    columns       = ["participant_id"] + extra_columns

    def _sort_key(sid: str):
        try:    return (0, int(sid))
        except: return (1, sid)

    lines = ["\t".join(columns) + "\n"]
    for sid in sorted(ordered, key=_sort_key):
        meta = metadata.get(sid, {})
        row  = [f"sub-{sid}"] + [str(meta.get(col, "n/a")) for col in extra_columns]
        lines.append("\t".join(row) + "\n")

    parts_path.write_text("".join(lines))
    info(f"  ✓ participants.tsv: {len(ordered)} subjects, columns: {columns}")


def _write_participants_from_subject_map(subject_map: Dict[str, str],
                                         out_dir: Path) -> None:
    """
    Write participants.tsv from the naming_structure expansion.
    subject_map: {subject_id: original_id}. Authoritative subject source.
    """
    parts_path = out_dir / "participants.tsv"
    if parts_path.exists():
        parts_path.unlink()

    def _sort_key(sid: str):
        try:    return (0, int(sid))
        except: return (1, sid)

    lines = ["participant_id\toriginal_id\n"]
    for sid in sorted(subject_map.keys(), key=_sort_key):
        lines.append(f"sub-{sid}\t{subject_map[sid]}\n")
    parts_path.write_text("".join(lines))
    info(f"  ✓ participants.tsv: {len(subject_map)} subjects (from naming_structure)")


def _merge_participants_from_llm_metadata(
    plan_yaml: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Append any extra columns from participant_metadata to existing participants.tsv."""
    parts_path = out_dir / "participants.tsv"
    if not parts_path.exists():
        return

    metadata      = plan_yaml.get("participant_metadata", {})
    extra_columns = _collect_extra_columns(metadata)
    if not extra_columns:
        info("  No extra columns from LLM metadata")
        return

    existing = parts_path.read_text().splitlines()
    if not existing:
        return

    header   = existing[0].split("\t")
    new_cols = [c for c in extra_columns if c not in header]
    if not new_cols:
        info("  participants.tsv already has all metadata columns")
        return

    info(f"  Adding columns to participants.tsv: {new_cols}")
    new_lines = ["\t".join(header + new_cols)]
    for line in existing[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        sid   = cells[0].replace("sub-", "")
        meta  = metadata.get(sid, {})
        new_lines.append("\t".join(cells + [str(meta.get(c, "n/a")) for c in new_cols]))

    parts_path.write_text("\n".join(new_lines) + "\n")
    info(f"  ✓ participants.tsv updated with {len(new_cols)} new column(s)")


MAT_MAPPING_FILE = "mat_mapping.json"

def _build_mat_mapping(
    model: str,
    all_files: List[str],
    data_root: Path,
    staging_dir: Path,
) -> Optional[str]:
    """
    Generate mat_mapping.json for every .mat file in the dataset.

    Strategy (minimise LLM calls):
    1. Inspect all .mat files → structural summary per file.
    2. Group files by structural fingerprint (same var names + shape patterns).
    3. One LLM call per group → one mapping per group.
    4. Copy group mapping to every file in that group.
    5. Write mat_mapping.json with per-file entries.

    Returns the relative path to mat_mapping.json ("_staging/mat_mapping.json"),
    or None if no .mat files exist.
    """
    from autobidsify.llm import llm_map_mat_to_snirf
    from autobidsify.converters.nirs_convert import inspect_mat_structure, _structure_fingerprint
    import json as _json

    mat_mapping_path = staging_dir / MAT_MAPPING_FILE

    mat_files = [f for f in all_files if f.lower().endswith(".mat")]
    if not mat_files:
        return None  # no .mat files — skip silently

    # Idempotent: skip if already generated
    if mat_mapping_path.exists():
        info("  mat_mapping.json already exists, skipping")
        return f"_staging/{MAT_MAPPING_FILE}"

    info(f"\nStep MAT: {len(mat_files)} .mat file(s) found — building mat_mapping.json")

    # ── Step 1: inspect all files ────────────────────────────────────
    file_summaries: Dict[str, Any] = {}   # relpath → summary
    for relpath in mat_files:
        abs_path = data_root / relpath
        if not abs_path.exists():
            warn(f"  .mat file not found: {relpath}")
            continue
        summary = inspect_mat_structure(abs_path)
        if summary is None:
            warn(f"  Cannot inspect: {relpath}")
            continue
        summary["relpath"] = relpath
        file_summaries[relpath] = summary

    if not file_summaries:
        warn("  No .mat files could be inspected — mat_mapping.json not generated")
        return None

    info(f"  Inspected {len(file_summaries)}/{len(mat_files)} files")

    # ── Step 2: group by structural fingerprint ───────────────────────
    groups: Dict[frozenset, List[str]] = {}
    for relpath, summary in file_summaries.items():
        fp = _structure_fingerprint(summary)
        groups.setdefault(fp, []).append(relpath)

    info(f"  Found {len(groups)} structural group(s) → {len(groups)} LLM call(s)")

    # ── Step 3: one LLM call per group ────────────────────────────────
    per_file_mappings: Dict[str, Dict] = {}  # relpath → mapping dict

    for group_idx, (fingerprint, group_relpaths) in enumerate(groups.items(), 1):
        info(f"  Group {group_idx}/{len(groups)}: {len(group_relpaths)} file(s)")

        # Use first file as representative
        rep_relpath = group_relpaths[0]
        rep_summary = file_summaries[rep_relpath]

        payload = _json.dumps(
            {
                "group_size": len(group_relpaths),
                "representative_file": rep_summary,
            },
            indent=2,
        )

        try:
            response = llm_map_mat_to_snirf(model, payload)
        except Exception as e:
            warn(f"  LLM call failed for group {group_idx}: {e}")
            # Leave files in this group without a mapping (heuristic fallback)
            continue

        # Parse response
        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3]
        try:
            mapping = _json.loads(text.strip())
        except Exception as e:
            warn(f"  Cannot parse LLM JSON for group {group_idx}: {e}")
            continue

        da = mapping.get("data_assembly") or {}
        info(f"    → data_assembly.type={da.get('type')}, "
             f"data_assembly.var={da.get('var')}, "
             f"confidence={mapping.get('confidence')}")

        # Copy this mapping to every file in the group
        for relpath in group_relpaths:
            per_file_mappings[relpath] = {
                # New assembly format (from updated prompt)
                "data_assembly":        mapping.get("data_assembly"),
                "time_assembly":        mapping.get("time_assembly"),
                "wavelengths_assembly": mapping.get("wavelengths_assembly"),
                # Legacy fields — kept for backward compatibility with old mat_mapping.json
                "data_var":             mapping.get("data_var"),
                "time_var":             mapping.get("time_var"),
                "wavelengths_var":      mapping.get("wavelengths_var"),
                # Common fields
                "wavelengths_default":  mapping.get("wavelengths_default", [760, 850]),
                "measlist_var":         mapping.get("measlist_var"),
                "sampling_rate_hz":     mapping.get("sampling_rate_hz"),
                "data_type_code":       mapping.get("data_type_code", 1),
                "confidence":           mapping.get("confidence", "unknown"),
                "notes":                mapping.get("notes", ""),
                # Multi-block support
                "n_blocks":             int(mapping.get("n_blocks", 1)),
                "block_data_field":     mapping.get("block_data_field", None),
                "n_sources_var":        mapping.get("n_sources_var", None),
                "n_detectors_var":      mapping.get("n_detectors_var", None),
            }

    # ── Step 4: write mat_mapping.json ────────────────────────────────
    mat_mapping_doc = {
        "generated_by":   "autobidsify plan stage",
        "model":          model,
        "total_mat_files": len(mat_files),
        "mapped_files":   len(per_file_mappings),
        "structural_groups": len(groups),
        "files":          per_file_mappings,
    }
    write_json(mat_mapping_path, mat_mapping_doc)
    info(f"  ✓ mat_mapping.json: {len(per_file_mappings)} file(s) mapped, "
         f"{len(groups)} LLM call(s)")

    return f"_staging/{MAT_MAPPING_FILE}"


EEG_EVENT_MAPPING_FILE = "eeg_event_mapping.json"

def _build_eeg_event_mapping(
    model: str,
    all_files: List[str],
    samples: List[Dict[str, Any]],
    staging_dir: Path,
) -> Optional[str]:
    """
    Generate eeg_event_mapping.json for EEG files that have associated event files.

    Strategy:
    1. Find all EEG files from samples that have associated_event_file info.
    2. Group by source_type + raw_head fingerprint (same structure = same mapping).
    3. One LLM call per group.
    4. Write eeg_event_mapping.json with per-EEG-file entries.

    Returns relative path to eeg_event_mapping.json, or None if no EEG event files found.
    """
    eeg_event_mapping_path = staging_dir / EEG_EVENT_MAPPING_FILE

    # Already generated
    if eeg_event_mapping_path.exists():
        info("  eeg_event_mapping.json already exists, skipping")
        return f"_staging/{EEG_EVENT_MAPPING_FILE}"

    # Collect EEG samples with event file info
    eeg_with_events = [
        s for s in samples
        if s.get("kind") == "eeg" and s.get("associated_event_file")
    ]

    if not eeg_with_events:
        return None

    info(f"\nStep EEG: {len(eeg_with_events)} EEG file(s) with event info — building eeg_event_mapping.json")

    # Group by source_type + raw_head fingerprint
    groups: Dict[str, List[Dict]] = {}
    for sample in eeg_with_events:
        ev = sample["associated_event_file"]
        source_type = ev.get("source_type", "unknown")
        raw_head = ev.get("raw_head", [])
        fingerprint = source_type + "|" + "|".join(raw_head[:5])
        groups.setdefault(fingerprint, []).append(sample)

    info(f"  Found {len(groups)} event structure group(s) → {len(groups)} LLM call(s)")

    per_file_mappings: Dict[str, Dict] = {}

    for group_idx, (fingerprint, group_samples) in enumerate(groups.items(), 1):
        rep = group_samples[0]
        ev  = rep["associated_event_file"]

        payload = json.dumps({
            "source_type": ev.get("source_type"),
            "extension":   ev.get("extension"),
            "raw_head":    ev.get("raw_head", []),
            "edf_file":    rep.get("relpath"),
        }, indent=2)

        try:
            response = llm_map_eeg_events(model, payload)
        except Exception as e:
            warn(f"  LLM call failed for EEG event group {group_idx}: {e}")
            continue

        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3]
        try:
            mapping = json.loads(text.strip())
        except Exception as e:
            warn(f"  Cannot parse LLM JSON for EEG event group {group_idx}: {e}")
            continue

        info(f"    → onset_col={mapping.get('onset_col')}, "
             f"trial_type_col={mapping.get('trial_type_col')}, "
             f"onset_unit={mapping.get('onset_unit')}")

        for sample in group_samples:
            ev_info = sample["associated_event_file"]
            per_file_mappings[sample["relpath"]] = {
                "eeg_relpath":      sample["relpath"],
                "event_file_path":  ev_info.get("path"),
                "source_type":      ev_info.get("source_type"),
                "onset_col":        mapping.get("onset_col"),
                "duration_col":     mapping.get("duration_col"),
                "trial_type_col":   mapping.get("trial_type_col"),
                "header_row":       mapping.get("header_row", True),
                "skip_rows":        mapping.get("skip_rows", 0),
                "separator":        mapping.get("separator", "tab"),
                "onset_unit":       mapping.get("onset_unit", "seconds"),
                "duration_unit":    mapping.get("duration_unit", "seconds"),
                "notes":            mapping.get("notes", ""),
            }

    if not per_file_mappings:
        return None

    eeg_event_doc = {
        "generated_by": "autobidsify plan stage",
        "model":        model,
        "files":        per_file_mappings,
    }
    write_json(eeg_event_mapping_path, eeg_event_doc)
    info(f"  ✓ eeg_event_mapping.json: {len(per_file_mappings)} file(s) mapped")
    return f"_staging/{EEG_EVENT_MAPPING_FILE}"


EEG_AUX_MAPPING_FILE = "eeg_aux_mapping.json"

def _build_eeg_aux_mapping(
    model: str,
    eeg_aux_files: List[Dict[str, Any]],
    staging_dir: Path,
) -> Optional[str]:
    """
    Analyze EEG auxiliary files and map them to BIDS sidecar targets.
    One LLM call for all files (send all raw_head samples together).
    Returns path to eeg_aux_mapping.json or None if no aux files.
    """
    if not eeg_aux_files:
        return None

    eeg_aux_path = staging_dir / EEG_AUX_MAPPING_FILE
    if eeg_aux_path.exists():
        info("  eeg_aux_mapping.json already exists, skipping")
        return f"_staging/{EEG_AUX_MAPPING_FILE}"

    info(f"\nStep EEG-AUX: analyzing {len(eeg_aux_files)} auxiliary file(s)")

    payload = json.dumps([
        {
            "relpath":       f["relpath"],
            "filename":      f["filename"],
            "extension":     f["extension"],
            "detected_type": f["detected_type"],
            "raw_head":      f["raw_head"],
        }
        for f in eeg_aux_files
    ], indent=2)

    try:
        response = llm_analyze_eeg_aux(model, payload)
    except Exception as e:
        warn(f"  LLM call failed for EEG aux analysis: {e}")
        return None

    text = response.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text[:-3]
    try:
        mapping_list = json.loads(text.strip())
    except Exception as e:
        warn(f"  Cannot parse LLM JSON for EEG aux mapping: {e}")
        return None

    # Index by relpath for easy lookup
    mapping_doc = {
        "generated_by": "autobidsify plan stage",
        "model":        model,
        "files":        {item["relpath"]: item for item in mapping_list
                         if isinstance(item, dict) and "relpath" in item},
    }
    write_json(eeg_aux_path, mapping_doc)
    n_useful = sum(1 for v in mapping_doc["files"].values()
                   if v.get("content_type") not in ("irrelevant", "unknown"))
    info(f"  ✓ eeg_aux_mapping.json: {len(mapping_doc['files'])} files analyzed, "
         f"{n_useful} with usable content")
    return f"_staging/{EEG_AUX_MAPPING_FILE}"


NAMING_STRUCTURE_KEY = "naming_structure"

def _layout_signature(relpath: str) -> tuple:
    parts = relpath.split('/')
    fname_toks = FilenameTokenizer.tokenize(parts[-1])
    shape = tuple('d' if t.isdigit() else 'a' for t in fname_toks)
    return (len(parts), len(fname_toks), shape)

def _build_naming_structure(model, data_files, describe, id_strategy):
    """One LLM call per layout signature. Returns list of naming_structure entries."""
    from autobidsify.llm import llm_label_naming_structure
    from autobidsify.filename_tokenizer import FilenamePatternAnalyzer
    import yaml as _yaml
    groups = {}
    for f in data_files:
        groups.setdefault(_layout_signature(f), []).append(f)
    entries = []
    for sig, files in groups.items():
        # build per-segment position distributions for this group
        max_depth = min(sig[0], 4)
        seg_stats = {}
        for seg in range(-1, -max_depth - 1, -1):
            names = [ff.split('/')[seg] for ff in files if abs(seg) <= len(ff.split('/'))]
            seg_stats[seg] = FilenamePatternAnalyzer(names).analyze_token_statistics().get("token_positions", {})
        payload = json.dumps({
            "segment_distributions": {str(k): v for k, v in seg_stats.items()},
            "samples": [{"path": ff, "tokens_seg_-1": FilenameTokenizer.tokenize(ff.split('/')[-1])}
                        for ff in files[:8]],
            "describe": describe,
            "id_strategy": id_strategy,
        }, indent=2)
        try:
            resp = llm_label_naming_structure(model, payload)
            txt = resp.strip()
            if txt.startswith("```"): txt = "\n".join(txt.split("\n")[1:])
            if txt.endswith("```"): txt = txt[:-3]
            parsed = _yaml.safe_load(txt.strip())
            # LLM may return a single dict or a list; normalise to a list
            candidates = parsed if isinstance(parsed, list) else [parsed]
            # Python OWNS matching entirely: applies_to is always derived from the
            # actual extensions present in THIS group's files. The LLM's own
            # applies_to is ignored, so it can never inject globs (e.g. *.dcm,
            # *.snirf) for file types this dataset does not contain.
            exts = sorted({('.' + ff.rsplit('.', 1)[-1]).lower()
                           for ff in files if '.' in ff})
            if len(exts) == 1:
                group_match = '**/*' + exts[0]
            elif len(exts) > 1:
                group_match = '**/*'      # mixed group: match any, roles still apply
            else:
                group_match = '**/*'
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                roles = cand.get("roles")
                if not isinstance(roles, dict) or not roles:
                    continue
                # take only the LLM's role labels; discard its applies_to
                ap = {"match": group_match}
                entries.append({"applies_to": ap, "roles": roles,
                                "_fname_shape": [sig[1], sig[2]]})
        except Exception as e:
            warn(f"  naming_structure: group {sig} failed: {e}")
    return entries


def _reconcile_naming_structure(entries):
    """
    LLM may label structurally-identical groups inconsistently (e.g. it assigns
    task to one modality but leaves another as mode:none even though both
    filenames share the same token layout). Deterministically borrow a missing
    role from a sibling entry that has the same filename token shape. No LLM.
    Returns (entries, notes).
    """
    from collections import defaultdict

    def _defined(spec):
        return (isinstance(spec, dict)
                and spec.get("mode") in ("from_path", "lookup", "constant"))

    buckets = defaultdict(list)
    for i, e in enumerate(entries):
        buckets[tuple(map(str, e.get("_fname_shape") or []))].append(i)

    notes = []
    ROLE_NAMES = ("subject", "session", "run", "task", "acq")
    for _key, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        for role in ROLE_NAMES:
            donor = None
            for i in idxs:
                if _defined((entries[i].get("roles") or {}).get(role)):
                    donor = entries[i]["roles"][role]
                    break
            if donor is None:
                continue
            for i in idxs:
                roles = entries[i].setdefault("roles", {})
                if not _defined(roles.get(role)):
                    roles[role] = donor
                    notes.append({"entry_index": i, "role": role,
                                  "action": "borrowed_from_same_shape_sibling"})

    for e in entries:
        e.pop("_fname_shape", None)
    return entries, notes


def _apply_evidence_gate(naming_structure, data_files):
    """Evidence gate for session/run. An entity that never distinguishes any
    file is spurious (weak models invent constant or single-valued session/run
    even when the data has neither). For each entry's 'session' and 'run' role:
      - mode 'constant'                       -> demote to none (zero contrast)
      - mode 'from_path' resolving to <2 distinct values across ALL files
                                              -> demote to none (no contrast)
      - mode 'from_path' with >=2 distinct    -> keep (real, e.g. ses1..ses6)
    Uses the same _pick the executor uses, so gate and execution never diverge.
    Returns a list of demotion records for provenance.
    """
    from autobidsify.filename_tokenizer import _pick

    demotions = []
    for entry in naming_structure or []:
        roles = entry.get("roles", {})
        if not isinstance(roles, dict):
            continue
        applies = (entry.get("applies_to") or {})
        for role_name in ("session", "run"):
            spec = roles.get(role_name)
            if not isinstance(spec, dict):
                continue
            mode = spec.get("mode", "none")
            if mode == "none":
                continue

            if mode == "constant":
                roles[role_name] = {"mode": "none"}
                demotions.append({
                    "role": role_name, "reason": "constant (zero contrast)",
                    "applies_to": applies,
                    "old": spec.get("value"),
                })
                continue

            if mode == "from_path":
                sources = spec.get("sources", [])
                join = spec.get("join", "")
                seen = set()
                for f in data_files:
                    v = _pick(f, sources, join)
                    if v not in (None, ""):
                        seen.add(v)
                if len(seen) < 2:
                    roles[role_name] = {"mode": "none"}
                    demotions.append({
                        "role": role_name,
                        "reason": f"from_path resolved to {len(seen)} distinct "
                                  f"value(s) across {len(data_files)} files",
                        "applies_to": applies,
                        "distinct_values": sorted(seen),
                    })
    return demotions


# ============================================================================
# Main entry point
# ============================================================================

def build_bids_plan(model: str, planning_inputs: Dict[str, Any],
                    out_dir: Path, id_strategy: str = "auto",
                    anonymize: bool = False,
                    describe: str = "") -> Dict[str, Any]:
    """
    Build BIDS conversion plan (LLM-first, Python-validates).

    Steps:
      1. Python extracts subject hints from directory/filename structure (advisory).
      2. Build LLM payload — data files only, non-data files filtered out.
      3. Call LLM to generate full BIDSPlan (assignment_rules, mappings, metadata).
      4. Validate subject count: trust LLM rules, only update count field if needed.
      5. Write participants.tsv from LLM plan.
      6. Merge any extra metadata columns from LLM.
      7. Save BIDSPlan.yaml.
    """
    info("=== Generating Unified BIDS Plan ===")

    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    all_files       = evidence_bundle.get("all_files", [])
    user_hints      = evidence_bundle.get("user_hints", {})
    user_n_subjects = user_hints.get("n_subjects")

    staging_dir = out_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Python structural hints (advisory only) ───────────────
    info("Step 1: Python extracting structural hints...")
    subject_info = _extract_subjects_from_directory_structure(all_files)
    if not subject_info["success"]:
        info("  Directory-level detection failed, trying flat filename analysis...")
        subject_info = _extract_subjects_from_flat_filenames(all_files)

    python_subject_count = subject_info.get("subject_count", 0)
    info(f"  Python hint: {python_subject_count} subjects "
         f"(method: {subject_info.get('method', 'unknown')})")

    # ── Step 2: Build LLM payload (data files only) ───────────────────
    info("\nStep 2: Building LLM payload...")
    data_files = [f for f in all_files if _is_data_file(f)]

    if len(data_files) <= 200:
        sample_files = data_files
    else:
        n       = len(data_files)
        indices = (list(range(0, min(50, n))) +
                   list(range(n // 2 - 25, n // 2 + 25)) +
                   list(range(max(0, n - 50), n)))
        sample_files = [data_files[i] for i in sorted(set(indices)) if i < n]

    info(f"  Data files for LLM: {len(sample_files)} "
         f"(filtered from {len(all_files)} total)")

    # Scrub --describe text if anonymize=True before passing to LLM
    describe_text = scrub_text(describe) if anonymize else describe

    optimized_bundle = {
        "root":          evidence_bundle.get("root"),
        "counts_by_ext": {
            k: v for k, v in evidence_bundle.get("counts_by_ext", {}).items()
            if k.lower() in _DATA_EXTS
        },
        "user_hints":    user_hints,
        "total_files":   len(all_files),
        "data_files":    len(data_files),
        "sample_files":  sample_files,
        "id_strategy":   id_strategy,
        "describe":      describe_text,
        "python_subject_analysis": {
            "success":       subject_info["success"],
            "method":        subject_info.get("method"),
            "subject_count": python_subject_count,
            "subject_examples": [
                {
                    "original":   rec["original_id"],
                    "numeric_id": rec.get("numeric_id"),
                    "site":       rec.get("site"),
                }
                for rec in subject_info.get("subject_records", [])[:20]
            ],
            "note": (
                "This is a HINT from Python's heuristic detection. "
                "Trust user_hints.n_subjects over this count. "
                "Use your own analysis of sample_files to determine "
                "the true subject structure."
            ),
        },
    }

    # ── Step 3: Call LLM ──────────────────────────────────────────────
    info("\nStep 3: Calling LLM for full plan generation...")
    evidence_json = json.dumps(optimized_bundle, indent=2)
    plan_response = llm_bids_plan(model, evidence_json, anonymize=anonymize)

    if not plan_response:
        fatal("LLM returned empty response for BIDS plan")

    try:
        text = plan_response.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3]
        plan_yaml = yaml.safe_load(text.strip())
    except yaml.YAMLError as e:
        fatal(f"BIDS plan YAML parsing failed: {e}")

    if not isinstance(plan_yaml, dict):
        fatal("BIDS plan is not a valid YAML dict")

    # ── Step 4: Validate subject count ────────────────────────────────
    info("\nStep 4: Validating subject count...")
    llm_count = plan_yaml.get("subjects", {}).get("count", 0)
    info(f"  LLM produced:  {llm_count} subjects")
    info(f"  User provided: {user_n_subjects} subjects (--nsubjects)")

    if user_n_subjects and llm_count != user_n_subjects:
        warn(f"  ⚠ LLM subject count ({llm_count}) ≠ user-provided count "
             f"({user_n_subjects}). Trusting LLM assignment_rules; updating count only.")
        plan_yaml["subjects"]["count"] = user_n_subjects
    
    # ── Step 4.5: naming_structure (authoritative for subject + entities) ──
    info("\nStep 4.5: Building naming_structure...")
    naming_structure = _build_naming_structure(model, data_files, describe, id_strategy)
    # Self-check: LLM may label structurally-identical groups inconsistently.
    # Deterministically borrow a missing role from a same-shape sibling group.
    naming_structure, recon_notes = _reconcile_naming_structure(naming_structure)
    if recon_notes:
        plan_yaml["naming_structure_reconciliation"] = recon_notes
        info(f"  reconciled {len(recon_notes)} role(s) across same-shape groups")

    # Evidence gate: strip session/run roles that never distinguish any file.
    _gate_demotions = _apply_evidence_gate(naming_structure, data_files)
    if _gate_demotions:
        plan_yaml["evidence_gate_demotions"] = _gate_demotions
        for _d in _gate_demotions:
            info(f"  ✓ demoted {_d['role']} to none — {_d['reason']}")

    plan_yaml[NAMING_STRUCTURE_KEY] = naming_structure

    # naming_structure is now the SINGLE authority for subject + entity
    # resolution. Drop the legacy assignment_rules / filename_rules the LLM
    # full-plan step may have emitted, so the executor cannot run two competing
    # naming mechanisms and produce inconsistent, colliding output.
    if naming_structure:
        if plan_yaml.pop("assignment_rules", None) is not None:
            info("  ✓ dropped legacy assignment_rules (naming_structure is authoritative)")
        for _m in plan_yaml.get("mappings", []):
            _m.pop("filename_rules", None)

    from autobidsify.filename_tokenizer import expand_naming_structure
    naming_per_file, naming_subject_map = expand_naming_structure(
        data_files, naming_structure, id_strategy, nsubjects=user_n_subjects)
    info(f"  naming_structure covered {len(naming_per_file)} file(s), "
         f"{len(naming_subject_map)} subject(s)")

    # ── Self-check (user --nsubjects is the authoritative oracle) ──────
    plan_questions: List[Dict[str, Any]] = []

    # Check 1: subject count. User-provided --nsubjects is assumed correct and
    # has highest priority; a mismatch means naming_structure mis-resolved
    # subjects. Block rather than silently emit a wrong dataset.
    if user_n_subjects and len(naming_subject_map) != user_n_subjects:
        got = len(naming_subject_map)
        warn(f"  ⚠ naming_structure produced {got} subjects, "
             f"but --nsubjects={user_n_subjects}")
        plan_questions.append({
            "severity": "block",
            "message": (
                f"naming_structure resolved {got} subject(s) but you specified "
                f"--nsubjects {user_n_subjects}. The subject field was likely taken "
                f"from the wrong part of the filename or path. Please clarify in "
                f"--describe which part of the name identifies the subject "
                f"(and, if applicable, which parts distinguish sites or sessions), "
                f"then re-run."
            ),
        })

    # Check 2: BIDS required-entity — nirs/eeg REQUIRE a task entity.
    # A missing task means files differing only by task would collide.
    _missing_task = [
        rel for rel, e in naming_per_file.items()
        if (rel.lower().endswith(('.snirf', '.nirs', '.mat',
                                  '.edf', '.vhdr', '.set', '.bdf'))
            and not e.get("task"))
    ]
    if _missing_task:
        warn(f"  ⚠ {len(_missing_task)} nirs/eeg file(s) have no 'task' entity")
        plan_questions.append({
            "severity": "block",
            "message": (
                f"{len(_missing_task)} nirs/eeg file(s) could not be assigned a "
                f"required 'task' entity (e.g. {_missing_task[0]}). Files that share "
                f"subject/session/run but differ only by task would overwrite each "
                f"other. Please explain in --describe how the task is encoded in the "
                f"filename or metadata, then re-run."
            ),
        })

    # ── Step 5: Write participants.tsv ────────────────────────────────
    info("\nStep 5: Generating participants.tsv from LLM plan...")
    if naming_subject_map:
        # authoritative subjects from naming_structure
        labels = sorted(naming_subject_map.keys(),
                        key=lambda s: (len(s), s))
        plan_yaml["subjects"] = {"count": len(labels), "labels": labels,
                                 "id_strategy": id_strategy, "source": "naming_structure"}
        # Replace entirely so stale LLM subject keys can't leak into the TSV
        plan_yaml["participant_metadata"] = {
            sid: {"original_id": raw} for sid, raw in naming_subject_map.items()
        }
        _write_participants_from_subject_map(naming_subject_map, out_dir)
    else:
        _write_participants_from_plan(plan_yaml, out_dir, user_n_subjects)
    
    if anonymize:
        parts_path = out_dir / "participants.tsv"
        if parts_path.exists():
            scrub_participants_tsv(parts_path)
            info("  ✓ participants.tsv scrubbed (anonymize=True)")

    # ── Step 6: Merge extra metadata columns ──────────────────────────
    info("\nStep 6: Merging participant metadata...")
    if "participant_metadata" in plan_yaml:
        _merge_participants_from_llm_metadata(plan_yaml, out_dir)
    
    # ── Step MAT: mat_mapping.json (only when .mat files present) ─────
    data_root_str = evidence_bundle.get("root", "")
    mat_mapping_relpath = None
    if data_root_str:
        mat_mapping_relpath = _build_mat_mapping(
            model=model,
            all_files=all_files,
            data_root=Path(data_root_str),
            staging_dir=staging_dir,
        )

    # Inject mat_mapping_path into any nirs mapping entries that cover .mat files
    if mat_mapping_relpath:
        for m in plan_yaml.get("mappings", []):
            if m.get("modality") == "nirs":
                # Check if this mapping covers .mat files
                patterns = m.get("match", [])
                covers_mat = any(
                    ".mat" in p.lower() or p == "**/*.mat"
                    for p in patterns
                )
                if covers_mat or not patterns:  # no patterns = catches all nirs
                    m["mat_mapping_path"] = mat_mapping_relpath
    
    # ── Step EEG: eeg_event_mapping.json (only when EEG files present) ─
    samples = evidence_bundle.get("samples", [])
    eeg_event_mapping_relpath = _build_eeg_event_mapping(
        model=model,
        all_files=all_files,
        samples=samples,
        staging_dir=staging_dir,
    )

    # Inject eeg_event_mapping_path into eeg mapping entries
    if eeg_event_mapping_relpath:
        for m in plan_yaml.get("mappings", []):
            if m.get("modality") == "eeg":
                m["eeg_event_mapping_path"] = eeg_event_mapping_relpath
    
    # ── Step EEG-AUX: eeg_aux_mapping.json ───────────────────────────
    eeg_aux_files = evidence_bundle.get("eeg_auxiliary_files", [])
    eeg_aux_mapping_relpath = _build_eeg_aux_mapping(
        model=model,
        eeg_aux_files=eeg_aux_files,
        staging_dir=staging_dir,
    )
    if eeg_aux_mapping_relpath:
        for m in plan_yaml.get("mappings", []):
            if m.get("modality") == "eeg":
                m["eeg_aux_mapping_path"] = eeg_aux_mapping_relpath

    # ── Step 7: Save plan ─────────────────────────────────────────────
    plan_yaml["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "model":        model,
        "id_strategy":  id_strategy,
        "anonymize":    anonymize,
    }
    plan_path = staging_dir / BIDS_PLAN
    write_yaml(plan_path, plan_yaml)
    info(f"\n✓ Plan saved: {plan_path}")

    final_count = plan_yaml.get("subjects", {}).get("count", llm_count)
    info(f"\n=== Complete: {final_count} subjects ===")
    if plan_questions:
        warn("  ⚠ Plan has blocking questions — execute stage will be skipped")
        return {"status": "blocked", "warnings": [], "questions": plan_questions}
    return {"status": "ok", "warnings": [], "questions": []}


# ============================================================================
# NIRS header planning (separate from main BIDS plan)
# ============================================================================

def nirs_plan_headers(model: str, planning_inputs: Dict[str, Any],
                      out_dir: Path) -> Dict[str, Any]:
    """Plan fNIRS header mappings via two-step LLM draft/normalize."""
    info("=== Planning NIRS headers ===")

    evidence_bundle = planning_inputs.get("evidence_bundle", {})
    evidence_json   = json.dumps(evidence_bundle, indent=2)

    draft_response = llm_nirs_draft(model, evidence_json)
    if not draft_response:
        return {"warnings": [], "questions": []}

    draft = _parse_llm_json_response(draft_response, "nirs_draft")
    if not draft:
        return {"warnings": [], "questions": []}

    staging_dir = out_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    write_json(staging_dir / HEADERS_DRAFT, draft)

    normalized_response = llm_nirs_normalize(model, json.dumps(draft, indent=2))
    if not normalized_response:
        return {"warnings": [], "questions": []}

    normalized = _parse_llm_json_response(normalized_response, "nirs_normalize")
    if not normalized:
        return {"warnings": [], "questions": []}

    write_json(staging_dir / HEADERS_NORMALIZED, normalized)
    info("✓ NIRS headers saved")
    return {"warnings": [], "questions": []}


def mri_plan_voxel_mappings(model: str, planning_inputs: Dict[str, Any],
                             out_dir: Path) -> Dict[str, Any]:
    """MRI voxel mapping planning (stub)."""
    return {"warnings": [], "questions": []}