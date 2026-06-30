# trio.py

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from autobidsify.utils import write_json, write_text, read_json, warn, info, fatal, debug
from autobidsify.constants import (
    TRIO_README, TRIO_PARTICIPANTS, TRIO_DATASET_DESC,
    LICENSE_WHITELIST, SEVERITY_WARN, SEVERITY_INFO
)
from autobidsify.llm import llm_trio_dataset_description, llm_trio_readme, llm_trio_participants
from autobidsify.anonymize import scrub_evidence_bundle, scrub_participants_tsv

DEBUG_MODE = True


# ============================================================================
# Trio status check
# ============================================================================

def check_trio_status(out_dir: Path) -> Dict[str, Any]:
    status = {
        "dataset_description": {"exists": False, "path": None, "data": None},
        "readme":               {"exists": False, "path": None, "variant": None},
        "participants":         {"exists": False, "path": None}
    }

    dd_path = out_dir / TRIO_DATASET_DESC
    if dd_path.exists():
        status["dataset_description"]["exists"] = True
        status["dataset_description"]["path"] = dd_path
        try:
            status["dataset_description"]["data"] = read_json(dd_path)
        except Exception:
            pass

    readme_variants = ['readme', 'readme.md', 'readme.txt', 'readme.rst']
    for item in out_dir.iterdir():
        if item.is_file() and item.name.lower() in readme_variants:
            status["readme"]["exists"] = True
            status["readme"]["path"] = item
            status["readme"]["variant"] = item.name
            break

    parts_path = out_dir / TRIO_PARTICIPANTS
    if parts_path.exists():
        status["participants"]["exists"] = True
        status["participants"]["path"] = parts_path

    return status


# ============================================================================
# License normalization — complete alias table
# ============================================================================

def normalize_license_locally(license_str: str) -> Optional[str]:
    """
    Normalize ANY license string to BIDS canonical form.

    Strategy:
    1. Strip input: remove hyphens / spaces / dots / underscores, uppercase
    2. Match against a complete alias table
    3. Return canonical BIDS value, or 'Non-Standard' if unrecognized

    Handles:
    - Exact canonical: "CC0", "CC-BY-4.0"
    - Case variants:   "cc0", "Cc-By-4.0"
    - Natural language: "creative commons zero", "public domain"
    - Verbose forms:   "Creative Commons Attribution 4.0 International"
    - Common typos:    "CC BY 4.0", "CCBY4"
    """
    if not license_str:
        return None

    # Normalize: remove separators and uppercase
    key = re.sub(r'[\s\-\._]+', '', license_str.upper())

    ALIAS_TABLE: Dict[str, List[str]] = {
        'CC0': [
            'CC0', 'CC010', 'CC01',
            'CREATIVECOMMONSZERO', 'CREATIVECOMMONS0',
            'CC0UNIVERSALPUBLICDOMAIN', 'CC010UNIVERSAL',
            'CC0UNIVERSAL',
            'ZERORIGHTSPUBLICDOMAIN', 'CC0LICENSE',
        ],
        'PD': [
            'PD', 'PUBLICDOMAIN', 'PUBLIEDOMAIN',
        ],
        'PDDL': [
            'PDDL', 'PDDL10', 'OPENDATACOMMONSPDD',
            'PUBLICDOMAINDEDICATIONLICENSE',
        ],
        'CC-BY-4.0': [
            'CCBY40', 'CCBY4', 'CCBY',
            'CREATIVECOMMONSATTRIBUTION40',
            'CREATIVECOMMONSATTRIBUTION4',
            'CREATIVECOMMONSATTRIBUTION40INTERNATIONAL',
        ],
        'CC-BY-SA-4.0': [
            'CCBYSA40', 'CCBYSA4', 'CCBYSA',
            'CREATIVECOMMONSATTRIBUTIONSHAREALIKE40',
            'CREATIVECOMMONSATTRIBUTIONSHAREALIKE4',
        ],
        'CC-BY-NC-4.0': [
            'CCBYNC40', 'CCBYNC4', 'CCBYNC',
            'CREATIVECOMMONSATTRIBUTIONNONCOMMERCIAL40',
            'CREATIVECOMMONSATTRIBUTIONNONCOMMERCIAL4',
        ],
        'CC-BY-NC-SA-4.0': [
            'CCBYNCSA40', 'CCBYNCSA4',
            'CREATIVECOMMONSATTRIBUTIONNONCOMMERCIALSHAREALIKE40',
        ],
        'CC-BY-NC-ND-4.0': [
            'CCBYNCND40', 'CCBYNCND4',
            'CREATIVECOMMONSATTRIBUTIONNONCOMMERCIALNODERIV40',
        ],
        'MIT': [
            'MIT', 'MITLICENSE', 'MITOPENSOURCE',
        ],
        'BSD-3-Clause': [
            'BSD3CLAUSE', 'BSD3', 'BSDNEW', 'BSDREVISED', 'BSD3CLAUSELICENSE',
        ],
        'BSD-2-Clause': [
            'BSD2CLAUSE', 'BSD2', 'BSDORIGINAL', 'BSDOLD', 'BSDSIMPLIFIED',
        ],
        'GPL-2.0': [
            'GPL20', 'GPL2', 'GNUGPL2', 'GNUGENERALPUBLICLICENSE2',
        ],
        'GPL-2.0+': [
            'GPL20+', 'GPL2+', 'GPL2ORLATER', 'GPL20ORLATER',
        ],
        'GPL-3.0': [
            'GPL30', 'GPL3', 'GNUGPL3', 'GNUGENERALPUBLICLICENSE3',
        ],
        'GPL-3.0+': [
            'GPL30+', 'GPL3+', 'GPL3ORLATER', 'GPL30ORLATER',
        ],
        'LGPL-3.0+': [
            'LGPL30+', 'LGPL3+', 'LGPL3ORLATER', 'GNULЕССЕРГPL3',
        ],
        'MPL': [
            'MPL', 'MPL20', 'MPL2', 'MOZILLAPUBLICLICENSE',
            'MOZILLAPUBLICLICENSE20',
        ],
        'CDDL-1.0': [
            'CDDL', 'CDDL10', 'COMMONDEVELOPMENTANDDISTRIBUTIONLICENSE',
        ],
        'GFDL-1.3': [
            'GFDL', 'GFDL13', 'GNUFREEDOCUMENTATIONLICENSE13',
        ],
        'Non-Standard': [
            'NONSTANDARD', 'NONSTANDARDLICENSE', 'CUSTOM', 'OTHER',
            'PROPRIETARY', 'RESTRICTED',
        ],
    }

    for canonical, variants in ALIAS_TABLE.items():
        if key in variants:
            return canonical

    return 'Non-Standard'


# ============================================================================
# Internal helpers
# ============================================================================

def _parse_llm_json_response(
    response_text: str,
    step_name: str,
    show_preview: bool = True
) -> Optional[Dict[str, Any]]:
    if not response_text or not response_text.strip():
        warn(f"{step_name}: LLM returned empty response")
        return None

    text = response_text.strip()

    # Strip markdown fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        lines = text.split('\n')
        text = '\n'.join(lines[1:])
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct parse
    try:
        obj = json.loads(text)
        if DEBUG_MODE:
            info(f"{step_name}: ✓ JSON parsed successfully")
        return obj
    except json.JSONDecodeError as e:
        if DEBUG_MODE:
            debug(f"{step_name}: Direct parse failed: {e}")

    # Try raw_decode
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text)
        if DEBUG_MODE:
            debug(f"{step_name}: ✓ JSON via raw_decode")
        return obj
    except Exception:
        pass

    # Try regex extraction
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            if DEBUG_MODE:
                debug(f"{step_name}: ✓ JSON via regex")
            return obj
    except Exception:
        pass

    warn(f"{step_name}: Failed to parse JSON")
    if show_preview:
        warn(f"Response preview: {text[:500]}...")
    return None


def _is_markdown_content(text: str) -> bool:
    t = text.strip()
    return any([
        t.startswith('#'), t.startswith('##'),
        '# ' in t[:100], '\n## ' in t[:200],
        t.startswith('**'), '- ' in t[:100], '\n- ' in t[:200],
    ])


def _validate_dataset_description(dd: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues = []
    if not dd.get("Name"):
        issues.append("Missing required field: Name")
    if not dd.get("BIDSVersion"):
        issues.append("Missing required field: BIDSVersion")
    if not dd.get("License"):
        issues.append("Missing required field: License")
    elif dd.get("License") not in LICENSE_WHITELIST:
        issues.append(f"License '{dd.get('License')}' not in BIDS whitelist")
    for f in ["Authors", "Funding", "EthicsApprovals"]:
        if f in dd and not isinstance(dd[f], list):
            issues.append(f"{f} must be an array")
    if dd.get("License") == "Non-Standard" and not dd.get("DataLicense"):
        issues.append("License='Non-Standard' requires DataLicense field")
    empty = [k for k, v in dd.items() if v == "" or v == []]
    if empty:
        issues.append(f"Empty fields (will be removed): {', '.join(empty)}")
    is_valid = not any("Missing required" in i or "must be an array" in i for i in issues)
    return is_valid, issues


def _fix_field_types(dd: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    fixed = dd.copy()
    fixes = []
    for field in ["Authors", "Funding", "EthicsApprovals"]:
        if field not in fixed:
            continue
        val = fixed[field]
        if isinstance(val, str):
            if val.strip():
                fixed[field] = [val]
                fixes.append(f"Converted {field} from string to array")
            else:
                del fixed[field]
        elif isinstance(val, list) and len(val) == 0:
            del fixed[field]
    keys_rm = [k for k, v in fixed.items()
               if v == "" and k not in ("Name", "BIDSVersion", "DatasetType", "License")]
    for k in keys_rm:
        del fixed[k]
    return fixed, fixes


# ============================================================================
# dataset_description.json
# ============================================================================

def generate_dataset_description(model: str, bundle: Dict[str, Any], out_dir: Path,
                                  anonymize: bool = False) -> Dict[str, Any]:
    """
    Generate dataset_description.json.

    Design:
    - LLM does ALL semantic extraction (Name, Authors, License, etc.)
      It outputs a 'raw_license' field (natural language, no format constraints)
    - Python does the final normalization:
        raw_license → normalize_license_locally() → BIDS canonical
    - This is universally robust: user can write anything, LLM understands it,
      Python maps it to the exact BIDS value.

    Priority for final value:
        Python-normalized License > existing_dd > llm_dd (other fields)
    """
    info("=== Generating dataset_description.json ===")
    dd_path = out_dir / TRIO_DATASET_DESC
    warnings_list: List[str] = []

    # ── Step 1: Load existing file ────────────────────────────────────
    existing_dd = None
    if dd_path.exists():
        try:
            existing_dd = read_json(dd_path)
            info(f"Found existing file: {dd_path}")
            _, issues = _validate_dataset_description(existing_dd)
            if issues:
                for issue in issues:
                    info(f"  ⚠ {issue}")
                existing_dd, fixes = _fix_field_types(existing_dd)
                for fix in fixes:
                    info(f"  ✓ {fix}")
        except Exception as e:
            warn(f"Could not read existing file: {e}")

    # ── Step 2: Call LLM ──────────────────────────────────────────────
    # LLM receives full user_text and documents.
    # Key instruction: output 'raw_license' as a plain string
    # (whatever the user said, verbatim or paraphrased — no format required).
    # Python will normalize it afterwards.
    payload = json.dumps({
        "user_hints":    bundle.get("user_hints", {}),
        "documents":     bundle.get("documents", []),
        "counts_by_ext": bundle.get("counts_by_ext", {}),
        "existing":      existing_dd,
        "task_instructions": (
            "Extract dataset metadata from user_hints.user_text and documents. "
            "For the license field: output it as 'raw_license' (plain string, "
            "exactly what the user wrote or what the document says — "
            "e.g. 'CC0', 'Creative Commons Zero', 'public domain', 'CC BY 4.0'). "
            "Do NOT try to normalize the license yourself. "
            "Python will handle normalization. "
            "For Authors: extract ONLY from user_hints.user_text citations/references. "
            "Do NOT extract authors from documents[]."
        )
    }, ensure_ascii=False)

    result = None
    llm_dd = None
    raw_license_from_llm: Optional[str] = None

    try:
        response_text = llm_trio_dataset_description(model, payload, anonymize=anonymize)
        if DEBUG_MODE:
            info(f"LLM response length: {len(response_text)} chars")
        result = _parse_llm_json_response(response_text, "dataset_description")
        if result:
            llm_dd = result.get("dataset_description", {})
            # Extract raw_license from top-level or nested
            raw_license_from_llm = (
                result.get("raw_license") or
                llm_dd.get("raw_license") or
                llm_dd.get("License")      # fallback if LLM put it in License
            )
            if DEBUG_MODE:
                info(f"LLM returned fields: {list(llm_dd.keys())}")
                info(f"raw_license from LLM: {raw_license_from_llm!r}")
    except Exception as e:
        warn(f"LLM call failed: {e}")

    # ── Step 3: Python normalizes the license ─────────────────────────
    # This is the ONLY place where license normalization happens.
    # Input can be anything the LLM returned; output is always a BIDS value.
    normalized_license: Optional[str] = None
    if raw_license_from_llm:
        normalized_license = normalize_license_locally(raw_license_from_llm)
        info(f"  License: '{raw_license_from_llm}' → '{normalized_license}'")
    elif existing_dd and existing_dd.get("License"):
        # Try to normalize what's already in the existing file
        existing_lic = existing_dd["License"]
        normalized_license = normalize_license_locally(existing_lic)
        if normalized_license != existing_lic:
            info(f"  License (from existing): '{existing_lic}' → '{normalized_license}'")
        else:
            normalized_license = existing_lic

    # ── Step 4: Merge: existing_dd < llm_dd, then apply normalized license ──
    base: Dict[str, Any] = {}

    if llm_dd:
        # Remove raw_license key if LLM put it there — we handle it separately
        llm_dd_clean = {k: v for k, v in llm_dd.items()
                        if k not in ("raw_license",)}
        base.update(llm_dd_clean)

    if existing_dd:
        for k, v in existing_dd.items():
            if v:
                base[k] = v

    # License: always use Python-normalized value (highest priority)
    if normalized_license:
        base["License"] = normalized_license

    if not base:
        fatal("No data available for dataset_description.json")
        return {"warnings": [], "questions": []}

    # ── Step 5: Build final structure ─────────────────────────────────
    final_dd: Dict[str, Any] = {
        "Name":        base.get("Name", ""),
        "BIDSVersion": "1.10.0",
        "DatasetType": base.get("DatasetType", "raw"),
        "License":     base.get("License", ""),
    }

    # Authors (must be array)
    authors = base.get("Authors")
    if authors:
        if isinstance(authors, str) and authors.strip():
            final_dd["Authors"] = [authors]
        elif isinstance(authors, list) and authors:
            final_dd["Authors"] = authors

    # Optional fields
    for field in ["Acknowledgements", "HowToAcknowledge", "Funding",
                  "EthicsApprovals", "ReferencesAndLinks", "DatasetDOI",
                  "HEDVersion", "GeneratedBy", "SourceDatasets"]:
        val = base.get(field)
        if val:
            final_dd[field] = val

    if base.get("License") == "Non-Standard" and base.get("DataLicense"):
        final_dd["DataLicense"] = base["DataLicense"]

    # ── Step 6: Validate ──────────────────────────────────────────────
    if not final_dd.get("Name"):
        warnings_list.append("WARNING: Missing 'Name' field (REQUIRED)")

    lic = final_dd.get("License", "")
    if not lic:
        warnings_list.append(
            "WARNING: License not found. "
            "Add 'License: CC0' (or other) to --describe, "
            "or include it in your dataset documentation."
        )
    elif lic not in LICENSE_WHITELIST:
        # One more normalization attempt on the final value
        again = normalize_license_locally(lic)
        if again and again in LICENSE_WHITELIST:
            final_dd["License"] = again
            info(f"  ✓ License re-normalized: '{lic}' → '{again}'")
        else:
            warnings_list.append(f"WARNING: License '{lic}' not in BIDS whitelist")

    if final_dd.get("License") == "Non-Standard" and not final_dd.get("DataLicense"):
        warnings_list.append("WARNING: License='Non-Standard' requires 'DataLicense' field")

    # Remove empty strings / empty arrays
    final_dd = {k: v for k, v in final_dd.items() if v != "" and v != []}

    # ── Step 7: Write ─────────────────────────────────────────────────
    write_json(dd_path, final_dd)
    action = "Updated" if existing_dd else "Created"
    info(f"✓ {action}: {dd_path}")
    info(f"  License : {final_dd.get('License', 'MISSING')}")
    info(f"  Name    : {final_dd.get('Name', 'MISSING')}")

    if result and "extraction_log" in result:
        info("LLM extraction log:")
        for field, source in result["extraction_log"].items():
            info(f"  {field}: {source}")

    if result and "warnings" in result:
        warnings_list.extend(result["warnings"])

    return {
        "warnings": warnings_list,
        "questions": result.get("questions", []) if result else []
    }


# ============================================================================
# README.md
# ============================================================================

def generate_readme(model: str, bundle: Dict[str, Any], out_dir: Path,
                    anonymize: bool = False) -> Dict[str, Any]:
    info("=== Generating README.md ===")

    readme_variants = ['readme', 'readme.md', 'readme.txt', 'readme.rst']
    for item in out_dir.iterdir():
        if item.is_file() and item.name.lower() in readme_variants:
            info(f"✓ Found existing: {item.name}")
            return {"warnings": [], "questions": []}

    payload = json.dumps({
        "documents":  bundle.get("documents", []),
        "user_hints": bundle.get("user_hints", {}),
        "existing_readme": None
    }, ensure_ascii=False)

    try:
        response_text = llm_trio_readme(model, payload, anonymize=anonymize)
        if _is_markdown_content(response_text):
            info("✓ LLM returned direct Markdown content")
            result = {"readme_content": response_text.strip()}
        else:
            result = _parse_llm_json_response(response_text, "README", show_preview=True)
            if result is None:
                result = {"readme_content": "# Dataset\n\nNeuroimaging dataset.\n"}
    except Exception as e:
        warn(f"README generation failed: {e}")
        result = {"readme_content": "# Dataset\n\nNeuroimaging dataset.\n"}

    readme_content = result.get("readme_content", "# Dataset\n\nNeuroimaging dataset.\n")
    write_text(out_dir / TRIO_README, readme_content)
    info(f"✓ Created: {TRIO_README}")
    return {"warnings": [], "questions": []}


# ============================================================================
# participants.tsv
# ============================================================================

def generate_participants(model: str, bundle: Dict[str, Any], out_dir: Path,
                          force_simple: bool = False,
                          anonymize: bool = False) -> Dict[str, Any]:
    info("=== Generating participants.tsv ===")

    parts_path = out_dir / TRIO_PARTICIPANTS
    if parts_path.exists():
        info(f"✓ Found existing: {parts_path}")
        return {"warnings": [], "questions": []}

    n_subjects = bundle.get("user_hints", {}).get("n_subjects", 1)
    all_files  = bundle.get("all_files", [])

    if not force_simple:
        if n_subjects > 100 or len(all_files) > 500:
            info(f"Complex dataset — deferring participants.tsv to Plan stage")
            return {"warnings": [], "questions": [], "deferred": True}

    info(f"Generating basic participants.tsv ({n_subjects} subjects)")
    lines = ["participant_id\n"] + [f"sub-{i:02d}\n" for i in range(1, n_subjects + 1)]
    write_text(parts_path, "".join(lines))
    info(f"✓ Created: {parts_path}")
    info(f"  Note: Plan stage may update this file with additional columns")

    if anonymize:
        scrub_participants_tsv(parts_path)
        info("  ✓ participants.tsv scrubbed (anonymize=True)")

    return {"warnings": [], "questions": []}


# ============================================================================
# Generate all trio files
# ============================================================================

def trio_generate_all(model: str, bundle: Dict[str, Any], out_dir: Path,
                      anonymize: bool = False) -> Dict[str, Any]:
    info("=== Generating BIDS Trio (all files) ===")

    status = check_trio_status(out_dir)
    info(f"  dataset_description.json: {'EXISTS' if status['dataset_description']['exists'] else 'MISSING'}")
    info(f"  README.md:                {'EXISTS' if status['readme']['exists'] else 'MISSING'}")
    info(f"  participants.tsv:         {'EXISTS' if status['participants']['exists'] else 'MISSING'}")
    info("")

    all_warnings: List[str] = []
    all_questions: List[str] = []

    for fn, label in [
        (generate_dataset_description, "dataset_description.json"),
        (generate_readme,              "README.md"),
        (generate_participants,        "participants.tsv"),
    ]:
        r = fn(model, bundle, out_dir, anonymize=anonymize)
        all_warnings.extend(r.get("warnings", []))
        all_questions.extend(r.get("questions", []))

    dd_exists    = (out_dir / TRIO_DATASET_DESC).exists()
    readme_exists = (out_dir / TRIO_README).exists()
    parts_exists  = (out_dir / TRIO_PARTICIPANTS).exists()
    generated     = sum([dd_exists, readme_exists, parts_exists])

    info("")
    info("=== Trio Generation Summary ===")
    info(f"{'✓' if dd_exists else '✗'} dataset_description.json")
    info(f"{'✓' if readme_exists else '✗'} README.md")
    info(f"{'✓' if parts_exists else '○'} participants.tsv {'(deferred to Plan)' if not parts_exists else ''}")
    info(f"Status: {generated}/3 generated")

    return {"warnings": all_warnings, "questions": all_questions}