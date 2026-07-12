# bids_grammar.py
# Single source of truth for BIDS filename grammar and required-metadata rules.
#
# Everything that decides "what a legal BIDS filename looks like" lives here, so
# that name assembly (executor) and sidecar generation share ONE rule set instead
# of each re-deriving order/legality with ad-hoc string surgery. Adding a modality
# means adding rows to these tables, not editing assembly logic.
#
# Rules encoded here come directly from the BIDS spec and the MRS-BIDS extension
# (Bouchard et al., Sci Data 2025, https://doi.org/10.1038/s41597-025-05543-2):
#   - filenames are  sub-<label>[_ses-..]_<entities>_<suffix>.<ext>
#   - all entities have ONE fixed order and each entity appears at most once
#   - which entities are legal is decided by the datatype (anat has NO task;
#     func/nirs/eeg REQUIRE task; mrs may carry nuc/voi/rec/echo/inv)
#   - the suffix is constrained by the datatype (mrs -> svs|mrsi|mrsref|unloc)
#   - the datatype also fixes which sidecar fields are REQUIRED

import re
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# 1) Canonical entity order.
#    run MUST come after task and acq. This single total order is consistent
#    with EVERY datatype row of the BIDS entity table:
#      MRI  task acq ce rec dir run      (nuc/voi absent)
#      dwi  acq dir run
#      MRS  task acq nuc voi rec run echo inv   (ce/dir absent, MRS-BIDS Fig.1)
#    No datatype uses both ce and nuc, so placing nuc/voi between ce and rec
#    keeps both the MRI and MRS orders intact. Assembling in this order makes
#    the CamCAN "run-3_task-rest" (run before task) bug structurally impossible.
# ---------------------------------------------------------------------------
CANONICAL_ORDER: List[str] = [
    "sub", "ses", "task", "acq", "ce", "nuc", "voi", "rec", "dir",
    "run", "echo", "inv",
]

# ---------------------------------------------------------------------------
# 2) Which entities each datatype is ALLOWED to carry (sub/ses are structural
#    and always allowed, so they are omitted here). Any entity a naming_entry
#    supplies that is not in this set is DROPPED before assembly — this is what
#    stops task- from leaking onto anat (T1w) files.
# ---------------------------------------------------------------------------
DATATYPE_ALLOWED_ENTITIES: Dict[str, set] = {
    "anat": {"acq", "rec", "run"},                 # no task
    "func": {"task", "acq", "rec", "run", "echo"}, # task required
    "dwi":  {"acq", "dir", "run"},
    "nirs": {"task", "acq", "run"},                # task required
    "eeg":  {"task", "acq", "run"},                # task required
    "mrs":  {"task", "acq", "nuc", "voi", "rec", "run", "echo", "inv"},
}

# ---------------------------------------------------------------------------
# 3) Entities that are REQUIRED for a datatype. A missing required entity means
#    files that differ only by that entity would collide, so the caller should
#    block rather than emit a colliding name.
# ---------------------------------------------------------------------------
DATATYPE_REQUIRED_ENTITIES: Dict[str, set] = {
    "func": {"task"},
    "nirs": {"task"},
    "eeg":  {"task"},
}

# ---------------------------------------------------------------------------
# 4) Legal suffixes per datatype and the default when only the datatype is known.
#    MRS-BIDS: SVS -> svs, MRSI -> mrsi, reference -> mrsref, no localization
#    -> unloc.
# ---------------------------------------------------------------------------
DATATYPE_SUFFIXES: Dict[str, set] = {
    "anat": {"T1w", "T2w", "T1rho", "T2star", "FLAIR", "FLASH", "PD",
             "PDT2", "inplaneT1", "inplaneT2", "angio"},
    "func": {"bold", "sbref"},
    "dwi":  {"dwi"},
    "nirs": {"nirs"},
    "eeg":  {"eeg"},
    "mrs":  {"svs", "mrsi", "mrsref", "unloc"},
}
DATATYPE_DEFAULT_SUFFIX: Dict[str, str] = {
    "anat": "T1w", "func": "bold", "dwi": "dwi",
    "nirs": "nirs", "eeg": "eeg", "mrs": "svs",
}

# ---------------------------------------------------------------------------
# 5) Required JSON sidecar fields per datatype, plus entity-conditioned
#    requirements. Values here are the KEY NAMES the sidecar must contain; the
#    caller fills the actual values from the source file / --describe. MRS-BIDS
#    mandates ResonantNucleus, SpectrometerFrequency, SpectralWidth, EchoTime.
# ---------------------------------------------------------------------------
DATATYPE_REQUIRED_SIDECAR: Dict[str, List[str]] = {
    "func": ["TaskName", "RepetitionTime"],
    "nirs": ["TaskName", "SamplingFrequency",
             "NIRSChannelCount", "NIRSSourceOptodeCount",
             "NIRSDetectorOptodeCount"],
    "eeg":  ["TaskName", "SamplingFrequency", "EEGReference",
             "PowerLineFrequency"],
    "mrs":  ["ResonantNucleus", "SpectrometerFrequency",
             "SpectralWidth", "EchoTime"],
    "anat": [],
    "dwi":  [],
}
# entity present in filename -> extra required sidecar keys (MRS-BIDS)
ENTITY_CONDITIONED_SIDECAR: Dict[str, List[str]] = {
    "nuc": ["ResonantNucleus"],
    "voi": ["BodyPart", "BodyPartDetails"],
}


def sanitize_label(value: str) -> str:
    """BIDS entity values must be purely alphanumeric — strip everything else."""
    return re.sub(r"[^A-Za-z0-9]", "", str(value))


def normalize_datatype(datatype: str) -> str:
    """Fold aliases onto canonical datatype folder names."""
    d = (datatype or "").lower()
    if d in ("fnirs",):
        return "nirs"
    if d in ("mri",):          # 'mri' is a modality, not a datatype
        return "anat"
    return d


def resolve_suffix(datatype: str, suffix: Optional[str]) -> str:
    """
    Return a legal suffix for the datatype. If the requested suffix is not
    legal for this datatype, fall back to the datatype default so we never emit
    an mrs file suffixed 'bold' or an anat file suffixed 'nirs'.
    """
    dt = normalize_datatype(datatype)
    legal = DATATYPE_SUFFIXES.get(dt, set())
    if suffix and suffix in legal:
        return suffix
    return DATATYPE_DEFAULT_SUFFIX.get(dt, suffix or "unknown")


def assemble_bids_name(
    subject: str,
    entities: Dict[str, str],
    datatype: str,
    suffix: str,
    ext: str,
) -> Tuple[str, List[str], List[str]]:
    """
    Build a BIDS filename from a subject id, a bag of entities, a datatype and a
    suffix, enforcing legality and canonical order.

    Returns (filename, dropped_entities, missing_required_entities).

      - entities not allowed for this datatype are DROPPED (e.g. task on anat)
      - remaining entities are emitted in CANONICAL_ORDER (run always after task)
      - required-but-absent entities are reported so the caller can block

    `subject` and entity values are sanitized to alphanumeric. `ext` should
    include the leading dot (e.g. '.nii.gz').
    """
    dt = normalize_datatype(datatype)
    allowed = DATATYPE_ALLOWED_ENTITIES.get(dt, set())
    required = DATATYPE_REQUIRED_ENTITIES.get(dt, set())

    clean: Dict[str, str] = {}
    dropped: List[str] = []
    for key, val in (entities or {}).items():
        if key in ("sub", "ses"):
            continue  # handled structurally below
        if val in (None, "", "none"):
            continue
        if key not in allowed:
            dropped.append(key)
            continue
        clean[key] = sanitize_label(val)

    missing = sorted(r for r in required if r not in clean)

    parts = [f"sub-{sanitize_label(subject)}"]
    ses = entities.get("ses") if entities else None
    if ses:
        parts.append(f"ses-{sanitize_label(ses)}")
    for key in CANONICAL_ORDER:
        if key in ("sub", "ses"):
            continue
        if key in clean:
            parts.append(f"{key}-{clean[key]}")

    suffix = resolve_suffix(dt, suffix)
    filename = "_".join(parts) + "_" + suffix + ext
    return filename, dropped, missing


def bids_stem(filename: str) -> str:
    """Strip the suffix+extension, keep the entity string. Used as a group key."""
    name = filename
    for e in (".nii.gz", ".snirf", ".nii", ".edf", ".vhdr", ".set", ".bdf",
              ".json", ".gz"):
        if name.endswith(e):
            name = name[: -len(e)]
            break
    return name


def required_sidecar_fields(datatype: str, entities: Dict[str, str]) -> List[str]:
    """Full list of sidecar keys that must be present for this file."""
    dt = normalize_datatype(datatype)
    fields = list(DATATYPE_REQUIRED_SIDECAR.get(dt, []))
    for ent, extra in ENTITY_CONDITIONED_SIDECAR.items():
        if entities and entities.get(ent):
            for k in extra:
                if k not in fields:
                    fields.append(k)
    return fields
