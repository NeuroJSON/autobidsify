# evidence.py

import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from autobidsify.utils import list_all_files, write_json, sha1_head, warn, info, fatal, copy_file, read_json
from autobidsify.constants import MAX_TEXT_SIZE, MAX_PDF_SIZE, MAX_DOCX_SIZE, MAX_PDF_PAGES
from autobidsify.universal_core import FileStructureAnalyzer
from autobidsify.filename_tokenizer import analyze_filenames_for_subjects
from autobidsify.anonymize import scrub_evidence_bundle, scrub_text

TEXT_EXT    = {".txt", ".md", ".rst", ".html", ".htm", ".log"}
TABLE_EXT   = {".csv", ".tsv", ".xlsx", ".xls"}
DOC_EXT     = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".odt"}
MRI_EXT     = {".nii", ".dcm"}
ARCHIVE_EXT = {".zip", ".tar", ".tar.gz", ".tgz"}
NIRS_EXT    = {".snirf", ".nirs", ".mat"}
EEG_EXT     = {".edf", ".vhdr", ".set", ".bdf"}
EEG_AUX_EXT = {".vmrk", ".eeg", ".fdt"}
EEG_EVENT_EXT = {".event", ".events", ".evt", ".mrk"}
# Known electrode coordinate file extensions
EEG_ELEC_EXT = {".elc", ".sfp", ".xyz", ".ced", ".loc", ".locs", ".elp"}
# Keywords that suggest a file contains electrode/channel metadata
EEG_AUX_KEYWORDS = {
    "electrode", "channel", "montage", "layout", "position",
    "location", "coord", "loc", "sensor", "cap"
}
ARRAY_EXT   = {".h5", ".hdf5", ".npy", ".npz"}
TRIO_NAMES  = {"readme.md", "participants.tsv", "dataset_description.json"}
JNIFTI_EXT  = {".jnii", ".bnii"}


def _is_trio_file(name: str) -> bool:
    return name.lower() in TRIO_NAMES


# ============================================================================
# JSON serialization safety
# ============================================================================

def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert any non-JSON-serializable object to a JSON-safe type.

    Handles:
    - bytes / numpy.bytes_  → utf-8 decoded string
    - numpy scalar types    → Python int / float
    - numpy ndarray         → list
    - sets                  → list
    - everything else       → str() fallback
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace').strip('\x00').strip()
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(i) for i in obj]

    # numpy types
    try:
        import numpy as np
        if isinstance(obj, np.bytes_):
            return obj.decode('utf-8', errors='replace').strip('\x00').strip()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return _make_json_serializable(obj.tolist())
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    if isinstance(obj, set):
        return [_make_json_serializable(i) for i in sorted(obj, key=str)]

    return str(obj)


# ============================================================================
# Document / table extraction
# ============================================================================

def _extract_text_content(path: Path) -> Optional[str]:
    try:
        size = path.stat().st_size
        if size > MAX_TEXT_SIZE:
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                return f.read(MAX_TEXT_SIZE) + f"\n\n[TRUNCATED: {size} bytes]"
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"[ERROR: {e}]"


def _extract_pdf_content(path: Path) -> Optional[str]:
    size = path.stat().st_size
    if size > MAX_PDF_SIZE:
        return f"[PDF too large: {size} bytes]"

    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text_parts = []
            max_pages = min(len(pdf.pages), MAX_PDF_PAGES)
            for page_num in range(max_pages):
                text = pdf.pages[page_num].extract_text()
                if text:
                    text_parts.append(text)
            result = "\n\n".join(text_parts)
            if max_pages < len(pdf.pages):
                result += f"\n\n[TRUNCATED: {max_pages}/{len(pdf.pages)} pages]"
            return result
    except ImportError:
        pass
    except Exception as e:
        warn(f"pdfplumber failed: {e}")

    try:
        import PyPDF2
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            max_pages = min(len(pdf_reader.pages), MAX_PDF_PAGES)
            text_parts = [pdf_reader.pages[i].extract_text() for i in range(max_pages)]
            return "\n\n".join(text_parts)
    except Exception:
        pass

    return "[ERROR: No PDF library available]"


def _extract_docx_content(path: Path) -> Optional[str]:
    try:
        import docx
        doc = docx.Document(path)
        text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(text_parts)[:100000]
    except Exception:
        return "[ERROR: python-docx not installed]"


def _extract_document_content(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXT:
        return _extract_text_content(path)
    elif suffix == ".pdf":
        return _extract_pdf_content(path)
    elif suffix == ".docx":
        return _extract_docx_content(path)
    return None


def _table_head(path: Path, max_rows: int = 5) -> Dict[str, Any]:
    head: Dict[str, Any] = {"rows": []}
    suf = path.suffix.lower()
    if suf not in {".csv", ".tsv"}:
        return head
    dialect = csv.excel_tab if suf == ".tsv" else csv.excel
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.reader(f, dialect=dialect)
            for i, row in enumerate(r):
                head["rows"].append(row)
                if i >= max_rows:
                    break
    except Exception:
        pass
    return head


# ============================================================================
# Header extraction — fNIRS
# ============================================================================

def _extract_snirf_header(path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from SNIRF file (HDF5 format).
    Reads only metadata groups, never loads signal arrays into memory.

    Extracts:
    - metaDataTags: SubjectID, MeasurementDate, MeasurementTime, units
    - probe: wavelengths_nm, n_sources, n_detectors
    - data: n_samples, n_channels, duration_s, sampling_rate_hz
    - n_measurement_list
    """
    try:
        import h5py
    except ImportError:
        return {"error": "h5py not installed"}

    try:
        result: Dict[str, Any] = {}

        with h5py.File(path, 'r') as f:
            nirs = f.get("nirs")
            if nirs is None:
                return {"error": "no /nirs group"}

            # metaDataTags
            meta = nirs.get("metaDataTags")
            if meta:
                tags: Dict[str, Any] = {}
                for key in meta.keys():
                    try:
                        val = meta[key][()]
                        if isinstance(val, bytes):
                            val = val.decode('utf-8', errors='ignore').strip()
                        elif hasattr(val, 'tolist'):
                            val = val.tolist()
                        if val:
                            tags[key] = val
                    except Exception:
                        pass
                result["metaDataTags"] = tags

            # probe
            probe = nirs.get("probe")
            if probe:
                probe_info: Dict[str, Any] = {}
                if "wavelengths" in probe:
                    wl = probe["wavelengths"][()]
                    probe_info["wavelengths_nm"] = wl.tolist() if hasattr(wl, 'tolist') else list(wl)
                if "sourcePos2D" in probe:
                    probe_info["n_sources"] = probe["sourcePos2D"].shape[0]
                elif "sourcePos3D" in probe:
                    probe_info["n_sources"] = probe["sourcePos3D"].shape[0]
                if "detectorPos2D" in probe:
                    probe_info["n_detectors"] = probe["detectorPos2D"].shape[0]
                elif "detectorPos3D" in probe:
                    probe_info["n_detectors"] = probe["detectorPos3D"].shape[0]
                if probe_info:
                    result["probe"] = probe_info

            # data1
            data1 = nirs.get("data1")
            if data1:
                data_info: Dict[str, Any] = {}
                if "dataTimeSeries" in data1:
                    shape = data1["dataTimeSeries"].shape
                    data_info["n_samples"]  = shape[0]
                    data_info["n_channels"] = shape[1] if len(shape) > 1 else 1
                if "time" in data1:
                    t = data1["time"]
                    n = t.shape[0]
                    if n >= 2:
                        t0 = float(t[0])
                        t1 = float(t[-1])
                        dur = round(t1 - t0, 3)
                        fs  = round((n - 1) / dur, 2) if dur > 0 else None
                        data_info["duration_s"]       = dur
                        data_info["sampling_rate_hz"] = fs
                if data_info:
                    result["data"] = data_info

                if "measurementList" in data1:
                    result["n_measurement_list"] = len(data1["measurementList"])

        return _make_json_serializable(result) if result else None

    except Exception as e:
        return {"error": str(e)}


def _extract_mat_nirs_header(path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from MATLAB .mat file containing fNIRS data.

    Supports:
    1. Homer3 .nirs layout: d, t, SD, s, CondNames
    2. Generic fNIRS layout: largest 2D float array as fallback

    Never loads full signal arrays — only reads shape and scalar metadata.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        return {"error": "scipy not installed"}

    try:
        mat = loadmat(str(path), squeeze_me=False)
        user_vars = {k: v for k, v in mat.items() if not k.startswith('__')}

        result: Dict[str, Any] = {
            "variables_found": list(user_vars.keys())
        }

        # Data matrix
        data_var_names = ['d', 'data', 'dOD', 'dConc', 'y', 'timeseries', 'nirs_data']
        for var in data_var_names:
            if var in user_vars:
                arr = user_vars[var]
                if hasattr(arr, 'shape') and len(arr.shape) >= 2:
                    result["data_shape"] = list(arr.shape)
                    result["n_samples"]  = int(arr.shape[0])
                    result["n_channels"] = int(arr.shape[1])
                    result["data_var"]   = var
                    result["data_dtype"] = str(arr.dtype)
                    break

        # Time vector → sampling rate
        time_var_names = ['t', 'time', 'times', 'time_vector']
        for var in time_var_names:
            if var in user_vars:
                t = user_vars[var].flatten()
                n = len(t)
                if n >= 2:
                    t0, t1 = float(t[0]), float(t[-1])
                    dur = round(t1 - t0, 3)
                    fs  = round((n - 1) / dur, 2) if dur > 0 else None
                    result["duration_s"]       = dur
                    result["sampling_rate_hz"] = fs
                    result["time_var"]         = var
                break

        # Homer3 SD structure
        if 'SD' in user_vars:
            SD = user_vars['SD']
            sd_info: Dict[str, Any] = {}
            try:
                if hasattr(SD, 'dtype') and SD.dtype.names:
                    names = SD.dtype.names
                    if 'Lambda' in names:
                        wl = SD['Lambda'][0, 0].flatten()
                        sd_info["wavelengths_nm"] = [round(float(w), 1) for w in wl]
                    if 'SrcPos' in names:
                        sd_info["n_sources"] = int(SD['SrcPos'][0, 0].shape[0])
                    if 'DetPos' in names:
                        sd_info["n_detectors"] = int(SD['DetPos'][0, 0].shape[0])
                    if 'MeasList' in names:
                        sd_info["n_measlist_rows"] = int(SD['MeasList'][0, 0].shape[0])
            except Exception:
                pass
            if sd_info:
                result["SD"] = sd_info

        # Stimulus markers
        if 's' in user_vars:
            s = user_vars['s']
            if hasattr(s, 'shape'):
                result["n_stimulus_types"] = int(s.shape[1]) if len(s.shape) > 1 else 1

        # Condition names
        for cname_var in ['CondNames', 'condNames', 'cond_names', 'conditions']:
            if cname_var in user_vars:
                try:
                    raw = user_vars[cname_var]
                    names_list = []
                    for item in raw.flatten():
                        if hasattr(item, 'flat'):
                            for s in item.flat:
                                names_list.append(str(s).strip())
                        else:
                            names_list.append(str(item).strip())
                    if names_list:
                        result["condition_names"] = names_list[:20]
                except Exception:
                    pass
                break

        # Generic fallback: largest 2D float array
        if "data_var" not in result:
            best_var, best_size = None, 0
            for vname, arr in user_vars.items():
                if (hasattr(arr, 'shape') and len(arr.shape) == 2
                        and arr.dtype.kind == 'f'
                        and arr.shape[0] > arr.shape[1]
                        and arr.size > best_size):
                    best_var, best_size = vname, arr.size
            if best_var:
                arr = user_vars[best_var]
                result["data_shape"]        = list(arr.shape)
                result["n_samples"]         = int(arr.shape[0])
                result["n_channels"]        = int(arr.shape[1])
                result["data_var"]          = best_var
                result["data_dtype"]        = str(arr.dtype)
                result["data_var_inferred"] = True

        return _make_json_serializable(result) if len(result) > 1 else None

    except Exception as e:
        return {"error": str(e)}


def _extract_edf_header(path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from EDF/EDF+ file without loading signal data.
    Reads only the fixed 256-byte global header + per-channel signal headers.
    No external library required — pure binary parsing.
    """
    try:
        with open(path, 'rb') as f:
            # Global header: 256 bytes
            raw = f.read(256)
            if len(raw) < 256:
                return {"error": "file too short"}

            version      = raw[0:8].decode('ascii', errors='ignore').strip()
            local_patient = raw[8:88].decode('ascii', errors='ignore').strip()
            local_record  = raw[88:168].decode('ascii', errors='ignore').strip()
            startdate    = raw[168:176].decode('ascii', errors='ignore').strip()
            starttime    = raw[176:184].decode('ascii', errors='ignore').strip()
            n_bytes_hdr  = int(raw[184:192].decode('ascii', errors='ignore').strip() or 0)
            reserved     = raw[192:236].decode('ascii', errors='ignore').strip()
            n_records    = int(raw[236:244].decode('ascii', errors='ignore').strip() or 0)
            duration     = float(raw[244:252].decode('ascii', errors='ignore').strip() or 0)
            n_signals    = int(raw[252:256].decode('ascii', errors='ignore').strip() or 0)

            # Per-channel signal headers
            if n_signals <= 0 or n_signals > 512:
                return {"error": f"invalid n_signals: {n_signals}"}

            # Read channel labels (16 bytes each)
            labels_raw = f.read(16 * n_signals)
            channel_labels = [
                labels_raw[i*16:(i+1)*16].decode('ascii', errors='ignore').strip()
                for i in range(n_signals)
            ]

            # Skip transducer types, physical dim, physical/digital min/max (skip 80+8+8+8+8 bytes per channel)
            f.read((80 + 8 + 8 + 8 + 8) * n_signals)

            # Prefiltering (80 bytes each)
            f.read(80 * n_signals)

            # Number of samples per record (8 bytes each)
            n_samples_raw = f.read(8 * n_signals)
            samples_per_record = []
            for i in range(n_signals):
                try:
                    spr = int(n_samples_raw[i*8:(i+1)*8].decode('ascii', errors='ignore').strip() or 0)
                    samples_per_record.append(spr)
                except Exception:
                    samples_per_record.append(0)

        # Compute sampling rates
        sampling_rates = []
        if duration > 0:
            for spr in samples_per_record:
                sampling_rates.append(round(spr / duration, 4) if spr > 0 else 0.0)

        # Classify channels
        eeg_ch, eog_ch, ecg_ch, misc_ch = [], [], [], []
        for label in channel_labels:
            lu = label.upper()
            if any(x in lu for x in ['EOG', 'VEOG', 'HEOG']):
                eog_ch.append(label)
            elif any(x in lu for x in ['ECG', 'EKG', 'HEART']):
                ecg_ch.append(label)
            elif lu in ('STATUS', 'TRIGGER', 'STI', 'STIM', 'ANNOTATIONS'):
                misc_ch.append(label)
            else:
                eeg_ch.append(label)

        total_duration = round(n_records * duration, 3) if n_records > 0 and duration > 0 else None

        result: Dict[str, Any] = {
            "version":          version,
            "n_signals":        n_signals,
            "channel_labels":   channel_labels,
            "eeg_channels":     eeg_ch,
            "eog_channels":     eog_ch,
            "ecg_channels":     ecg_ch,
            "misc_channels":    misc_ch,
            "n_eeg_channels":   len(eeg_ch),
            "n_eog_channels":   len(eog_ch),
            "n_ecg_channels":   len(ecg_ch),
            "n_records":        n_records,
            "record_duration_s": duration,
            "total_duration_s": total_duration,
            "sampling_rates":   sampling_rates,
            "is_edf_plus":      "EDF+C" in reserved or "EDF+D" in reserved,
        }

        # Dominant sampling rate (most channels)
        if sampling_rates:
            from collections import Counter
            most_common = Counter(sampling_rates).most_common(1)[0][0]
            result["dominant_sampling_rate"] = most_common

        return _make_json_serializable(result)

    except Exception as e:
        return {"error": str(e)}


def _find_associated_event_files(edf_path: Path, data_root: Path) -> Optional[Dict[str, Any]]:
    """
    Find event files associated with an EDF file.

    Search order (priority 1→5):
      1. Same stem, EEG event extensions: .event .events .evt .mrk
      2. Same stem, text extensions: .tsv .csv .txt
      3. Same directory, filename contains 'event'/'marker'/'trigger'
      4. EDF+ annotations (detected from header — no external file)
      5. BrainVision .vmrk with same stem

    Returns dict with 'path' (relative to data_root), 'raw_head' (first 20 lines),
    and 'source_type'. Returns None if nothing found.
    """
    stem = edf_path.stem
    parent = edf_path.parent

    # Priority 1: same stem, event-specific extensions
    for ext in ['.event', '.events', '.evt', '.mrk']:
        candidate = parent / (stem + ext)
        if candidate.exists():
            try:
                lines = candidate.read_text(encoding='utf-8', errors='ignore').splitlines()[:20]
            except Exception:
                lines = []
            return {
                "path": str(candidate.relative_to(data_root)),
                "raw_head": lines,
                "source_type": "external_event_file",
                "extension": ext,
            }

    # Priority 2: same stem, text table extensions
    for ext in ['.tsv', '.csv', '.txt']:
        candidate = parent / (stem + ext)
        if candidate.exists():
            try:
                lines = candidate.read_text(encoding='utf-8', errors='ignore').splitlines()[:20]
            except Exception:
                lines = []
            return {
                "path": str(candidate.relative_to(data_root)),
                "raw_head": lines,
                "source_type": "external_table_file",
                "extension": ext,
            }

    # Priority 3: any file in same dir with event/marker/trigger in name
    try:
        for f in parent.iterdir():
            if f.is_file() and f != edf_path:
                name_lower = f.name.lower()
                if any(kw in name_lower for kw in ['event', 'marker', 'trigger']):
                    if f.suffix.lower() in ['.tsv', '.csv', '.txt', '.event', '.events', '.evt', '.mrk']:
                        try:
                            lines = f.read_text(encoding='utf-8', errors='ignore').splitlines()[:20]
                        except Exception:
                            lines = []
                        return {
                            "path": str(f.relative_to(data_root)),
                            "raw_head": lines,
                            "source_type": "shared_event_file",
                            "extension": f.suffix.lower(),
                        }
    except Exception:
        pass

    # Priority 4: BrainVision .vmrk with same stem
    vmrk = parent / (stem + '.vmrk')
    if vmrk.exists():
        try:
            lines = vmrk.read_text(encoding='utf-8', errors='ignore').splitlines()[:20]
        except Exception:
            lines = []
        return {
            "path": str(vmrk.relative_to(data_root)),
            "raw_head": lines,
            "source_type": "brainvision_vmrk",
            "extension": ".vmrk",
        }

    return None


def _find_eeg_auxiliary_files(
    data_root: Path,
    all_files: List[str],
) -> List[Dict[str, Any]]:
    """
    Scan dataset for EEG auxiliary files that may contain electrode coordinates,
    recording metadata, or participant info usable to enrich BIDS sidecars.

    Detection is content-based and keyword-based — not filename-specific.
    Works for any dataset regardless of naming convention.

    Returns list of candidate files with raw_head and detected_content_type.
    """
    candidates = []
    seen_paths = set()

    for relpath in all_files:
        p = data_root / relpath
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        name_lower = p.name.lower()
        stem_lower = p.stem.lower()

        # Skip known primary data and BIDS sidecar files
        if ext in EEG_EXT or ext in {".nii", ".nii.gz", ".dcm", ".snirf", ".nirs", ".mat"}:
            continue
        if ext in EEG_EVENT_EXT:
            continue
        if relpath in seen_paths:
            continue

        detected_type = None

        # Known electrode coordinate formats — always collect
        if ext in EEG_ELEC_EXT:
            detected_type = "electrode_coordinates"

        # Text/table files — check name keywords and content
        elif ext in {".csv", ".tsv", ".txt", ".json"}:
            # Name-based keyword detection
            if any(kw in name_lower for kw in EEG_AUX_KEYWORDS):
                detected_type = "electrode_or_channel_metadata"
            else:
                # Content-based: read first 20 lines and check for spatial/channel keywords
                try:
                    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]
                    combined = " ".join(lines).lower()
                    if any(kw in combined for kw in
                           ["electrode", "x\t", "y\t", "z\t", " x ", " y ", " z ",
                            "impedance", "channel", "label", "theta", "phi",
                            "age", "sex", "gender", "subject"]):
                        detected_type = "possible_metadata"
                except Exception:
                    continue

        if detected_type is None:
            continue

        try:
            raw_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]
        except Exception:
            raw_lines = []

        candidates.append({
            "relpath":            relpath,
            "filename":           p.name,
            "extension":          ext,
            "detected_type":      detected_type,
            "raw_head":           raw_lines,
            "size_bytes":         p.stat().st_size,
        })
        seen_paths.add(relpath)

    return candidates


# ============================================================================
# Header extraction — MRI
# ============================================================================

def _extract_nifti_header(path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from NIfTI file (.nii or .nii.gz).
    Uses nibabel header only — pixel data is never loaded.

    Extracts: shape, voxel_size_mm, TR_s (4D), data_dtype,
              qform_code, sform_code, descrip
    """
    try:
        import nibabel as nib
    except ImportError:
        return {"error": "nibabel not installed"}

    try:
        img   = nib.load(str(path))
        hdr   = img.header
        shape = tuple(int(x) for x in img.shape)
        zooms = hdr.get_zooms()

        result: Dict[str, Any] = {
            "shape":         shape,
            "voxel_size_mm": [round(float(z), 4) for z in zooms[:3]],
            "data_dtype":    str(hdr.get_data_dtype()),
        }

        if len(shape) == 4 and len(zooms) >= 4:
            tr = float(zooms[3])
            if tr > 0:
                result["TR_s"] = round(tr, 4)

        try:
            result["qform_code"] = int(hdr["qform_code"])
            result["sform_code"] = int(hdr["sform_code"])
        except Exception:
            pass

        try:
            descrip = hdr["descrip"].tobytes().decode('utf-8', errors='ignore').strip('\x00').strip()
            if descrip:
                result["descrip"] = descrip
        except Exception:
            pass

        return _make_json_serializable(result)

    except Exception as e:
        return {"error": str(e)}


def _extract_dicom_header(path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a single DICOM file.
    Uses stop_before_pixels=True — pixel data never read.
    """
    try:
        import pydicom
    except ImportError:
        return {"error": "pydicom not installed"}

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)

        fields = [
            "Modality", "StudyDescription", "SeriesDescription",
            "ProtocolName", "AcquisitionDate",
            "PatientSex", "PatientAge", "PatientID",
            "RepetitionTime", "EchoTime", "FlipAngle",
            "SliceThickness", "PixelSpacing",
            "Rows", "Columns", "NumberOfSlices",
            "MagneticFieldStrength", "Manufacturer",
            "ManufacturerModelName", "SoftwareVersions",
        ]

        result: Dict[str, Any] = {}
        for field in fields:
            try:
                val = getattr(ds, field, None)
                if val is not None:
                    val_str = str(val).strip()
                    if val_str:
                        result[field] = val_str
            except Exception:
                pass

        return _make_json_serializable(result) if result else None

    except Exception as e:
        return {"error": str(e)}


def _extract_jnifti_header(path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from JNIfTI file (.jnii = JSON, .bnii = binary JSON).
    Only reads NIFTIHeader fields, not NIFTIData.

    Extracts: Dim, VoxelSize, DataType, Intent, QForm, SForm,
              Description, NIIFormat
    """
    try:
        if path.suffix.lower() == '.jnii':
            import json as _json
            with open(path, 'r', encoding='utf-8') as f:
                jnii = _json.load(f)
        elif path.suffix.lower() == '.bnii':
            try:
                import bjdata
                with open(path, 'rb') as f:
                    jnii = bjdata.load(f)
            except ImportError:
                return {"error": "bjdata not installed for .bnii"}
        else:
            return None

        hdr = jnii.get("NIFTIHeader", {})
        if not hdr:
            return {"error": "no NIFTIHeader"}

        result: Dict[str, Any] = {}
        for field in ["Dim", "VoxelSize", "DataType", "Intent",
                      "QForm", "SForm", "Description", "NIIFormat"]:
            val = hdr.get(field)
            if val is not None:
                result[field] = val

        return _make_json_serializable(result) if result else None

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# File kind detection
# ============================================================================

def detect_kind(p: Path) -> str:
    """
    Detect file type/kind for classification.

    Detection order (highest → lowest priority):
    user_trio → jnifti → nirs → mri → table → array → text_doc → document → archive → other

    CRITICAL: .mat is NIRS only — checked before MRI.
    """
    s    = p.suffix.lower()
    name = p.name.lower()

    if _is_trio_file(name):
        return "user_trio"
    if s in JNIFTI_EXT:
        return "jnifti"
    if s in NIRS_EXT:
        return "nirs"
    if s in EEG_EXT:
        return "eeg"
    if s in EEG_AUX_EXT or s in EEG_EVENT_EXT:
        return "eeg_aux"
    if name.endswith(".nii.gz") or s in MRI_EXT:
        return "mri"
    if s in TABLE_EXT:
        return "table"
    if s in ARRAY_EXT:
        return "array"
    if s in TEXT_EXT:
        return "text_doc"
    if s in DOC_EXT:
        return "document"
    if s in ARCHIVE_EXT:
        return "archive"
    return "other"


# ============================================================================
# Trio file promotion
# ============================================================================

def _promote_trio_files(data_root: Path, output_dir: Path) -> Dict[str, List[str]]:
    promoted: Dict[str, List[str]] = {
        "dataset_description": [], "readme": [], "participants": []
    }

    dd_candidates = list(data_root.glob("**/dataset_description.json"))
    if dd_candidates:
        source = dd_candidates[0]
        dest   = output_dir / "dataset_description.json"
        if not dest.exists():
            copy_file(source, dest)
            promoted["dataset_description"].append(str(source.relative_to(data_root)))
            info(f"✓ Promoted existing dataset_description.json")

    readme_variants = ['readme', 'readme.md', 'readme.txt', 'readme.rst']
    for variant in readme_variants:
        candidates = list(data_root.glob(f"**/{variant}"))
        candidates.extend(list(data_root.glob(f"**/{variant.upper()}")))
        if candidates:
            source = candidates[0]
            dest   = output_dir / "README.md"
            if not dest.exists():
                copy_file(source, dest)
                promoted["readme"].append(str(source.relative_to(data_root)))
                info(f"✓ Promoted existing {source.name} → README.md")
            break

    parts_candidates = list(data_root.glob("**/participants.tsv"))
    if parts_candidates:
        source = parts_candidates[0]
        dest   = output_dir / "participants.tsv"
        if not dest.exists():
            copy_file(source, dest)
            promoted["participants"].append(str(source.relative_to(data_root)))
            info(f"✓ Promoted existing participants.tsv")

    return promoted


# ============================================================================
# Intelligent file sampling
# ============================================================================

def _intelligent_file_sampling(
    files_by_ext: Dict[str, List[Path]],
    target_samples_per_ext: int = 5,
    ensure_full_coverage: bool = False
) -> Tuple[List[Path], Dict]:

    samples: List[Path] = []
    pattern_summary: Dict[str, Any] = {}

    for ext, file_list in files_by_ext.items():
        pattern_groups: Dict[str, List[Path]] = defaultdict(list)
        for filepath in file_list:
            pattern = re.sub(r'\d+', 'N', filepath.name)
            pattern = re.sub(r'\s*\([^)]*\)', '', pattern)
            pattern_groups[pattern].append(filepath)

        n_patterns = len(pattern_groups)

        if ensure_full_coverage:
            ext_samples = [files[0] for _, files in sorted(pattern_groups.items())]
            info(f"  {ext}: Full coverage mode - {len(ext_samples)} patterns sampled")
        else:
            spp = max(1, target_samples_per_ext // n_patterns) if n_patterns > 0 else target_samples_per_ext
            ext_samples = []
            for _, files in sorted(pattern_groups.items()):
                ext_samples.extend(files[:min(len(files), spp)])

            if len(ext_samples) < target_samples_per_ext:
                remaining = target_samples_per_ext - len(ext_samples)
                for _, files in sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True):
                    if remaining <= 0:
                        break
                    available = [f for f in files if f not in ext_samples]
                    take = min(remaining, len(available))
                    if take:
                        ext_samples.extend(available[:take])
                        remaining -= take

        samples.extend(ext_samples)

        pattern_info = []
        for pattern, files in sorted(pattern_groups.items()):
            pattern_info.append({
                "pattern":       pattern,
                "total_files":   len(files),
                "sampled":       sum(1 for f in ext_samples if f in files),
                "example_files": [f.name for f in files[:2]]
            })

        pattern_summary[ext] = {
            "total_patterns": n_patterns,
            "total_files":    len(file_list),
            "sampled_files":  len(ext_samples),
            "patterns":       pattern_info
        }

    return samples, pattern_summary


# ============================================================================
# Participant metadata evidence collection
# ============================================================================

def _collect_participant_metadata_evidence(
    data_root: Path,
    all_files: List[str],
    documents: List[Dict]
) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}

    # Evidence 1: explicit metadata files
    info("  [Evidence 1/5] Scanning for explicit metadata files...")
    metadata_files = []
    for pattern in ['**/participants.*', '**/subjects.*', '**/metadata.*',
                    '**/demographics.*', '**/phenotype.*',
                    '**/participant_data.*', '**/subject_info.*']:
        try:
            for match in data_root.glob(pattern):
                if match.is_file() and match.suffix in ['.csv', '.tsv', '.json', '.txt', '.xlsx']:
                    metadata_files.append({
                        "filename":   match.name,
                        "path":       str(match.relative_to(data_root)),
                        "extension":  match.suffix,
                        "size_bytes": match.stat().st_size
                    })
        except Exception as e:
            warn(f"    Error scanning pattern {pattern}: {e}")

    if metadata_files:
        evidence["explicit_metadata_files"] = {
            "found": True, "count": len(metadata_files), "files": metadata_files,
            "note": "These files may contain participant demographics"
        }
        info(f"    ✓ Found {len(metadata_files)} potential metadata file(s)")
    else:
        evidence["explicit_metadata_files"] = {"found": False}
        info("    - No explicit metadata files found")

    # Evidence 2: DICOM headers
    info("  [Evidence 2/5] Sampling DICOM headers...")
    dicom_patient_info = []
    dicom_files = [f for f in all_files if f.lower().endswith('.dcm')]

    if dicom_files:
        sample_size = min(20, len(dicom_files))
        step        = max(1, len(dicom_files) // sample_size)
        sampled     = [dicom_files[i] for i in range(0, len(dicom_files), step)][:sample_size]

        try:
            import pydicom
            pydicom_ok = True
        except ImportError:
            pydicom_ok = False
            warn("    pydicom not installed, skipping DICOM header analysis")

        if pydicom_ok:
            for dcm_str in sampled:
                try:
                    ds = pydicom.dcmread(str(data_root / dcm_str), stop_before_pixels=True)
                    info_dict: Dict[str, str] = {"filename": Path(dcm_str).name}
                    for attr, field in [
                        ('PatientID', 'PatientID'), ('PatientSex', 'PatientSex'),
                        ('PatientAge', 'PatientAge'), ('PatientName', 'PatientName'),
                        ('StudyDescription', 'StudyDescription'),
                        ('PatientBirthDate', 'PatientBirthDate'),
                        ('PatientWeight', 'PatientWeight'),
                    ]:
                        try:
                            val = getattr(ds, attr, None)
                            if val is not None and str(val).strip():
                                info_dict[field] = str(val).strip()
                        except Exception:
                            pass
                    if len(info_dict) > 1:
                        dicom_patient_info.append(info_dict)
                except Exception:
                    continue

        if dicom_patient_info:
            evidence["dicom_headers"] = {
                "found": True, "sampled_count": len(dicom_patient_info),
                "total_dicom_files": len(dicom_files),
                "samples": dicom_patient_info[:10],
                "note": "DICOM headers may contain PatientSex, PatientAge, PatientID"
            }
            info(f"    ✓ Extracted headers from {len(dicom_patient_info)}/{sample_size} DICOM files")
        else:
            evidence["dicom_headers"] = {
                "found": False, "total_dicom_files": len(dicom_files),
                "note": "DICOM files exist but no patient info extracted"
            }
            info(f"    - DICOM files present ({len(dicom_files)}) but no patient info found")
    else:
        evidence["dicom_headers"] = {"found": False}
        info("    - No DICOM files in dataset")

    # Evidence 3: filename semantic patterns
    info("  [Evidence 3/5] Analyzing filename semantic patterns...")
    filename_patterns: Dict[str, List] = {
        "gender_keywords": [], "age_patterns": [], "group_keywords": []
    }
    gender_kws  = ['male', 'female', 'man', 'woman', 'boy', 'girl',
                   '_m_', '_f_', '_m.', '_f.', 'VHM', 'VHF',
                   'male_', 'female_', '_male', '_female']
    age_regexes = [r'\d{2}yo', r'\d{2}y\b', r'age\d{2}', r'_\d{2}_', r'y\d{2}']
    group_kws   = ['patient', 'control', 'healthy', 'disease', 'hc', 'pt',
                   'ctrl', 'case', 'normal', 'treated', 'untreated']

    for filepath in all_files[:200]:
        fn = filepath.split('/')[-1].lower()
        for kw in gender_kws:
            if kw.lower() in fn:
                filename_patterns["gender_keywords"].append({"keyword": kw, "filename": fn})
                break
        for pat in age_regexes:
            if re.search(pat, fn):
                filename_patterns["age_patterns"].append({"pattern": pat, "filename": fn})
                break
        for kw in group_kws:
            if kw in fn:
                filename_patterns["group_keywords"].append({"keyword": kw, "filename": fn})
                break

    # deduplicate
    for key in filename_patterns:
        seen: set = set()
        unique = []
        for item in filename_patterns[key]:
            ident = item.get('keyword') or item.get('pattern')
            if ident not in seen:
                seen.add(ident)
                unique.append(item)
        filename_patterns[key] = unique[:10]

    if any(filename_patterns.values()):
        total_hints = sum(len(v) for v in filename_patterns.values())
        evidence["filename_semantic_patterns"] = {
            "found": True, "patterns": filename_patterns,
            "note": "Filenames may contain demographic keywords"
        }
        info(f"    ✓ Found {total_hints} semantic patterns in filenames")
    else:
        evidence["filename_semantic_patterns"] = {"found": False}
        info("    - No semantic patterns found in filenames")

    # Evidence 4: demographic keywords in documents
    info("  [Evidence 4/5] Searching documents for demographic keywords...")
    demo_terms = ['male', 'female', 'sex', 'gender', 'age', 'years old',
                  'patient', 'control', 'healthy', 'diagnosis', 'participants',
                  'subjects', 'volunteers', 'cohort', 'population',
                  'cadaver', 'human', 'adult', 'child']
    demo_hits = []
    for doc in documents:
        content  = doc.get('content', '').lower()
        doc_name = doc.get('filename', 'unknown')
        found = []
        for term in demo_terms:
            idx = content.find(term)
            if idx != -1:
                snippet = content[max(0, idx-100):idx+100].strip()
                found.append({"term": term, "context_snippet": ' '.join(snippet.split())[:200]})
        if found:
            demo_hits.append({"document": doc_name, "found_terms": found[:5]})

    if demo_hits:
        evidence["document_demographic_keywords"] = {
            "found": True, "documents_with_keywords": len(demo_hits),
            "details": demo_hits[:5], "note": "Documents mention demographic terms"
        }
        info(f"    ✓ Found demographic keywords in {len(demo_hits)} document(s)")
    else:
        evidence["document_demographic_keywords"] = {"found": False}
        info("    - No demographic keywords found in documents")

    # Evidence 5: balanced prefix distribution
    info("  [Evidence 5/5] Analyzing subject grouping patterns...")
    from autobidsify.filename_tokenizer import FilenamePatternAnalyzer
    filenames = [f.split('/')[-1] for f in all_files]
    stats     = FilenamePatternAnalyzer(filenames).analyze_token_statistics()
    dominant  = stats.get('dominant_prefixes', [])

    if len(dominant) == 2:
        p1, p2 = dominant[0], dominant[1]
        ratio  = min(p1['percentage'], p2['percentage']) / max(p1['percentage'], p2['percentage'])
        if ratio > 0.8:
            evidence["balanced_prefix_distribution"] = {
                "found": True,
                "prefix_1": p1['prefix'], "prefix_1_percentage": p1['percentage'], "prefix_1_count": p1['count'],
                "prefix_2": p2['prefix'], "prefix_2_percentage": p2['percentage'], "prefix_2_count": p2['count'],
                "distribution_ratio": round(ratio, 2),
                "note": "Two balanced groups may indicate gender/group split"
            }
            info(f"    ✓ Found balanced distribution: {p1['prefix']} ({p1['percentage']}%) vs {p2['prefix']} ({p2['percentage']}%)")
        else:
            evidence["balanced_prefix_distribution"] = {"found": False}
            info(f"    - Two prefixes found but not balanced ({ratio:.2f})")
    else:
        evidence["balanced_prefix_distribution"] = {"found": False}
        info("    - No balanced distribution pattern detected")

    evidence_count = sum(1 for v in evidence.values() if isinstance(v, dict) and v.get('found'))
    evidence["summary"] = {
        "total_evidence_types_found": evidence_count,
        "evidence_types": [k for k, v in evidence.items() if isinstance(v, dict) and v.get('found')]
    }
    info(f"\n  Summary: Found {evidence_count}/5 types of evidence")
    return evidence


# ============================================================================
# Core bundle builder
# ============================================================================

def _build_evidence_bundle_internal(
    data_root: Path,
    user_n_subjects: Optional[int],
    modality_hint: str,
    user_text: str,
    sample_per_ext: int = 5
) -> Dict[str, Any]:

    root  = Path(data_root)
    files = list_all_files(root)
    info(f"Scanning {len(files)} files in {root}")

    all_file_paths = [str(p.relative_to(root)).replace("\\", "/") for p in files]

    # Directory structure analysis
    info("Analyzing file structure with universal engine...")
    analyzer     = FileStructureAnalyzer(all_file_paths)
    dir_structure = analyzer.analyze_directory_structure()
    info(f"  Directory structure: {dir_structure['max_depth']} levels, template: {dir_structure['structure_template']}")
    info(f"  Unique directories: {dir_structure['total_unique_dirs']}")

    subject_detection_result = analyzer.detect_subject_identifiers(user_n_subjects)
    if subject_detection_result["best_candidate"]:
        best = subject_detection_result["best_candidate"]
        info(f"  Best subject pattern: {best['pattern_display']}")
        info(f"  Detected: {best['count']} subjects (confidence: {subject_detection_result['confidence']})")
        info(f"  Avg files/subject: {best['avg_files_per_subject']:.1f}")
    else:
        warn("  ⚠ No subject pattern detected from directory structure")

    duplicates = analyzer.detect_duplicate_filenames()
    if duplicates:
        info(f"  Found {len(duplicates)} duplicate filenames across different paths")
        for fname, paths in list(duplicates.items())[:2]:
            info(f"    '{fname}' appears in {len(paths)} locations")

    tree_summary = analyzer.build_directory_tree_summary(max_subjects=50)
    info(f"  Structure summary: {tree_summary['sampled_subjects']}/{tree_summary['total_subjects_detected']} subjects")

    # Filename token analysis
    info("\nAnalyzing filename token patterns...")
    filename_analysis = analyze_filenames_for_subjects(all_file_paths, {
        "n_subjects": user_n_subjects, "user_text": user_text
    })
    info(f"  Token-based analysis: {filename_analysis['confidence']} confidence")
    info(f"  {filename_analysis['recommendation']}")
    dominant_prefixes = filename_analysis['python_statistics'].get('dominant_prefixes', [])
    if dominant_prefixes:
        info("  Dominant filename prefixes:")
        for p in dominant_prefixes[:5]:
            info(f"    '{p['prefix']}': {p['count']} files ({p['percentage']}%)")

    # Group files by extension
    by_ext: Dict[str, List[Path]] = {}
    for p in files:
        key = ".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower()
        by_ext.setdefault(key, []).append(p)

    # Sample files
    info("\nSampling files for document extraction...")
    sampled_files, pattern_summary = _intelligent_file_sampling(by_ext, sample_per_ext)
    info("Sampling summary:")
    for ext, summary in pattern_summary.items():
        info(f"  {ext}: {summary['total_patterns']} patterns, {summary['sampled_files']}/{summary['total_files']} files")

    # Build samples list with header extraction
    samples:   List[Dict[str, Any]] = []
    documents: List[Dict[str, Any]] = []

    for p in sampled_files:
        ext  = ".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower()
        kind = detect_kind(p)

        entry: Dict[str, Any] = {
            "relpath":   str(p.relative_to(root)).replace("\\", "/"),
            "size":      p.stat().st_size,
            "suffix":    ext,
            "kind":      kind,
            "sha1_head": sha1_head(p)
        }

        if kind in {"text_doc", "document"}:
            content = _extract_document_content(p)
            if content:
                documents.append({
                    "relpath":  entry["relpath"],
                    "filename": p.name,
                    "type":     ext,
                    "size":     entry["size"],
                    "content":  content,
                    "purpose":  "experimental_protocol_or_metadata"
                })
                entry["has_full_content"] = True
                entry["content_length"]   = len(content)

        elif kind == "table":
            entry["table_head"] = _table_head(p)

        elif kind == "nirs":
            if ext == ".snirf":
                hdr = _extract_snirf_header(p)
            elif ext in (".nirs", ".mat"):
                hdr = _extract_mat_nirs_header(p)
            else:
                hdr = None
            if hdr:
                entry["header_info"] = hdr

        elif kind == "mri" and ext in (".nii", ".nii.gz"):
            hdr = _extract_nifti_header(p)
            if hdr:
                entry["header_info"] = hdr

        elif kind == "mri" and ext == ".dcm":
            hdr = _extract_dicom_header(p)
            if hdr:
                entry["header_info"] = hdr

        elif kind == "jnifti":
            hdr = _extract_jnifti_header(p)
            if hdr:
                entry["header_info"] = hdr

        elif kind == "jnifti":
            hdr = _extract_jnifti_header(p)
            if hdr:
                entry["header_info"] = hdr

        elif kind == "eeg":
            if p.suffix.lower() == ".edf":
                hdr = _extract_edf_header(p)
                if hdr:
                    entry["header_info"] = hdr
                # Find associated event file
                event_info = _find_associated_event_files(p, root)
                if event_info:
                    entry["associated_event_file"] = event_info
                elif hdr and hdr.get("is_edf_plus"):
                    entry["associated_event_file"] = {
                        "source_type": "edf_plus_annotations",
                        "path": None,
                    }

        samples.append(entry)

    # EEG auxiliary files scan (only when EEG files present)
    eeg_aux_files: List[Dict[str, Any]] = []
    has_eeg = any(
        s.get("kind") == "eeg" for s in samples
    )
    if has_eeg:
        eeg_aux_files = _find_eeg_auxiliary_files(root, all_file_paths)
        if eeg_aux_files:
            info(f"\n  Found {len(eeg_aux_files)} potential EEG auxiliary file(s)")

    # Participant metadata evidence
    info("\n=== Collecting Participant Metadata Evidence ===")
    participant_evidence = _collect_participant_metadata_evidence(root, all_file_paths, documents)
    evidence_count = participant_evidence['summary']['total_evidence_types_found']
    info(f"\n✓ Evidence collection complete: {evidence_count}/5 types found")

    # Subject count decision
    path_based_count      = subject_detection_result["best_candidate"]["count"] if subject_detection_result["best_candidate"] else 0
    path_based_confidence = subject_detection_result["confidence"]
    filename_based_count  = len(filename_analysis['python_statistics'].get('dominant_prefixes', []))
    filename_based_conf   = filename_analysis['confidence']

    if user_n_subjects is not None:
        final_count, count_source = user_n_subjects, "user_provided"
        info(f"\nUsing user-provided subject count: {final_count}")
    elif path_based_confidence == "high":
        final_count, count_source = path_based_count, "path_based_high_confidence"
        info(f"\nUsing path-based detection (high confidence): {final_count}")
    elif filename_based_conf in ("high", "medium") and path_based_count == 0:
        final_count, count_source = filename_based_count, "filename_based"
        info(f"\nUsing filename token analysis: {final_count}")
    elif path_based_count > 0:
        final_count, count_source = path_based_count, "path_based"
        info(f"\nUsing path-based detection: {final_count}")
    else:
        final_count, count_source = 1, "fallback"
        warn("\n⚠ Could not detect subject count, using fallback: 1")

    bundle: Dict[str, Any] = {
        "root":          str(root),
        "counts_by_ext": {ext: len(lst) for ext, lst in by_ext.items()},
        "samples":       samples,
        "documents":     documents,
        "all_files":     all_file_paths,
        "trio_found":    {name: (root / name).exists() for name in TRIO_NAMES},

        "structure_analysis": {
            "directory_structure":  dir_structure,
            "subject_detection":    subject_detection_result,
            "duplicate_files":      {k: v for k, v in list(duplicates.items())[:20]},
            "tree_summary_for_llm": tree_summary,
            "analyzer_confidence":  subject_detection_result["confidence"]
        },

        "filename_analysis":             filename_analysis,
        "participant_metadata_evidence": participant_evidence,

        "user_hints": {
            "n_subjects":    final_count,
            "modality_hint": str(modality_hint) if modality_hint else "",
            "user_text":     str(user_text) if user_text else ""
        },

        "subject_detection": {
            "method":                   "hybrid_analysis",
            "path_based_count":         path_based_count,
            "path_based_confidence":    path_based_confidence,
            "filename_based_count":     filename_based_count,
            "filename_based_confidence": filename_based_conf,
            "final_count":              final_count,
            "count_source":             count_source,
            "best_pattern":             subject_detection_result["best_candidate"]["pattern_display"]
                                        if subject_detection_result["best_candidate"] else "none"
        },

        "document_summary": {
            "total_documents":  len(documents),
            "document_types":   list(set(d["type"] for d in documents)),
            "total_text_length": sum(len(d["content"]) for d in documents)
        },

        "sampling_strategy": {
            "method":                   "pattern_based",
            "target_per_ext":           sample_per_ext,
            "total_patterns_detected":  sum(s["total_patterns"] for s in pattern_summary.values()),
            "pattern_summary":          pattern_summary
        },

        "eeg_auxiliary_files": eeg_aux_files
    }

    info(f"\nExtracted {len(documents)} documents")
    if documents:
        info(f"Total document text: {bundle['document_summary']['total_text_length']:,} characters")

    info("\n=== Universal Analysis Summary ===")
    info(f"Subject detection (hybrid):")
    info(f"  Path-based: {path_based_count} subjects ({path_based_confidence} confidence)")
    info(f"  Filename-based: {filename_based_count} subjects ({filename_based_conf} confidence)")
    info(f"  Final decision: {final_count} subjects (source: {count_source})")
    info(f"Duplicate handling: {len(duplicates)} duplicate filenames detected")
    info(f"Sampling: {sum(s['total_patterns'] for s in pattern_summary.values())} unique patterns")

    return bundle


# ============================================================================
# Public entry point
# ============================================================================

def build_evidence_bundle(output_dir: Path, user_hints: Dict[str, Any],
                          anonymize: bool = False) -> None:
    output_dir = Path(output_dir)

    ingest_info_path = output_dir / "_staging" / "ingest_info.json"
    if not ingest_info_path.exists():
        fatal(f"Ingest info not found: {ingest_info_path}")
        return

    ingest_info = read_json(ingest_info_path)
    actual_data_path = ingest_info.get("actual_data_path") or ingest_info.get("staging_dir")
    if not actual_data_path:
        fatal("Cannot determine data location from ingest_info")
        return

    data_root = Path(actual_data_path)
    if not data_root.exists():
        fatal(f"Data directory not found: {data_root}")
        return

    info(f"Using data from: {data_root}")

    info("\nChecking for existing trio files...")
    promoted = _promote_trio_files(data_root, output_dir)
    total = sum(len(f) for f in promoted.values())
    info(f"Promoted {total} trio file(s)" if total > 0 else "No existing trio files found")

    bundle = _build_evidence_bundle_internal(
        data_root=data_root,
        user_n_subjects=user_hints.get("n_subjects"),
        modality_hint=user_hints.get("modality_hint", ""),
        user_text=user_hints.get("user_text", "")
    )

    bundle["trio_promoted"] = promoted
    bundle["data_source"] = {
        "type":          ingest_info.get("input_type"),
        "original_path": ingest_info.get("input_path"),
        "actual_path":   str(data_root)
    }

    # Apply de-identification before writing if anonymize=True.
    # scrub_evidence_bundle returns a deep copy — original bundle unchanged.
    if anonymize:
        info("\n  Anonymizing evidence bundle (PHI scrubbing)...")
        bundle_to_write = scrub_evidence_bundle(bundle)
        info("  ✓ PHI scrubbed from evidence bundle")
    else:
        bundle_to_write = bundle

    write_json(output_dir / "_staging" / "evidence_bundle.json", bundle_to_write)
    info(f"\n✓ Evidence bundle saved")

    info("\n=== Evidence Bundle Summary ===")
    info(f"Total files: {len(bundle['all_files'])}")
    info(f"File types: {len(bundle['counts_by_ext'])}")
    info(f"Subject count: {bundle['subject_detection']['final_count']} (source: {bundle['subject_detection']['count_source']})")
    info(f"Detection confidence: hybrid ({bundle['subject_detection']['path_based_confidence']} path, {bundle['subject_detection']['filename_based_confidence']} filename)")