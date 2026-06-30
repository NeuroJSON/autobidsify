# converters/eeg_convert.py
# EEG format support: read metadata, generate BIDS sidecars
# Supports: EDF (.edf), BrainVision (.vhdr), EEGLAB (.set), Biosemi (.bdf)
# No MNE dependency — uses pure binary parsing for EDF, text parsing for others.

import re
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

from autobidsify.utils import ensure_dir, warn, info, write_json



# ============================================================================
# EDF header reader (pure Python, no external lib)
# ============================================================================

def _read_edf_metadata(path: Path) -> Dict[str, Any]:
    """
    Read EDF/EDF+ metadata from the fixed binary header.
    Never loads signal data — header only.

    EDF global header layout (256 bytes):
      [0:8]   version
      [8:88]  local patient identification
      [88:168] local recording identification
      [168:176] startdate
      [176:184] starttime
      [184:192] number of bytes in header record
      [192:236] reserved (EDF+C / EDF+D marker lives here)
      [236:244] number of data records
      [244:252] duration of a data record (seconds)
      [252:256] number of signals
    Then per-signal fields follow (n_signals * field_size bytes each).
    """
    result: Dict[str, Any] = {"format": "edf", "error": None}

    try:
        with open(path, 'rb') as f:
            raw = f.read(256)
            if len(raw) < 256:
                result["error"] = "file too short for EDF header"
                return result

            reserved     = raw[192:236].decode('ascii', errors='ignore').strip()
            n_records    = _parse_int(raw[236:244])
            duration     = _parse_float(raw[244:252])
            n_signals    = _parse_int(raw[252:256])

            if n_signals <= 0 or n_signals > 512:
                result["error"] = f"invalid n_signals: {n_signals}"
                return result

            # Per-signal headers
            labels_raw  = f.read(16 * n_signals)
            f.read(80 * n_signals)   # transducer type
            phys_dim    = f.read(8  * n_signals)   # physical dimension (units)
            f.read(8  * n_signals)   # physical min
            f.read(8  * n_signals)   # physical max
            f.read(8  * n_signals)   # digital min
            f.read(8  * n_signals)   # digital max
            f.read(80 * n_signals)   # prefiltering
            ns_raw      = f.read(8  * n_signals)   # nr of samples per record

        channel_labels = [
            labels_raw[i*16:(i+1)*16].decode('ascii', errors='ignore').strip()
            for i in range(n_signals)
        ]
        channel_units = [
            phys_dim[i*8:(i+1)*8].decode('ascii', errors='ignore').strip()
            for i in range(n_signals)
        ]
        samples_per_record = [
            _parse_int(ns_raw[i*8:(i+1)*8]) for i in range(n_signals)
        ]

        # Sampling rates
        sampling_rates = []
        if duration > 0:
            for spr in samples_per_record:
                sampling_rates.append(round(spr / duration, 4) if spr > 0 else 0.0)
        dominant_fs = Counter(sampling_rates).most_common(1)[0][0] if sampling_rates else 0.0

        # Classify channels
        eeg_ch, eog_ch, ecg_ch, misc_ch = [], [], [], []
        eeg_units = []
        for i, label in enumerate(channel_labels):
            lu = label.upper()
            unit = channel_units[i] if i < len(channel_units) else ""
            if any(x in lu for x in ['EOG', 'VEOG', 'HEOG']):
                eog_ch.append(label)
            elif any(x in lu for x in ['ECG', 'EKG', 'HEART']):
                ecg_ch.append(label)
            elif lu in ('STATUS', 'TRIGGER', 'STI', 'STIM', 'ANNOTATIONS', 'EDF ANNOTATIONS'):
                misc_ch.append(label)
            else:
                eeg_ch.append(label)
                eeg_units.append(unit)

        total_duration = round(n_records * duration, 3) if n_records > 0 and duration > 0 else None
        is_edf_plus = "EDF+C" in reserved or "EDF+D" in reserved

        result.update({
            "n_signals":           n_signals,
            "channel_labels":      channel_labels,
            "channel_units":       channel_units,
            "eeg_channels":        eeg_ch,
            "eog_channels":        eog_ch,
            "ecg_channels":        ecg_ch,
            "misc_channels":       misc_ch,
            "n_eeg_channels":      len(eeg_ch),
            "n_eog_channels":      len(eog_ch),
            "n_ecg_channels":      len(ecg_ch),
            "n_misc_channels":     len(misc_ch),
            "sampling_rates":      sampling_rates,
            "dominant_sampling_rate": dominant_fs,
            "n_records":           n_records,
            "record_duration_s":   duration,
            "total_duration_s":    total_duration,
            "is_edf_plus":         is_edf_plus,
            "eeg_units":           eeg_units,
        })

        # Read EDF+ annotations if present
        if is_edf_plus:
            result["has_annotations"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def _read_vhdr_metadata(path: Path) -> Dict[str, Any]:
    """
    Read BrainVision .vhdr header file (text INI-like format).
    """
    result: Dict[str, Any] = {"format": "vhdr", "error": None}
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
        lines = text.splitlines()

        sampling_interval_us = None
        n_channels = None
        channel_names = []

        for line in lines:
            line = line.strip()
            if line.lower().startswith('samplinginterval='):
                try:
                    sampling_interval_us = float(line.split('=', 1)[1].strip())
                except Exception:
                    pass
            elif line.lower().startswith('numberofchannels='):
                try:
                    n_channels = int(line.split('=', 1)[1].strip())
                except Exception:
                    pass
            elif re.match(r'^CH\d+=', line, re.IGNORECASE):
                # CH1=Fp1,,0.1,µV  or  Ch1=Fp1,Ref,0.1
                parts = line.split('=', 1)[1].split(',')
                if parts:
                    channel_names.append(parts[0].strip())

        fs = None
        if sampling_interval_us and sampling_interval_us > 0:
            fs = round(1_000_000 / sampling_interval_us, 4)

        result.update({
            "sampling_rate":  fs,
            "n_channels":     n_channels or len(channel_names),
            "channel_labels": channel_names,
        })

        # Check for associated .vmrk
        vmrk = path.parent / (path.stem + '.vmrk')
        result["has_vmrk"] = vmrk.exists()

    except Exception as e:
        result["error"] = str(e)
    return result


def _read_eeg_metadata(path: Path) -> Dict[str, Any]:
    """
    Unified EEG metadata reader — routes by extension.
    Returns a normalized metadata dict.
    """
    ext = path.suffix.lower()
    if ext == '.edf':
        return _read_edf_metadata(path)
    elif ext == '.vhdr':
        return _read_vhdr_metadata(path)
    else:
        # .set and .bdf: basic fallback (no heavy dependency)
        return {"format": ext.lstrip('.'), "error": None}


# ============================================================================
# EDF+ annotations reader
# ============================================================================

def _read_edf_annotations(path: Path, sampling_rate: float = 1.0) -> List[Dict[str, Any]]:
    """
    Read EDF+ TAL (Time-stamped Annotations Lists) from EDF+ file.
    Returns list of {onset, duration, trial_type} dicts.
    """
    events = []
    try:
        with open(path, 'rb') as f:
            raw_hdr = f.read(256)
            n_signals  = _parse_int(raw_hdr[252:256])
            n_records  = _parse_int(raw_hdr[236:244])
            duration   = _parse_float(raw_hdr[244:252])
            n_bytes_hdr = _parse_int(raw_hdr[184:192])

            if n_signals <= 0 or n_signals > 512:
                return events

            # Read samples per record
            f.seek(n_bytes_hdr - 8 * n_signals)
            ns_raw = f.read(8 * n_signals)
            samples_per_record = [_parse_int(ns_raw[i*8:(i+1)*8]) for i in range(n_signals)]

            # Find annotation channel index
            f.seek(256)
            labels_raw = f.read(16 * n_signals)
            ann_ch_idx = None
            for i in range(n_signals):
                label = labels_raw[i*16:(i+1)*16].decode('ascii', errors='ignore').strip().upper()
                if 'ANNOTATIONS' in label or 'EDF ANNOTATIONS' in label:
                    ann_ch_idx = i
                    break

            if ann_ch_idx is None:
                return events

            # Bytes per sample = 2 (EDF uses 16-bit integers)
            bytes_per_record = [spr * 2 for spr in samples_per_record]
            ann_offset_in_record = sum(bytes_per_record[:ann_ch_idx])
            ann_bytes = bytes_per_record[ann_ch_idx]

            f.seek(n_bytes_hdr)
            for _ in range(n_records):
                record_data = f.read(sum(bytes_per_record))
                ann_data = record_data[ann_offset_in_record:ann_offset_in_record + ann_bytes]
                # Parse TAL: +onset\x14duration\x14annotation\x00
                raw_str = ann_data.decode('utf-8', errors='ignore').replace('\x15', '').rstrip('\x00')
                for tal in raw_str.split('\x14\x00'):
                    if not tal.strip():
                        continue
                    parts = tal.split('\x14')
                    if not parts:
                        continue
                    onset_str = parts[0].lstrip('+')
                    try:
                        onset = float(onset_str)
                    except Exception:
                        continue
                    dur = 0.0
                    label = ""
                    if len(parts) >= 2:
                        try:
                            dur = float(parts[1])
                        except Exception:
                            label = parts[1]
                    if len(parts) >= 3:
                        label = parts[2].strip()
                    if not label:
                        continue
                    events.append({
                        "onset":      round(onset, 6),
                        "duration":   round(dur, 6),
                        "trial_type": label,
                    })
    except Exception:
        pass

    events.sort(key=lambda x: x["onset"])
    return events


# ============================================================================
# External event file reader
# ============================================================================

def _read_external_event_file(
    event_path: Path,
    mapping: Dict[str, Any],
    sampling_rate: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Read an external event file using the column mapping produced by the LLM.
    Handles tab/comma/space separators, optional headers, comment lines.
    Converts onset/duration units to seconds.
    """
    events = []

    sep_map = {"tab": "\t", "comma": ",", "space": None}
    sep_char = sep_map.get(mapping.get("separator", "tab"), "\t")

    onset_col      = mapping.get("onset_col")
    duration_col   = mapping.get("duration_col")
    trial_type_col = mapping.get("trial_type_col")
    header_row     = mapping.get("header_row", True)
    skip_rows      = int(mapping.get("skip_rows", 0))
    onset_unit     = mapping.get("onset_unit", "seconds")
    duration_unit  = mapping.get("duration_unit", "seconds")

    if not onset_col:
        warn(f"  EEG events: no onset_col in mapping for {event_path.name}")
        return events

    # Handle BrainVision .vmrk format separately
    source_type = mapping.get("source_type", "")
    if source_type == "brainvision_vmrk" or event_path.suffix.lower() == ".vmrk":
        return _read_vmrk_events(event_path, sampling_rate)

    try:
        text = event_path.read_text(encoding='utf-8', errors='ignore')
        lines = text.splitlines()

        # Skip comment lines and blank lines
        data_lines = []
        skipped = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('%'):
                continue
            if skipped < skip_rows:
                skipped += 1
                continue
            data_lines.append(stripped)

        if not data_lines:
            return events

        # Parse header
        header = []
        start_idx = 0
        if header_row and data_lines:
            raw_header = data_lines[0]
            if sep_char:
                header = [h.strip() for h in raw_header.split(sep_char)]
            else:
                header = raw_header.split()
            start_idx = 1

        def _get_col_idx(col_name: Optional[str]) -> Optional[int]:
            if col_name is None:
                return None
            if header:
                # Case-insensitive match
                for i, h in enumerate(header):
                    if h.lower() == col_name.lower():
                        return i
                # Try numeric index
                try:
                    return int(col_name)
                except Exception:
                    return None
            else:
                try:
                    return int(col_name)
                except Exception:
                    return None

        onset_idx    = _get_col_idx(onset_col)
        duration_idx = _get_col_idx(duration_col)
        tt_idx       = _get_col_idx(trial_type_col)

        if onset_idx is None:
            warn(f"  EEG events: cannot find onset column '{onset_col}'")
            return events

        for line in data_lines[start_idx:]:
            if not line.strip():
                continue
            if sep_char:
                cols = [c.strip() for c in line.split(sep_char)]
            else:
                cols = line.split()

            try:
                raw_onset = float(cols[onset_idx])
            except Exception:
                continue

            # Unit conversion
            if onset_unit == "milliseconds":
                onset_sec = round(raw_onset / 1000.0, 6)
            elif onset_unit == "samples" and sampling_rate > 0:
                onset_sec = round(raw_onset / sampling_rate, 6)
            else:
                onset_sec = round(raw_onset, 6)

            dur_sec = "n/a"
            if duration_idx is not None and duration_idx < len(cols):
                try:
                    raw_dur = float(cols[duration_idx])
                    if duration_unit == "milliseconds":
                        dur_sec = round(raw_dur / 1000.0, 6)
                    elif duration_unit == "samples" and sampling_rate > 0:
                        dur_sec = round(raw_dur / sampling_rate, 6)
                    else:
                        dur_sec = round(raw_dur, 6)
                except Exception:
                    pass

            trial_type = "n/a"
            if tt_idx is not None and tt_idx < len(cols):
                trial_type = cols[tt_idx].strip() or "n/a"

            events.append({
                "onset":      onset_sec,
                "duration":   dur_sec,
                "trial_type": trial_type,
            })

    except Exception as e:
        warn(f"  EEG events: failed to read {event_path.name}: {e}")

    events.sort(key=lambda x: x["onset"])
    return events


def _read_vmrk_events(vmrk_path: Path, sampling_rate: float = 1.0) -> List[Dict[str, Any]]:
    """
    Read BrainVision .vmrk marker file.
    Format: Mk{n}={type},{description},{position},{length},{channel}
    position is in samples; sampling_rate used for conversion to seconds.
    """
    events = []
    try:
        text = vmrk_path.read_text(encoding='utf-8', errors='ignore')
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('['):
                continue
            m = re.match(r'^Mk\d+=(.+)', line, re.IGNORECASE)
            if not m:
                continue
            parts = m.group(1).split(',')
            if len(parts) < 3:
                continue
            ev_type  = parts[0].strip()
            ev_desc  = parts[1].strip() if len(parts) > 1 else ev_type
            try:
                position = int(parts[2].strip())
            except Exception:
                continue
            try:
                length = int(parts[3].strip()) if len(parts) > 3 else 0
            except Exception:
                length = 0

            trial_type = ev_desc if ev_desc else ev_type
            # Clean up common BrainVision label formatting: "S  1" → "S1"
            trial_type = re.sub(r'\s+', '', trial_type)

            onset_sec = round(position / sampling_rate, 6) if sampling_rate > 0 else position
            dur_sec   = round(length  / sampling_rate, 6) if sampling_rate > 0 and length > 0 else "n/a"

            events.append({
                "onset":      onset_sec,
                "duration":   dur_sec,
                "trial_type": trial_type,
            })
    except Exception as e:
        warn(f"  EEG events: failed to read {vmrk_path.name}: {e}")

    events.sort(key=lambda x: x["onset"])
    return events


# ============================================================================
# BIDS sidecar generators
# ============================================================================

def _generate_eeg_json(
    eeg_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
) -> None:
    """
    Generate *_eeg.json sidecar (run-level, REQUIRED).

    Required BIDS EEG fields:
      TaskName, SamplingFrequency, EEGChannelCount, RecordingDuration,
      EEGReference, PowerLineFrequency, SoftwareFilters
    """
    m = re.search(r"task-([A-Za-z0-9]+)", bids_stem)
    task_name = m.group(1) if m else "unknown"

    fs    = meta.get("dominant_sampling_rate") or meta.get("sampling_rate") or 0
    n_eeg = meta.get("n_eeg_channels", meta.get("n_channels", 0))
    n_eog = meta.get("n_eog_channels", 0)
    n_ecg = meta.get("n_ecg_channels", 0)
    n_misc = meta.get("n_misc_channels", 0)
    duration = meta.get("total_duration_s")

    content: Dict[str, Any] = {
        "TaskName":            task_name,
        "SamplingFrequency":   fs,
        "EEGChannelCount":     n_eeg,
        "RecordingDuration":   duration if duration else "n/a",
        "EEGReference":        "unknown",
        "PowerLineFrequency":  "n/a",
        "SoftwareFilters":     "n/a",
    }
    if n_eog:
        content["EOGChannelCount"] = n_eog
    if n_ecg:
        content["ECGChannelCount"] = n_ecg
    if n_misc:
        content["MiscChannelCount"] = n_misc

    out_path = eeg_path.parent / f"{bids_stem}_eeg.json"
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return
    write_json(out_path, content)
    info(f"  ✓ {out_path.name}")


def _generate_eeg_channels_tsv(
    eeg_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
) -> None:
    """
    Generate *_channels.tsv sidecar (task-level, REQUIRED).
    Strip run- entity — shared across runs.
    """
    channel_labels = meta.get("channel_labels", [])
    channel_units  = meta.get("channel_units", [])
    sampling_rates = meta.get("sampling_rates", [])

    if not channel_labels:
        warn(f"  _channels.tsv: no channel info for {eeg_path.name}, skipping")
        return

    # task-level: strip run- entity
    stem_no_run = re.sub(r"_run-[A-Za-z0-9]+", "", bids_stem)
    out_path = eeg_path.parent / f"{stem_no_run}_channels.tsv"
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return

    eeg_set  = set(meta.get("eeg_channels",  []))
    eog_set  = set(meta.get("eog_channels",  []))
    ecg_set  = set(meta.get("ecg_channels",  []))
    misc_set = set(meta.get("misc_channels", []))

    header = ["name", "type", "units", "sampling_frequency", "status"]
    rows = ["\t".join(header)]

    for i, label in enumerate(channel_labels):
        if label in eeg_set:
            ch_type = "EEG"
        elif label in eog_set:
            ch_type = "EOG"
        elif label in ecg_set:
            ch_type = "ECG"
        else:
            ch_type = "MISC"

        unit = channel_units[i] if i < len(channel_units) else "n/a"
        fs_val = str(sampling_rates[i]) if i < len(sampling_rates) else "n/a"
        rows.append("\t".join([label, ch_type, unit, fs_val, "good"]))

    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    info(f"  ✓ {out_path.name} ({len(channel_labels)} channels)")


def _generate_eeg_events_tsv(
    eeg_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
    eeg_mapping_doc: Optional[Dict[str, Any]],
    data_root: Optional[Path],
) -> None:
    """
    Generate *_events.tsv sidecar (run-level, RECOMMENDED).

    Priority:
    1. External event file (using LLM column mapping)
    2. EDF+ internal annotations
    3. No events → skip silently
    """
    out_path = eeg_path.parent / f"{bids_stem}_events.tsv"
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return

    events: List[Dict[str, Any]] = []
    fs = meta.get("dominant_sampling_rate") or meta.get("sampling_rate") or 1.0

    # Priority 1: external event file via LLM mapping
    if eeg_mapping_doc and data_root:
        eeg_relpath = None
        # Try to match by stem
        for relpath_key in eeg_mapping_doc.get("files", {}).keys():
            if Path(relpath_key).stem == eeg_path.stem or relpath_key in str(eeg_path):
                eeg_relpath = relpath_key
                break

        if eeg_relpath:
            file_mapping = eeg_mapping_doc["files"][eeg_relpath]
            source_type  = file_mapping.get("source_type", "")

            if source_type != "edf_plus_annotations":
                event_file_relpath = file_mapping.get("event_file_path")
                if event_file_relpath:
                    event_file_abs = data_root / event_file_relpath
                    if event_file_abs.exists():
                        events = _read_external_event_file(event_file_abs, file_mapping, fs)
                    else:
                        warn(f"  EEG events: event file not found: {event_file_relpath}")

    # Priority 2: EDF+ annotations
    if not events and meta.get("is_edf_plus") and eeg_path.suffix.lower() == ".edf":
        events = _read_edf_annotations(eeg_path, fs)

    if not events:
        return  # No events to write

    header = ["onset", "duration", "trial_type"]
    lines = ["\t".join(header)]
    for ev in events:
        lines.append("\t".join(str(ev[h]) for h in header))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"  ✓ {out_path.name} ({len(events)} event(s))")


def _parse_aux_electrode_file(
    aux_path: Path,
    column_order: Optional[List[str]] = None,
) -> Dict[str, tuple]:
    """
    Parse an auxiliary electrode coordinate file.
    Returns dict of {label: (x, y, z)} or {label: (x, y, "n/a")} for 2D.
    Supports CSV, TSV, space-delimited, and common EEG electrode formats.
    """
    coords: Dict[str, tuple] = {}
    try:
        text = aux_path.read_text(encoding="utf-8", errors="ignore")
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith(("#", "%", ";"))]
        if not lines:
            return coords

        # Detect separator
        sep = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)

        def split_line(line: str) -> List[str]:
            if sep:
                return [c.strip() for c in line.split(sep)]
            return line.split()

        # Detect header
        first = split_line(lines[0])
        has_header = any(c.lower() in ("name", "label", "channel", "electrode", "x", "y", "z")
                         for c in first)

        col_names = column_order
        data_start = 0
        if has_header and not col_names:
            col_names = [c.lower() for c in first]
            data_start = 1

        for line in lines[data_start:]:
            parts = split_line(line)
            if not parts:
                continue

            if col_names:
                row = dict(zip(col_names, parts))
                label = row.get("name") or row.get("label") or row.get("channel") or row.get("electrode")
                try:
                    x = float(row.get("x", "n/a"))
                    y = float(row.get("y", "n/a"))
                    z_raw = row.get("z", "n/a")
                    z = float(z_raw) if z_raw != "n/a" else "n/a"
                except (ValueError, TypeError):
                    x, y, z = "n/a", "n/a", "n/a"
            else:
                # Positional: assume name x y z or name theta phi
                if len(parts) >= 4:
                    label = parts[0]
                    try:
                        x, y = float(parts[1]), float(parts[2])
                        z = float(parts[3])
                    except ValueError:
                        x, y, z = "n/a", "n/a", "n/a"
                elif len(parts) == 3:
                    label = parts[0]
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), "n/a"
                    except ValueError:
                        x, y, z = "n/a", "n/a", "n/a"
                else:
                    continue

            if label:
                coords[label] = (x, y, z)

    except Exception as e:
        warn(f"  _parse_aux_electrode_file: {e}")

    return coords


def _generate_electrodes_tsv(
    eeg_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
    aux_mapping_doc: Optional[Dict[str, Any]],
    data_root: Optional[Path],
) -> bool:
    """
    Generate *_electrodes.tsv (subject-level, RECOMMENDED).

    Coordinates are taken only from a real auxiliary coordinate file
    identified by the LLM in eeg_aux_mapping.json. Template/idealized
    positions (e.g. a standard 10-20 lookup table) are intentionally NOT
    used, following MNE-BIDS guidance: electrodes.tsv is optional and
    should reflect measured positions, not idealized ones. If no aux
    coordinate file is available, this function returns False and no
    electrodes.tsv is written. Channels absent from the aux file get n/a.

    Returns True if file was created.
    """
    channel_labels = meta.get("channel_labels", [])
    if not channel_labels:
        return False

    # subject-level: strip task- and run- entities
    m = re.search(r"(sub-[A-Za-z0-9]+)", bids_stem)
    sub_label = m.group(1) if m else bids_stem
    out_path = eeg_path.parent / f"{sub_label}_electrodes.tsv"
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return True

    # Try to load coordinates from aux file
    aux_coords: Dict[str, tuple] = {}
    if aux_mapping_doc and data_root:
        for relpath, file_info in aux_mapping_doc.get("files", {}).items():
            if file_info.get("content_type") == "electrode_coordinates":
                aux_file = data_root / relpath
                if aux_file.exists():
                    col_order = file_info.get("column_order")
                    aux_coords = _parse_aux_electrode_file(aux_file, col_order)
                    if aux_coords:
                        info(f"  Using electrode coordinates from: {aux_file.name}")
                    break
    
    # Only generate electrodes.tsv when real measured coordinates are available.
    # Do NOT use template/idealized positions (e.g. standard 10-20 lookup table).
    # This follows MNE-BIDS guidance and BIDS spec (electrodes.tsv is optional).
    if not aux_coords:
        return False

    # Build rows: real measured coordinates from aux file only.
    # n/a for any channels not found in the aux file.
    eeg_set  = set(meta.get("eeg_channels",  []))
    eog_set  = set(meta.get("eog_channels",  []))
    ecg_set  = set(meta.get("ecg_channels",  []))

    header = ["name", "x", "y", "z", "type"]
    rows = ["\t".join(header)]
    n_matched = 0

    for label in channel_labels:
        if label in aux_coords:
            x, y, z = aux_coords[label]
            n_matched += 1
        else:
            x, y, z = "n/a", "n/a", "n/a"

        if label in eeg_set:
            ch_type = "EEG"
        elif label in eog_set:
            ch_type = "EOG"
        elif label in ecg_set:
            ch_type = "ECG"
        else:
            ch_type = "MISC"

        rows.append("\t".join([label, str(x), str(y), str(z), ch_type]))

    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    info(f"  ✓ {out_path.name} ({len(channel_labels)} electrodes, "
         f"{n_matched} with coordinates)")
    return True


def _generate_eeg_coordsystem_json(
    eeg_path: Path,
    bids_stem: str,
    aux_mapping_doc: Optional[Dict[str, Any]],
) -> None:
    """
    Generate *_coordsystem.json (subject-level, REQUIRED when electrodes.tsv exists).
    """
    m = re.search(r"(sub-[A-Za-z0-9]+)", bids_stem)
    sub_label = m.group(1) if m else bids_stem
    out_path = eeg_path.parent / f"{sub_label}_coordsystem.json"
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return

    # Determine coordinate system from aux mapping if available
    coord_system = "CapTrak"  # default for standard 10-20
    coord_units  = "mm"
    if aux_mapping_doc:
        for file_info in aux_mapping_doc.get("files", {}).values():
            if file_info.get("content_type") == "electrode_coordinates":
                detected_sys = file_info.get("coordinate_system", "")
                if "mni" in detected_sys.lower():
                    coord_system = "MNI152Lin"
                elif "ctf" in detected_sys.lower():
                    coord_system = "CTF"
                break

    content = {
        "EEGCoordinateSystem":      coord_system,
        "EEGCoordinateUnits":       coord_units,
        "EEGCoordinateSystemDescription": (
            "Electrode coordinates derived from an auxiliary coordinate "
            f"file provided with the dataset (coordinate system: {coord_system})."
        ),
    }
    write_json(out_path, content)
    info(f"  ✓ {out_path.name}")


# ============================================================================
# Public entry point
# ============================================================================

def generate_eeg_bids_sidecars(
    eeg_path: Path,
    bids_stem: str,
    eeg_mapping_doc: Optional[Dict[str, Any]] = None,
    data_root: Optional[Path] = None,
    eeg_aux_mapping_doc: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate all EEG-BIDS sidecar files for a single EEG file.

    Called by executor immediately after each EEG file is copied.
    Reads file header once, then generates:
      *_eeg.json        (REQUIRED, run-level)
      *_channels.tsv    (REQUIRED, task-level — strips run-)
      *_events.tsv      (RECOMMENDED, run-level — if events available)

    Args:
        eeg_path:        Absolute path to the .edf/.vhdr/.set/.bdf file.
        bids_stem:       Filename without extension,
                         e.g. "sub-1_task-rest_eeg"
        eeg_mapping_doc: Parsed eeg_event_mapping.json (from planner), or None.
        data_root:       Original data root (for resolving event file paths), or None.
    """
    # Strip trailing _eeg if present (same pattern as NIRS)
    if bids_stem.endswith("_eeg"):
        bids_stem = bids_stem[:-4]

    try:
        meta = _read_eeg_metadata(eeg_path)
        _generate_eeg_json(eeg_path, bids_stem, meta)
        _generate_eeg_channels_tsv(eeg_path, bids_stem, meta)
        _generate_eeg_events_tsv(eeg_path, bids_stem, meta, eeg_mapping_doc, data_root)
        created = _generate_electrodes_tsv(eeg_path, bids_stem, meta,
                                           eeg_aux_mapping_doc, data_root)
        if created:
            _generate_eeg_coordsystem_json(eeg_path, bids_stem, eeg_aux_mapping_doc)
    except Exception as e:
        warn(f"  generate_eeg_bids_sidecars failed for {eeg_path.name}: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Internal helpers
# ============================================================================

def _parse_int(raw: bytes) -> int:
    try:
        return int(raw.decode('ascii', errors='ignore').strip())
    except Exception:
        return 0


def _parse_float(raw: bytes) -> float:
    try:
        return float(raw.decode('ascii', errors='ignore').strip())
    except Exception:
        return 0.0
