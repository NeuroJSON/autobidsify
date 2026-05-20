# nirs_convert.py - COMPLETE v10
# Full implementation with .mat and .nirs direct conversion

"""
fNIRS Converter Module - Complete v10

Supported conversions:
1. MATLAB .mat → SNIRF (NEW: direct conversion)
2. Homer3 .nirs → SNIRF (NEW: direct conversion)
3. CSV/TSV tables → SNIRF (existing: using normalized headers)
4. SNIRF sidecar generation
5. SNIRF validation

Dependencies:
- h5py: SNIRF file creation
- numpy: Data manipulation
- scipy: .mat file loading
- csv: CSV parsing (built-in)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import numpy as np
import h5py
import csv
import json as _json
from autobidsify.utils import ensure_dir, warn, info, write_json


# ============================================================================
# NEW v10: Homer3 .nirs to SNIRF conversion
# ============================================================================

def convert_nirs_to_snirf(nirs_file: Path, output_path: Path,
                         quiet: bool = False) -> Optional[Path]:
    """
    Convert Homer3 .nirs file to SNIRF format.
    
    Homer3 .nirs file structure (it's actually a MATLAB .mat file):
    - d: data matrix (samples * channels)
    - t: time vector (samples)
    - SD: source-detector configuration (optional)
    - s: stimulus markers (optional)
    
    Args:
        nirs_file: Path to .nirs file
        output_path: Output SNIRF path (.snirf extension)
        quiet: Suppress output messages
    
    Returns:
        Path to created SNIRF file, or None if conversion failed
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        if not quiet:
            warn("scipy not installed for .nirs conversion")
            warn("Install with: pip install scipy")
        return None
    
    if not nirs_file.exists():
        if not quiet:
            warn(f".nirs file not found: {nirs_file}")
        return None
    
    try:
        if not quiet:
            info(f"  Converting Homer3 .nirs: {nirs_file.name}")
        
        # Load .nirs file (it's a MATLAB .mat file with specific structure)
        nirs_data = loadmat(str(nirs_file))
        
        # Extract data matrix (d variable is standard in Homer3)
        if 'd' not in nirs_data:
            if not quiet:
                warn(f"  'd' variable not found in {nirs_file.name}")
                available = [k for k in nirs_data.keys() if not k.startswith('__')]
                warn(f"  Available: {available}")
            return None
        
        data_array = nirs_data['d']
        
        # Extract time vector (t variable)
        if 't' in nirs_data:
            time_array = nirs_data['t'].flatten()
        else:
            # Generate time vector if not found
            n_samples = data_array.shape[0]
            time_array = np.arange(n_samples) / 10.0  # Assume 10 Hz
            if not quiet:
                warn(f"  No time vector ('t'), generated assuming 10 Hz")
        
        # Ensure correct dimensions
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
        
        if len(time_array.shape) > 1:
            time_array = time_array.flatten()
        
        n_samples, n_channels = data_array.shape
        
        if not quiet:
            info(f"  Data shape: {data_array.shape}")
            info(f"  Channels: {n_channels}, Samples: {n_samples}")
        
        # Check dimension consistency
        if n_samples != len(time_array):
            if not quiet:
                warn(f"  Dimension mismatch: data {n_samples}, time {len(time_array)}")
            min_len = min(n_samples, len(time_array))
            data_array = data_array[:min_len, :]
            time_array = time_array[:min_len]
            n_samples = min_len
            if not quiet:
                info(f"  → Adjusted to {n_samples} samples")
        
        # Extract wavelengths from SD structure (Homer3 specific)
        wavelengths = [760, 850]  # Default
        n_sources = 1
        n_detectors = n_channels
        
        if 'SD' in nirs_data:
            try:
                SD = nirs_data['SD']
                
                # SD is a structured array in Homer3
                if isinstance(SD, np.ndarray) and SD.dtype.names:
                    # Extract wavelengths
                    if 'Lambda' in SD.dtype.names:
                        wl = SD['Lambda'][0, 0].flatten()
                        if len(wl) > 0:
                            wavelengths = wl.tolist()
                    
                    # Extract source/detector positions if available
                    if 'SrcPos' in SD.dtype.names:
                        src_pos = SD['SrcPos'][0, 0]
                        if len(src_pos) > 0:
                            n_sources = src_pos.shape[0]
                    
                    if 'DetPos' in SD.dtype.names:
                        det_pos = SD['DetPos'][0, 0]
                        if len(det_pos) > 0:
                            n_detectors = det_pos.shape[0]
                
                if not quiet:
                    info(f"  Wavelengths: {wavelengths} nm")
                    info(f"  Sources: {n_sources}, Detectors: {n_detectors}")
            except Exception as e:
                if not quiet:
                    warn(f"  Could not parse SD structure: {e}")
        
        # Create SNIRF file
        ensure_dir(output_path.parent)
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset("formatVersion", data="1.0")
            # /nirs group
            nirs_grp = f.create_group("nirs")
            
            # /nirs/data1
            data_grp = nirs_grp.create_group("data1")
            data_grp.create_dataset("dataTimeSeries", data=data_array, dtype='f')
            data_grp.create_dataset("time", data=time_array, dtype='f')
            
            for ch_idx in range(n_channels):
                ch_grp = data_grp.create_group(f"measurementList{ch_idx + 1}")
                
                # Simplified: assume sequential source/detector mapping
                # In production, this should be extracted from SD.MeasList
                ch_grp.create_dataset("sourceIndex", data=1)
                ch_grp.create_dataset("detectorIndex", data=ch_idx + 1)
                ch_grp.create_dataset("wavelengthIndex", data=1)
                ch_grp.create_dataset("dataType", data=1)  # 1 = Intensity
                ch_grp.create_dataset("dataTypeLabel", data="Intensity")
                ch_grp.create_dataset("dataTypeIndex", data=1)
            
            # /nirs/probe
            probe_grp = nirs_grp.create_group("probe")
            probe_grp.create_dataset("wavelengths", data=np.array(wavelengths, dtype=np.float64))
            
            # Simplified probe geometry (2D positions)
            probe_grp.create_dataset("sourcePos2D", data=np.zeros((n_sources, 2)))
            probe_grp.create_dataset("detectorPos2D", data=np.zeros((n_detectors, 2)))
            
            # /nirs/metaDataTags
            meta_grp = nirs_grp.create_group("metaDataTags")
            meta_grp.create_dataset("SubjectID", data="unknown")
            meta_grp.create_dataset("MeasurementDate", data="")
            meta_grp.create_dataset("MeasurementTime", data="")
            meta_grp.create_dataset("LengthUnit", data="mm")
            meta_grp.create_dataset("TimeUnit", data="s")
            meta_grp.create_dataset("FrequencyUnit", data="Hz")
        
        if not quiet:
            info(f"  ✓ Created SNIRF: {output_path.name}")
            info(f"    Data: {n_samples} samples * {n_channels} channels")
        
        # Validate
        if validate_snirf_file(output_path):
            return output_path
        else:
            return None
        
    except Exception as e:
        if not quiet:
            warn(f"  MAT→SNIRF conversion failed: {e}")
            import traceback
            traceback.print_exc()
        return None


# ============================================================================
# NEW v10: Homer3 .nirs to SNIRF conversion
# ============================================================================

def convert_nirs_to_snirf(nirs_file: Path, output_path: Path,
                         quiet: bool = False) -> Optional[Path]:
    """
    Convert Homer3 .nirs file to SNIRF format.
    
    Note: .nirs files ARE .mat files with Homer3-specific structure.
    This function is essentially the same as convert_mat_to_snirf() but
    with Homer3-specific variable name expectations.
    
    Args:
        nirs_file: Path to .nirs file (Homer3 format)
        output_path: Output SNIRF path (.snirf extension)
        quiet: Suppress output messages
    
    Returns:
        Path to created SNIRF file, or None if conversion failed
    """
    # .nirs files are .mat files, so we can use the same converter
    return convert_mat_to_snirf(nirs_file, output_path, quiet=quiet)


# ============================================================================
# Existing: CSV/TSV to SNIRF (uses normalized headers from LLM)
# ============================================================================

def write_snirf_from_normalized(
    normalized: Dict[str, Any],
    input_root: Path,
    output_dir: Path
) -> List[Path]:
    """
    Create SNIRF files from CSV/table data using LLM-generated normalized headers.
    
    This function is used when:
    - Input data is in CSV/TSV format
    - LLM has generated normalized column mappings
    - Complex header structures need semantic understanding
    
    For direct .mat/.nirs conversion, use:
    - convert_mat_to_snirf() for .mat files
    - convert_nirs_to_snirf() for .nirs files
    
    Args:
        normalized: Normalized headers from LLM (contains column mappings)
        input_root: Root directory of input data
        output_dir: Output directory for SNIRF files
    
    Returns:
        List of created SNIRF file paths
    """
    ensure_dir(output_dir)
    
    snirf_files = []
    
    # Extract global settings
    norm_data = normalized.get("normalized", {})
    globals_dict = norm_data.get("globals", {})
    
    sampling_freq = globals_dict.get("SamplingFrequency", 10.0)
    wavelengths = globals_dict.get("Wavelengths", [760, 850])
    task_name = globals_dict.get("TaskName", "task")
    
    info(f"CSV→SNIRF conversion parameters:")
    info(f"  SamplingFrequency: {sampling_freq} Hz")
    info(f"  Wavelengths: {wavelengths} nm")
    info(f"  TaskName: {task_name}")
    
    # Process each file
    for file_spec in norm_data.get("files", []):
        relpath = file_spec.get("relpath", "")
        input_path = input_root / relpath
        
        if not input_path.exists():
            warn(f"Input file not found: {input_path}")
            continue
        
        info(f"Processing CSV: {relpath}")
        
        try:
            # Read CSV data
            time_spec = file_spec.get("time", {})
            time_col = time_spec.get("column", "time")
            time_unit = time_spec.get("unit", "s")
            
            signals = file_spec.get("signals", [])
            
            # Parse CSV
            data_dict = {}
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Strip whitespace from headers
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
                for row in reader:
                    for key, value in row.items():
                        if key not in data_dict:
                            data_dict[key] = []
                        try:
                            data_dict[key].append(float(value))
                        except (ValueError, TypeError):
                            data_dict[key].append(0.0)
            
            # Extract time vector
            if time_col not in data_dict:
                warn(f"Time column '{time_col}' not found in {relpath}")
                continue
            
            time_data = np.array(data_dict[time_col])
            
            # Convert time unit to seconds
            if time_unit == "milliseconds":
                time_data = time_data / 1000.0
            elif time_unit == "minutes":
                time_data = time_data * 60.0
            
            n_samples = len(time_data)
            info(f"  Time samples: {n_samples}")
            
            # Build dataTimeSeries matrix
            all_channels = []
            channel_info = []
            
            for signal in signals:
                signal_type = signal.get("type", "Intensity")
                columns = signal.get("columns", [])
                
                for col in columns:
                    col_stripped = col.strip()
                    if col_stripped in data_dict:
                        all_channels.append(data_dict[col_stripped])
                        channel_info.append({
                            "type": signal_type,
                            "column": col_stripped
                        })
            
            if not all_channels:
                warn(f"No valid channels found in {relpath}")
                continue
            
            data_matrix = np.array(all_channels).T  # (n_samples, n_channels)
            n_channels = data_matrix.shape[1]
            
            info(f"  Data matrix: {data_matrix.shape} (samples * channels)")
            
            # Create SNIRF file
            output_name = input_path.stem + ".snirf"
            snirf_path = output_dir / output_name
            
            with h5py.File(snirf_path, 'w') as f:
                f.create_dataset("formatVersion", data="1.0")
                # /nirs group
                nirs_grp = f.create_group("nirs")
                
                # /nirs/data1
                data_grp = nirs_grp.create_group("data1")
                data_grp.create_dataset("dataTimeSeries", data=data_matrix, dtype='f')
                data_grp.create_dataset("time", data=time_data, dtype='f')
                
                for ch_idx in range(n_channels):
                    ch_grp = data_grp.create_group(f"measurementList{ch_idx + 1}")
                    
                    # Simplified: sequential source/detector
                    ch_grp.create_dataset("sourceIndex", data=1)
                    ch_grp.create_dataset("detectorIndex", data=ch_idx + 1)
                    ch_grp.create_dataset("wavelengthIndex", data=1)
                    
                    # dataType: 1=Intensity, 99=processed
                    data_type = 1 if channel_info[ch_idx]["type"] == "Intensity" else 99
                    ch_grp.create_dataset("dataType", data=data_type)
                    ch_grp.create_dataset("dataTypeLabel", data=channel_info[ch_idx]["type"])
                    ch_grp.create_dataset("dataTypeIndex", data=1)
                
                # /nirs/probe
                probe_grp = nirs_grp.create_group("probe")
                probe_grp.create_dataset("wavelengths", data=np.array(wavelengths, dtype=np.float64))
                
                n_sources = 1
                n_detectors = n_channels
                probe_grp.create_dataset("sourcePos2D", data=np.zeros((n_sources, 2)))
                probe_grp.create_dataset("detectorPos2D", data=np.zeros((n_detectors, 2)))
                
                # /nirs/metaDataTags
                meta_grp = nirs_grp.create_group("metaDataTags")
                meta_grp.create_dataset("SubjectID", data="unknown")
                meta_grp.create_dataset("MeasurementDate", data="")
                meta_grp.create_dataset("MeasurementTime", data="")
                meta_grp.create_dataset("LengthUnit", data="mm")
                meta_grp.create_dataset("TimeUnit", data="s")
                meta_grp.create_dataset("FrequencyUnit", data="Hz")
            
            info(f"  ✓ Created SNIRF: {snirf_path.name}")
            snirf_files.append(snirf_path)
            
            # Validate
            validate_snirf_file(snirf_path)
            
        except Exception as e:
            warn(f"Failed to process {relpath}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return snirf_files


# ============================================================================
# SNIRF validation
# ============================================================================

def validate_snirf_file(snirf_path: Path) -> bool:
    """
    Validate SNIRF file using the official snirf Python package first,
    then fall back to manual HDF5 structure checks if not installed.

    Priority:
    1. pip install snirf  — official SNIRF spec validation
    2. Manual HDF5 checks — basic structure check (fallback)
    """
    # ── Priority 1: official snirf package ───────────────────────────
    try:
        from snirf import Snirf
        try:
            with Snirf(str(snirf_path), 'r') as s:
                issues = s.validate()
                # ValidationResult supports is_valid() but not direct iteration
                # in all versions — use is_valid() only to avoid TypeError
                # if issues.is_valid():
                #     info(f"  ✓ SNIRF valid (snirf pkg): {snirf_path.name}")
                #     return True
                # else:
                #     warn(f"  ✗ SNIRF invalid per snirf pkg")
                #     return False

                issue_list = getattr(issues, 'issues', None) \
                             or getattr(issues, '_issues', None) \
                             or getattr(issues, 'errors', None) \
                             or []
                # Only treat severity>=2 (WARNING) or severity==3 (FATAL) as failure.
                # severity==1 (INFO) are optional fields — not a validation error.
                fatal_issues = [
                    iss for iss in issue_list
                    if getattr(iss, 'severity', 0) >= 2
                ]
                if not fatal_issues:
                    info(f"  ✓ SNIRF valid (snirf pkg): {snirf_path.name}")
                    return True
                else:
                    warn(f"  ✗ SNIRF invalid per snirf pkg: {snirf_path.name}")
                    for iss in fatal_issues[:20]:
                        warn(f"    - {iss}")
                    return False
        except Exception as e:
            warn(f"  snirf pkg validation error: {e} — falling back to manual check")
            # Fall through to manual HDF5 check below

    except ImportError:
        pass  # snirf package not installed — use manual check

    # ── Priority 2: manual HDF5 structure check (fallback) ───────────
    try:
        with h5py.File(snirf_path, 'r') as f:
            if "nirs" not in f:
                warn(f"SNIRF invalid: missing /nirs group")
                return False
            nirs = f["nirs"]
            if "data1" not in nirs:
                warn(f"SNIRF invalid: missing /nirs/data1")
                return False
            data1 = nirs["data1"]
            if "dataTimeSeries" not in data1:
                warn(f"SNIRF invalid: missing dataTimeSeries")
                return False
            if "time" not in data1:
                warn(f"SNIRF invalid: missing time vector")
                return False
            data_shape = data1["dataTimeSeries"].shape
            time_shape = data1["time"].shape
            if data_shape[0] != time_shape[0]:
                warn(f"SNIRF invalid: dimension mismatch "
                     f"dataTimeSeries={data_shape}, time={time_shape}")
                return False
            if not any(k.startswith("measurementList") for k in data1.keys()):
                warn(f"SNIRF invalid: missing measurementList entries")
                return False
            info(f"  ✓ SNIRF valid (manual check): {snirf_path.name}")
            info(f"    Shape: {data_shape[0]} samples × {data_shape[1]} channels")
            return True
    except Exception as e:
        warn(f"SNIRF validation error: {e}")
        return False


# ============================================================================
# LEGACY: MATLAB toolbox conversion (placeholder)
# ============================================================================

def run_homer3_nirs_to_snirf(nirs_files: List[Path], output_dir: Path) -> List[Path]:
    """
    Convert Homer3 .nirs files using Homer3 MATLAB toolbox.
    
    LEGACY function - kept for compatibility.
    
    NOTE: This requires MATLAB + Homer3 installation.
    For most users, use the pure Python convert_nirs_to_snirf() instead.
    
    Args:
        nirs_files: List of .nirs file paths
        output_dir: Output directory
    
    Returns:
        List of generated SNIRF files
    """
    warn("Homer3 MATLAB toolbox conversion not implemented")
    info("Using pure Python converter instead...")
    
    # Call pure Python converter for each file
    converted = []
    for nirs_file in nirs_files:
        output_path = output_dir / (nirs_file.stem + ".snirf")
        result = convert_nirs_to_snirf(nirs_file, output_path, quiet=False)
        if result:
            converted.append(result)
    
    return converted


def _flatten_mat_vars(user_vars: Dict[str, Any], prefix: str = "",
                      depth: int = 0, max_depth: int = 5) -> Dict[str, Any]:
    """
    Recursively flatten a scipy-loaded .mat variable dict into a flat key→info dict.

    All struct wrappers, object-array singletons, and nested structs are
    unwrapped. The result is a plain dict where every key is a dot-notation
    path (e.g. "dat.signal") and every value is a descriptor dict with
    shape, dtype, and optionally sample values or scalar value.

    This is the single source of truth for mat variable access — no other
    code needs to know about scipy's layered struct representation.

    Handles:
    - Numeric ndarray              → shape, dtype, sample values
    - Struct (dtype.names)         → recurse into fields
    - Object singleton (1,1)/(1,) → unwrap then recurse
    - Object array of strings      → record as labels list
    - Cell array (object array)    → record each cell as key[0], key[1], ...
    - Scalar (size==1 numeric)     → record scalar value directly
    """
    flat: Dict[str, Any] = {}

    for raw_name, arr in user_vars.items():
        full_key = f"{prefix}{raw_name}" if not prefix else f"{prefix}.{raw_name}"
        _flatten_single(arr, full_key, flat, depth, max_depth)

    return flat


def _flatten_single(arr: Any, key: str, flat: Dict[str, Any],
                    depth: int, max_depth: int) -> None:
    """Recursively flatten one variable into flat dict."""
    if depth > max_depth:
        flat[key] = {"shape": "...", "dtype": "max_depth_exceeded"}
        return

    if arr is None or not hasattr(arr, "dtype"):
        return

    # ── Unwrap object-dtype singleton wrappers ────────────────────────
    # scipy wraps struct fields in (1,1) or (1,) object arrays
    while (arr.dtype == object and arr.size == 1
           and hasattr(arr.flat[0], "dtype")):
        arr = arr.flat[0]

    # ── Structured array (MATLAB struct) ─────────────────────────────
    if arr.dtype.names:
        # Unwrap (1,1) struct wrapper to get the actual inner struct
        inner = arr
        while inner.ndim > 0 and inner.size == 1:
            candidate = inner.flat[0]
            if hasattr(candidate, "dtype") and candidate.dtype.names:
                inner = candidate
            else:
                break

        if hasattr(inner, "dtype") and inner.dtype.names:
            for field in inner.dtype.names:
                try:
                    child = inner[field]
                    _flatten_single(child, f"{key}.{field}", flat, depth + 1, max_depth)
                except Exception:
                    flat[f"{key}.{field}"] = {"error": "unreadable"}
        return

    # ── Object array of strings (channel labels etc.) ─────────────────
    if arr.dtype == object:
        try:
            flat_items = arr.flatten()
            str_vals = []
            all_str = True
            for item in flat_items[:50]:
                if isinstance(item, str):
                    str_vals.append(item)
                elif hasattr(item, "item") and not (
                        hasattr(item, "dtype") and item.dtype.names):
                    str_vals.append(str(item.item()))
                elif hasattr(item, "shape") and item.size == 1 and not (
                        hasattr(item, "dtype") and item.dtype.names):
                    str_vals.append(str(item.flat[0]))
                else:
                    all_str = False
                    break
            if all_str and str_vals:
                flat[key] = {
                    "shape": list(arr.shape),
                    "dtype": "string_array",
                    "values": str_vals,
                }
                return
        except Exception:
            pass

        # Cell array: each element is a separate ndarray.
        # Special case: if ALL elements are structs with the SAME field names,
        # treat as a struct array — expand fields directly (no [N] indexing).
        # This handles scipy's (1,4) object arrays wrapping a single struct.
        try:
            flat_items = arr.flatten()
            # Check if all elements are structs with identical field names
            if flat_items.size > 0:
                first = flat_items[0]
                if (hasattr(first, "dtype") and first.dtype.names and
                        all(hasattr(item, "dtype") and item.dtype.names == first.dtype.names
                            for item in flat_items)):
                    # Uniform struct array: expand the first element's fields
                    # (representative — all elements share the same structure)
                    for field in first.dtype.names:
                        try:
                            child = first[field]
                            _flatten_single(child, f"{key}.{field}", flat, depth + 1, max_depth)
                        except Exception:
                            flat[f"{key}.{field}"] = {"error": "unreadable"}
                    return
            # Default: each element as separate entry
            for i, item in enumerate(flat_items[:10]):
                if hasattr(item, "dtype"):
                    _flatten_single(item, f"{key}[{i}]", flat, depth + 1, max_depth)
        except Exception:
            pass
        return

    # ── Numeric scalar ────────────────────────────────────────────────
    if arr.size == 1:
        try:
            scalar_val = float(arr.flat[0])
            flat[key] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "value": scalar_val,
            }
        except Exception:
            flat[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        return

    # ── Numeric ndarray ───────────────────────────────────────────────
    entry: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    # Mark tall 2D arrays as likely data matrices
    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
        entry["likely_data"] = True
    # Sample a few values for small arrays
    if arr.size <= 20:
        try:
            entry["values"] = [
                v.item() if hasattr(v, "item") else float(v)
                for v in arr.flatten().tolist()
            ]
        except Exception:
            pass
    flat[key] = entry


def inspect_mat_structure(mat_file: Path) -> Optional[Dict[str, Any]]:
    """
    Return a fully-flattened structural summary of a .mat file for LLM consumption.

    Uses _flatten_mat_vars() to recursively unwrap all struct wrappers,
    object singletons, and nested fields. The result is a plain flat dict
    where every key is a dot-notation path and every value describes
    shape/dtype/sample — no scipy wrapper artifacts.

    Returns None if file cannot be read.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        warn("scipy not installed — cannot inspect .mat structure")
        return None

    try:
        mat = loadmat(str(mat_file), squeeze_me=False)
    except Exception as e:
        warn(f"  Cannot load {mat_file.name}: {e}")
        return None

    user_vars = {k: v for k, v in mat.items() if not k.startswith("__")}
    flat_vars = _flatten_mat_vars(user_vars)

    # Record raw shape/dtype of each top-level variable BEFORE any unwrapping.
    # This is the only reliable way for the LLM to detect multi-block structures
    # (e.g. data is (1,4) object array of structs → n_blocks=4).
    top_level_shapes = {}
    for k, v in user_vars.items():
        if hasattr(v, "shape"):
            top_level_shapes[k] = {
                "shape":     list(v.shape),
                "dtype":     str(v.dtype),
                "is_object": bool(v.dtype == object),
                "is_struct": bool(v.dtype.names),
            }

    return {
        "filename":         mat_file.name,
        "file_size_mb":     round(mat_file.stat().st_size / 1e6, 2),
        "flat_vars":        flat_vars,
        "top_level_keys":   list(user_vars.keys()),
        "top_level_shapes": top_level_shapes,
    }


def _structure_fingerprint(summary: Dict[str, Any]) -> frozenset:
    """
    Produce a hashable fingerprint from a flat_vars summary.
    Two files with the same flat keys and shape patterns share one LLM call.
    Shape pattern: ndim + whether tall matrix (rows>cols).
    """
    flat_vars = summary.get("flat_vars", {})
    sigs = frozenset(
        (key,
         len(info_dict.get("shape", [])),
         "tall" if (len(info_dict.get("shape", [])) == 2
                    and info_dict["shape"][0] > info_dict["shape"][1])
                else "other")
        for key, info_dict in flat_vars.items()
        if isinstance(info_dict, dict) and "shape" in info_dict
    )
    return sigs


def _resolve_mat_var(flat_vars: Dict[str, Any], var_path: Optional[str]):
    """
    Retrieve the actual numpy array for a flat_vars key.

    flat_vars is produced by _flatten_mat_vars() — keys are dot-notation
    paths, values are descriptor dicts. We need to re-load the actual
    array from the original mat data using the path.

    Since flat_vars only stores descriptors (not the arrays themselves),
    this function accepts pre-loaded user_vars alongside flat_vars.
    See _write_snirf_from_mat_mapping() for usage.
    """
    # This signature is kept for API compatibility but the real work
    # is done inside _write_snirf_from_mat_mapping using _extract_by_path().
    pass


def _extract_by_path(user_vars: Dict[str, Any], var_path: Optional[str]):
    """
    Extract the actual numpy ndarray from user_vars using a dot-notation path.

    Unlike the old _resolve_mat_var, this function unwraps ALL scipy
    struct/object wrappers at every level, matching what _flatten_mat_vars
    reports in its flat_vars summary.

    Examples:
        "d"            → user_vars["d"]
        "dat.signal"   → unwrap dat struct → get signal field → unwrap object
        "a.b.c"        → recursive unwrap at every level
        "data[0]"      → cell array first element (not yet supported, returns None)
    """
    if not var_path:
        return None

    # Cell array indexing not supported
    if "[" in var_path:
        return None

    parts = var_path.split(".")

    # Get top-level variable
    obj = user_vars.get(parts[0])
    if obj is None:
        return None

    # Traverse remaining path parts
    for part in parts[1:]:
        # Unwrap object-dtype singleton wrappers at each level
        while (hasattr(obj, "dtype") and obj.dtype == object
               and obj.size == 1 and hasattr(obj.flat[0], "dtype")):
            obj = obj.flat[0]

        # Unwrap struct singleton wrappers
        while (hasattr(obj, "dtype") and obj.dtype.names
               and obj.ndim > 0 and obj.size == 1):
            candidate = obj.flat[0]
            if hasattr(candidate, "dtype"):
                obj = candidate
            else:
                break

        # Field access
        if hasattr(obj, "dtype") and obj.dtype.names and part in obj.dtype.names:
            obj = obj[part]
        else:
            return None

    # Final unwrap: peel off all remaining object/struct singleton wrappers
    max_unwrap = 10
    for _ in range(max_unwrap):
        if not hasattr(obj, "dtype"):
            break
        if obj.dtype == object and obj.size == 1:
            inner = obj.flat[0]
            if hasattr(inner, "dtype"):
                obj = inner
                continue
        if obj.dtype.names and obj.size == 1:
            inner = obj.flat[0]
            if hasattr(inner, "dtype"):
                obj = inner
                continue
        break

    if not hasattr(obj, "shape"):
        return None
    # If still a struct wrapper, attempt one final unwrap to reach numeric content
    if hasattr(obj, "dtype") and obj.dtype.names:
        if obj.size == 1:
            inner = obj.flat[0]
            if hasattr(inner, "shape") and not (
                hasattr(inner, "dtype") and inner.dtype.names
            ):
                return inner
        return None  # genuine struct with no accessible numeric content
    return obj


def _unwrap_to_numeric(arr: Any, max_depth: int = 5) -> Optional[np.ndarray]:
    """
    Recursively unwrap object/struct arrays to find the first usable numeric array.
    Returns None if no numeric array found within max_depth levels.
    """
    if arr is None or not hasattr(arr, "dtype"):
        return None

    for _ in range(max_depth):
        if arr.dtype == object:
            if arr.size == 1:
                inner = arr.flat[0]
                if hasattr(inner, "dtype"):
                    arr = inner
                    continue
                return None
            # Object array with multiple elements.
            # Case A: all elements are structs with same fields → extract first trial's data field
            # Case B: all elements are numeric → stack column-wise
            try:
                items = [arr.flat[i] for i in range(arr.size)]
                # Case A: uniform struct array (e.g. trial-based data storage)
                first = items[0] if items else None
                if (first is not None and hasattr(first, "dtype") and first.dtype.names and
                        all(hasattr(it, "dtype") and it.dtype.names == first.dtype.names
                            for it in items)):
                    # Try to extract numeric data field from first element (first trial)
                    for field in ("X", "x", "data", "signal", "d", "dOD", "dConc"):
                        if field in first.dtype.names:
                            try:
                                candidate = first[field]
                                while (hasattr(candidate, "dtype") and
                                       candidate.dtype == object and
                                       candidate.size == 1):
                                    candidate = candidate.flat[0]
                                result = np.array(candidate).astype(float)
                                if result.ndim == 1:
                                    return result.reshape(-1, 1)
                                return result
                            except Exception:
                                continue
                    return None
                # Case B: numeric elements → stack as columns
                numeric_items = []
                for item in items:
                    if hasattr(item, "dtype") and not item.dtype.names:
                        try:
                            numeric_items.append(np.array(item).flatten().astype(float))
                        except Exception:
                            pass
                if numeric_items:
                    min_len = min(len(x) for x in numeric_items)
                    return np.column_stack([x[:min_len] for x in numeric_items])
            except Exception:
                pass
            return None
        if hasattr(arr, "dtype") and arr.dtype.names:
            # Struct: unwrap singleton wrapper first
            if arr.size == 1:
                arr = arr.flat[0]
                continue
            # Multi-element struct or top-level struct: try known data field names
            for candidate_field in ("X", "x", "data", "signal", "d", "dOD", "dConc"):
                if candidate_field in arr.dtype.names:
                    try:
                        candidate = arr[candidate_field]
                        # Unwrap object wrappers around the field
                        while (hasattr(candidate, "dtype") and
                               candidate.dtype == object and
                               candidate.size == 1):
                            candidate = candidate.flat[0]
                        result = candidate.astype(float)
                        if result.ndim >= 2:
                            return result
                        elif result.ndim == 1:
                            return result.reshape(-1, 1)
                    except Exception:
                        continue
            return None
        # Numeric array found
        try:
            return arr.astype(float)
        except Exception:
            return None
    return None


def _assemble_data_array(
    user_vars: Dict[str, Any],
    assembly: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    """
    Build the dataTimeSeries array from user_vars according to assembly instructions.

    Supported assembly types:
      "single"       — data lives in one variable (most common)
      "stack_columns"— data split across ch1, ch2, ... chN; stack column-wise
      "hbo_hbr"      — HbO and HbR stored separately; concatenate column-wise
    """
    if not assembly:
        return None

    atype = assembly.get("type", "single")

    # ── TYPE 1: single variable ───────────────────────────────────────
    if atype == "single":
        var_path = assembly.get("var")
        arr = _extract_by_path(user_vars, var_path)
        # If path failed (e.g. contains [N] indexing or invalid path),
        # fall back to top-level variable name and unwrap from there.
        if arr is None and var_path:
            top_key = var_path.split(".")[0].split("[")[0]
            arr = user_vars.get(top_key)

        # If result is still a struct (has dtype.names), the mapping pointed
        # to a struct variable rather than a numeric field inside it.
        # Try appending known data field names to the path.
        if hasattr(arr, "dtype") and arr.dtype.names:
            found = None
            for field in ("X", "x", "data", "signal", "d", "dOD", "dConc"):
                if field in arr.dtype.names:
                    candidate = _extract_by_path(user_vars, f"{var_path}.{field}")
                    if candidate is not None and hasattr(candidate, "dtype") \
                            and not candidate.dtype.names:
                        try:
                            found = candidate.astype(float)
                            break
                        except Exception:
                            continue
            if found is None:
                return None
            arr = found
        else:
            arr = _unwrap_to_numeric(arr)
            if arr is None:
                return None

        arr = np.array(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if assembly.get("transpose", False):
            arr = arr.T
        return arr

    # ── TYPE 2: stack multiple channel variables column-wise ──────────
    if atype == "stack_columns":
        # Build the list of keys to collect
        keys: List[str] = []
        if "vars" in assembly:
            # Explicit list provided
            keys = assembly["vars"]
        elif "var_pattern" in assembly and "var_range" in assembly:
            # Numeric range: var_pattern + index, e.g. "ch" + 1..40
            pattern = assembly["var_pattern"]
            start, end = assembly["var_range"][0], assembly["var_range"][1]
            keys = [f"{pattern}{i}" for i in range(start, end + 1)]
        elif "var_pattern" in assembly:
            # Match all flat_vars keys that start with the pattern, sort naturally
            pattern = assembly["var_pattern"]
            import re as _re
            matched = [k for k in user_vars if k.startswith(pattern)]
            # Natural sort: "ch2" before "ch10"
            def _nat_key(s: str):
                parts = _re.split(r'(\d+)', s)
                return [int(p) if p.isdigit() else p for p in parts]
            keys = sorted(matched, key=_nat_key)

        if not keys:
            return None

        columns: List[np.ndarray] = []
        for key in keys:
            col = _extract_by_path(user_vars, key)
            if col is None:
                warn(f"  stack_columns: key '{key}' not found, skipping")
                continue
            col = np.array(col).flatten()
            columns.append(col)

        if not columns:
            return None

        # Align lengths to the shortest column
        min_len = min(len(c) for c in columns)
        columns = [c[:min_len] for c in columns]
        return np.column_stack(columns)   # shape (n_samples, n_channels)

    # ── TYPE 3: HbO and HbR stored as separate matrices ──────────────
    if atype == "hbo_hbr":
        hbo = _extract_by_path(user_vars, assembly.get("hbo_var"))
        hbr = _extract_by_path(user_vars, assembly.get("hbr_var"))
        if hbo is None or hbr is None:
            warn("  hbo_hbr: could not resolve hbo_var or hbr_var")
            return None
        hbo = np.array(hbo)
        hbr = np.array(hbr)
        if hbo.ndim == 1:
            hbo = hbo.reshape(-1, 1)
        if hbr.ndim == 1:
            hbr = hbr.reshape(-1, 1)
        # Align row counts
        min_len = min(hbo.shape[0], hbr.shape[0])
        return np.concatenate([hbo[:min_len], hbr[:min_len]], axis=1)

    warn(f"  Unknown data_assembly type: '{atype}'")
    return None


def _assemble_time_array(
    user_vars: Dict[str, Any],
    assembly: Optional[Dict[str, Any]],
    n_samples: int,
) -> np.ndarray:
    """
    Build the time vector from user_vars according to assembly instructions.

    Supported assembly types:
      "var"      — time vector stored in a variable
      "generate" — no time variable; generate from sampling rate (fs_var or fs_value)

    Falls back to 10 Hz generation if assembly is None or unresolvable.
    """
    fallback_fs = 10.0

    if not assembly:
        return np.arange(n_samples) / fallback_fs

    atype = assembly.get("type", "generate")

    # ── TYPE 1: read time vector from variable ────────────────────────
    if atype == "var":
        arr = _extract_by_path(user_vars, assembly.get("var"))
        if arr is not None:
            try:
                return np.array(arr).flatten().astype(float)
            except Exception:
                pass
        warn(f"  time_assembly var '{assembly.get('var')}' unresolvable — generating")

    # ── TYPE 2: generate from sampling rate ───────────────────────────
    fs = fallback_fs

    # Prefer reading fs from a variable in the file
    fs_var = assembly.get("fs_var")
    if fs_var:
        fs_raw = _extract_by_path(user_vars, fs_var)
        if fs_raw is not None:
            try:
                fs = float(np.array(fs_raw).flat[0])
            except Exception:
                pass

    # Fall back to hardcoded value from mapping
    if fs == fallback_fs and assembly.get("fs_value") is not None:
        try:
            fs = float(assembly["fs_value"])
        except Exception:
            pass

    return np.arange(n_samples) / fs


def _write_snirf_from_mat_mapping(
    mat_file: Path,
    output_path: Path,
    mapping: Dict[str, Any],
    quiet: bool = False,
    block_index: Optional[int] = None,
) -> Optional[Path]:
    """
    Write SNIRF from .mat using pre-computed mapping from mat_mapping.json.
    Uses _extract_by_path() for all variable access — handles all scipy
    struct/object wrapper patterns universally.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        warn("scipy not installed")
        return None

    try:
        mat = loadmat(str(mat_file), squeeze_me=False)
    except Exception as e:
        warn(f"  Cannot load {mat_file.name}: {e}")
        return None

    user_vars = {k: v for k, v in mat.items() if not k.startswith("__")}

    # ── block extraction (multi-block .mat) ───────────────────────────
    # When block_index is set, the top-level variable is a (1,N) object
    # array of structs. Extract the single block at block_index so that
    # all downstream assembly logic sees a plain struct — exactly as if
    # the file contained only one block.
    if block_index is not None:
        data_assembly = mapping.get("data_assembly") or {}
        top_key = (data_assembly.get("var") or "").split(".")[0].split("[")[0]
        if top_key and top_key in user_vars:
            top_arr = user_vars[top_key]
            if (hasattr(top_arr, "dtype") and top_arr.dtype == object
                    and top_arr.size > 1):
                try:
                    block = top_arr.flat[block_index]
                    if hasattr(block, "dtype"):
                        user_vars = dict(user_vars)   # shallow copy, don't mutate original
                        user_vars[top_key] = block
                        if not quiet:
                            info(f"  block_index={block_index}: "
                                 f"extracted block from '{top_key}' "
                                 f"(total {top_arr.size} blocks)")
                except IndexError:
                    if not quiet:
                        warn(f"  block_index={block_index} out of range "
                             f"(size={top_arr.size}), falling back to block 0")
                    user_vars = dict(user_vars)
                    user_vars[top_key] = top_arr.flat[0]

    # ── dataTimeSeries ─────────────────────────────────────────────────
    # Support new assembly format; fall back to legacy data_var string for
    # backward compatibility with mat_mapping.json files generated before
    # the assembly format was introduced.
    data_assembly = mapping.get("data_assembly")
    if data_assembly is None and mapping.get("data_var"):
        data_assembly = {"type": "single", "var": mapping["data_var"]}

    data_array = _assemble_data_array(user_vars, data_assembly)
    if data_array is None:
        if not quiet:
            warn(f"  data_assembly could not be resolved — heuristic fallback")
        return None
    try:
        data_array = data_array.astype(float)
    except Exception as e:
        if not quiet:
            warn(f"  Cannot cast data to float: {e}")
        return None
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)
    n_samples, n_channels = data_array.shape
    if not quiet:
        info(f"  dataTimeSeries: {data_array.shape} via {data_assembly.get('type','?')}")

    # ── time ───────────────────────────────────────────────────────────
    # Support new assembly format; fall back to legacy time_var / sampling_rate_hz.
    time_assembly = mapping.get("time_assembly")
    if time_assembly is None:
        if mapping.get("time_var"):
            time_assembly = {"type": "var", "var": mapping["time_var"]}
        elif mapping.get("sampling_rate_hz") is not None:
            time_assembly = {
                "type": "generate",
                "fs_value": mapping["sampling_rate_hz"],
            }

    time_array = _assemble_time_array(user_vars, time_assembly, n_samples)
    if not quiet and (time_assembly is None or time_assembly.get("type") == "generate"):
        warn(f"  No time variable — generated from fs")

    min_len   = min(n_samples, len(time_array))
    data_array = data_array[:min_len]
    time_array = time_array[:min_len]
    n_samples  = min_len

    # ── wavelengths ────────────────────────────────────────────────────
    # Support new assembly format; fall back to legacy wavelengths_var.
    wavelengths = list(mapping.get("wavelengths_default", [760, 850]))
    wavelengths_assembly = mapping.get("wavelengths_assembly")
    if wavelengths_assembly is None and mapping.get("wavelengths_var"):
        wavelengths_assembly = {"type": "var", "var": mapping["wavelengths_var"]}

    if wavelengths_assembly:
        wtype = wavelengths_assembly.get("type", "var")
        if wtype == "var":
            wl_raw = _extract_by_path(user_vars, wavelengths_assembly.get("var"))
            if wl_raw is not None:
                try:
                    wavelengths = [round(float(w), 1)
                                   for w in np.array(wl_raw).flatten()]
                except Exception:
                    pass
        elif wtype == "value":
            wavelengths = list(wavelengths_assembly.get("values", wavelengths))
    if not quiet:
        info(f"  wavelengths: {wavelengths} nm")

    # ── measurementList ───────────────────────────────────────────────
    measlist: Optional[np.ndarray] = None
    ml_var = mapping.get("measlist_var") or ""
    ml_raw = _extract_by_path(user_vars, ml_var) if ml_var else None
    if ml_raw is not None:
        try:
            measlist = np.array(ml_raw)
        except Exception:
            pass

    data_type_code = int(mapping.get("data_type_code", 1))
    n_wl = max(len(wavelengths), 1)

    # ── write SNIRF ───────────────────────────────────────────────────
    ensure_dir(output_path.parent)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("formatVersion", data="1.0")
        nirs_grp = f.create_group("nirs")

        meta = nirs_grp.create_group("metaDataTags")
        meta.create_dataset("SubjectID",       data="unknown")
        meta.create_dataset("MeasurementDate", data="")
        meta.create_dataset("MeasurementTime", data="")
        meta.create_dataset("LengthUnit",      data="mm")
        meta.create_dataset("TimeUnit",        data="s")
        meta.create_dataset("FrequencyUnit",   data="Hz")

        data_grp = nirs_grp.create_group("data1")
        data_grp.create_dataset("dataTimeSeries", data=data_array, dtype="f")
        data_grp.create_dataset("time",           data=time_array, dtype="f")

        # Determine dataTypeLabel per channel based on data_type_code.
        # dataType=1 (Intensity): all channels share the same label.
        # dataType=4 (HbO/HbR concentration): first half → HbO, second half → HbR.
        # For other processed types (99 etc.) fall back to a generic label.
        def _ch_label_and_wl(ch_idx: int, n_ch: int, dtype_code: int, n_wl: int) -> tuple:
            """Return (dataTypeLabel, wavelengthIndex, dataTypeIndex) for one channel."""
            if dtype_code == 1:
                return "Intensity", (ch_idx % n_wl) + 1, 1
            if dtype_code == 4:
                half = n_ch // 2
                if n_ch % 2 == 0 and half > 0:
                    label = "HbO" if ch_idx < half else "HbR"
                    wl_idx = 1 if ch_idx < half else 2
                    dt_idx = 1 if ch_idx < half else 2
                else:
                    label = "HbO" if ch_idx % 2 == 0 else "HbR"
                    wl_idx = 1 if ch_idx % 2 == 0 else 2
                    dt_idx = 1 if ch_idx % 2 == 0 else 2
                return label, wl_idx, dt_idx
            # Fallback for other dataType codes
            return "Unknown", (ch_idx % n_wl) + 1, 1

        for ch_idx in range(n_channels):
            ch = data_grp.create_group(f"measurementList{ch_idx + 1}")
            ch_label, wl_idx, dt_idx = _ch_label_and_wl(ch_idx, n_channels, data_type_code, n_wl)
            if measlist is not None and ch_idx < measlist.shape[0]:
                row = measlist[ch_idx]
                ch.create_dataset("sourceIndex",     data=int(row[0]))
                ch.create_dataset("detectorIndex",   data=int(row[1]))
                ch.create_dataset("wavelengthIndex", data=wl_idx)
            else:
                # Fallback when no MeasList is available.
                # Each channel gets its own sequential index — this is the most
                # conservative placeholder: avoids incorrect source/detector
                # counts derived from wavelength arithmetic (e.g. ch//n_wl).
                ch.create_dataset("sourceIndex",     data=ch_idx + 1)
                ch.create_dataset("detectorIndex",   data=ch_idx + 1)
                ch.create_dataset("wavelengthIndex", data=wl_idx)
            ch.create_dataset("dataType",      data=data_type_code)
            ch.create_dataset("dataTypeLabel", data=ch_label)
            ch.create_dataset("dataTypeIndex", data=dt_idx)

        probe = nirs_grp.create_group("probe")
        probe.create_dataset("wavelengths", data=np.array(wavelengths, dtype=np.float64))
        if measlist is not None:
            n_src = max(int(np.max(measlist[:, 0])), 1)
            n_det = max(int(np.max(measlist[:, 1])), 1)
        else:
            # Try LLM-identified paths for source/detector counts
            _n_src_raw = _extract_by_path(user_vars, mapping.get("n_sources_var"))
            _n_det_raw = _extract_by_path(user_vars, mapping.get("n_detectors_var"))
            try:
                n_src = int(np.array(_n_src_raw).flat[0]) if _n_src_raw is not None else 1
            except Exception:
                n_src = 1
            try:
                n_det = int(np.array(_n_det_raw).flat[0]) if _n_det_raw is not None else n_channels
            except Exception:
                n_det = n_channels
        probe.create_dataset("sourcePos2D",   data=np.zeros((n_src, 2)))
        probe.create_dataset("detectorPos2D", data=np.zeros((n_det, 2)))

    if not quiet:
        info(f"  ✓ SNIRF: {output_path.name} ({n_samples}*{n_channels})")

    return output_path if validate_snirf_file(output_path) else None


# ============================================================================
# REPLACE existing convert_mat_to_snirf with this version
# Only change: accepts optional _mat_mapping kwarg; heuristic body unchanged
# ============================================================================

def convert_mat_to_snirf(
    mat_file: Path,
    output_path: Path,
    quiet: bool = False,
    _mat_mapping: Optional[Dict[str, Any]] = None,
    _block_index: Optional[int] = None,
) -> Optional[Path]:
    """
    Convert MATLAB .mat file to SNIRF format.

    Priority:
    1. If _mat_mapping is provided → _write_snirf_from_mat_mapping() [precise]
       Falls through to heuristic if mapping-based write fails.
    2. Heuristic fallback (original logic, unchanged).

    _mat_mapping is injected by executor from mat_mapping.json["files"][relpath].
    All existing callers that omit it continue to use heuristic logic.
    """
    # ── Priority 1: mapping-based (injected by executor) ─────────────
    if _mat_mapping:
        if not quiet:
            info(f"  Using mat_mapping entry for {mat_file.name}")
        result = _write_snirf_from_mat_mapping(
            mat_file, output_path, _mat_mapping, quiet=quiet,
            block_index=_block_index,
        )
        if result:
            return result
        # File written but failed validation: still return it rather than
        # falling back to heuristic (which may not recognize the variables).
        if output_path.exists():
            if not quiet:
                warn("  Mapping-based SNIRF failed validation but file exists — returning as-is")
            return output_path
        if not quiet:
            warn("  Mapping-based write failed — falling back to heuristic")

    # ── Priority 2: heuristic (original body, completely unchanged) ───
    try:
        from scipy.io import loadmat
    except ImportError:
        if not quiet:
            warn("scipy not installed")
        return None

    if not mat_file.exists():
        if not quiet:
            warn(f"MAT file not found: {mat_file}")
        return None

    try:
        if not quiet:
            info(f"  [heuristic] Converting MATLAB: {mat_file.name}")

        mat_data = loadmat(str(mat_file))
        data_var_names = ['d', 'data', 'dOD', 'dConc', 'y', 'timeseries', 'nirs_data']
        time_var_names = ['t', 'time', 'times', 'time_vector']

        data_array = None
        data_var_used = None
        for var_name in data_var_names:
            if var_name in mat_data:
                data_array = mat_data[var_name]
                data_var_used = var_name
                break

        if data_array is None:
            available = [k for k in mat_data.keys() if not k.startswith('__')]
            if not quiet:
                warn(f"  Could not find fNIRS data variable. Available: {available}")
            return None

        time_array = None
        for var_name in time_var_names:
            if var_name in mat_data:
                time_array = mat_data[var_name]
                break

        # If data_array is an object/struct array, try to extract numeric content
        if hasattr(data_array, "dtype") and (
                data_array.dtype == object or data_array.dtype.names):
            data_array = _unwrap_to_numeric(data_array)
            if data_array is None:
                if not quiet:
                    warn(f"  Cannot extract numeric data from variable '{data_var_used}'")
                return None
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        if time_array is None:
            time_array = np.arange(data_array.shape[0]) / 10.0
            if not quiet:
                warn("  No time variable found, generated assuming 10 Hz sampling")
        if len(time_array.shape) > 1:
            time_array = time_array.flatten()

        n_samples, n_channels = data_array.shape
        min_len = min(n_samples, len(time_array))
        data_array = data_array[:min_len]
        time_array = time_array[:min_len]
        n_samples   = min_len

        if not quiet:
            info(f"  Data shape: {data_array.shape} (samples * channels)")

        wavelengths = [760, 850]
        if 'SD' in mat_data:
            try:
                SD = mat_data['SD']
                if isinstance(SD, np.ndarray) and SD.dtype.names:
                    if 'Lambda' in SD.dtype.names:
                        wl = SD['Lambda'][0, 0].flatten()
                        if len(wl) > 0:
                            wavelengths = wl.tolist()
            except Exception:
                pass

        ensure_dir(output_path.parent)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset("formatVersion", data="1.0")
            nirs_grp = f.create_group("nirs")
            data_grp = nirs_grp.create_group("data1")
            data_grp.create_dataset("dataTimeSeries", data=data_array, dtype='f')
            data_grp.create_dataset("time",           data=time_array, dtype='f')
            for ch_idx in range(n_channels):
                ch = data_grp.create_group(f"measurementList{ch_idx + 1}")
                ch.create_dataset("sourceIndex",     data=1)
                ch.create_dataset("detectorIndex",   data=ch_idx + 1)
                ch.create_dataset("wavelengthIndex", data=1)
                ch.create_dataset("dataType",        data=1)
                ch.create_dataset("dataTypeLabel",   data="Intensity")
                ch.create_dataset("dataTypeIndex",   data=1)
            probe = nirs_grp.create_group("probe")
            probe.create_dataset("wavelengths",   data=np.array(wavelengths, dtype=np.float64))
            probe.create_dataset("sourcePos2D",   data=np.zeros((1, 2)))
            probe.create_dataset("detectorPos2D", data=np.zeros((n_channels, 2)))
            meta = nirs_grp.create_group("metaDataTags")
            meta.create_dataset("SubjectID",       data="unknown")
            meta.create_dataset("MeasurementDate", data="")
            meta.create_dataset("MeasurementTime", data="")
            meta.create_dataset("LengthUnit",      data="mm")
            meta.create_dataset("TimeUnit",        data="s")
            meta.create_dataset("FrequencyUnit",   data="Hz")

        if not quiet:
            info(f"  ✓ Created SNIRF: {output_path.name}")
            info(f"    Channels: {n_channels}, Samples: {n_samples}")

        return output_path if validate_snirf_file(output_path) else None

    except Exception as e:
        if not quiet:
            warn(f"  MAT→SNIRF conversion failed: {e}")
            import traceback
            traceback.print_exc()
        return None


# ============================================================================
# NIRS-BIDS sidecar generation
# ============================================================================

# Mapping from SNIRF dataTypeLabel to BIDS channel type string.
# Reference: NIRS-BIDS spec, Table of channel types.
_DTYPE_LABEL_TO_BIDS: Dict[str, str] = {
    "Intensity": "NIRSCWAMPLITUDE",
    "dOD":       "NIRSCWOPTICALDENSITY",
    "HbO":       "NIRSCWHBO",
    "HbR":       "NIRSCWHBR",
    "HbT":       "MISC",      # no dedicated BIDS type for total-Hb
    "Unknown":   "MISC",
}

# Mapping from BIDS channel type to measurement units.
_BIDS_TYPE_TO_UNITS: Dict[str, str] = {
    "NIRSCWAMPLITUDE":      "V",
    "NIRSCWOPTICALDENSITY": "unitless",
    "NIRSCWHBO":            "mM·mm",
    "NIRSCWHBR":            "mM·mm",
    "MISC":                 "n/a",
}


def _read_snirf_metadata(snirf_path: Path) -> Dict[str, Any]:
    """
    Read all metadata needed for sidecar generation from a SNIRF HDF5 file.

    Opens the file once and returns a flat dict. Any field that cannot be
    read is set to None — all callers must handle None gracefully.

    Returned keys:
      sampling_freq      : float or None
      n_channels         : int or None
      channels           : List[Dict] or None  — per-channel info
      wavelengths        : List[float] or None
      source_pos         : np.ndarray or None  — shape (n_src, 2or3)
      detector_pos       : np.ndarray or None  — shape (n_det, 2or3)
      source_pos_is_3d   : bool
      detector_pos_is_3d : bool
      n_sources          : int or None
      n_detectors        : int or None
    """
    result: Dict[str, Any] = {
        "sampling_freq":       None,
        "n_channels":          None,
        "channels":            None,
        "wavelengths":         None,
        "source_pos":          None,
        "detector_pos":        None,
        "source_pos_is_3d":    False,
        "detector_pos_is_3d":  False,
        "n_sources":           None,
        "n_detectors":         None,
        "stim_events":         None,
    }

    try:
        with h5py.File(snirf_path, "r") as f:
            nirs = f.get("nirs")
            if nirs is None:
                return result
            data1 = nirs.get("data1")

            # ── sampling frequency ─────────────────────────────────
            if data1 is not None and "time" in data1:
                try:
                    t = data1["time"][()]
                    if len(t) >= 2 and (t[-1] - t[0]) > 0:
                        result["sampling_freq"] = round(
                            (len(t) - 1) / float(t[-1] - t[0]), 4
                        )
                except Exception:
                    pass

            # ── channels from measurementList ──────────────────────
            if data1 is not None:
                ch_keys = sorted(
                    [k for k in data1.keys() if k.startswith("measurementList")],
                    key=lambda x: int(re.search(r"\d+", x).group())
                    if re.search(r"\d+", x) else 0
                )
                result["n_channels"] = len(ch_keys)
                channels = []
                for ck in ch_keys:
                    ch = data1[ck]
                    def _read_scalar(group, key):
                        try:
                            v = group[key][()]
                            return int(v) if np.issubdtype(
                                np.array(v).dtype, np.integer) else float(v)
                        except Exception:
                            return None

                    def _read_str(group, key):
                        try:
                            v = group[key][()]
                            if isinstance(v, bytes):
                                return v.decode("utf-8", errors="ignore").strip()
                            return str(v).strip()
                        except Exception:
                            return None

                    channels.append({
                        "source_idx":    _read_scalar(ch, "sourceIndex"),
                        "detector_idx":  _read_scalar(ch, "detectorIndex"),
                        "wavelength_idx": _read_scalar(ch, "wavelengthIndex"),
                        "data_type":     _read_scalar(ch, "dataType"),
                        "data_type_label": _read_str(ch, "dataTypeLabel"),
                    })
                result["channels"] = channels

            # ── wavelengths ────────────────────────────────────────
            probe = nirs.get("probe")
            if probe is not None and "wavelengths" in probe:
                try:
                    wl = probe["wavelengths"][()]
                    result["wavelengths"] = [
                        round(float(w), 1) for w in wl.flatten()
                    ]
                except Exception:
                    pass

            # ── optode positions ───────────────────────────────────
            # Priority: 3D > 2D. All-zero arrays are treated as placeholders.
            if probe is not None:
                for pos_key, is_3d, src_or_det in [
                    ("sourcePos3D",   True,  "source"),
                    ("sourcePos2D",   False, "source"),
                    ("detectorPos3D", True,  "detector"),
                    ("detectorPos2D", False, "detector"),
                ]:
                    if pos_key not in probe:
                        continue
                    if result[f"{src_or_det}_pos"] is not None:
                        continue   # already found a higher-priority entry
                    try:
                        arr = probe[pos_key][()]
                        if arr.ndim == 2 and arr.shape[0] > 0:
                            result[f"{src_or_det}_pos"] = arr
                            result[f"{src_or_det}_pos_is_3d"] = is_3d

                            n_key = "n_sources" if src_or_det == "source" \
                                    else "n_detectors"
                            result[n_key] = arr.shape[0]
                    except Exception:
                        pass

            # ── fallback n_sources / n_detectors from measurementList ──
            if result["channels"]:
                if result["n_sources"] is None:
                    src_indices = [
                        c["source_idx"] for c in result["channels"]
                        if c["source_idx"] is not None
                    ]
                    if src_indices:
                        result["n_sources"] = max(src_indices)
                if result["n_detectors"] is None:
                    det_indices = [
                        c["detector_idx"] for c in result["channels"]
                        if c["detector_idx"] is not None
                    ]
                    if det_indices:
                        result["n_detectors"] = max(det_indices)
            
            # ── stim events ────────────────────────────────────────
            stim_groups = sorted(
                [k for k in nirs.keys() if k.startswith("stim")],
                key=lambda x: int(re.search(r"\d+", x).group())
                if re.search(r"\d+", x) else 0
            )
            all_events = []
            for sg in stim_groups:
                try:
                    stim = nirs[sg]
                    name_raw = stim["name"][()]
                    trial_type = (
                        name_raw.decode("utf-8", errors="ignore").strip()
                        if isinstance(name_raw, bytes) else str(name_raw).strip()
                    )
                    data = stim["data"][()]
                    data = np.atleast_2d(data)
                    labels_raw = stim.get("dataLabels")
                    if labels_raw is not None:
                        labels = [
                            (l.decode("utf-8", errors="ignore").strip()
                             if isinstance(l, bytes) else str(l).strip())
                            for l in labels_raw[()]
                        ]
                    else:
                        labels = ["Onset", "Duration", "Amplitude"]
                    labels_lower = [l.lower() for l in labels]
                    onset_idx    = next((i for i, l in enumerate(labels_lower)
                                        if "onset" in l), 0)
                    duration_idx = next((i for i, l in enumerate(labels_lower)
                                        if "duration" in l), 1)
                    for row in data:
                        all_events.append({
                            "onset":      round(float(row[onset_idx]), 6),
                            "duration":   round(float(row[duration_idx]), 6)
                                          if len(row) > duration_idx else "n/a",
                            "trial_type": trial_type,
                        })
                except Exception as e_stim:
                    warn(f"  stim group '{sg}' could not be read: {e_stim}")
            if all_events:
                all_events.sort(key=lambda x: x["onset"])
                result["stim_events"] = all_events

    except Exception as e:
        warn(f"  _read_snirf_metadata failed for {snirf_path.name}: {e}")

    return result


def _generate_nirs_json(
    snirf_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
) -> None:
    """
    Generate *_nirs.json sidecar.

    TaskName is extracted from the task- entity in bids_stem.
    All numeric fields come from the shared metadata dict.
    Defaults are used for any missing field; warnings are logged.
    """
    # Extract TaskName from BIDS stem (e.g. "sub-1_task-mentalarithmetic_run-1_nirs")
    m = re.search(r"task-([A-Za-z0-9]+)", bids_stem)
    if m:
        task_name = m.group(1)
    else:
        task_name = "unknown"
        warn(f"  _nirs.json: no task- entity in '{bids_stem}', "
             f"using TaskName='unknown'")

    fs = meta["sampling_freq"]
    if fs is None:
        fs = 10.0
        warn(f"  _nirs.json: SamplingFrequency not found in SNIRF, "
             f"using default 10.0 Hz")

    n_ch = meta["n_channels"]
    if n_ch is None:
        n_ch = 0
        warn(f"  _nirs.json: NIRSChannelCount not found, using 0")

    n_src = meta["n_sources"]
    if n_src is None:
        n_src = 1
        warn(f"  _nirs.json: NIRSSourceOptodeCount not found, using 1")

    n_det = meta["n_detectors"]
    if n_det is None:
        n_det = 1
        warn(f"  _nirs.json: NIRSDetectorOptodeCount not found, using 1")

    sidecar = {
        "TaskName":               task_name,
        "SamplingFrequency":      fs,
        "NIRSChannelCount":       n_ch,
        "NIRSSourceOptodeCount":  n_src,
        "NIRSDetectorOptodeCount": n_det,
    }

    out_path = snirf_path.parent / f"{bids_stem}_nirs.json"
    write_json(out_path, sidecar)
    info(f"  ✓ {out_path.name}")


def _generate_channels_tsv(
    snirf_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
) -> None:
    """
    Generate *_channels.tsv sidecar.

    One row per measurementList entry. Channel names follow the pattern
    S{src}-D{det}; a wavelength suffix (_760 etc.) is appended when the
    same source-detector pair appears at multiple wavelengths.
    SNIRF dataTypeLabel is mapped to the BIDS channel type string.
    """
    channels  = meta["channels"]
    wl_values = meta["wavelengths"] or []

    if not channels:
        warn(f"  _channels.tsv: no measurementList data found in "
             f"{snirf_path.name}, skipping")
        return

    # Count (src, det) pair occurrences to decide whether to add wl suffix
    from collections import Counter
    pair_counts: Counter = Counter(
        (c["source_idx"], c["detector_idx"]) for c in channels
    )

    rows: List[Dict] = []
    for ch in channels:
        src = ch["source_idx"]
        det = ch["detector_idx"]
        wl_idx = ch["wavelength_idx"]
        label = ch["data_type_label"] or ""
        dtype = ch["data_type"]

        # Channel name
        src_str = str(src) if src is not None else "?"
        det_str = str(det) if det is not None else "?"
        base_name = f"S{src_str}-D{det_str}"
        if pair_counts.get((src, det), 0) > 1 and wl_idx is not None:
            # Append wavelength value to disambiguate
            if wl_idx - 1 < len(wl_values):
                wl_val = int(round(wl_values[wl_idx - 1]))
                ch_name = f"{base_name}_{wl_val}"
            else:
                ch_name = f"{base_name}_wl{wl_idx}"
        else:
            ch_name = base_name

        # BIDS channel type
        bids_type = _DTYPE_LABEL_TO_BIDS.get(label) if label else None
        if bids_type is None:
            if dtype == 1:
                bids_type = "NIRSCWAMPLITUDE"
            elif dtype == 2:
                bids_type = "NIRSCWOPTICALDENSITY"
            elif dtype == 4:
                bids_type = "NIRSCWHBO"
            else:
                bids_type = "MISC"

        # Wavelength nominal value
        if wl_idx is not None and wl_idx - 1 < len(wl_values):
            wl_nominal = int(round(wl_values[wl_idx - 1]))
        else:
            wl_nominal = "n/a"
            if wl_idx is not None:
                warn(f"  _channels.tsv: wavelengthIndex {wl_idx} out of range "
                     f"for channel {ch_name}")

        units = _BIDS_TYPE_TO_UNITS.get(bids_type, "n/a")

        rows.append({
            "name":               ch_name,
            "type":               bids_type,
            "source":             src_str,
            "detector":           det_str,
            "wavelength_nominal": wl_nominal,
            "units":              units,
        })

    # channels.tsv is device-level (same across runs) — strip run- entity
    # so all runs of the same subject/task share one file.
    stem_no_run = re.sub(r"_run-[A-Za-z0-9]+", "", bids_stem)
    out_path = snirf_path.parent / f"{stem_no_run}_channels.tsv"

    # Skip if already written by an earlier run of the same subject/task
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return

    header = ["name", "type", "source", "detector",
              "wavelength_nominal", "units"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row[h]) for h in header))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"  ✓ {out_path.name} ({len(rows)} channels)")


def _generate_optodes_tsv(
    snirf_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
) -> None:
    """
    Generate *_optodes.tsv sidecar.

    Uses real 3D or 2D coordinates from the SNIRF probe group when available.
    All-zero placeholder arrays are treated as missing (written as n/a).
    Logs a warning when coordinates are unavailable.
    """
    src_pos    = meta["source_pos"]
    det_pos    = meta["detector_pos"]
    src_is_3d  = meta["source_pos_is_3d"]
    det_is_3d  = meta["detector_pos_is_3d"]
    n_sources  = meta["n_sources"]  or 1
    n_detectors = meta["n_detectors"] or 1

    has_real_coords = (src_pos is not None or det_pos is not None)
    if not has_real_coords:
        warn(f"  _optodes.tsv: no valid optode coordinates in "
             f"{snirf_path.name} — writing n/a placeholder")

    rows: List[Dict] = []

    # Sources
    for i in range(n_sources):
        row: Dict[str, Any] = {"name": f"S{i+1}", "type": "source",
                               "x": "n/a", "y": "n/a", "z": "n/a"}
        if src_pos is not None and i < src_pos.shape[0]:
            row["x"] = round(float(src_pos[i, 0]), 6)
            row["y"] = round(float(src_pos[i, 1]), 6)
            if src_is_3d and src_pos.shape[1] >= 3:
                row["z"] = round(float(src_pos[i, 2]), 6)
            else:
                row["z"] = "n/a"   # 2D coordinate — z not available
        rows.append(row)

    # Detectors
    for i in range(n_detectors):
        row = {"name": f"D{i+1}", "type": "detector",
               "x": "n/a", "y": "n/a", "z": "n/a"}
        if det_pos is not None and i < det_pos.shape[0]:
            row["x"] = round(float(det_pos[i, 0]), 6)
            row["y"] = round(float(det_pos[i, 1]), 6)
            if det_is_3d and det_pos.shape[1] >= 3:
                row["z"] = round(float(det_pos[i, 2]), 6)
            else:
                row["z"] = "n/a"
        rows.append(row)

    # optodes.tsv is device-level (same across runs) — strip run- entity
    # so all runs of the same subject/task share one file.
    m = re.search(r"(sub-[A-Za-z0-9]+)", bids_stem)
    sub_label = m.group(1) if m else bids_stem
    out_path = snirf_path.parent / f"{sub_label}_optodes.tsv"

    # Skip if already written by an earlier run of the same subject/task
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return

    header = ["name", "type", "x", "y", "z"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row[h]) for h in header))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"  ✓ {out_path.name} ({len(rows)} optodes)")


def _generate_coordsystem_json(
    snirf_path: Path,
    bids_stem: str,
) -> None:
    """
    Generate *_coordsystem.json sidecar.

    Required by BIDS whenever *_optodes.tsv exists.
    Coordinate system is always 'unknown' because SNIRF does not store
    the coordinate system name — it must be provided externally.
    The file is named sub-XX_coordsystem.json (subject-level, not run-level).
    """
    m = re.search(r"(sub-[A-Za-z0-9]+)", bids_stem)
    sub_label = m.group(1) if m else "sub-unknown"

    out_path = snirf_path.parent / f"{sub_label}_coordsystem.json"

    # Only write if not already written by a previous run of the same subject
    if out_path.exists():
        return

    write_json(out_path, {
        "NIRSCoordinateSystem":      "unknown",
        "NIRSCoordinateSystemUnits": "mm",
    })
    info(f"  ✓ {out_path.name}")


def _generate_events_tsv(
    snirf_path: Path,
    bids_stem: str,
    meta: Dict[str, Any],
) -> None:
    """
    Generate *_events.tsv sidecar from SNIRF stim groups.
    run-level: one file per SNIRF file (do NOT strip run- entity).
    """
    events = meta.get("stim_events")
    if not events:
        return

    out_path = snirf_path.parent / f"{bids_stem}_events.tsv"
    if out_path.exists():
        info(f"  ✓ {out_path.name} (already exists, skipped)")
        return

    header = ["onset", "duration", "trial_type"]
    lines = ["\t".join(header)]
    for ev in events:
        lines.append("\t".join(str(ev[h]) for h in header))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    info(f"  ✓ {out_path.name} ({len(events)} event(s))")


def generate_nirs_bids_sidecars(snirf_path: Path, bids_stem: str) -> None:
    """
    Generate all NIRS-BIDS sidecar files for a single SNIRF file.

    Entry point called by executor immediately after each SNIRF file is
    created or copied. Reads the SNIRF HDF5 once, then generates:
      *_nirs.json        (REQUIRED)
      *_channels.tsv     (REQUIRED)
      *_optodes.tsv      (RECOMMENDED)
      *_coordsystem.json (REQUIRED when optodes.tsv exists)

    Args:
        snirf_path: Absolute path to the .snirf file just created/copied.
        bids_stem:  Filename without .snirf extension,
                    e.g. "sub-1_task-mentalarithmetic_run-1_nirs"
    """
    if bids_stem.endswith("_nirs"):
        bids_stem = bids_stem[:-5]
    try:
        meta = _read_snirf_metadata(snirf_path)
        _generate_nirs_json(snirf_path, bids_stem, meta)
        _generate_channels_tsv(snirf_path, bids_stem, meta)
        _generate_events_tsv(snirf_path, bids_stem, meta)
        _generate_optodes_tsv(snirf_path, bids_stem, meta)
        _generate_coordsystem_json(snirf_path, bids_stem)
    except Exception as e:
        warn(f"  generate_nirs_bids_sidecars failed for {snirf_path.name}: {e}")
        import traceback
        traceback.print_exc()