# mri_convert.py
# MRI format converters with batch DICOM conversion support

from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import shutil
import tempfile
import numpy as np
import nibabel as nib
from scipy.io import loadmat
from autobidsify.utils import ensure_dir, warn, info, write_json

def check_dcm2niix_available() -> bool:
    """
    Check if dcm2niix binary is available.

    Detection order:
    1. pip-installed dcm2niix  → binary lands in venv/bin/, found by shutil.which()
    2. System-level dcm2niix   → apt/brew install, also found by shutil.which()

    Both routes resolve through the same shutil.which() call because the pip
    package places the compiled binary on PATH automatically upon installation.
    No 'import dcm2niix' is needed — the package exposes no Python API.
    """
    return shutil.which("dcm2niix") is not None

def run_dcm2niix_batch(dicom_files: List[Path], output_path: Path, 
                       temp_dir: Optional[Path] = None,
                       quiet: bool = False) -> Optional[Path]:  # ADDED quiet parameter
    """
    Convert a batch of DICOM files to a single NIfTI volume.
    
    Args:
        dicom_files: List of DICOM files to convert (should be from same series)
        output_path: Desired output NIfTI path
        temp_dir: Temporary directory for conversion (will create if None)
        quiet: If True, suppress verbose output (NEW)
    
    Returns:
        Path to created NIfTI file, or None if conversion failed
    """
    if not check_dcm2niix_available():
        if not quiet:
            warn("dcm2niix not found. Install via one of:")
            warn("  Option 1 (recommended): pip install dcm2niix")
            warn("  Option 2 (system):      apt-get install dcm2niix  # Ubuntu/Debian")
            warn("  Option 3 (system):      brew install dcm2niix     # macOS")
        return None
    
    if not dicom_files:
        if not quiet:
            warn("No DICOM files provided for conversion")
        return None
    
    # Create temp directory if not provided
    cleanup_temp = False
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
        cleanup_temp = True
    
    ensure_dir(temp_dir)
    
    try:
        # Copy DICOM files to temp directory
        if not quiet:
            info(f"  Converting {len(dicom_files)} DICOM files...")
        
        for dcm_file in dicom_files:
            dst = temp_dir / dcm_file.name
            shutil.copy2(dcm_file, dst)
        
        # Run dcm2niix on temp directory
        result = subprocess.run(
            [
                "dcm2niix",
                "-o", str(temp_dir),
                "-f", "temp_output",
                "-z", "y",
                "-b", "y",
                str(temp_dir)
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            if not quiet:
                warn(f"  dcm2niix failed: {result.stderr}")
            return None
        
        # Find generated NIfTI file
        nifti_files = list(temp_dir.glob("temp_output*.nii.gz"))
        
        if not nifti_files:
            if not quiet:
                warn(f"  dcm2niix did not generate NIfTI file")
            return None
        
        # Move to final location
        ensure_dir(output_path.parent)
        shutil.move(str(nifti_files[0]), str(output_path))
        
        # Move JSON sidecar if exists
        json_files = list(temp_dir.glob("temp_output*.json"))
        if json_files:
            json_output = output_path.parent / (output_path.stem.replace('.nii', '') + '.json')
            shutil.move(str(json_files[0]), str(json_output))
            if not quiet:
                info(f"  ✓ Created sidecar: {json_output.name}")
        
        if not quiet:
            info(f"  ✓ Created: {output_path.name}")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        if not quiet:
            warn(f"  dcm2niix timed out")
        return None
    except Exception as e:
        if not quiet:
            warn(f"  dcm2niix error: {e}")
        return None
    finally:
        # Cleanup temp directory if we created it
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

def run_dcm2niix(dicom_dir: Path, output_dir: Path) -> List[Path]:
    """
    Run dcm2niix tool on DICOM directory.
    
    Args:
        dicom_dir: Directory containing DICOM files
        output_dir: Output directory for NIfTI files
    
    Returns:
        List of generated NIfTI file paths
    """
    dcm2niix_path = shutil.which("dcm2niix")
    if not dcm2niix_path:
        warn("dcm2niix not found. Skipping DICOM conversion.")
        warn("Install via one of:")
        warn("  Option 1 (recommended): pip install dcm2niix")
        warn("  Option 2 (system):      apt-get install dcm2niix  # Ubuntu/Debian")
        warn("  Option 3 (system):      brew install dcm2niix     # macOS")
        return []
    
    ensure_dir(output_dir)
    
    info(f"Running dcm2niix on {dicom_dir}")
    
    try:
        # Run dcm2niix
        result = subprocess.run(
            [
                dcm2niix_path,
                "-o", str(output_dir),
                "-f", "%p_%s",  # protocol_seriesNum
                "-z", "y",      # gzip compression
                str(dicom_dir)
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            warn(f"dcm2niix failed: {result.stderr}")
            return []
        
        # Find generated NIfTI files
        nifti_files = list(output_dir.glob("*.nii.gz"))
        info(f"✓ dcm2niix generated {len(nifti_files)} NIfTI files")
        
        return nifti_files
        
    except subprocess.TimeoutExpired:
        warn(f"dcm2niix timed out for {dicom_dir}")
        return []
    except Exception as e:
        warn(f"dcm2niix error: {e}")
        return []

def arrays_to_nifti(
    final_plan: Dict[str, Any],
    input_root: Path,
    output_path: Path
) -> Path:
    """
    Convert MATLAB arrays to NIfTI based on final mapping plan.
    
    Args:
        final_plan: Final mapping plan from LLM
        input_root: Root directory of input data
        output_path: Output NIfTI file path
    
    Returns:
        Path to created NIfTI file
    """
    ensure_dir(output_path.parent)
    
    info(f"Converting MATLAB arrays to NIfTI: {output_path.name}")
    
    # Get conversions
    conversions = final_plan.get("conversions", [])
    if not conversions:
        warn("No conversions specified in final_plan")
        return output_path
    
    conversion = conversions[0]
    
    # Parse source
    source_file = conversion.get("source_file", "")
    source_var = conversion.get("source_variable", "")
    operations = conversion.get("operations", [])
    sidecar = conversion.get("sidecar", {})
    
    # Load MATLAB file
    mat_path = input_root / source_file
    
    if not mat_path.exists():
        warn(f"MAT file not found: {mat_path}")
        return output_path
    
    info(f"  Loading: {mat_path}")
    info(f"  Variable: {source_var}")
    
    try:
        mat_data = loadmat(str(mat_path))
        
        if source_var not in mat_data:
            available = [k for k in mat_data.keys() if not k.startswith('__')]
            warn(f"Variable '{source_var}' not found. Available: {available}")
            return output_path
        
        img_data = mat_data[source_var]
        info(f"  Loaded shape: {img_data.shape}, dtype: {img_data.dtype}")
        
        # Default affine
        affine = np.eye(4)
        
        # Apply operations
        for op in operations:
            op_type = op.get("type")
            
            if op_type == "transpose":
                axes = op.get("axes", [0, 1, 2])
                info(f"  Transposing: {axes}")
                img_data = np.transpose(img_data, axes)
            
            elif op_type == "flip":
                flip_axes = op.get("axes", [])
                for axis in flip_axes:
                    info(f"  Flipping axis: {axis}")
                    img_data = np.flip(img_data, axis=axis)
            
            elif op_type == "build_affine":
                voxel_size = op.get("voxel_size_mm", [1.0, 1.0, 1.0])
                coord_sys = op.get("coordinate_system", "RAS")
                
                info(f"  Building affine: voxel={voxel_size}, coord={coord_sys}")
                
                # Build affine matrix
                affine = np.eye(4)
                affine[0, 0] = voxel_size[0]
                affine[1, 1] = voxel_size[1]
                affine[2, 2] = voxel_size[2]
                
                # Handle coordinate system
                if coord_sys == "LPS":
                    affine[0, 0] = -affine[0, 0]
                    affine[1, 1] = -affine[1, 1]
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(img_data, affine)
        
        # Set TR if available
        tr = sidecar.get("RepetitionTime")
        if tr and len(img_data.shape) > 3:
            zooms = list(nifti_img.header.get_zooms()[:3]) + [tr]
            nifti_img.header.set_zooms(zooms)
            info(f"  Set TR: {tr}s")
        
        # Save NIfTI
        nib.save(nifti_img, str(output_path))
        info(f"  ✓ Created NIfTI: {output_path}")
        
        # Create sidecar JSON
        sidecar_path = output_path.parent / (output_path.stem.replace('.nii', '') + '.json')
        write_json(sidecar_path, sidecar)
        info(f"  ✓ Created sidecar: {sidecar_path.name}")
        
        return output_path
        
    except Exception as e:
        warn(f"Failed to convert arrays to NIfTI: {e}")
        import traceback
        traceback.print_exc()
        return output_path

def validate_nifti(nifti_path: Path) -> bool:
    """
    Validate NIfTI file integrity.
    
    Args:
        nifti_path: Path to NIfTI file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        img = nib.load(str(nifti_path))
        shape = img.shape
        dtype = img.get_data_dtype()
        
        info(f"NIfTI valid: {nifti_path.name} - shape {shape}, dtype {dtype}")
        return True
        
    except Exception as e:
        warn(f"NIfTI validation failed for {nifti_path}: {e}")
        return False
    

def generate_mri_bids_sidecar(nifti_path: Path, bids_filename: str,
                              subdirectory: str) -> None:
    """
    Write the BIDS-required JSON sidecar for an MRI file when its datatype needs one.

    func (bold): TaskName and RepetitionTime are REQUIRED by the validator.
      - TaskName comes from the task- entity in the filename.
      - RepetitionTime is read from the NIfTI header's 4th voxel dimension (TR
        in seconds). If the header has no TR, we warn honestly rather than
        fabricate a value.
    anat / dwi: no validator-required field, so no sidecar is forced.
    """
    import re
    from autobidsify.bids_grammar import bids_stem, required_sidecar_fields
    stem     = bids_stem(bids_filename)
    suffix   = stem.split("_")[-1]
    datatype = subdirectory.split("/")[-1]        # strip any ses-XX/ prefix

    required = required_sidecar_fields(datatype, {})
    if not required:                              # anat / dwi
        return

    sidecar: Dict[str, Any] = {}
    m = re.search(r"task-([A-Za-z0-9]+)", bids_filename)
    if "TaskName" in required:
        sidecar["TaskName"] = m.group(1) if m else "rest"

    if "RepetitionTime" in required:
        try:
            img   = nib.load(str(nifti_path))
            zooms = img.header.get_zooms()
            if len(zooms) >= 4 and float(zooms[3]) > 0:
                sidecar["RepetitionTime"] = round(float(zooms[3]), 6)
        except Exception:
            pass
        if "RepetitionTime" not in sidecar:
            warn(f"  ⚠ {nifti_path.name}: no RepetitionTime in NIfTI header; "
                 f"validator will flag it. Add TR to the source data or --describe.")

    json_path = nifti_path.parent / (stem + ".json")
    write_json(json_path, sidecar)
    info(f"  ✓ MRI sidecar: {json_path.name} ({', '.join(sidecar.keys())})")
