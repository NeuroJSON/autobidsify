# ingest.py
# Data ingestion: extract archives or reference directories
# Optimized: No copying for directory inputs

"""
Data Ingestion Module Workflow:
1. Detect Input Type
2. Compress File → Decompress to Staging
3. Directory → Create only ingest_info.json, recording the original path
4. The Evidence step reads the actual data path from ingest_info.
"""

import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from autobidsify.utils import ensure_dir, info, fatal, warn, write_json

def ingest_data(input_path: str, output_dir: Path,
                anonymize: bool = False, deface: bool = False) -> None:
    """
    High-level ingest function.
    
    Optimized behavior:
    - Archive files: Extract to staging directory
    - Directories: Record path only (no copying)
    
    Args:
        input_path: Path to input data (string from CLI)
        output_dir: Output BIDS directory
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not input_path.exists():
        fatal(f"Input path not found: {input_path}")
        return
    
    ensure_dir(output_dir / "_staging")
    
    # Determine input type
    if input_path.is_file():
        # Archive file: extract to staging
        info(f"Detected archive file: {input_path.name}")
        staging_dir = _extract_archive(input_path, output_dir)
        input_type = "archive"
        actual_data_path = staging_dir
        
    elif input_path.is_dir():
        # Directory: no copying, just reference
        info(f"Detected directory: {input_path}")
        info(f"Optimization: No copying performed (using original location)")
        
        staging_dir = None
        input_type = "directory"
        actual_data_path = input_path
        
    else:
        fatal(f"Input path is neither file nor directory: {input_path}")
        return
    
    # Create ingest metadata
    ingest_info = {
        "step": "ingest",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "input_type": input_type,
        "output_dir": str(output_dir),
        "staging_dir": str(staging_dir) if staging_dir else None,
        "actual_data_path": str(actual_data_path),
        "anonymize": anonymize,
        "deface": deface,
        "status": "complete"
    }
    
    # Save ingest info
    ingest_info_path = output_dir / "_staging" / "ingest_info.json"
    write_json(ingest_info_path, ingest_info)

    # Write session file so subsequent steps can find output_dir automatically
    session_path = Path.cwd() / ".autobidsify_session"
    try:
        session_path.write_text(str(output_dir), encoding="utf-8")
    except Exception as e:
        warn(f"Could not write session file {session_path}: {e}")

    info(f"✓ Data ingestion complete")
    info(f"  Input type: {input_type}")
    info(f"  Data location: {actual_data_path}")
    info(f"  Ingest info saved: {ingest_info_path}")
    if anonymize:
        info(f"  Anonymization: enabled (metadata de-identification active)")
    if deface:
        info(f"  Defacing: enabled (MRI anat NIfTI files will be defaced at execute stage)")

def _extract_archive(archive_path: Path, output_dir: Path) -> Path:
    """
    Extract archive to staging directory.
    
    Args:
        archive_path: Path to archive file
        output_dir: Output directory
    
    Returns:
        Path to staging directory
    """
    staging_dir = output_dir / "_staging" / "extracted"
    ensure_dir(staging_dir)
    
    suffix = archive_path.suffix.lower()
    
    if suffix == '.zip':
        info(f"Extracting ZIP archive...")
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(staging_dir)
            info(f"✓ Extracted {len(zip_ref.namelist())} files")
        except zipfile.BadZipFile:
            fatal(f"Invalid ZIP file: {archive_path}")
        except Exception as e:
            fatal(f"Failed to extract ZIP: {e}")
    
    elif suffix in {'.tar', '.gz', '.tgz', '.bz2'}:
        info(f"Extracting TAR archive...")
        try:
            if suffix == '.tar':
                mode = 'r:'
            elif suffix in {'.gz', '.tgz'}:
                mode = 'r:gz'
            elif suffix == '.bz2':
                mode = 'r:bz2'
            else:
                mode = 'r:*'
            
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(staging_dir)
            info(f"✓ Extracted to {staging_dir}")
        except tarfile.TarError:
            fatal(f"Invalid TAR file: {archive_path}")
        except Exception as e:
            fatal(f"Failed to extract TAR: {e}")
    
    else:
        fatal(f"Unsupported archive format: {suffix}")
    
    return staging_dir
