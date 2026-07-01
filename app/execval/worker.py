"""
worker.py
=========

Runs the local conversion (execute + optional Tier-1 validate) for the
ExecVal desktop app.

Unlike the old Qt version, this worker runs *in-process* on a background
thread, so it cannot rely on a subprocess's stdout being captured. Instead it
accepts a `log` callback and forwards every line to it. Because the underlying
autobidsify library prints via plain print()/sys.stdout, this module
temporarily redirects stdout/stderr while the library runs and pipes each
captured line into the same callback. That makes the library's info()/warn()
output appear live in the UI log.

This module imports autobidsify directly from the same repository and never
modifies it.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Callable, Dict, Optional


LogFn = Callable[[str], None]

# Matches ANSI color / control escape sequences (e.g. "\033[92m", "\033[0m").
# The autobidsify library colorizes terminal output with these; they render as
# garbage in the HTML log view, so we strip them from captured lines.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

# bundle filename -> where execute expects it
_STAGING_FILES = {
    "BIDSPlan.yaml", "mat_mapping.json",
    "headers_normalized.json", "voxel_final_plan.json",
}
_TRIO_FILES = {"dataset_description.json", "README.md", "participants.tsv"}


class _LineStream(io.TextIOBase):
    """A writable stream that splits incoming text into lines and forwards
    each completed line to a callback. Used to capture library print output."""

    def __init__(self, emit: LogFn) -> None:
        super().__init__()
        self._emit = emit
        self._buffer = ""

    def write(self, s: str) -> int:  # type: ignore[override]
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(_strip_ansi(line))
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer:
            self._emit(_strip_ansi(self._buffer))
            self._buffer = ""


def _stage_bundle(found: Dict[str, str], output_dir: Path, log: LogFn) -> None:
    """Copy each resolved bundle file to the location execute expects."""
    staging = output_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, src_str in found.items():
        src = Path(src_str)
        if not src.exists():
            continue
        if name in _STAGING_FILES:
            dst = staging / name
        elif name in _TRIO_FILES:
            dst = output_dir / name
        else:
            continue
        shutil.copy2(src, dst)
        log(f"[WORKER]   staged {name}")


def _build_aux_inputs(staging: Path) -> Dict[str, Any]:
    """Assemble aux_inputs from optional staged JSON files."""
    aux: Dict[str, Any] = {}
    norm = staging / "headers_normalized.json"
    if norm.exists():
        aux["normalized_headers"] = json.loads(norm.read_text(encoding="utf-8"))
    voxel = staging / "voxel_final_plan.json"
    if voxel.exists():
        aux["final_mapping_plan"] = json.loads(voxel.read_text(encoding="utf-8"))
    return aux


def run(found: Dict[str, str], input_root: Path, output_dir: Path,
        do_validate: bool, log: Optional[LogFn] = None) -> int:
    """
    Execute the full local conversion. Returns 0 on success, non-zero on error.

    Parameters
    ----------
    found : dict   canonical bundle filename -> path
    input_root : Path   user's extracted dataset root
    output_dir : Path   output directory
    do_validate : bool  run Tier-1 validation after execute
    log : callable      receives each log line (defaults to print)
    """
    if log is None:
        log = print

    # Library print output is redirected into our callback while it runs.
    sink = _LineStream(log)

    try:
        import yaml
        from autobidsify.converters.executor import execute_bids_plan
    except Exception as e:  # noqa: BLE001
        log(f"[FATAL] Could not import autobidsify: {e}")
        return 2

    staging = output_dir / "_staging"

    log("[WORKER] === Staging plan bundle ===")
    _stage_bundle(found, output_dir, log)

    plan_path = staging / "BIDSPlan.yaml"
    if not plan_path.exists():
        log("[FATAL] BIDSPlan.yaml not found after staging.")
        return 2

    try:
        plan_dict = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        log(f"[FATAL] BIDSPlan.yaml is not valid YAML: {e}")
        return 2

    aux_inputs = _build_aux_inputs(staging)

    log("")
    log("[WORKER] === Executing BIDS plan ===")
    log(f"[WORKER] input_root : {input_root}")
    log(f"[WORKER] output_dir : {output_dir}")
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            execute_bids_plan(input_root, output_dir, plan_dict, aux_inputs)
        sink.flush()
    except SystemExit:
        # autobidsify's fatal() calls sys.exit(1); treat as failure, not crash.
        sink.flush()
        log("[FATAL] Execution aborted by library (fatal error above).")
        return 1
    except Exception:  # noqa: BLE001
        sink.flush()
        log("[FATAL] Execution raised an exception:")
        log(traceback.format_exc())
        return 1

    bids_root = output_dir / "bids_compatible"
    log(f"[WORKER] Execution complete. BIDS dataset: {bids_root}")

    if do_validate:
        log("")
        log("[WORKER] === Validating BIDS dataset (Tier 1) ===")
        try:
            from autobidsify.converters.validators import validate_bids_compatible
            with redirect_stdout(sink), redirect_stderr(sink):
                validate_bids_compatible(output_dir)
            sink.flush()
        except SystemExit:
            sink.flush()
            log("[WARN] Validation aborted by library.")
        except Exception:  # noqa: BLE001
            sink.flush()
            log("[WARN] Validation raised an exception:")
            log(traceback.format_exc())

    log("")
    log("[WORKER] Done.")
    return 0
