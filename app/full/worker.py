"""
worker.py  (Full pipeline)
==========================

Runs the COMPLETE autobidsify pipeline (ingest -> evidence -> trio -> plan ->
execute -> validate) in-process for the Full desktop app.

Unlike ExecVal (which vendors a trimmed library and only runs execute+validate),
Full uses the full pip-installed `autobidsify` package and invokes its CLI
entry point in-process by setting sys.argv and calling
`autobidsify.__main__.main()`. This reuses the library's own `full` command
exactly, and works after PyInstaller packaging (no subprocess / no external
Python needed).

AI configuration is supplied by the USER (bring-your-own-AI). The selected
engine's credentials are injected as environment variables for the duration of
the run and removed afterwards — API keys are never written to disk.
"""

from __future__ import annotations

import io
import os
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Callable, Dict, List, Optional


LogFn = Callable[[str], None]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _LineStream(io.TextIOBase):
    """Splits captured output into lines and forwards each (ANSI-stripped)
    line to a callback, so the library's print()/info() output streams live
    into the UI log."""

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


# Environment variables we may set per engine; cleared after the run.
_AI_ENV_KEYS = ["OPENAI_API_KEY", "DASHSCOPE_API_KEY", "OLLAMA_BASE_URL"]


def _apply_ai_env(engine: str, api_key: str, base_url: str) -> Dict[str, Optional[str]]:
    """Set the env vars for the chosen engine. Returns the previous values so
    they can be restored. Keys are never persisted to disk."""
    prev: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in _AI_ENV_KEYS}

    # Clear all first so a previous engine's leftovers don't bleed in.
    for k in _AI_ENV_KEYS:
        os.environ.pop(k, None)

    if engine == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    elif engine == "dashscope":
        if api_key:
            os.environ["DASHSCOPE_API_KEY"] = api_key
    elif engine == "ollama-remote":
        if base_url:
            os.environ["OLLAMA_BASE_URL"] = base_url
    elif engine == "ollama-local":
        # local Ollama; honor an edited URL if provided (default localhost:11434)
        if base_url:
            os.environ["OLLAMA_BASE_URL"] = base_url
    return prev


def _restore_ai_env(prev: Dict[str, Optional[str]]) -> None:
    """Restore env vars to their previous state (removing any keys we set)."""
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _build_argv(input_dir: Path, output_dir: Path, model: str,
                modality: Optional[str], nsubjects: Optional[int],
                id_strategy: str, describe: str) -> List[str]:
    """Assemble sys.argv to mimic: autobidsify full --input ... --output ..."""
    argv: List[str] = [
        "autobidsify", "full",
        "--input", str(input_dir),
        "--output", str(output_dir),
        "--model", model,
        "--id-strategy", id_strategy,
    ]
    if modality:
        argv += ["--modality", modality]
    if nsubjects is not None:
        argv += ["--nsubjects", str(nsubjects)]
    if describe:
        argv += ["--describe", describe]
    return argv


def run(input_dir: Path, output_dir: Path, *, engine: str, model: str,
        api_key: str = "", base_url: str = "",
        modality: Optional[str] = None, nsubjects: Optional[int] = None,
        id_strategy: str = "auto", describe: str = "",
        log: Optional[LogFn] = None) -> int:
    """
    Run the full pipeline. Returns 0 on success, non-zero on error.

    AI config (engine/model/api_key/base_url) is provided by the user.
    Credentials are injected as env vars only for this run and removed after.
    """
    if log is None:
        log = print

    sink = _LineStream(log)

    # Import the library's CLI entry point (full pip-installed package).
    try:
        from autobidsify.__main__ import main as autobidsify_main
    except Exception as e:  # noqa: BLE001
        log(f"[FATAL] Could not import autobidsify: {e}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    argv = _build_argv(input_dir, output_dir, model, modality,
                       nsubjects, id_strategy, describe)

    log("[WORKER] === Running full pipeline ===")
    log(f"[WORKER] engine   : {engine}")
    log(f"[WORKER] model    : {model}")
    log(f"[WORKER] input    : {input_dir}")
    log(f"[WORKER] output   : {output_dir}")
    log(f"[WORKER] modality : {modality or '(auto-detect)'}")
    log("")

    prev_env = _apply_ai_env(engine, api_key, base_url)
    saved_argv = sys.argv
    try:
        sys.argv = argv
        with redirect_stdout(sink), redirect_stderr(sink):
            autobidsify_main()
        sink.flush()
    except SystemExit as se:
        # __main__.main() calls sys.exit(1) on fatal errors.
        sink.flush()
        code = se.code if isinstance(se.code, int) else 1
        if code == 0:
            log("[WORKER] Done.")
            return 0
        log("[FATAL] Pipeline aborted (see messages above).")
        return code or 1
    except Exception:  # noqa: BLE001
        sink.flush()
        log("[FATAL] Pipeline raised an exception:")
        log(traceback.format_exc())
        return 1
    finally:
        sys.argv = saved_argv
        _restore_ai_env(prev_env)  # remove API keys from the environment

    log("")
    log(f"[WORKER] Pipeline complete. BIDS dataset: {output_dir / 'bids_compatible'}")
    log("[WORKER] Done.")
    return 0