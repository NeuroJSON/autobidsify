# llm.py
# Unified LLM caller supporting OpenAI and Qwen (Ollama / REST API / DashScope).

import os
import json
from typing import Any, Optional
from autobidsify.utils import warn, fatal, info


# ============================================================================
# Exception
# ============================================================================

class LLMHardFail(Exception):
    def __init__(self, step: str, error_type: str, message: str):
        self.step       = step
        self.error_type = error_type
        self.message    = message
        super().__init__(f"[{step}] {error_type}: {message}")


# ============================================================================
# Provider detection
# ============================================================================

def is_qwen_model(model: str) -> bool:
    """Return True for any Qwen model (qwen*)."""
    return model.lower().startswith('qwen')


def is_openai_model(model: str) -> bool:
    """Return True for OpenAI models (gpt*, o1*, o3*)."""
    return model.lower().startswith(('gpt', 'o1', 'o3'))


def is_reasoning_model(model: str) -> bool:
    """Return True for reasoning-style models that skip temperature."""
    m = model.lower()
    return m.startswith("o1") or m.startswith("o3") or m.startswith("gpt-5")


# ============================================================================
# OpenAI
# ============================================================================

def _get_openai_client():
    """Build and return an OpenAI client.  Calls fatal() on missing key."""
    try:
        from openai import OpenAI
    except ImportError:
        fatal("openai library not installed.  Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        fatal(
            "OPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "If you want to use Qwen instead, pass --model qwen3-coder-next:latest "
            "(or any qwen* model name)."
        )

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise LLMHardFail("Initialization", "ClientError", str(e))


def _call_openai(model: str, system_prompt: str, user_payload: str,
                 step: str, temperature: Optional[float] = None) -> str:
    """Call the OpenAI chat-completions endpoint."""
    client = _get_openai_client()

    try:
        from openai import OpenAIError
    except ImportError:
        fatal("openai library not installed")

    try:
        params: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_payload},
            ],
        }
        if is_reasoning_model(model):
            params["max_completion_tokens"] = 32000
        else:
            params["max_completion_tokens"] = 16000
            if temperature is not None:
                params["temperature"] = temperature

        response = client.chat.completions.create(**params)

        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                if content and content.strip():
                    return content.strip()
                raise LLMHardFail(step, "EmptyResponse", "OpenAI returned empty content")
            raise LLMHardFail(step, "InvalidResponse", "Response has no message.content")
        raise LLMHardFail(step, "InvalidResponse", "Response has no choices")

    except OpenAIError as e:
        raise LLMHardFail(step, e.__class__.__name__, str(e))
    except LLMHardFail:
        raise
    except Exception as e:
        raise LLMHardFail(step, "UnexpectedError", str(e))


# ============================================================================
# Qwen — local Ollama package
# ============================================================================

def _call_qwen_ollama(model: str, system_prompt: str, user_payload: str,
                      step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via the local Ollama Python package.

    Requires:
      pip install ollama
      ollama serve
      ollama pull <model>
    """
    try:
        import ollama
    except ImportError:
        raise LLMHardFail(step, "OllamaNotInstalled",
                          "ollama library not installed.  Run: pip install ollama")

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_payload},
        ]
        options: dict = {"top_p": 0.8}
        if temperature is not None:
            options["temperature"] = temperature

        response = ollama.chat(model=model, messages=messages, options=options)

        # ollama >= 0.6: object-style response
        if hasattr(response, "message") and hasattr(response.message, "content"):
            content = response.message.content
            if content and content.strip():
                return content.strip()
            raise LLMHardFail(step, "EmptyResponse", "Qwen (Ollama) returned empty content")

        # ollama < 0.6: dict-style response
        if isinstance(response, dict) and "message" in response:
            content = response["message"].get("content", "")
            if content and content.strip():
                return content.strip()
            raise LLMHardFail(step, "EmptyResponse", "Qwen (Ollama) returned empty content")

        raise LLMHardFail(
            step, "InvalidResponse",
            f"Unexpected Ollama response type: {type(response)}"
        )

    except ImportError:
        raise
    except LLMHardFail:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg:
            raise LLMHardFail(step, "OllamaNotRunning",
                              "Cannot connect to Ollama.  Run: ollama serve")
        if "not found" in msg or "pull" in msg:
            raise LLMHardFail(step, "ModelNotFound",
                              f"Model '{model}' not found.  Run: ollama pull {model}")
        raise LLMHardFail(step, "QwenError", str(e))


# ============================================================================
# Qwen — remote Ollama REST API
# ============================================================================

def _call_qwen_rest_api(model: str, system_prompt: str, user_payload: str,
                        step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via a remote Ollama REST API endpoint.

    No local Ollama installation required.  Only needs the requests library.

    Setup:
      export OLLAMA_BASE_URL=http://your-server.com:11434
    """
    try:
        import requests
    except ImportError:
        raise LLMHardFail(step, "RequestsNotInstalled",
                          "requests library not installed.  Run: pip install requests")

    base_url = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
    if not base_url:
        raise LLMHardFail(step, "MissingOllamaBaseURL",
                          "OLLAMA_BASE_URL is not set.")

    payload: dict = {
        "model":   model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_payload},
        ],
        "stream":  False,
        "options": {"top_p": 0.8},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        if content and content.strip():
            return content.strip()
        raise LLMHardFail(step, "EmptyResponse",
                          "Ollama REST API returned empty content")
    except requests.exceptions.ConnectionError:
        raise LLMHardFail(step, "OllamaRESTUnreachable",
                          f"Cannot reach Ollama REST API at {base_url}.")
    except requests.exceptions.Timeout:
        raise LLMHardFail(step, "OllamaRESTTimeout",
                          f"Ollama REST API timed out at {base_url}.")
    except requests.exceptions.HTTPError as e:
        raise LLMHardFail(step, "OllamaRESTHTTPError", str(e))
    except LLMHardFail:
        raise
    except Exception as e:
        raise LLMHardFail(step, "OllamaRESTError", str(e))


# ============================================================================
# Qwen — DashScope cloud API
# ============================================================================

def _call_qwen_api(model: str, system_prompt: str, user_payload: str,
                   step: str, temperature: Optional[float] = None) -> str:
    """
    Call Qwen via the Alibaba DashScope cloud API.

    Setup:
      pip install dashscope
      export DASHSCOPE_API_KEY='sk-...'
    """
    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        raise LLMHardFail(step, "DashScopeNotInstalled",
                          "dashscope not installed.  Run: pip install dashscope")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise LLMHardFail(step, "MissingAPIKey",
                          "DASHSCOPE_API_KEY is not set.  "
                          "Get one at: https://dashscope.console.aliyun.com/")

    dashscope.api_key = api_key

    model_mapping = {
        "qwen-max":     "qwen-max",
        "qwen-plus":    "qwen-plus",
        "qwen-turbo":   "qwen-turbo",
        "qwen2.5-coder": "qwen-coder-plus",
    }
    ds_model = model_mapping.get(model, model)

    try:
        response = Generation.call(
            model=ds_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_payload},
            ],
            result_format="message",
            temperature=temperature if temperature is not None else 0.85,
            top_p=0.8,
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            if content and content.strip():
                return content.strip()
            raise LLMHardFail(step, "EmptyResponse",
                              "DashScope returned empty content")
        raise LLMHardFail(step, "APIError",
                          f"DashScope error {response.code}: {response.message}")
    except LLMHardFail:
        raise
    except Exception as e:
        raise LLMHardFail(step, "QwenAPIError", str(e))


# ============================================================================
# Qwen dispatcher
# ============================================================================

def _call_qwen(model: str, system_prompt: str, user_payload: str,
               step: str, temperature: Optional[float] = None) -> str:
    """
    Route a Qwen call to the appropriate backend.

    Priority:
      1. OLLAMA_BASE_URL set          → remote Ollama REST API
      2. ollama Python package usable → local Ollama process
      3. DASHSCOPE_API_KEY set        → DashScope cloud API
      4. Nothing available            → fatal with clear instructions

    IMPORTANT: if none of the above conditions are met, the function calls
    fatal() with a Qwen-specific message.  This function is ONLY reached
    when is_qwen_model() returns True — it is never called for OpenAI models.
    """
    # Priority 1 ── Remote Ollama REST API
    if os.getenv("OLLAMA_BASE_URL"):
        info(f"Using remote Ollama REST API: {os.getenv('OLLAMA_BASE_URL')}")
        return _call_qwen_rest_api(model, system_prompt, user_payload, step, temperature)

    # Priority 2 ── Local Ollama Python package
    try:
        return _call_qwen_ollama(model, system_prompt, user_payload, step, temperature)
    except LLMHardFail as e:
        if e.error_type not in ("OllamaNotInstalled", "OllamaNotRunning", "ModelNotFound"):
            raise   # unexpected error — re-raise as-is

        # Priority 3 ── DashScope cloud API
        if os.getenv("DASHSCOPE_API_KEY"):
            warn(f"Ollama not available ({e.error_type}), falling back to DashScope API...")
            return _call_qwen_api(model, system_prompt, user_payload, step, temperature)

        # Priority 4 ── Nothing available
        fatal(
            f"Cannot call Qwen model '{model}' — no backend is available.\n"
            "\n"
            "Choose one of the following options:\n"
            "\n"
            "  Option 1 — Remote Ollama REST API (no local install required)\n"
            "    export OLLAMA_BASE_URL=http://your-server.com:11434\n"
            "\n"
            "  Option 2 — Local Ollama\n"
            "    ollama serve\n"
            f"    ollama pull {model}\n"
            "\n"
            "  Option 3 — DashScope cloud API\n"
            "    export DASHSCOPE_API_KEY='sk-...'\n"
            "    (get a key at https://dashscope.console.aliyun.com/)\n"
            "\n"
            "If you intended to use OpenAI instead, pass --model gpt-4o\n"
            "and make sure OPENAI_API_KEY is set."
        )


# ============================================================================
# Temperature inference for Qwen
# ============================================================================

def _infer_qwen_temperature(model: str,
                            base_temperature: Optional[float]) -> Optional[float]:
    """
    Adjust temperature for Qwen based on model-name keywords.

    Rules:
      think / careful / compare / reason  → cap at 0.15  (reasoning model)
      next  / fast    / turbo   / lite    → floor at 0.4  (speed model)
      everything else                     → floor at 0.3  (avoid too-low temps)
    """
    if base_temperature is None:
        return None

    m = model.lower()

    if any(kw in m for kw in ("think", "careful", "compare", "reason")):
        return min(base_temperature, 0.15)

    if any(kw in m for kw in ("next", "fast", "turbo", "lite")):
        return max(base_temperature, 0.4)

    return max(base_temperature, 0.3)


# ============================================================================
# Unified entry point
# ============================================================================

def _call_llm(model: str, system_prompt: str, user_payload: str,
              step: str, temperature: Optional[float] = None) -> str:
    """
    Route an LLM call to the correct provider based on model name.

    Routing:
      qwen*        → _call_qwen()    (Ollama / REST API / DashScope)
      gpt* o1* o3* → _call_openai() (OpenAI API)
      anything else → LLMHardFail(UnknownProvider)

    The two providers are completely independent — using a Qwen model
    never touches OPENAI_API_KEY, and vice versa.
    """
    if is_qwen_model(model):
        temp = _infer_qwen_temperature(model, temperature)
        return _call_qwen(model, system_prompt, user_payload, step, temp)

    if is_openai_model(model):
        return _call_openai(model, system_prompt, user_payload, step, temperature)

    raise LLMHardFail(
        step, "UnknownProvider",
        f"Unrecognised model name: '{model}'.\n"
        "  OpenAI models start with: gpt, o1, o3\n"
        "  Qwen models start with:   qwen\n"
        "Examples: --model gpt-4o   or   --model qwen3-coder-next:latest"
    )


# ============================================================================
# Prompts
# ============================================================================

PROMPT_TRIO_DATASET_DESC = """You are a BIDS dataset_description.json metadata extractor.

═══════════════════════════════════════════════════════
YOUR JOB
═══════════════════════════════════════════════════════

Extract dataset metadata from the input. Return ONLY valid JSON, no markdown.

═══════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════

1. LICENSE — output as "raw_license" (plain string, NOT normalized):
   - Copy exactly what the user wrote, e.g. "CC0", "CC BY 4.0",
     "Creative Commons Zero", "public domain", "MIT license"
   - Do NOT try to normalize or format it — Python will do that
   - If the user wrote "License: CC0" → raw_license: "CC0"
   - If the document says "released under Creative Commons" → raw_license: "Creative Commons"
   - If no license mentioned anywhere → omit raw_license

2. AUTHORS — extract from ALL available sources:
   - Search in order: user_hints.user_text → documents[]
   - Look for: explicit author lists, citation patterns, "Created by",
     "Principal Investigator", "Contact", "Contributors" sections
   - If full names are available, use them: ["Last FM", "Last FM"]
   - If only "et al." citation exists, keep first author + et al.: ["Shafto MA et al."]
   - Do NOT infer, guess, or use outside knowledge to expand author lists
   - Do NOT fabricate names not present in any input source
   - If no author information found anywhere, omit Authors field entirely

   EXAMPLES (follow exactly):

   Input: "Smith et al. (2023). A neuroimaging study..."
   Output: "Authors": ["Smith et al."]

   Input: "Created by John Doe, Jane Smith and Bob Lee"
   Output: "Authors": ["John Doe", "Jane Smith", "Bob Lee"]

   Input: "Data collected by the CamCAN team. Contact: info@cam.ac.uk"
   Output: (omit Authors field)

   Input: "Shafto et al. (2014). The Cambridge Centre for Ageing..."
   Output: "Authors": ["Shafto et al."]

3. NAME — infer from context:
   - Look for explicit dataset name in user_hints.user_text
   - If not found, infer from the scientific context
   - Keep it short and descriptive

4. MISSING FIELDS — omit rather than guess:
   - If you cannot determine a field with reasonable confidence, omit it
   - Never invent information not present in the input

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════

{
  "dataset_description": {
    "Name": "...",
    "BIDSVersion": "1.10.0",
    "DatasetType": "raw",
    "Authors": ["First Last", "First Last"]
  },
  "raw_license": "CC0",
  "extraction_log": {
    "Name": "inferred from user_text: '...'",
    "raw_license": "found in user_text: 'License: CC0'",
    "Authors": "extracted from citation in user_text"
  },
  "questions": []
}

Notes:
- raw_license goes at the TOP LEVEL (not inside dataset_description)
- dataset_description should NOT contain a "License" field — Python adds it after normalization
- BIDSVersion must always be "1.10.0"
- DatasetType must always be "raw"
- Output ONLY valid JSON, no extra text, no markdown fences

FIELD SOURCE RULES (STRICT - violations cause data integrity failure):
┌─────────────────┬────────────────────────────────────────────────────┐
│ Field           │ Allowed sources                                    │
├─────────────────┼────────────────────────────────────────────────────┤
│ Authors         │ user_hints.user_text or documents[] ONLY           │
│                 │ NEVER use training knowledge to expand et al.      │
│ raw_license     │ user_hints.user_text or documents[] ONLY           │
│ Name            │ may infer from context if not explicit             │
│ BIDSVersion     │ always "1.10.0" (fixed)                            │
│ DatasetType     │ always "raw" (fixed)                               │
└─────────────────┴────────────────────────────────────────────────────┘
"""

PROMPT_TRIO_README = """Generate README.md for BIDS dataset.

CRITICAL: Use user_hints.user_text as primary source for README content.

Create comprehensive README with sections:
- Overview
- Dataset Description
- Data Acquisition
- File Organization
- Usage Notes
- References

Output: Direct Markdown text (no JSON wrapper)"""

PROMPT_TRIO_PARTICIPANTS = """You are a BIDS participants.tsv generator.

CRITICAL: Extract participant metadata from user_hints.user_text!

Examples:
- "1 male, 1 female" → sex column: M, F
- "ages 25-65" → age column
- "patients and controls" → group column

Return column structure (Python will generate rows):

Output JSON:
{
  "columns": [
    {"name": "participant_id", "required": true},
    {"name": "sex", "levels": ["M", "F"]},
    {"name": "group", "levels": ["patient", "control"]}
  ]
}"""

PROMPT_NIRS_DRAFT = """fNIRS-to-SNIRF mapper (Draft).

Output JSON (ONLY valid JSON):
{
  "draft": {...},
  "confidence": 0.8,
  "questions": [...]
}"""

PROMPT_NIRS_NORMALIZE = """fNIRS-to-SNIRF mapper (Normalize).

Output JSON (ONLY valid JSON):
{
  "normalized": {...},
  "questions": [...]
}"""

PROMPT_MRI_VOXEL_DRAFT = """MRI voxelization planner (Draft).

Output JSON (ONLY valid JSON):
{
  "volume_candidates": [...],
  "meta_candidates": {...},
  "confidence": 0.8
}"""

PROMPT_MRI_VOXEL_FINAL = """MRI voxelization planner (Final).

Output JSON (ONLY valid JSON):
{
  "conversions": [...],
  "questions": []
}"""

PROMPT_BIDS_PLAN = """You are a BIDS dataset architect with complete decision-making authority.

═══════════════════════════════════════════════════════════════════════
SUPPORTED FORMATS AND CONVERSION RULES
═══════════════════════════════════════════════════════════════════════

MRI FORMATS (modality: mri):
  • DICOM (.dcm)           → convert_to: nifti   (dcm2niix)
  • NIfTI (.nii, .nii.gz)  → format_ready: true  (copy directly)
  • JNIfTI (.jnii, .bnii)  → convert_to: nifti

fNIRS FORMATS (modality: nirs):
  • SNIRF (.snirf)         → format_ready: true  (copy directly)
  • Homer3 (.nirs)         → convert_to: snirf
  • MATLAB (.mat)          → convert_to: snirf

EEG FORMATS (modality: eeg):
  • EDF/EDF+ (.edf)        → format_ready: true  (copy directly)
  • BrainVision (.vhdr)    → format_ready: true  (copy directly)
  • EEGLAB (.set)          → format_ready: true  (copy directly)
  • Biosemi (.bdf)         → format_ready: true  (copy directly)
  CRITICAL: EEG files are NEVER converted. Always format_ready: true, convert_to: none.
  CRITICAL: EEG bids_template MUST end with '_eeg.<original_ext>' (e.g. '_eeg.edf').
            NEVER use NIfTI suffixes (T1w, T2w, bold) for EEG data.

═══════════════════════════════════════════════════════════════════════
SUBJECT IDENTIFICATION — MOST IMPORTANT STEP
═══════════════════════════════════════════════════════════════════════

Your first job is to correctly identify all subjects from the file list.
The dataset may use ANY of the following structures:

STRUCTURE 1 — Already BIDS (sub-XX directories)
  sub-01/nirs/sub-01_task-rest_nirs.snirf
  sub-02/nirs/sub-02_task-rest_nirs.snirf
  → Use 'already_bids' strategy. Strip 'sub-' prefix.

STRUCTURE 2 — Site-prefixed directories
  Beijing_sub82352/anat/scan.nii.gz
  Newark_sub41006/anat/scan.nii.gz
  → Use directory names as subject identifiers.

STRUCTURE 3 — Flat files with numeric suffix
  VHMCT1mm-Hip (134).dcm  (prefix VHM = subject 1)
  VHFCT1mm-Hip (45).dcm   (prefix VHF = subject 2)
  → Use filename prefix as subject identifier.

STRUCTURE 4 — Group/subject nested directories
  PD/PD_01.snirf
  PD/PD_02.snirf
  control/control_01.snirf
  control/control_20.snirf
  → Each unique filename base (PD_01, PD_02 ... control_01 ... control_20)
    is ONE subject. The parent directory (PD / control) is the GROUP,
    not the subject. Add 'group' column to participant_metadata.
  → Assign numeric IDs: PD_01→1, PD_02→2 ... control_01→21 ... control_20→40

STRUCTURE 5 — Task/group/subject nested directories
  walking/PD/PD_01.snirf
  walking/control/control_01.snirf
  → Same as Structure 4. Ignore the task-level directory when identifying subjects.
    The task name goes into the BIDS filename (task-walking), not the subject ID.

STRUCTURE 6 — Pure numeric directories
  001/scan.dcm
  002/scan.dcm
  → Use directory number as subject ID.

CRITICAL RULES FOR SUBJECT COUNTING:
1. python_subject_analysis.subject_count is a HINT, not authoritative.
2. user_hints.n_subjects is the AUTHORITATIVE count.
   If provided, your assignment_rules MUST produce exactly that many subjects.
3. Count the actual unique files/directories to determine the true number.
4. For group/subject nested structures: count UNIQUE FILES, not directories.
   (PD/ and control/ are 2 directories but may contain 40 subjects total)

═══════════════════════════════════════════════════════════════════════
GROUP METADATA
═══════════════════════════════════════════════════════════════════════

When the dataset has clinically meaningful groups (PD vs control,
patient vs healthy, treated vs untreated):
- Add a 'group' column to participant_metadata for EVERY subject.
- Use the exact group label from the directory or filename.

Example for PD dataset with 40 subjects:
  participant_metadata:
    '1':  {original_id: 'PD_01',      group: 'PD'}
    '2':  {original_id: 'PD_02',      group: 'PD'}
    ...
    '21': {original_id: 'control_01', group: 'control'}
    ...
    '40': {original_id: 'control_20', group: 'control'}

═══════════════════════════════════════════════════════════════════════
ASSIGNMENT RULES
═══════════════════════════════════════════════════════════════════════

Each rule maps source files to one BIDS subject ID.

CRITICAL: 'subject' field must be BARE ID — no 'sub-' prefix.
  ✓ subject: '1'      → executor creates sub-1
  ✗ subject: 'sub-1'  → executor creates sub-sub-1

For group/subject nested structures, use the filename as the match token:
  assignment_rules:
    - subject: '1'
      original: 'PD_01'
      match: ['*PD_01*']
    - subject: '21'
      original: 'control_01'
      match: ['*control_01*']

For prefix-based flat structures:
  assignment_rules:
    - subject: '1'
      original: 'VHM'
      match: ['*VHM*']
    - subject: '2'
      original: 'VHF'
      match: ['*VHF*']

═══════════════════════════════════════════════════════════════════════
FORMAT_READY AND CONVERT_TO RULES
═══════════════════════════════════════════════════════════════════════

format_ready: true  → .nii/.nii.gz (MRI) or .snirf (fNIRS) — copy directly
format_ready: false → needs conversion:
  .dcm / .jnii / .bnii → convert_to: nifti
  .mat / .nirs         → convert_to: snirf
convert_to: "none"   → only when format_ready: true

═══════════════════════════════════════════════════════════════════════
FILENAME RULES — TASK INFERENCE
═══════════════════════════════════════════════════════════════════════

For fNIRS: infer task name from directory structure or user description.
  walking/ directory → task-walking
  fingertapping/ or tapping/ → task-fingertapping
  resting/ or rest/ → task-rest

For MRI: use acq- to distinguish different scan series from same subject.
  VHFCT1mm-Ankle.dcm → acq-ankle_T1w
  VHFCT1mm-Head.dcm  → acq-head_T1w

For EEG: infer task and run from filename suffixes or directory names.
  RULE 1 — If each subject has multiple EEG files, each file is a separate scan.
    Identify what differs between files of the same subject (suffix, keyword, directory).
    Map each variant to a distinct task- or run- label from user description.
    If task labels cannot be inferred, use run-1, run-2, run-N.
  RULE 2 — Create one mapping entry per unique file variant across subjects.
  RULE 3 — BIDS directory for EEG is always 'eeg/', never 'anat/' or 'nirs/'.
  RULE 4 — BIDS filename suffix is always '_eeg' + original extension.

EEG FILENAME EXAMPLES (CRITICAL — follow exactly):
  ✓ sub-01_task-rest_eeg.edf
  ✓ sub-01_task-arithmetic_eeg.edf
  ✓ sub-01_run-1_eeg.edf
  ✗ sub-01_T1w.nii.gz          ← NEVER for EEG
  ✗ sub-01_unknown.nii.gz      ← NEVER for EEG
  ✗ sub-01_bold.nii.gz         ← NEVER for EEG

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════

subjects:
  labels: [list of bare BIDS IDs, e.g. ['1','2',...,'40']]
  count: N
  source: llm_analysis
  id_strategy: numeric / semantic / already_bids

assignment_rules:
  - subject: 'bare_id'
    original: 'exact_identifier_from_filename_or_dirname'
    match: ['*identifier*']

participant_metadata:
  'bare_id':
    original_id: 'xxx'
    group: 'PD'          # if applicable
    sex: 'M'             # if available
    age: '65'            # if available

mappings:
  - modality: nirs
    match: ['**/*.snirf']
    exclude: []
    format_ready: true
    convert_to: none
    filename_rules:
      - match_pattern: '.*'
        bids_template: 'sub-X_task-walking_nirs.snirf'

  # EEG example — when each subject has ONE edf file:
  - modality: eeg
    match: ['**/*.edf']
    exclude: []
    format_ready: true
    convert_to: none
    filename_rules:
      - match_pattern: '.*'
        bids_template: 'sub-X_task-rest_eeg.edf'

  # EEG example — when each subject has MULTIPLE edf files (different tasks/runs):
  # Create one mapping entry per task/run, use match_pattern to distinguish them.
  # The match_pattern must be derived from what actually differs in the filenames.

OUTPUT: Raw YAML only (no markdown, no explanation)
"""

# ── Anonymize instruction snippets (appended to prompts when anonymize=True) ─

_ANON_INSTRUCTION_TRIO = """
═══════════════════════════════════════════════════════════════════════
DE-IDENTIFICATION MODE (anonymize=True)
═══════════════════════════════════════════════════════════════════════

The input data has been pre-scrubbed of PHI. Follow these rules strictly:

1. AUTHORS: Do NOT include real personal names in the Authors field.
   Use institutional abbreviations or "et al." forms only if explicitly
   present in the input text. If unsure, omit the Authors field entirely.

2. INSTITUTION: Do NOT output real institution names.
   If an institution is referenced, use the site label already in the
   input (e.g. "site-01"). Never invent or restore institution names.

3. DATES: Do NOT include specific dates (month/day). Year only is acceptable.

4. ACKNOWLEDGEMENTS / FUNDING: Omit if they contain personal names or
   specific institution names not already anonymized in the input.

5. GENERAL: If a field value looks like PHI (name, address, phone, email),
   omit that field entirely rather than including it.
"""

_ANON_INSTRUCTION_PLAN = """
═══════════════════════════════════════════════════════════════════════
DE-IDENTIFICATION MODE (anonymize=True)
═══════════════════════════════════════════════════════════════════════

The evidence bundle and file paths have been pre-scrubbed of PHI.
Follow these rules:

1. SUBJECT IDs: Use only the anonymized identifiers already in the input.
   Do NOT attempt to restore or infer original patient names or IDs.

2. PARTICIPANT METADATA: Do NOT include real names, addresses, or contact
   information in participant_metadata. Use group/sex/age fields only.

3. INSTITUTION: If site labels (site-01, site-02) appear in the input,
   use them as-is. Do NOT restore original institution names.

4. DESCRIBE TEXT: The user description may have been partially scrubbed.
   Work with what is provided; do not attempt to fill in redacted parts.
"""

# MAT → SNIRF semantic mapping

PROMPT_MAT_SNIRF_MAPPING = """You are an fNIRS data format expert.

You will receive a JSON summary of one or more MATLAB .mat files from the
same structural group. The summary contains a "flat_vars" dict where all
scipy struct wrappers have already been unwrapped — what you see reflects
the actual data shape and content.

flat_vars key conventions:
- Top-level variable:     "d", "t", "fs"
- Struct field:           "dat.signal", "SD.Lambda", "dat.fs"
- "likely_data": true     marks tall 2D float arrays (n_samples > n_channels)
- "value"                 means scalar
- "values"                means small array with known content
- "string_array" dtype    means channel labels or string metadata

Use flat_vars keys EXACTLY as they appear. Do not invent new paths.

═══════════════════════════════════════════════════════════
SNIRF REQUIRED FIELDS
═══════════════════════════════════════════════════════════

dataTimeSeries  — 2D float (n_samples × n_channels)
time            — 1D float (n_samples,), unit: seconds
wavelengths     — 1D array of wavelength values in nm
measurementList — per-channel source/detector/wavelength/dataType indices

═══════════════════════════════════════════════════════════
DATA ASSEMBLY TYPES
═══════════════════════════════════════════════════════════

Choose the correct type based on how the data is stored:

TYPE 1 — "single": data is in one variable (most common)
  Use when: one tall 2D array holds all channels
  Example: Homer3 "d", or "dat.signal"
  {
    "type": "single",
    "var": "d",
    "transpose": false
  }
  Set transpose: true if shape is (n_channels, n_samples) instead of (n_samples, n_channels)
  FORBIDDEN: Do NOT use array indexing syntax like "data.values[0]" or "data[0]".
  The Python executor does not support cell array indexing.
  Only dot-notation paths are supported: "data.X", "dat.signal", "SD.Lambda".

  CRITICAL — struct variables: if the top-level variable is a MATLAB struct
  (i.e. flat_vars shows sub-fields like "data.X", "data.fs", "data.trial"),
  you MUST use the full dot-notation path to the numeric field, NOT the
  struct variable name itself.

  Example: flat_vars shows:
    "data.X":     {"shape": [N, C], "likely_data": true}
    "data.fs":    {"value": 10.0}
    "data.trial": {"shape": [1, 75]}
  Correct:   "var": "data.X"     ← full dot-notation path
  WRONG:     "var": "data"       ← this is the struct, not the data array

  Similarly for time:
    "data.fs" is a scalar → use as fs_var in time_assembly
    Correct: {"type": "generate", "fs_var": "data.fs"}

TYPE 2 — "stack_columns": data split across ch1, ch2, ... chN variables
  Use when: flat_vars contains many variables named ch1, ch2, ch3 ... chN
  each being a 1D or column vector of the same length
  {
    "type": "stack_columns",
    "var_pattern": "ch",
    "var_range": [1, 40]
  }
  var_pattern: the common prefix (e.g. "ch", "channel", "nirs")
  var_range: [first_index, last_index] inclusive
  Use "vars" list instead of var_pattern+var_range if naming is non-numeric:
  {
    "type": "stack_columns",
    "vars": ["left_pfc", "right_pfc", "motor"]
  }

TYPE 3 — "hbo_hbr": HbO and HbR stored as separate matrices
  Use when: two 2D arrays named HbO/HbR or oxy/deoxy exist with same shape
  {
    "type": "hbo_hbr",
    "hbo_var": "HbO",
    "hbr_var": "HbR"
  }
  Result: columns are concatenated [HbO | HbR] → (n_samples, n_channels)

═══════════════════════════════════════════════════════════
TIME ASSEMBLY TYPES
═══════════════════════════════════════════════════════════

TYPE 1 — "var": time vector exists as a variable
  {
    "type": "var",
    "var": "t"
  }

TYPE 2 — "generate": no time variable, generate from sampling rate
  Prefer fs_var (read from file) over fs_value (hardcoded)
  {
    "type": "generate",
    "fs_var": "dat.fs",
    "fs_value": 13.33
  }
  If neither fs_var nor fs_value is known, set fs_value to null
  (executor will default to 10.0 Hz)

═══════════════════════════════════════════════════════════
WAVELENGTHS ASSEMBLY TYPES
═══════════════════════════════════════════════════════════

TYPE 1 — "var": wavelengths stored in a variable
  {
    "type": "var",
    "var": "SD.Lambda"
  }

TYPE 2 — "value": hardcode the values
  Use when no wavelength variable found, or data is already concentration (HbO/HbR)
  {
    "type": "value",
    "values": [760, 850]
  }

═══════════════════════════════════════════════════════════
OTHER FIELDS
═══════════════════════════════════════════════════════════

measlist_var:
  2D array shape (n_channels, 4), cols = [srcIdx, detIdx, aux, dataTypeCode]
  Common: "SD.MeasList"
  null if not found

n_sources_var:
  dot-notation path to a scalar variable whose value is the number of sources (optodes).
  Look in flat_vars for a key whose:
    - value is a small integer (typically 2–64)
    - name semantically suggests source count: contains "nSrc", "nSource", "source",
      "Src", "nS" or similar
  Use the EXACT key as it appears in flat_vars. Do NOT invent paths.
  null if no such variable found.

n_detectors_var:
  dot-notation path to a scalar variable whose value is the number of detectors (optodes).
  Look in flat_vars for a key whose:
    - value is a small integer (typically 2–64)
    - name semantically suggests detector count: contains "nDet", "nDetector",
      "detector", "Det", "nD" or similar
  Use the EXACT key as it appears in flat_vars. Do NOT invent paths.
  null if no such variable found.

data_type_code:
  1 = raw intensity (default)
  2 = dOD (optical density change)
  4 = HbO/HbR concentration
  Set to 4 if data_assembly type is "hbo_hbr" or var names suggest concentration

confidence: "high" | "medium" | "low"

═══════════════════════════════════════════════════════
DECISION GUIDE
═══════════════════════════════════════════════════════

Step 0 — Detect multi-block structure:
  Use "top_level_shapes" (NOT flat_vars) to detect multi-block structures.
  top_level_shapes shows the RAW shape of each variable BEFORE any unwrapping,
  which is the only reliable way to see that e.g. "data" is a (1,4) cell array.

  Detection rule — ALL three conditions must be true:
    1. top_level_shapes[key].is_object == true
    2. top_level_shapes[key].shape == [1, N] with N > 1
    3. flat_vars contains sub-fields of that key (e.g. "data.X", "data.fs")
       meaning each element of the cell array is a struct with data fields

  If all three conditions are met:
    → n_blocks = N  (the second dimension of the shape)
    → block_data_field = the sub-field name holding the signal matrix
      (look for the tall 2D array in flat_vars, e.g. "data.X" with likely_data=true)
    → data_assembly.var = full dot-notation path to signal field in ONE block
      (e.g. "data.X") — the executor iterates over blocks automatically

  If the top-level variable is a plain 2D float matrix: n_blocks=1.
  If uncertain: n_blocks=1  (safe default — no data is lost).

  EXAMPLES:
    top_level_shapes: {"data": {"shape": [1,4], "is_object": true, "is_struct": false}}
    flat_vars has: "data.X" (likely_data=true), "data.fs" (scalar), "data.trial"
    → n_blocks=4, block_data_field="X", data_assembly.var="data.X"

    top_level_shapes: {"d": {"shape": [3000, 52], "is_object": false}}
    → n_blocks=1, standard single-block processing

Step 1 — Identify data_assembly type:
  - Is there one tall 2D float array?        → "single"
  - Are there many ch1...chN variables?      → "stack_columns"
  - Are there HbO and HbR arrays?            → "hbo_hbr"

Step 2 — Identify time_assembly type:
  - Is there a 1D array matching n_samples?  → "var"
  - Is there a scalar fs/Fs/srate?           → "generate" with fs_var
  - Neither?                                 → "generate" with fs_value from notes or null

Step 3 — Identify wavelengths_assembly type:
  - Is there a small float array 600-1000?   → "var"
  - No wavelength info found?                → "value" with [760, 850]

Step 4 — Set data_type_code:
  - Raw NIR intensity data                   → 1
  - Optical density (log ratio)              → 2
  - Hemoglobin concentration (HbO/HbR)       → 4

═══════════════════════════════════════════════════════════
OUTPUT FORMAT — JSON only, no markdown, no explanation
═══════════════════════════════════════════════════════════

{
  "data_assembly": {
    "type": "single",
    "var": "d",
    "transpose": false
  },
  "time_assembly": {
    "type": "var",
    "var": "t"
  },
  "wavelengths_assembly": {
    "type": "var",
    "var": "SD.Lambda"
  },
  "wavelengths_default": [760, 850],
  "measlist_var": "SD.MeasList",
  "n_sources_var": null,
  "n_detectors_var": null,
  "n_blocks": 1,
  "block_data_field": null,
  "data_type_code": 1,
  "notes": "Homer3 format: standard d/t/SD structure detected",
  "confidence": "high"
}

Additional examples:

stack_columns case (ch1...ch40):
{
  "data_assembly": {
    "type": "stack_columns",
    "var_pattern": "ch",
    "var_range": [1, 40]
  },
  "time_assembly": {
    "type": "generate",
    "fs_var": "nfo.fs",
    "fs_value": 13.33
  },
  "wavelengths_assembly": {
    "type": "value",
    "values": [760, 850]
  },
  "wavelengths_default": [760, 850],
  "measlist_var": null,
  "n_sources_var": null,
  "n_detectors_var": null,
  "n_blocks": 1,
  "block_data_field": null,
  "data_type_code": 4,
  "notes": "Data split across 40 channel variables ch1-ch40, concentration format",
  "confidence": "medium"
}

hbo_hbr case:
{
  "data_assembly": {
    "type": "hbo_hbr",
    "hbo_var": "HbO",
    "hbr_var": "HbR"
  },
  "time_assembly": {
    "type": "var",
    "var": "time"
  },
  "wavelengths_assembly": {
    "type": "value",
    "values": [760, 850]
  },
  "wavelengths_default": [760, 850],
  "measlist_var": null,
  "n_sources_var": null,
  "n_detectors_var": null,
  "n_blocks": 1,
  "block_data_field": null,
  "data_type_code": 4,
  "notes": "HbO and HbR stored separately, will be concatenated column-wise",
  "confidence": "high"
}
"""

PROMPT_EEG_EVENT_MAPPING = """You are an expert in EEG data formats and BIDS events.tsv specification.

You will be given a sample of raw lines from an EEG event file (or BrainVision .vmrk file),
along with the source type. Your job is to identify the column structure and output a
standardized mapping so Python can read this file deterministically.

TASK: Identify which columns map to:
- onset: event start time
- duration: event duration (may be absent → use "n/a")  
- trial_type: event label/condition name

Also identify:
- header_row: true if the first non-comment line is a header
- skip_rows: number of lines to skip before data starts (comment lines, blank lines)
- separator: "tab", "comma", or "space"
- onset_unit: "seconds", "milliseconds", or "samples"
- duration_unit: "seconds", "milliseconds", or "samples" (or "n/a" if absent)

For BrainVision .vmrk format:
- Lines look like: Mk1=Stimulus,S  1,1000,1,0
- onset is the 3rd comma-separated value (sample index, not seconds)
- trial_type is the 2nd value (e.g. "S  1" → "S1")
- onset_unit must be "samples"

OUTPUT: JSON only, no markdown, no explanation.

{
  "onset_col": "time",
  "duration_col": "duration",
  "trial_type_col": "condition",
  "header_row": true,
  "skip_rows": 0,
  "separator": "tab",
  "onset_unit": "seconds",
  "duration_unit": "seconds",
  "source_type": "external_event_file",
  "notes": "standard TSV with header"
}

If onset or trial_type cannot be identified, set them to null.
If this is EDF+ annotations (source_type=edf_plus_annotations), return:
{"source_type": "edf_plus_annotations", "notes": "read from EDF+ annotations directly"}
"""


PROMPT_EEG_AUX_MAPPING = """You are an EEG-BIDS expert analyzing auxiliary files from an EEG dataset.

You will receive a list of candidate auxiliary files, each with its filename, extension,
and the first 30 lines of content. Your job is to analyze each file and determine:

1. What BIDS sidecar fields can be filled from this file?
2. What is the primary content type?

For each file output a JSON object. Combine all files into a single JSON array.

CONTENT TYPES to detect:
- "electrode_coordinates": file contains x/y/z or theta/phi positions per electrode/channel
- "participant_demographics": file contains age, sex, group, or other per-subject info
- "recording_metadata": file contains EEGReference, PowerLineFrequency, filter settings, amplifier info
- "channel_labels": file maps channel numbers to names
- "irrelevant": file is documentation, checksums, or unrelated metadata

For electrode_coordinates files, also detect:
- coordinate_system: "cartesian_3d", "cartesian_2d", "spherical", or "unknown"
- has_impedance: true/false
- column_order: list of column names in order (e.g. ["name","x","y","z"])

For participant_demographics files, detect:
- columns: list of column names found
- subject_id_col: which column is the subject identifier

For recording_metadata files, detect:
- fields: dict mapping detected field → value or "present"

OUTPUT: JSON array only, no markdown, no explanation.

[
  {
    "relpath": "subject-info.csv",
    "content_type": "participant_demographics",
    "bids_targets": ["participants.tsv"],
    "subject_id_col": "subject_id",
    "columns": ["subject_id", "age", "sex", "group"],
    "notes": "Contains age and sex for all subjects"
  },
  {
    "relpath": "electrodes.loc",
    "content_type": "electrode_coordinates",
    "bids_targets": ["*_electrodes.tsv", "*_coordsystem.json"],
    "coordinate_system": "cartesian_3d",
    "has_impedance": false,
    "column_order": ["name", "x", "y", "z"],
    "notes": "Standard 10-20 positions"
  }
]

If a file is irrelevant, still include it with content_type: "irrelevant".
If content cannot be determined from the sample, use content_type: "unknown".
"""


# ============================================================================
# Public LLM call wrappers
# ============================================================================

def llm_trio_dataset_description(model: str, payload: str,
                                  anonymize: bool = False) -> str:
    prompt = PROMPT_TRIO_DATASET_DESC
    if anonymize:
        prompt = prompt + _ANON_INSTRUCTION_TRIO
    return _call_llm(model, prompt, payload,
                     "Trio_DatasetDesc", temperature=0.1)

def llm_trio_readme(model: str, payload: str,
                    anonymize: bool = False) -> str:
    prompt = PROMPT_TRIO_README
    if anonymize:
        prompt = prompt + _ANON_INSTRUCTION_TRIO
    return _call_llm(model, prompt, payload,
                     "Trio_README", temperature=0.4)

def llm_trio_participants(model: str, payload: str,
                           anonymize: bool = False) -> str:
    prompt = PROMPT_TRIO_PARTICIPANTS
    if anonymize:
        prompt = prompt + _ANON_INSTRUCTION_TRIO
    return _call_llm(model, prompt, payload,
                     "Trio_Participants", temperature=0.2)

def llm_nirs_draft(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_NIRS_DRAFT, payload,
                     "NIRS_Draft", temperature=0.2)

def llm_nirs_normalize(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_NIRS_NORMALIZE, payload,
                     "NIRS_Normalize", temperature=0.1)

def llm_map_mat_to_snirf(model: str, payload: str) -> str:
    """Ask LLM to map .mat variable structure to SNIRF fields."""
    return _call_llm(
        model,
        PROMPT_MAT_SNIRF_MAPPING,
        payload,
        "MAT_SNIRF_Mapping",
        temperature=0.05,
    )

def llm_mri_voxel_draft(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_MRI_VOXEL_DRAFT, payload,
                     "MRI_Voxel_Draft", temperature=0.2)

def llm_mri_voxel_final(model: str, payload: str) -> str:
    return _call_llm(model, PROMPT_MRI_VOXEL_FINAL, payload,
                     "MRI_Voxel_Final", temperature=0.1)

def llm_map_eeg_events(model: str, payload: str) -> str:
    """Ask LLM to map EEG event file columns to BIDS events.tsv fields."""
    return _call_llm(
        model,
        PROMPT_EEG_EVENT_MAPPING,
        payload,
        "EEG_Event_Mapping",
        temperature=0.05,
    )

def llm_analyze_eeg_aux(model: str, payload: str) -> str:
    """Ask LLM to analyze EEG auxiliary files and map them to BIDS targets."""
    return _call_llm(
        model,
        PROMPT_EEG_AUX_MAPPING,
        payload,
        "EEG_Aux_Mapping",
        temperature=0.05,
    )

def llm_bids_plan(model: str, payload: str,
                  anonymize: bool = False) -> str:
    prompt = PROMPT_BIDS_PLAN
    if anonymize:
        prompt = prompt + _ANON_INSTRUCTION_PLAN
    return _call_llm(model, prompt, payload,
                     "BIDSPlan", temperature=0.15)