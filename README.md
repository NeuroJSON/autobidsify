# autobidsify

Automated Brain Imaging Data Structure (BIDS) standardization tool powered by LLM-first architecture.

[![Website](https://img.shields.io/badge/Website-AutoBIDSify-blue)](https://neurojson.org/Page/autobidsify)
[![PyPI version](https://badge.fury.io/py/autobidsify.svg)](https://pypi.org/project/autobidsify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **General compatibility**: Handles diverse dataset structures (flat, hierarchical, multi-site)
- **Multi-modal support**: MRI, fNIRS, EEG, and mixed modality datasets
- **Intelligent metadata extraction**: Automatic participant demographics from DICOM headers, documents, and filenames
- **Format conversion**: DICOM→NIfTI, JNIfTI→NIfTI, .mat/.nirs→SNIRF, and more
- **Multi-LLM support**: OpenAI (gpt-4o, gpt-5.1) and Qwen (via Ollama locally, REST API, or DashScope)
- **Evidence-based reasoning**: Confidence scoring and provenance tracking for all decisions

## Supported Formats

**Input formats:**
- MRI: DICOM (.dcm), NIfTI (.nii, .nii.gz), JNIfTI (.jnii, .bnii)
- fNIRS: SNIRF (.snirf), Homer3 (.nirs), MATLAB (.mat)
- EEG: EDF/EDF+ (.edf), BrainVision (.vhdr), EEGLAB (.set), Biosemi (.bdf)
- Documents: PDF, DOCX, TXT, Markdown

**Output:** Compliant to [BIDS specification (v1.10.0)](https://bids-specification.readthedocs.io/en/stable/)

## Installation

```bash
pip install autobidsify
```

**Optional dependencies:**
```bash
# For BIDS validation
npm install -g bids-validator

# For DICOM conversion
pip install dcm2niix          # or: apt-get install dcm2niix / brew install dcm2niix
```

**Set API key:**
```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"

# Qwen via DashScope (optional cloud alternative to Ollama)
export DASHSCOPE_API_KEY="your-key-here"
```

**Run all testing datasets:**
```bash
./run_all_tests.sh
```

## Quick Start

```bash
# Full pipeline (one command)
autobidsify full \
  --input /path/to/your/data \
  --output outputs/my_dataset \
  --model gpt-4o \
  --modality mri \
  --nsubjects 10 \
  --id-strategy auto \
  --describe "Your dataset description here"

# Step-by-step execution
autobidsify ingest   --input data/ --output outputs/run
autobidsify evidence --output outputs/run --modality mri
autobidsify trio     --output outputs/run --model gpt-4o
autobidsify plan     --output outputs/run --model gpt-4o
autobidsify execute  --output outputs/run
autobidsify validate --output outputs/run
```

## Command Options

```
--input PATH            Input data (archive or directory)
--output PATH           Output directory
--model MODEL           LLM model (default: gpt-4o)
--modality TYPE         Data modality: mri | nirs | eeg | mixed
--nsubjects N           Number of subjects (optional, auto-detected if omitted)
--describe "TEXT"       Dataset description (recommended for metadata accuracy)
--id-strategy STRATEGY  Subject ID strategy: auto | numeric | semantic (default: auto)
```

## Supported Models

**OpenAI:**
```bash
--model gpt-4o           # Recommended, stable
--model gpt-4o-mini      # Faster, cheaper
--model gpt-5.1          # Latest
```

**Qwen (via local Ollama):**
```bash
--model qwen3-coder-next:latest     # Recommended
--model qwen3-coder-careful:latest  # Recommended
--model qwen2.5-coder:7b            # Not recommended, slow and sometimes inaccurate
```

**Qwen (via remote Ollama REST API):**
```bash
export OLLAMA_BASE_URL=http://your-server.com:xxxx
--model qwen3-coder-next:latest
```

**Qwen (via DashScope cloud API):**
```bash
export DASHSCOPE_API_KEY="your-key-here"
--model qwen-max
```

## Pipeline Stages

| Stage | Command | Input | Output | Purpose |
|-------|---------|-------|--------|---------|
| 1 | `ingest` | Raw data | `ingest_info.json` | Extract/reference data |
| 2 | `evidence` | All files | `evidence_bundle.json` | Analyze structure, detect subjects, scan auxiliary files |
| 3 | `classify` | Mixed data | `classification_plan.json`, pool directories | Separate MRI/fNIRS/EEG (optional, mixed only) |
| 4 | `trio` | Evidence | BIDS trio files | Generate dataset_description.json, README, participants.tsv |
| 5 | `plan` | Evidence + trio | `BIDSPlan.yaml` | Create conversion strategy, generate modality-specific mappings |
| 6 | `execute` | Plan | `bids_compatible/`, `conversion_log.json`, `BIDSManifest.yaml` | Execute conversions, generate BIDS sidecars |
| 7 | `validate` | BIDS dataset | Validation report | Check compliance (Tier 1: Python bids_validator, Tier 2: npm bids-validator) |

## Output Structure

```
outputs/my_dataset/
├── bids_compatible/              # Final BIDS dataset
│   ├── dataset_description.json
│   ├── README.md
│   ├── participants.tsv
│   ├── sub-001/
│   │   ├── anat/
│   │   │   └── sub-001_T1w.nii.gz
│   │   ├── func/
│   │   │   └── sub-001_task-rest_bold.nii.gz
│   │   ├── nirs/
│   │   │   ├── sub-001_task-rest_nirs.snirf
│   │   │   └── sub-001_task-rest_nirs.json
│   │   └── eeg/
│   │       ├── sub-001_task-rest_eeg.edf
│   │       ├── sub-001_task-rest_eeg.json
│   │       ├── sub-001_task-rest_channels.tsv
│   │       ├── sub-001_optodes.tsv        # fNIRS only
│   │       ├── sub-001_electrodes.tsv     # EEG only
│   │       └── sub-001_coordsystem.json
│   └── derivatives/              # Unprocessed files (original structure)
└── _staging/                     # Intermediate files
    ├── evidence_bundle.json
    ├── BIDSPlan.yaml
    ├── mat_mapping.json           # fNIRS .mat datasets only
    ├── eeg_event_mapping.json     # EEG datasets with event files
    ├── eeg_aux_mapping.json       # EEG datasets with auxiliary metadata
    └── conversion_log.json
```

## Examples

### MRI dataset
```bash
autobidsify full \
  --input brain_scans/ \
  --output outputs/study1 \
  --model gpt-4o \
  --modality mri \
  --nsubjects 30 \
  --id-strategy numeric \
  --describe "Single-site T1w MRI study, 30 healthy adults"
```

### fNIRS dataset
```bash
autobidsify full \
  --input fnirs_data/ \
  --output outputs/fnirs \
  --model gpt-4o \
  --modality nirs \
  --describe "Prefrontal fNIRS, 20 subjects, resting state and finger tapping"
```

### EEG dataset
```bash
autobidsify full \
  --input eeg_data/ \
  --output outputs/eeg \
  --model gpt-4o \
  --modality eeg \
  --nsubjects 36 \
  --describe "EEG during mental arithmetic tasks, 36 subjects, EDF format"
```

### Using Qwen (local, no API cost)
```bash
ollama serve
autobidsify full \
  --input data/ \
  --output outputs/run \
  --model qwen3-coder-next:latest \
  --modality mri
```

## Architecture

**LLM-First Design:**
- **Python**: Deterministic operations — file I/O, regex-based subject detection, format conversion, BIDS validation, standard 10-20 electrode lookup
- **LLM**: Semantic understanding — dataset description, metadata extraction, scan type classification, license normalization, event file column mapping, auxiliary file analysis
- **Hybrid**: Python analyzes ALL files for completeness; LLM sees representative samples for semantic decisions

## Requirements

- Python 3.10+
- OpenAI API key (or Ollama for local Qwen models)
- `bids-validator` (npm) for full structural validation (optional)
- `dcm2niix` for DICOM conversion (optional)

## Current Status

**Version:** 0.9.5

## Contributing

We need YOUR datasets to improve robustness. Please test and report issues at:
https://github.com/cotilab/autobidsify/issues