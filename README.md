# autobidsify

Automated Brain Imaging Data Structure (BIDS) standardization tool powered by LLM-first architecture.

[![Website](https://img.shields.io/badge/Website-AutoBIDSify-blue)](https://neurojson.org/Page/autobidsify)
[![PyPI version](https://badge.fury.io/py/autobidsify.svg)](https://pypi.org/project/autobidsify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Features

- **General compatibility**: Handles diverse dataset structures (flat, hierarchical, multi-site)
- **Multi-modal support**: MRI, fNIRS, and mixed modality datasets
- **Intelligent metadata extraction**: Automatic participant demographics from DICOM headers, documents, and filenames
- **Format conversion**: DICOM→NIfTI, JNIfTI→NIfTI, .mat/.nirs→SNIRF, and more
- **Multi-LLM support**: OpenAI (gpt-4o, gpt-5.1) and Qwen (via Ollama locally or with rest-api or DashScope)
- **Evidence-based reasoning**: Confidence scoring and provenance tracking for all decisions

## Supported Formats

**Input formats:**
- MRI: DICOM (.dcm), NIfTI (.nii, .nii.gz), JNIfTI (.jnii, .bnii)
- fNIRS: SNIRF (.snirf), Homer3 (.nirs), MATLAB (.mat)
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
```

**Set API key:**
```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"

# Qwen via DashScope (optional cloud alternative to Ollama)
export DASHSCOPE_API_KEY="your-key-here"
```

## Quick Start

```bash
# Full pipeline (one command)
# With dataset description (recommended for better metadata extraction)
autobidsify full \
  --input /path/to/your/data \
  --output outputs/my_dataset \
  --model gpt-4o \
  --modality mri \
  --nsubjects 10 \
  --id-strategy auto \
  --describe "Your dataset description here"

# Step-by-step execution
autobidsify ingest  --input data/ --output outputs/run
autobidsify evidence --output outputs/run --modality mri
autobidsify trio   --output outputs/run --model gpt-4o
autobidsify plan   --output outputs/run --model gpt-4o
autobidsify execute  --output outputs/run
autobidsify validate --output outputs/run
```

## Command Options

```
--input PATH            Input data (archive or directory)
--output PATH           Output directory
--model MODEL           LLM model (default: gpt-4o)
--modality TYPE         Data modality: mri | nirs | mixed
--nsubjects N           Number of subjects (optional, auto-detected if omitted)
--describe "TEXT"       Dataset description (recommended for metadata accuracy)
--id-strategy STRATEGY  Subject ID strategy: auto | numeric | semantic (default: auto)
```

## Supported Models

**OpenAI:**
```bash
--model gpt-4o           # Highly recommended, stable
--model gpt-4o-mini      # Faster, cheaper
--model gpt-5.1          # Not that ecommended, latest
```

**Qwen (via Ollama, local):**
```bash
--model qwen3-coder-next:latest     # Recommended
--model qwen3-coder-careful:latest  # Recommended
--model qwen2.5-coder:7b            # Not recommended, slow and sometimes inaccurate, 
```

**Qwen (via rest-api):**
```bash
export OLLAMA_BASE_URL=http://your-server.com:xxxx
```

## Pipeline Stages

| Stage | Command | Input | Output | Purpose |
|-------|---------|-------|--------|---------|
| 1 | `ingest` | Raw data | `ingest_info.json` | Extract/reference data |
| 2 | `evidence` | All files | `evidence_bundle.json` | Analyze structure, detect subjects |
| 3 | `classify` | Mixed data | `classification_plan.json`, `nirs_pool/`, `mri_pool/`, `unknown/` | Separate MRI/fNIRS (optional) |
| 4 | `trio` | Evidence | BIDS trio files | Generate metadata files |
| 5 | `plan` | Evidence + trio | `BIDSPlan.yaml`, `subject_analysis.json` | Create conversion strategy |
| 6 | `execute` | Plan | `bids_compatible/`, `coversion_log.json`, `BIDSManifest.yaml` | Execute conversions |
| 7 | `validate` | BIDS dataset | Validation report | Check compliance |

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
│   │   └── func/
│   │       └── sub-001_task-rest_bold.nii.gz
│   └── derivatives/              # Unprocessed files (original structure)
│       └── sub-001/
│           └── ...
└── _staging/                     # Intermediate files
    ├── evidence_bundle.json
    ├── BIDSPlan.yaml
    └── conversion_log.json
```

## Architecture

**LLM-First Design:**
- **Python**: Deterministic operations — file I/O, regex-based subject detection, format conversion, BIDS validation
- **LLM**: Semantic understanding — dataset description, metadata extraction, scan type classification, license normalization
- **Hybrid**: Python analyzes ALL files for completeness; LLM sees representative samples for semantic decisions

## Requirements

- Python
- OpenAI API key (or Ollama for local Qwen models)
- `bids-validator` for validation

## Current Status

**Version:** 0.9.2

**Tested datasets:**
- Visible Human Project (flat structure, DICOM CT)
- CamCAN (hierarchical, multi-site, 30+ subjects)
- FRESH-Motor (fNIRS, existing BIDS format)
- fNIRS tinnitus dataset (flat structure, .nirs files)

**Known limitations:**
- Mixed modality classification (Stage 3) is experimental
- .mat fNIRS conversion assumes Homer3-compatible variable naming

## Contributing

We need YOUR datasets to improve robustness. Please test and report issues at:
https://github.com/cotilab/autobidsify/issues
