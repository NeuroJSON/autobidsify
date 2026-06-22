# AutoBIDSify

Automated Brain Imaging Data Structure (BIDS) standardization tool powered by an LLM-first architecture.

[![Website](https://img.shields.io/badge/Website-AutoBIDSify-blue)](https://neurojson.org/Page/autobidsify)
[![PyPI version](https://badge.fury.io/py/autobidsify.svg)](https://pypi.org/project/autobidsify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoBIDSify helps convert raw neuroimaging datasets into BIDS-compatible structures. It is designed for real-world datasets with inconsistent folder layouts, incomplete metadata, heterogeneous naming conventions, and mixed modalities.

## Key Features

- **Multi-modal support**: MRI, fNIRS, EEG, and mixed-modality datasets
- **Flexible input handling**: Supports flat, nested, multi-subject, and multi-site dataset structures
- **Format conversion**: DICOM → NIfTI, JNIfTI → NIfTI, `.mat`/`.nirs` → SNIRF, and EEG formats such as EDF/EDF+
- **Metadata extraction**: Uses file headers, filenames, folder structure, and auxiliary documents to infer BIDS metadata
- **LLM-assisted reasoning**: Uses LLMs for semantic decisions such as scan classification, task inference, event-file interpretation, and metadata normalization
- **Provenance-aware output**: Tracks decisions, confidence, intermediate evidence, and conversion logs
- **Validation support**: Supports BIDS validation after conversion

## Supported Data Types and Formats

### Input formats

| Modality | Supported input formats |
|---|---|
| MRI | DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`), JNIfTI (`.jnii`, `.bnii`) |
| fNIRS | SNIRF (`.snirf`), Homer/Homer3 (`.nirs`), MATLAB (`.mat`) |
| EEG | EDF/EDF+ (`.edf`), BrainVision (`.vhdr`), EEGLAB (`.set`), Biosemi (`.bdf`) |
| Documents | PDF, DOCX, TXT, Markdown |

### Output

AutoBIDSify generates a BIDS-compatible dataset following the [BIDS specification](https://bids-specification.readthedocs.io/en/stable/).

## Installation

```bash
pip install autobidsify
```

### Optional dependencies

```bash
# For full BIDS validation
npm install -g bids-validator

# For DICOM conversion
pip install dcm2niix
# Alternative installation options:
# apt-get install dcm2niix
# brew install dcm2niix
```

## API Keys and Model Setup

### OpenAI

```bash
export OPENAI_API_KEY="your-key-here"
```

### Qwen via DashScope

```bash
export DASHSCOPE_API_KEY="your-key-here"
```

### Qwen via local or remote Ollama

```bash
ollama serve
# Optional: use a remote Ollama endpoint
export OLLAMA_BASE_URL=http://your-server.com:xxxx
```

## Quick Start

### Run the full pipeline

```bash
autobidsify full \
  --input /path/to/your/data \
  --output outputs/my_dataset \
  --model gpt-4o \
  --modality mri \
  --nsubjects 10 \
  --id-strategy auto \
  --describe "Your dataset description here"
```

### Run step by step

```bash
autobidsify ingest   --input data/ --output outputs/run
autobidsify evidence --output outputs/run --modality mri
autobidsify trio     --output outputs/run --model gpt-4o
autobidsify plan     --output outputs/run --model gpt-4o
autobidsify execute  --output outputs/run
autobidsify validate --output outputs/run
```

## Command Options

| Option | Description |
|---|---|
| `--input PATH` | Input data directory or archive |
| `--output PATH` | Output directory |
| `--model MODEL` | LLM model to use, for example `gpt-4o`, `gpt-4o-mini`, `gpt-5.1`, `qwen3-coder-next:latest`, or `qwen-max` |
| `--modality TYPE` | Data modality: `mri`, `nirs`, `eeg`, or `mixed` |
| `--nsubjects N` | Number of subjects; optional and auto-detected if omitted |
| `--id-strategy STRATEGY` | Subject ID strategy: `auto`, `numeric`, or `semantic`; default is `auto` |
| `--describe "TEXT"` | Dataset description; recommended for more accurate metadata inference |

## Supported Models

### OpenAI

```bash
--model gpt-4o           # Recommended, stable
--model gpt-4o-mini      # Faster, cheaper
--model gpt-5.1          # Latest supported OpenAI model option
```

### Qwen via local Ollama

```bash
--model qwen3-coder-next:latest     # Recommended
--model qwen3-coder-careful:latest  # Recommended
--model qwen2.5-coder:7b            # Available but slower and less reliable
```

### Qwen via remote Ollama REST API

```bash
export OLLAMA_BASE_URL=http://your-server.com:xxxx
autobidsify full \
  --input data/ \
  --output outputs/run \
  --model qwen3-coder-next:latest \
  --modality mri
```

### Qwen via DashScope cloud API

```bash
export DASHSCOPE_API_KEY="your-key-here"
autobidsify full \
  --input data/ \
  --output outputs/run \
  --model qwen-max \
  --modality mri
```

## Pipeline Stages

| Stage | Command | Main input | Main output | Purpose |
|---|---|---|---|---|
| 1 | `ingest` | Raw data | `ingest_info.json` | Index input files and extract basic file information |
| 2 | `evidence` | Indexed files | `evidence_bundle.json` | Analyze dataset structure, detect subjects, and collect metadata evidence |
| 3 | `classify` | Mixed-modality data | `classification_plan.json`, modality pools | Separate MRI, fNIRS, and EEG files; mainly used for mixed datasets |
| 4 | `trio` | Evidence bundle | BIDS trio files | Generate `dataset_description.json`, `README`, and `participants.tsv` |
| 5 | `plan` | Evidence + BIDS trio | `BIDSPlan.yaml` | Create the conversion plan and modality-specific mappings |
| 6 | `execute` | BIDS plan | `bids_compatible/`, `conversion_log.json`, `BIDSManifest.yaml` | Execute conversion and generate BIDS files/sidecars |
| 7 | `validate` | BIDS dataset | Validation report | Validate the generated BIDS dataset |

## Example Output Structure

```text
outputs/my_dataset/
├── bids_compatible/                    # Final BIDS-compatible dataset
│   ├── dataset_description.json
│   ├── README.md
│   ├── participants.tsv
│   ├── sub-001/
│   │   ├── anat/                       # MRI anatomical data
│   │   │   └── sub-001_T1w.nii.gz
│   │   ├── func/                       # MRI functional data, if present
│   │   │   └── sub-001_task-rest_bold.nii.gz
│   │   ├── nirs/                       # fNIRS data, if present
│   │   │   ├── sub-001_task-rest_nirs.snirf
│   │   │   ├── sub-001_task-rest_nirs.json
│   │   │   └── sub-001_optodes.tsv
│   │   └── eeg/                        # EEG data, if present
│   │       ├── sub-001_task-rest_eeg.edf
│   │       ├── sub-001_task-rest_eeg.json
│   │       ├── sub-001_task-rest_channels.tsv
│   │       ├── sub-001_electrodes.tsv
│   │       └── sub-001_coordsystem.json
│   └── derivatives/                    # Unprocessed or copied auxiliary files, when needed
└── _staging/                           # Intermediate AutoBIDSify files
    ├── ingest_info.json
    ├── evidence_bundle.json
    ├── classification_plan.json         # Mixed-modality datasets only
    ├── BIDSPlan.yaml
    ├── mat_mapping.json                 # fNIRS .mat datasets only
    ├── eeg_event_mapping.json           # EEG datasets with event files
    ├── eeg_aux_mapping.json             # EEG datasets with auxiliary metadata
    ├── conversion_log.json
    └── BIDSManifest.yaml
```

## Architecture

AutoBIDSify uses a hybrid architecture that combines deterministic Python operations with LLM-assisted semantic reasoning.

- **Python-based components** handle file I/O, archive extraction, subject detection, format conversion, BIDS file generation, validation, and standard EEG electrode lookup.
- **LLM-based components** handle semantic tasks such as dataset description generation, metadata interpretation, scan type classification, task label inference, license normalization, event-file mapping, and auxiliary-file interpretation.
- **Hybrid reasoning** allows Python to inspect all files for completeness while the LLM receives representative evidence for higher-level decisions.

## Requirements

- Python 3.8+
- OpenAI API key, local Ollama model, remote Ollama endpoint, or DashScope API key
- `bids-validator` for full BIDS validation, optional
- `dcm2niix` for DICOM conversion, optional

## Current Status

**Version:** 0.9.6

AutoBIDSify is under active development. Current work focuses on improving robustness across MRI, fNIRS, EEG, and mixed-modality datasets with diverse real-world structures.

## Desktop Application

A graphical desktop application is available for users who prefer not to use the command line.
Two versions are provided:

**AutoBIDSify ExecVal** — Execute and validate only. No AI or API key required.
Runs the execute and validate stages locally using a pre-generated BIDS plan.

| Platform | Download |
|---|---|
| Windows | [AutoBIDSify-ExecVal-Windows.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-execval/AutoBIDSify-ExecVal-Windows.zip) |
| macOS (Apple Silicon) | [AutoBIDSify-ExecVal-macOS-arm64.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-execval/AutoBIDSify-ExecVal-macOS-arm64.zip) |
| Linux | [AutoBIDSify-ExecVal-Linux.tar.gz](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-execval/AutoBIDSify-ExecVal-Linux.tar.gz) |

**AutoBIDSify Full** — Complete pipeline with AI. Requires an OpenAI API key, local Ollama model, or DashScope API key.
Runs the full autobidsify pipeline from ingestion to validation.

| Platform | Download |
|---|---|
| Windows | [AutoBIDSify-Full-Windows.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-full/AutoBIDSify-Full-Windows.zip) |
| macOS (Apple Silicon) | [AutoBIDSify-Full-macOS-arm64.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-full/AutoBIDSify-Full-macOS-arm64.zip) |
| Linux | [AutoBIDSify-Full-Linux.tar.gz](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-full/AutoBIDSify-Full-Linux.tar.gz) |

All releases: [https://github.com/yiyiliu-rose/autobidsifyAPP/releases](https://github.com/yiyiliu-rose/autobidsifyAPP/releases)

## Contributing Test Datasets

We welcome representative examples that can help improve AutoBIDSify across real-world dataset structures, metadata conventions, scanner/vendor differences, and modality-specific edge cases.

Full DICOM image data is **not required** in most cases. We understand that sharing patient DICOM data is usually not feasible for many centers. The most practical and recommended contribution is a small, de-identified example that includes the raw organization, metadata, and expected BIDS-compatible output structure.

A possible contribution format is:

```text
example_dataset/
├── raw_tree.txt              # Original file/folder structure, e.g., output from the tree command
├── dicom_metadata/           # De-identified or pseudonymized DICOM metadata, without image data
│   ├── series_001.json       # One metadata JSON file per series
│   ├── series_002.json       # Include fields such as ProtocolName, SeriesDescription, Modality, etc.
│   └── ...
├── expected_bids_tree.txt    # Expected BIDS-compatible output directory structure
└── notes.md                  # Mapping decisions, ambiguous cases, manual corrections, or other notes
```

For EEG and fNIRS datasets, the same idea applies: a raw folder tree, representative de-identified metadata or header information, relevant auxiliary files if shareable, and an expected BIDS-compatible output tree are very helpful for testing and debugging.

The expected BIDS structure does not need to be an official existing ground truth. Since many messy datasets do not already have ground-truth BIDS labels, it can be manually designed during testing or manually corrected after running AutoBIDSify.

Please do not share identifiable or sensitive data. If you are unsure whether a dataset can be shared, please only provide de-identified metadata, a simplified folder tree, and a manually prepared expected BIDS tree.

Please test AutoBIDSify and report issues at:
https://github.com/cotilab/autobidsify/issues

## License

This project is released under the MIT License.
