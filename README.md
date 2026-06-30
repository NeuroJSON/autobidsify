# AutoBIDSify

Automated Brain Imaging Data Structure (BIDS) standardization tool powered by an LLM-first architecture.

[![Website](https://img.shields.io/badge/Website-AutoBIDSify-blue)](https://neurojson.org/Page/autobidsify)
[![PyPI version](https://badge.fury.io/py/autobidsify.svg)](https://pypi.org/project/autobidsify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AutoBIDSify helps convert raw neuroimaging datasets into BIDS-compatible structures. It is designed for real-world datasets with inconsistent folder layouts, incomplete metadata, heterogeneous naming conventions, and mixed modalities. A [graphical desktop application](#desktop-application) is also available for users who prefer not to use the command line.

## Key Features

- **Multi-modal support**: MRI, fNIRS, EEG, and mixed-modality datasets
- **Flexible input handling**: Supports flat, nested, multi-subject, and multi-site dataset structures
- **Format conversion**: DICOM → NIfTI, JNIfTI → NIfTI, `.mat`/`.nirs` → SNIRF, and EEG formats such as EDF/EDF+
- **Metadata extraction**: Uses file headers, filenames, folder structure, and auxiliary documents to infer BIDS metadata
- **LLM-assisted reasoning**: Uses LLMs for semantic decisions such as scan classification, task inference, event-file interpretation, and metadata normalization
- **De-identification support**: Optional HIPAA-aligned metadata scrubbing and MRI defacing via `--anonymize` and `--deface`
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
# Full BIDS validation
npm install -g bids-validator

# DICOM conversion
pip install dcm2niix
# or: apt-get install dcm2niix / brew install dcm2niix

# MRI anatomical defacing — required only when using --deface
# Also requires FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
pip install pydeface
```

De-identification by modality:

| Modality | Extra dependency for `--anonymize` | Extra dependency for `--deface` |
|---|---|---|
| MRI | none (pydicom already included) | `pydeface` + FSL |
| fNIRS | none (h5py already included) | not applicable |
| EEG | none (built-in EDF parser) | not applicable |

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

`--output` is only required at `ingest`. Subsequent steps read the output directory from the session automatically. All options shown below are optional unless marked required.

```bash
# Stage 1 — ingest (--input and --output required)
autobidsify ingest \
  --input data/ \
  --output outputs/run \
  [--anonymize true|false] \   # enable de-identification for this session; default false
  [--deface]                   # enable MRI anat defacing for execute stage

# Stage 2 — evidence
autobidsify evidence \
  [--output PATH] \            # default: session value from ingest
  [--modality mri|nirs|eeg|mixed] \
  [--nsubjects N] \
  [--describe "TEXT"]

# Stage 3 — classify (mixed-modality datasets only)
autobidsify classify \
  [--output PATH]

# Stage 4 — trio
autobidsify trio \
  [--output PATH] \
  --model gpt-4o \             # must be local Ollama model if --anonymize true
  [--file dataset_description|readme|participants|all]   # default: all

# Stage 5 — plan
autobidsify plan \
  [--output PATH] \
  --model gpt-4o \             # must be local Ollama model if --anonymize true
  [--id-strategy auto|numeric|semantic]   # default: auto

# Stage 6 — execute
autobidsify execute \
  [--output PATH] \
  [--deface]                   # overrides session value if provided here

# Stage 7 — validate
autobidsify validate \
  [--output PATH]

# --output can be provided at any step to override the session
autobidsify evidence --output outputs/other_run --modality mri
```

### Run with de-identification

`--anonymize true` requires a local Ollama model. Online models (OpenAI, DashScope) are not permitted because they send data to external servers.

```bash
# Metadata de-identification only
autobidsify full \
  --input data/ \
  --output outputs/anon \
  --model qwen3-coder-next:latest \
  --modality mri \
  --anonymize true

# Metadata de-identification + MRI defacing (requires pydeface + FSL)
autobidsify full \
  --input data/ \
  --output outputs/anon \
  --model qwen3-coder-next:latest \
  --modality mri \
  --anonymize true \
  --deface

# Step by step with de-identification
# --anonymize is set once at ingest and remembered for all subsequent steps
autobidsify ingest --input data/ --output outputs/anon --anonymize true
autobidsify evidence --modality mri
autobidsify trio     --model qwen3-coder-next:latest
autobidsify plan     --model qwen3-coder-next:latest
autobidsify execute  --deface   # --deface can also be added here
autobidsify validate
```

## CLI Reference

### `autobidsify full` — all options

Run the complete pipeline (stages 1–7) in one command. For step-by-step usage see [Run step by step](#run-step-by-step) above.

| Option | Required | Default | Description |
|---|---|---|---|
| `--input PATH` | Yes | — | Input data directory or archive file |
| `--output PATH` | Yes | — | Output directory |
| `--model MODEL` | No | `gpt-4o` | LLM model (see Supported Models) |
| `--modality TYPE` | No | auto | `mri`, `nirs`, `eeg`, or `mixed` |
| `--nsubjects N` | No | auto | Number of subjects |
| `--id-strategy STRATEGY` | No | `auto` | `auto`, `numeric`, or `semantic` |
| `--describe "TEXT"` | No | — | Dataset description; strongly recommended |
| `--anonymize true\|false` | No | `false` | Enable HIPAA-aligned metadata de-identification; requires local Ollama model |
| `--deface` | No | off | Deface MRI anat images (T1w/T2w/FLAIR); requires pydeface + FSL |

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

> **Note:** `--anonymize true` requires a local Ollama model. OpenAI and DashScope models send data to external servers and are not permitted with de-identification enabled.

## De-identification

When `--anonymize true` is set at `ingest` or `full`, AutoBIDSify applies HIPAA Safe Harbor-aligned de-identification to all metadata produced during the pipeline.

**What is scrubbed:** PatientName, PatientID, PatientBirthDate, InstitutionName/Address (replaced with `site-XX`), AcquisitionDate/StudyDate/SeriesDate (year retained, month/day removed), PatientAge (retained for ≤89; replaced with `90+` for ≥90), DeviceSerialNumber, StationName, AccessionNumber, and PHI patterns in free-text documents.

**What is not scrubbed automatically:** MRI image pixel data (facial features). Use `--deface` to run pydeface on anat NIfTI files after conversion. fNIRS and EEG signal data do not require defacing.

## Pipeline Stages

| Stage | Command | Main input | Main output | Purpose |
|---|---|---|---|---|
| 1 | `ingest` | Raw data | `ingest_info.json` | Index input files and set session output directory |
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

- **Python-based components** handle file I/O, archive extraction, subject detection, format conversion, BIDS file generation, validation, de-identification scrubbing, and standard EEG electrode lookup.
- **LLM-based components** handle semantic tasks such as dataset description generation, metadata interpretation, scan type classification, task label inference, license normalization, event-file mapping, and auxiliary-file interpretation.
- **Hybrid reasoning** allows Python to inspect all files for completeness while the LLM receives representative evidence for higher-level decisions.

## Requirements

- Python 3.8+
- OpenAI API key, local Ollama model, remote Ollama endpoint, or DashScope API key
- `bids-validator` (npm) for full BIDS validation, optional
- `dcm2niix` for DICOM conversion, optional
- `pydeface` + FSL for MRI anatomical defacing, optional

## Current Status

**Version:** 0.9.6

AutoBIDSify is under active development. Current work focuses on improving robustness across MRI, fNIRS, EEG, and mixed-modality datasets with diverse real-world structures.

## Desktop Application

A graphical desktop application is available for users who prefer not to use the command line. All releases are available at:
https://github.com/yiyiliu-rose/autobidsifyAPP/releases

Two versions are provided:

**AutoBIDSify ExecVal** — Execute and validate only. No AI or API key required.
Runs the `execute` and `validate` stages locally using a pre-generated BIDS plan.

| Platform              | Download                                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Windows               | [AutoBIDSify-ExecVal-Windows.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-execval/AutoBIDSify-ExecVal-Windows.zip)         |
| macOS (Apple Silicon) | [AutoBIDSify-ExecVal-macOS-arm64.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-execval/AutoBIDSify-ExecVal-macOS-arm64.zip) |
| Linux                 | [AutoBIDSify-ExecVal-Linux.tar.gz](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-execval/AutoBIDSify-ExecVal-Linux.tar.gz)       |

**AutoBIDSify Full** — Complete pipeline with AI. Requires an OpenAI API key, local Ollama model, or DashScope API key.
Runs the full AutoBIDSify pipeline from ingestion to validation.

| Platform              | Download                                                                                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Windows               | [AutoBIDSify-Full-Windows.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-full/AutoBIDSify-Full-Windows.zip)         |
| macOS (Apple Silicon) | [AutoBIDSify-Full-macOS-arm64.zip](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-full/AutoBIDSify-Full-macOS-arm64.zip) |
| Linux                 | [AutoBIDSify-Full-Linux.tar.gz](https://github.com/yiyiliu-rose/autobidsifyAPP/releases/download/latest-full/AutoBIDSify-Full-Linux.tar.gz)       |

Linux note: both desktop apps require a desktop environment with Tk and are built for GLIBC 2.35+.

## Contributing Test Datasets

We welcome representative examples that can help improve AutoBIDSify across real-world dataset structures, metadata conventions, scanner/vendor differences, and modality-specific edge cases.

Full DICOM image data is **not required** in most cases. We understand that sharing patient DICOM data is usually not feasible for many centers. The most practical and recommended contribution is a small, de-identified example that includes the raw organization, metadata, expected BIDS-compatible output structure, and any useful run feedback.

A possible contribution format is:

```text
example_dataset/
├── raw_tree.txt              # Original file/folder structure, e.g., output from the tree command
├── dicom_metadata/           # De-identified or pseudonymized DICOM metadata, without image data
│   ├── series_001.json       # One metadata JSON file per series
│   ├── series_002.json       # Include fields such as ProtocolName, SeriesDescription, Modality, etc.
│   └── ...
├── expected_bids_tree.txt    # Expected BIDS-compatible output directory structure
├── notes.md                  # Dataset description, mapping decisions, ambiguous cases, or manual corrections
└── logs/                     # Optional but very useful
    ├── run_log.txt           # AutoBIDSify run log, including LLM questions, errors, and warnings
    ├── classification_plan.json
    ├── BIDSPlan.yaml
    └── conversion_log.json
```

For EEG and fNIRS datasets, the same idea applies: a raw folder tree, representative de-identified metadata or header information, relevant auxiliary files if shareable, and an expected BIDS-compatible output tree are very helpful for testing and debugging.

The expected BIDS structure does not need to be an official existing ground truth. Since many messy datasets do not already have ground-truth BIDS labels, it can be manually designed during testing or manually corrected after running AutoBIDSify.

### Useful Feedback to Include

In addition to the minimal dataset example, the following feedback is especially helpful and should not require much extra manual work.

#### 1. AutoBIDSify's LLM questions

When AutoBIDSify is not confident about how the input data should map to a BIDS label based on the available information, such as the protocol name, series description, folder name, or file header, it asks a clarification question instead of silently guessing.

These questions mainly appear in two stages:

* `classify`, when the modality or data type is ambiguous;
* `plan`, when the BIDS entity mapping is uncertain while building the per-file conversion plan (`BIDSPlan`).

The questions are printed in the run log and are also saved in the output of the corresponding stage. Sharing these questions helps us understand which cases are genuinely ambiguous and where the pipeline needs better rules or better LLM context.

#### 2. Errors and warnings in the log

If a conversion fails, the log should contain the related errors or warnings, such as DICOM-to-NIfTI conversion failures, unmatched BIDS entities, or duplicate/missing participant IDs.

These messages are very useful because they often point directly to the messy real-world cases that AutoBIDSify is designed to handle, including varied acquisition protocols, inconsistent series names, missing metadata, and MR-tech naming habits.

#### 3. A clear dataset description

The dataset description is also important. AutoBIDSify uses the natural-language description to give the LLM context during the `classify` and `plan` stages.

A short but clear description of the dataset can make the mapping more reliable and reduce ambiguous questions. Useful details include:

* the modality or modalities included;
* the acquisition context, such as task, resting state, anatomical scan, or calibration data;
* known naming conventions;
* expected subject/session/task labels, if known;
* any files or series that should be ignored;
* any manually corrected mappings.

This description can be provided in `notes.md` or through the runtime `--describe` field.

Together, the dataset example, LLM questions, logs, and description help us inspect failed or unclear cases directly and distinguish pipeline issues from missing-description issues.

Please do not share identifiable or sensitive data. If you are unsure whether a dataset can be shared, please only provide de-identified metadata, a simplified folder tree, logs without sensitive paths, and a manually prepared expected BIDS tree.

Please test AutoBIDSify and report issues at:
https://github.com/cotilab/autobidsify/issues

## License

This project is released under the MIT License.
