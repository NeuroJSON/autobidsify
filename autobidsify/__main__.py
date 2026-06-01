#!/usr/bin/env python3
"""
Command-line interface for BIDS Pipeline
NEW: Support for both OpenAI and Qwen (Ollama) models
"""

import argparse
import sys
from pathlib import Path

from autobidsify.stages.ingest import ingest_data
from autobidsify.stages.evidence import build_evidence_bundle
from autobidsify.stages.classification import classify_files
from autobidsify.stages.trio import (
    trio_generate_all,
    generate_dataset_description,
    generate_readme,
    generate_participants
)
from autobidsify.converters.planner import build_bids_plan
from autobidsify.converters.executor import execute_bids_plan
from autobidsify.converters.validators import validate_bids_compatible
from autobidsify.utils import info, warn, fatal, read_json, read_yaml
from autobidsify.constants import QWEN_RECOMMENDED_MODELS


def is_qwen_model(model: str) -> bool:
    """Check if model is a Qwen model."""
    return model.startswith('qwen')


def is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (o1/o3/gpt-5 series)."""
    return (
        model.startswith("o1") or 
        model.startswith("o3") or
        model.startswith("gpt-5")
    )


def validate_model(model: str) -> None:
    """Validate and display model information."""
    if is_qwen_model(model):
        info(f"Using Qwen model (via Ollama): {model}")
        info(f"  Make sure Ollama is running: ollama serve")
        info(f"  Make sure model is pulled: ollama pull {model}")
    elif is_reasoning_model(model):
        info(f"Using OpenAI reasoning model: {model}")
    else:
        info(f"Using OpenAI model: {model}")


def setup_parser():
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="autobidsify",
        description=(
            "autobidsify v0.9.5 — Automated BIDS Standardization Tool\n"
            "Powered by LLM-first architecture.\n"
            "\n"
            "Supports MRI (.dcm, .nii, .nii.gz, .jnii, .bnii),\n"
            "fNIRS (.snirf, .nirs, .mat), and\n"
            "EEG (.edf, .vhdr, .set, .bdf) datasets.\n"
            "Output complies with BIDS specification v1.10.0.\n"
            "\n"
            "Website:  https://neurojson.org/Page/autobidsify\n"
            "Issues:   https://github.com/cotilab/autobidsify/issues"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Run full pipeline in one command
  autobidsify full \\
    --input /path/to/data \\
    --output outputs/my_dataset \\
    --model gpt-4o \\
    --modality mri \\
    --nsubjects 10 \\
    --id-strategy auto \\
    --describe "Your dataset description here"

  # Run step by step
  autobidsify ingest   --input data/ --output outputs/run
  autobidsify evidence --output outputs/run --modality mri
  autobidsify trio     --output outputs/run --model gpt-4o
  autobidsify plan     --output outputs/run --model gpt-4o
  autobidsify execute  --output outputs/run
  autobidsify validate --output outputs/run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPPORTED MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OpenAI (requires OPENAI_API_KEY):
    --model gpt-4o              Recommended, stable
    --model gpt-4o-mini         Faster, cheaper
    --model gpt-5.1             Latest

  Qwen via local Ollama:
    --model qwen3-coder-next:latest     Recommended
    --model qwen3-coder-careful:latest  Recommended
    --model qwen2.5-coder:7b            Not recommended

  Qwen via remote Ollama REST API (no local install required):
    export OLLAMA_BASE_URL=http://your-server.com:11434
    --model qwen3-coder-next:latest

  Qwen via DashScope cloud API (requires DASHSCOPE_API_KEY):
    --model qwen-max / qwen-plus / qwen-turbo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE STAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Stage 1  ingest    Extract or reference raw data
  Stage 2  evidence  Analyze structure, detect subjects
  Stage 3  classify  Separate MRI/fNIRS (mixed modality only)
  Stage 4  trio      Generate dataset_description.json, README, participants.tsv
  Stage 5  plan      Create BIDSPlan.yaml conversion strategy
  Stage 6  execute   Run conversions, output bids_compatible/
  Stage 7  validate  Check BIDS compliance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT VARIABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OPENAI_API_KEY      Required for OpenAI models
  DASHSCOPE_API_KEY   Required for Qwen via DashScope
  OLLAMA_BASE_URL     Remote Ollama server (e.g. http://host:11434)
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')

    # ── full ──────────────────────────────────────────────────────────────────
    full_parser = subparsers.add_parser(
        'full',
        help='Run full pipeline (stages 1-7)',
        description='Run the complete autobidsify pipeline from raw data to validated BIDS dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # MRI dataset
  autobidsify full \\
    --input brain_scans/ \\
    --output outputs/study1 \\
    --model gpt-4o \\
    --modality mri \\
    --nsubjects 30 \\
    --id-strategy numeric \\
    --describe "Single-site T1w MRI study, 30 healthy adults"

  # fNIRS dataset
  autobidsify full \\
    --input fnirs_data/ \\
    --output outputs/fnirs \\
    --model gpt-4o \\
    --modality nirs \\
    --describe "Prefrontal fNIRS, 20 subjects, resting state"

  # Using Qwen via remote Ollama (no local install required)
  export OLLAMA_BASE_URL=http://your-server.com:11434
  autobidsify full \\
    --input data/ \\
    --output outputs/run \\
    --model qwen3-coder-next:latest \\
    --modality mri
"""
    )
    full_parser.add_argument('--input',  type=str, required=True,
                             help='Input data path (directory or archive file)')
    full_parser.add_argument('--output', type=str, required=True,
                             help='Output directory for the BIDS dataset')
    full_parser.add_argument('--model',  type=str, default='gpt-4o',
                             help='LLM model (default: gpt-4o). '
                                  'OpenAI: gpt-4o, gpt-4o-mini, gpt-5.1. '
                                  'Qwen: qwen3-coder-next:latest, qwen2.5-coder:7b')
    full_parser.add_argument('--modality', choices=['mri', 'nirs', 'eeg', 'mixed'],
                             help='Data modality: mri | nirs | mixed '
                                  '(auto-detected if omitted)')
    full_parser.add_argument('--nsubjects', type=int, default=None,
                             help='Number of subjects (auto-detected if omitted)')
    full_parser.add_argument('--describe', type=str,
                             help='Dataset description text — strongly recommended '
                                  'for accurate metadata extraction')
    full_parser.add_argument('--id-strategy', type=str,
                             choices=['auto', 'numeric', 'semantic'],
                             default='auto',
                             help='Subject ID strategy: '
                                  'auto (default) | numeric (sub-1, sub-2) | '
                                  'semantic (preserve original IDs)')

    # ── ingest ────────────────────────────────────────────────────────────────
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Stage 1 — Extract or reference raw data',
        description='Stage 1: Ingest raw data. '
                    'Archives (.zip, .tar.gz) are extracted to _staging/extracted/. '
                    'Directories are referenced in-place (no copying).',
    )
    ingest_parser.add_argument('--input',  type=str, required=True,
                               help='Input path (directory or archive)')
    ingest_parser.add_argument('--output', type=str, required=True,
                               help='Output directory')

    # ── evidence ──────────────────────────────────────────────────────────────
    evidence_parser = subparsers.add_parser(
        'evidence',
        help='Stage 2 — Analyze structure, detect subjects',
        description='Stage 2: Build evidence bundle. '
                    'Scans all files, detects subject identifiers, '
                    'extracts DICOM/NIfTI/SNIRF headers, and collects '
                    'participant metadata evidence. '
                    'Saves _staging/evidence_bundle.json.',
    )
    evidence_parser.add_argument('--output',    type=str, required=True,
                                 help='Output directory (must contain _staging/ingest_info.json)')
    evidence_parser.add_argument('--nsubjects', type=int, default=None,
                                 help='Number of subjects (auto-detected if omitted)')
    evidence_parser.add_argument('--modality',  choices=['mri', 'nirs', 'eeg', 'mixed'],
                                 help='Data modality')
    evidence_parser.add_argument('--describe',  type=str,
                                 help='Dataset description')

    # ── classify ──────────────────────────────────────────────────────────────
    classify_parser = subparsers.add_parser(
        'classify',
        help='Stage 3 — Separate MRI/fNIRS files (mixed modality only)',
        description='Stage 3: Classify files by extension into MRI/fNIRS/unknown pools. '
                    'Only needed for mixed-modality datasets. '
                    'Single-modality datasets skip this stage automatically.',
    )
    classify_parser.add_argument('--output', type=str, required=True,
                                 help='Output directory')

    # ── trio ──────────────────────────────────────────────────────────────────
    trio_parser = subparsers.add_parser(
        'trio',
        help='Stage 4 — Generate BIDS trio files (dataset_description, README, participants)',
        description='Stage 4: Generate the three required BIDS metadata files: '
                    'dataset_description.json, README.md, and participants.tsv. '
                    'Uses LLM to extract metadata from the evidence bundle.',
    )
    trio_parser.add_argument('--output', type=str, required=True,
                             help='Output directory')
    trio_parser.add_argument('--model',  type=str, default='gpt-4o',
                             help='LLM model (default: gpt-4o)')
    trio_parser.add_argument('--file',
                             choices=['dataset_description', 'readme', 'participants', 'all'],
                             default='all',
                             help='Which file to generate (default: all)')

    # ── plan ──────────────────────────────────────────────────────────────────
    plan_parser = subparsers.add_parser(
        'plan',
        help='Stage 5 — Generate BIDSPlan.yaml conversion strategy',
        description='Stage 5: Generate the BIDS conversion plan. '
                    'Python extracts subject IDs from all files; '
                    'LLM determines file mappings, modality assignments, and filename rules. '
                    'Also generates/updates participants.tsv with complete subject list.',
    )
    plan_parser.add_argument('--output', type=str, required=True,
                             help='Output directory')
    plan_parser.add_argument('--model',  type=str, default='gpt-4o',
                             help='LLM model (default: gpt-4o)')
    plan_parser.add_argument('--id-strategy', type=str,
                             choices=['auto', 'numeric', 'semantic'],
                             default='auto',
                             help='Subject ID strategy (default: auto)')

    # ── execute ───────────────────────────────────────────────────────────────
    execute_parser = subparsers.add_parser(
        'execute',
        help='Stage 6 — Execute conversions, output bids_compatible/',
        description='Stage 6: Execute the BIDS conversion plan. '
                    'Converts DICOM→NIfTI (via dcm2niix), JNIfTI→NIfTI, '
                    '.mat/.nirs→SNIRF, and copies SNIRF/NIfTI files. '
                    'Unprocessed files are copied verbatim to bids_compatible/derivatives/.',
    )
    execute_parser.add_argument('--output', type=str, required=True,
                                help='Output directory (must contain _staging/BIDSPlan.yaml)')

    # ── validate ──────────────────────────────────────────────────────────────
    validate_parser = subparsers.add_parser(
        'validate',
        help='Stage 7 — Validate BIDS compliance',
        description='Stage 7: Validate the generated BIDS dataset. '
                    'Uses the official bids-validator (npm install -g bids-validator) '
                    'if available, otherwise falls back to internal validation.',
    )
    validate_parser.add_argument('--output', type=str, required=True,
                                 help='Output directory (must contain bids_compatible/)')

    return parser


def run_full_pipeline(args):
    """Run complete BIDS conversion pipeline."""
    info("=== Starting Full Pipeline v10 ===")
    validate_model(args.model)
    info(f"Subject ID strategy: {args.id_strategy}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    needs_classification = (args.modality == 'mixed' or args.modality is None)
    
    # Stage 1: Ingest
    info("\n[1/7] Ingesting data...")
    ingest_data(args.input, output_dir)
    
    # Stage 2: Evidence
    info("\n[2/7] Building evidence bundle...")
    user_hints = {
        "n_subjects": args.nsubjects,
        "modality_hint": args.modality,
        "user_text": args.describe or ""
    }
    build_evidence_bundle(output_dir, user_hints)
    
    # Stage 3: Classification (if needed)
    if needs_classification:
        info("\n[3/7] Classifying files (mixed modality detected)...")
        classify_files(output_dir)
    else:
        info(f"\n[3/7] Skipping classification (single modality: {args.modality})...")
        info("  ✓ No classification needed for single-modality datasets")
    
    # Stage 4: Trio
    info("\n[4/7] Generating BIDS trio files...")
    bundle = read_json(output_dir / "_staging" / "evidence_bundle.json")
    
    count_source = bundle.get("subject_detection", {}).get("count_source")
    if count_source == "user_provided":
        info(f"✓ Using user-provided subject count: {args.nsubjects}")
    
    dd_result = generate_dataset_description(args.model, bundle, output_dir)
    readme_result = generate_readme(args.model, bundle, output_dir)
    parts_result = generate_participants(args.model, bundle, output_dir)
    
    all_warnings = []
    all_warnings.extend(dd_result.get("warnings", []))
    all_warnings.extend(readme_result.get("warnings", []))
    all_warnings.extend(parts_result.get("warnings", []))
    
    for w in all_warnings:
        warn(f"  {w}")
    
    # Stage 5: Plan
    info("\n[5/7] Generating BIDS plan...")
    trio_status = {
        "dataset_description": (output_dir / "dataset_description.json").exists(),
        "readme": (output_dir / "README.md").exists(),
        "participants": (output_dir / "participants.tsv").exists()
    }
    planning_inputs = {"evidence_bundle": bundle, "trio_status": trio_status}
    
    plan_result = build_bids_plan(args.model, planning_inputs, output_dir, id_strategy=args.id_strategy)
    
    if plan_result.get("status") == "blocked":
        fatal("\n⚠ BLOCKING QUESTIONS DETECTED:")
        for q in plan_result.get("questions", []):
            if q.get("severity") == "block":
                fatal(f"  • {q.get('message')}")
        fatal("\nPlease resolve these issues and re-run the plan command")
        return
    
    if not (output_dir / "participants.tsv").exists():
        warn("WARNING: participants.tsv was not created by Plan stage")
    
    # Stage 6: Execute
    info("\n[6/7] Executing conversions...")
    
    ingest_info = read_json(output_dir / "_staging" / "ingest_info.json")
    actual_data_path = Path(ingest_info.get("actual_data_path", 
                                            output_dir / "_staging" / "extracted"))
    
    plan_dict = read_yaml(output_dir / "_staging" / "BIDSPlan.yaml")
    execute_bids_plan(actual_data_path, output_dir, plan_dict, {})
    
    # Stage 7: Validate
    info("\n[7/7] Validating BIDS dataset...")
    validate_bids_compatible(output_dir)
    
    info("\n=== Pipeline Complete ===")
    info(f"Output: {output_dir / 'bids_compatible'}")


def run_ingest(args):
    """Run ingest stage."""
    info("=== Running Ingest ===")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_data(args.input, output_dir)
    info("✓ Ingest complete")


def run_evidence(args):
    """Run evidence bundle generation."""
    info("=== Building Evidence Bundle ===")
    output_dir = Path(args.output)
    user_hints = {
        "n_subjects": args.nsubjects,
        "modality_hint": args.modality,
        "user_text": args.describe or ""
    }
    build_evidence_bundle(output_dir, user_hints)
    info("✓ Evidence bundle complete")


def run_classify(args):
    """Run file classification."""
    info("=== Classifying Files ===")
    output_dir = Path(args.output)
    classify_files(output_dir)
    info("✓ Classification complete")


def run_trio(args):
    """Run trio file generation."""
    info("=== Generating BIDS Trio ===")
    validate_model(args.model)
    output_dir = Path(args.output)
    
    bundle_path = output_dir / "_staging" / "evidence_bundle.json"
    if not bundle_path.exists():
        fatal(f"Evidence bundle not found: {bundle_path}")
        return
    
    bundle = read_json(bundle_path)
    
    if args.file == 'dataset_description':
        result = generate_dataset_description(args.model, bundle, output_dir)
    elif args.file == 'readme':
        result = generate_readme(args.model, bundle, output_dir)
    elif args.file == 'participants':
        result = generate_participants(args.model, bundle, output_dir)
    else:
        result = trio_generate_all(args.model, bundle, output_dir)
    
    if result.get("warnings"):
        warn("\nWarnings:")
        for w in result["warnings"]:
            warn(f"  {w}")
    
    info("✓ Trio generation complete")


def run_plan(args):
    """Run BIDS plan generation."""
    info("=== Generating BIDS Plan ===")
    validate_model(args.model)
    info(f"Subject ID strategy: {args.id_strategy}")
    
    output_dir = Path(args.output)
    
    bundle_path = output_dir / "_staging" / "evidence_bundle.json"
    if not bundle_path.exists():
        fatal(f"Evidence bundle not found: {bundle_path}")
        return
    
    bundle = read_json(bundle_path)
    
    trio_status = {
        "dataset_description": (output_dir / "dataset_description.json").exists(),
        "readme": (output_dir / "README.md").exists(),
        "participants": (output_dir / "participants.tsv").exists()
    }
    
    planning_inputs = {"evidence_bundle": bundle, "trio_status": trio_status}
    
    result = build_bids_plan(args.model, planning_inputs, output_dir, id_strategy=args.id_strategy)
    
    if result.get("status") == "ok":
        info("✓ BIDS plan generation complete")
    elif result.get("status") == "blocked":
        warn("\n⚠ BLOCKING QUESTIONS DETECTED:")
        for q in result.get("questions", []):
            if q.get("severity") == "block":
                warn(f"  • {q.get('message')}")
        warn("\nPlease resolve these issues and re-run this command")
    else:
        warn("BIDS plan generation encountered errors")


def run_execute(args):
    """Run BIDS plan execution."""
    info("=== Executing BIDS Plan ===")
    output_dir = Path(args.output)
    
    plan_path = output_dir / "_staging" / "BIDSPlan.yaml"
    if not plan_path.exists():
        fatal(f"BIDS plan not found: {plan_path}")
        return
    
    plan_dict = read_yaml(plan_path)
    
    ingest_info_path = output_dir / "_staging" / "ingest_info.json"
    if not ingest_info_path.exists():
        fatal(f"Ingest info not found: {ingest_info_path}")
        return
    
    ingest_info = read_json(ingest_info_path)
    actual_data_path = Path(ingest_info.get("actual_data_path", 
                                            output_dir / "_staging" / "extracted"))
    
    aux_inputs = {}
    
    result = execute_bids_plan(actual_data_path, output_dir, plan_dict, aux_inputs)
    
    info("✓ Execution complete")
    info(f"  BIDS dataset: {result.get('bids_root')}")


def run_validate(args):
    """Run BIDS validation."""
    info("=== Validating BIDS Dataset ===")
    output_dir = Path(args.output)
    
    result = validate_bids_compatible(output_dir)
    
    if result.get("status") == "complete":
        info("")
        info("✓ Validation complete")
    else:
        warn("Validation encountered errors")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'full':
            run_full_pipeline(args)
        elif args.command == 'ingest':
            run_ingest(args)
        elif args.command == 'evidence':
            run_evidence(args)
        elif args.command == 'classify':
            run_classify(args)
        elif args.command == 'trio':
            run_trio(args)
        elif args.command == 'plan':
            run_plan(args)
        elif args.command == 'execute':
            run_execute(args)
        elif args.command == 'validate':
            run_validate(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        warn("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        fatal(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()