#!/usr/bin/env bash

# autobidsify batch validation runner — multi-model
#
# Runs every dataset with every specified model, collects BIDS-validator
# results, and writes ONE consolidated HTML report covering all
# datasets x models.
#
# Directory layout (X = dataset index, matches its position in DATASETS):
#   outputs/autochecks/
#     run1/run1-<label>/   <- dataset 1, model <label>, BIDS output directly here
#     run1/run1-<label2>/
#     run2/run2-<label>/   <- dataset 2
#     ...
#     run8/run8-<label>/
#     validation_report.html   <- single report, all datasets x models
#
# Usage:
#   ./run_all_tests.sh
#
# To add a new dataset:
#   Append a full autobidsify command to DATASETS. Its position in the
#   array determines its run number (1-based). Use OUTPUT_PLACEHOLDER as
#   the --output value — the script substitutes the correct path per model.
#
# To add/remove models:
#   Edit the MODELS array.

set -uo pipefail

# Model list
# The script runs every dataset against every model listed.
MODELS=(
  "gpt-4o"
  # "gpt-4o-mini"
  # "gpt-5.1"
  # "qwen3-coder-next:latest"
  # "qwen3-coder-careful:latest"
)

# Naming: model -> short label (used in paths)
model_label() {
  local m="$1"
  case "$m" in
    "gpt-4o")                     echo "4o" ;;
    "gpt-4o-mini")                echo "4o-mini" ;;
    "gpt-5.1")                    echo "5-1" ;;
    "qwen3-coder-next:latest")    echo "qwen3-next" ;;
    "qwen3-coder-careful:latest") echo "qwen3-careful" ;;
    *)
      # Generic fallback: strip special chars
      echo "$m" | sed 's/[^a-zA-Z0-9]/-/g; s/--*/-/g; s/-$//'
      ;;
  esac
}

OUTPUT_PLACEHOLDER="OUTPUT_PLACEHOLDER"

# Dataset registry
# Each entry is a complete autobidsify full command.
# Use OUTPUT_PLACEHOLDER for --output (required).
# Position in this array = dataset index = run number.
# Final path: outputs/autochecks/run<N>/run<N>-<model-label>

declare -a DATASETS=(

  "autobidsify full \
    --input datasets/1-FRESH-Motor-snirf \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --nsubjects 10 \
    --modality nirs \
    --id-strategy numeric \
    --describe 'References, Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896, In preperation. Paper describes two datasets, but here we only one of them. License: CC0'"

  "autobidsify full \
    --input datasets/2-CamCAN-no-sidebar \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --nsubjects 30 \
    --modality mri \
    --id-strategy semantic \
    --describe 'Cambridge Centre for Ageing and Neuroscience (CamCAN): The Cambridge Centre for Ageing and Neuroscience (Cam-CAN) is a large-scale project using epidemiological, cognitive, and neuroimaging data to understand how individuals can best retain cognitive abilities into old age. There are 5 phases, with data from Phases 1-3 available now. Phase 2 Arm 1 (CC700): MRI T1 653, T2 653, DWI 642. T1 MPRAGE TR 2250 ms TE 2.99 ms 1 mm iso. T2 SPACE TR 2800 ms TE 408 ms 1 mm iso. Shafto et al. (2014). The Cambridge Centre for Ageing and Neuroscience study protocol. BMC Neurology 14(204). doi: 10.1186/s12883-014-0204-1. Here we only use 30 subjects of this dataset.'"

  "autobidsify full \
    --input datasets/3-Visible-Human-dcm \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --modality mri \
    --nsubjects 2 \
    --id-strategy numeric \
    --describe "The NLM Visible Human Project has created publicly-available complete, anatomically detailed, three-dimensional representations of a human male body and a human female body. Specifically, the VHP provides a public-domain library of cross-sectional cryosection, CT, and MRI images obtained from one male cadaver and one female cadaver. The Visible Man data set was publicly released in 1994 and the Visible Woman in 1995. The Visible Human Male data set consists of MRI, CT, and anatomical images. Axial MRI images of the head and neck, and longitudinal sections of the rest of the body were obtained at 4mm intervals. The MRI images are 256 by 256 pixel resolution with each pixel made up of 12 bits of gray tone. The CT data consist of axial CT scans of the entire body taken at 1mm intervals at a pixel resolution of 512 by 512 with each pixel made up of 12 bits of gray tone. The Visible Human Female data set has the same characteristics as the Visible Human Male. However, the axial anatomical images were obtained at 0.33 mm intervals. Spacing in the “Z” dimension was reduced to 0.33mm in order to match the 0.33mm pixel sizing in the “X-Y” plane. As a result, developers interested in three-dimensional reconstructions are able to work with cubic voxels. There are 5,189 anatomical images in the Visible Human Female data set. The data set size is approximately 40 gigabytes."

  "autobidsify full \
    --input datasets/4-openfnirs-parkinson-snirf \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --modality nirs \
    --nsubjects 40 \
    --id-strategy semantic \
    --describe 'This dataset is a publicly available fNIRS dataset (Guevara et al., 2023; DOI: 10.5281/zenodo.7966830) that investigates cortical activity and functional connectivity in Parkinson disease. It includes recordings from 20 PD patients and 20 age- and sex-matched healthy controls. Three conditions: 10-second finger-tapping task, 2-minute walking task, 6-minute resting-state session. Each subject has three SNIRF files: (1) resting state: file named 1_resting_seg_1.snirf, BIDS task name: rest, (2) finger tapping: file named 2_finger_tapping.snirf, BIDS task name: fingertapping, (3) walking: file named 3_walking.snirf, BIDS task name: walking. All three files should be included for each subject. CC BY 4.0 license.'"

  "autobidsify full \
    --input datasets/5-RSFC-harvard_dataverse-nirs \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --modality nirs \
    --nsubjects 13 \
    --id-strategy numeric \
    --describe 'Replication Data for: fNIRS RSFC in Tinnitus, San Juan, Juan, 2017, https://doi.org/10.7910/DVN/ZNZZBV, Harvard Dataverse V1. Manuscript: Tinnitus Alters Resting State Functional Connectivity (RSFC) in Human Auditory and Non-Auditory Brain Regions as Measured by Functional Near-Infrared Spectroscopy (fNIRS). San Juan J, Hu X-S, Issa M, Bisconti S, Kovelman I, Kileny P, et al. (2017) PLoS ONE 12(6): e0179150. doi: 10.1371/journal.pone.0179150'"

  "autobidsify full \
    --input datasets/6-figshare-9783755-mat \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --modality nirs \
    --nsubjects 30 \
    --id-strategy numeric \
    --describe 'Open Access fNIRS Dataset for Classification of Unilateral Finger- and Foot-Tapping. https://doi.org/10.6084/m9.figshare.9783755, posted 2019-12-04, authored by Sujin Bak, Jinwoo Park, Jaeyoung Shin, Jichai Jeong. Neurosciences not elsewhere classified. Licence: CC BY 4.0'"

  "autobidsify full \
    --input datasets/7-mental-arithmetic-mat \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --modality nirs \
    --nsubjects 8 \
    --id-strategy numeric \
    --describe 'Mental arithmetic (003-2014) on BNCI Horizon 2020. Participants: 8. Signals: 52 fNIRS. Data: S01-S08. License: CC BY-ND 4.0. Licensor: Institute for Knowledge Discovery, Graz University of Technology. Bauernfeind G, Scherer R, Pfurtscheller G et al. Single-trial classification of antagonistic oxyhemoglobin responses during mental arithmetic. Med Biol Eng Comput 49, 979-984 (2011). https://doi.org/10.1007/s11517-011-0792-5'"

  "autobidsify full \
    --input datasets/8-eeg-mental-arithmetic-edf \
    --output OUTPUT_PLACEHOLDER \
    --model MODEL_PLACEHOLDER \
    --modality eeg \
    --nsubjects 36 \
    --id-strategy numeric \
    --describe 'EEG During Mental Arithmetic Tasks. Published Dec 2018. EEG recordings before and during mental arithmetic tasks. Neurocom EEG 23-channel system. Electrodes placed according to International 10/20 scheme. Each subject has 2 EDF files: _1 suffix = background EEG before task, _2 suffix = EEG during arithmetic task. Group G: 24 subjects good quality count. Group B: 12 subjects bad quality count. DOI: https://doi.org/10.13026/C2JQ1P. License: Open Data Commons Attribution License v1.0. Zyma I et al. Electroencephalograms during Mental Arithmetic Task Performance. Data. 2019; 4(1):14.'"

)

# Helpers

# Strip ANSI/VT100 escape sequences (color codes, cursor moves, etc.)
# so they do not leak into the HTML report as garbage characters.
strip_ansi() {
  sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g; s/\x1b\][^\x07]*\x07//g; s/\r//g'
}

colorize_log_lines() {
  # Input is assumed already ANSI-stripped.
  while IFS= read -r line; do
    esc=$(printf '%s' "$line" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')
    if echo "$line" | grep -qiE "error|fatal|failed|✗|non-bids-compliant|non-compliant"; then
      printf '<span class="error-line">%s</span>\n' "$esc"
    elif echo "$line" | grep -qiE "warning|warn|⚠"; then
      printf '<span class="warn-line">%s</span>\n' "$esc"
    elif echo "$line" | grep -qiE "✓|all.*compliant|pipeline complete|pass"; then
      printf '<span class="ok-line">%s</span>\n' "$esc"
    else
      printf '%s\n' "$esc"
    fi
  done
}

extract_input_name() {
  # Extract dataset name from --input value
  echo "$1" | grep -oP '(?<=--input\s)\S+' | head -1 | xargs basename
}

RUN_ROOT="outputs/autochecks"
REPORT_FILE="${RUN_ROOT}/validation_report.html"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

mkdir -p "${RUN_ROOT}"

echo ""
echo "============================================================"
echo "        autobidsify batch validation runner"
echo "============================================================"
printf "  Timestamp : %s\n" "${TIMESTAMP}"
printf "  Models    : %d  (%s)\n" "${#MODELS[@]}" "${MODELS[*]}"
printf "  Datasets  : %d\n" "${#DATASETS[@]}"
printf "  Output    : %s\n\n" "${RUN_ROOT}"

# Result accumulators (flat lists, one entry per dataset x model combo)
declare -a ALL_RUN=()
declare -a ALL_NAMES=()
declare -a ALL_MODELS=()
declare -a ALL_MODALITIES=()
declare -a ALL_NSUBJECTS=()
declare -a ALL_STATUSES=()
declare -a ALL_CHECKED=()
declare -a ALL_NONCOMPLIANT=()
declare -a ALL_PIPELINE_LOGS=()
declare -a ALL_VALIDATOR_LOGS=()

# Outer loop: datasets (index N = run number)
DSIDX=0
for CMD_TEMPLATE in "${DATASETS[@]}"; do
  DSIDX=$((DSIDX + 1))
  RUN_TAG="run${DSIDX}"

  DS_NAME=$(extract_input_name "$CMD_TEMPLATE")
  MODALITY=$(echo "$CMD_TEMPLATE" | grep -oP '(?<=--modality\s)\S+' | head -1)
  NSUBJECTS=$(echo "$CMD_TEMPLATE" | grep -oP '(?<=--nsubjects\s)\S+' | head -1)
  NSUBJECTS=${NSUBJECTS:-"auto"}

  echo "============================================================"
  printf "  [%d/%d] %s  (%s)\n\n" "${DSIDX}" "${#DATASETS[@]}" "${DS_NAME}" "${RUN_TAG}"

  # Inner loop: models
  for MODEL in "${MODELS[@]}"; do
    LABEL=$(model_label "$MODEL")
    # BIDS output goes DIRECTLY here — no extra dataset subdirectory.
    OUT_DIR="${RUN_ROOT}/${RUN_TAG}/${RUN_TAG}-${LABEL}"
    mkdir -p "${OUT_DIR}"

    # Substitute placeholders
    CMD="${CMD_TEMPLATE/OUTPUT_PLACEHOLDER/${OUT_DIR}}"
    CMD="${CMD/MODEL_PLACEHOLDER/${MODEL}}"

    printf "    -> model %s  (label %s)\n" "${MODEL}" "${LABEL}"

    # Run; strip ANSI immediately so stored logs are clean.
    PIPELINE_LOG=$(eval "$CMD" 2>&1 | strip_ansi) || true
    echo "$PIPELINE_LOG" | tail -5 | sed 's/^/      /'

    # Extract validator section
    VALIDATOR_LOG=$(echo "$PIPELINE_LOG" | \
      awk '/\[7\/7\] Validating BIDS dataset/,/=== Pipeline Complete ===/' | \
      grep -v "^$" | head -80)

    # Parse metrics
    CHECKED=$(echo "$VALIDATOR_LOG" | grep -oP 'checked \K[0-9]+' | head -1)
    NONCOMPLIANT=$(echo "$VALIDATOR_LOG" | grep -oP '⚠ \K[0-9]+(?= file)' | head -1)
    CHECKED=${CHECKED:-"?"}
    NONCOMPLIANT=${NONCOMPLIANT:-"0"}

    if echo "$VALIDATOR_LOG" | grep -q "All.*checked files have BIDS-compliant"; then
      STATUS="PASS"
    elif [ "${NONCOMPLIANT}" != "0" ] && [ "${NONCOMPLIANT}" != "?" ]; then
      STATUS="FAIL"
    elif echo "$PIPELINE_LOG" | grep -qiE "Traceback|Fatal|Exception"; then
      STATUS="ERROR"; NONCOMPLIANT="?"
    else
      STATUS="PASS"
    fi

    printf "       => %s  (checked=%s, non-compliant=%s)\n\n" \
      "${STATUS}" "${CHECKED}" "${NONCOMPLIANT}"

    ALL_RUN+=("${RUN_TAG}")
    ALL_NAMES+=("${DS_NAME}")
    ALL_MODELS+=("${MODEL}")
    ALL_MODALITIES+=("${MODALITY}")
    ALL_NSUBJECTS+=("${NSUBJECTS}")
    ALL_STATUSES+=("${STATUS}")
    ALL_CHECKED+=("${CHECKED}")
    ALL_NONCOMPLIANT+=("${NONCOMPLIANT}")
    ALL_PIPELINE_LOGS+=("${PIPELINE_LOG}")
    ALL_VALIDATOR_LOGS+=("${VALIDATOR_LOG}")
  done
done

# Generate the single consolidated HTML report
{
cat << 'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>autobidsify Validation Report</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     margin:0;padding:24px 32px;background:#f5f5f5;color:#222;line-height:1.5}
h1{color:#1a1a2e;margin-bottom:4px;font-size:1.8em}
.meta{color:#666;font-size:.9em;margin-bottom:28px}
.card{background:#fff;border-radius:10px;padding:20px 24px;
      margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.09)}
.card-header{display:flex;align-items:center;gap:12px;
             margin-bottom:14px;flex-wrap:wrap}
.badge{padding:4px 12px;border-radius:12px;font-size:.78em;font-weight:700;
       white-space:nowrap;letter-spacing:.02em}
.badge-pass{background:#d4edda;color:#155724}
.badge-fail{background:#f8d7da;color:#721c24}
.badge-error{background:#fff3cd;color:#856404}
h2{margin:0;font-size:1.05em;color:#1a1a2e}
.section-label{font-weight:700;color:#555;font-size:.78em;text-transform:uppercase;
               letter-spacing:.06em;margin:14px 0 4px}
pre{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:5px;
    padding:10px 14px;font-size:.8em;white-space:pre-wrap;word-break:break-all;
    margin:0;max-height:340px;overflow-y:auto;
    font-family:"SFMono-Regular",Consolas,monospace}
.error-line{color:#c0392b;font-weight:700}
.warn-line{color:#d35400}
.ok-line{color:#1e8449}
table{width:100%;border-collapse:collapse;margin-top:12px}
thead th{background:#1a1a2e;color:#fff;padding:9px 14px;
         text-align:left;font-size:.82em;font-weight:600}
tbody td{padding:8px 14px;border-bottom:1px solid #eee;font-size:.86em}
tbody tr:last-child td{border-bottom:none}
.pass-row{background:#f0fff4}
.fail-row{background:#fff5f5}
.error-row{background:#fffbf0}
hr{border:none;border-top:1px solid #ddd;margin:28px 0}
code.tag{background:#eef;border-radius:4px;padding:1px 6px;font-size:.85em}
</style>
</head>
<body>
HTML

  printf '<h1>autobidsify Validation Report</h1>\n'
  printf '<p class="meta">Generated: %s &nbsp;|&nbsp; Models: <strong>%s</strong> &nbsp;|&nbsp; Datasets: %d &nbsp;|&nbsp; Runs: %d</p>\n' \
    "${TIMESTAMP}" "${MODELS[*]}" "${#DATASETS[@]}" "${#ALL_NAMES[@]}"

  printf '<hr>\n<h2 style="margin-bottom:12px">Summary</h2>\n'
  printf '<table><thead><tr>\n'
  printf '  <th>Run</th><th>Dataset</th><th>Model</th><th>Modality</th><th>Subjects</th>'
  printf '<th>Files checked</th><th>Non-compliant</th><th>Status</th>\n'
  printf '</tr></thead><tbody>\n'

  for i in "${!ALL_NAMES[@]}"; do
    st="${ALL_STATUSES[$i]}"
    rc="pass-row"; [[ "$st" == "FAIL" ]] && rc="fail-row"
    [[ "$st" == "ERROR" ]] && rc="error-row"
    bc="badge-pass"; [[ "$st" == "FAIL" ]] && bc="badge-fail"
    [[ "$st" == "ERROR" ]] && bc="badge-error"
    printf '<tr class="%s"><td><code class="tag">%s</code></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>' \
      "$rc" "${ALL_RUN[$i]}" "${ALL_NAMES[$i]}" "${ALL_MODELS[$i]}" "${ALL_MODALITIES[$i]}" "${ALL_NSUBJECTS[$i]}"
    printf '<td>%s</td><td>%s</td>' "${ALL_CHECKED[$i]}" "${ALL_NONCOMPLIANT[$i]}"
    printf '<td><span class="badge %s">%s</span></td></tr>\n' "$bc" "$st"
  done

  printf '</tbody></table>\n<hr>\n'
  printf '<h2 style="margin-bottom:16px">Dataset Details</h2>\n'

  for i in "${!ALL_NAMES[@]}"; do
    st="${ALL_STATUSES[$i]}"
    bc="badge-pass"; [[ "$st" == "FAIL" ]] && bc="badge-fail"
    [[ "$st" == "ERROR" ]] && bc="badge-error"

    printf '<div class="card">\n'
    printf '  <div class="card-header">\n'
    printf '    <span class="badge %s">%s</span>\n' "$bc" "$st"
    printf '    <h2>%s &nbsp;·&nbsp; %s</h2>\n' "${ALL_RUN[$i]}" "${ALL_NAMES[$i]}"
    printf '    <span style="color:#888;font-size:.85em">%s &nbsp;|&nbsp; %s &nbsp;|&nbsp; %s subjects</span>\n' \
      "${ALL_MODELS[$i]}" "${ALL_MODALITIES[$i]}" "${ALL_NSUBJECTS[$i]}"
    printf '  </div>\n'

    printf '  <div class="section-label">Pipeline log (last 40 lines)</div>\n  <pre>\n'
    echo "${ALL_PIPELINE_LOGS[$i]}" | tail -40 | colorize_log_lines
    printf '  </pre>\n'

    printf '  <div class="section-label">BIDS Validator output</div>\n  <pre>\n'
    if [ -n "${ALL_VALIDATOR_LOGS[$i]}" ]; then
      echo "${ALL_VALIDATOR_LOGS[$i]}" | colorize_log_lines
    else
      printf '<span class="warn-line">Validator section not found in pipeline log.</span>\n'
    fi
    printf '  </pre>\n</div>\n'
  done

  printf '</body></html>\n'

} > "${REPORT_FILE}"

echo "============================================================"
printf "  All runs complete.\n"
printf "  Report : %s\n" "${REPORT_FILE}"
printf "  Layout :\n"
printf "    %s/\n" "${RUN_ROOT}"
DSIDX=0
for CMD_TEMPLATE in "${DATASETS[@]}"; do
  DSIDX=$((DSIDX + 1))
  printf "      run%d/\n" "${DSIDX}"
  for MODEL in "${MODELS[@]}"; do
    LABEL=$(model_label "$MODEL")
    printf "        run%d-%s/   (bids_compatible/, _staging/)\n" "${DSIDX}" "${LABEL}"
  done
done
printf "      validation_report.html\n"
echo "============================================================"