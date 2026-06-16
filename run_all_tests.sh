#!/usr/bin/env bash
# AutoBIDSify batch validation runner — multi-model
#
# Runs every dataset against every model, writes one consolidated HTML report.
#
# Output layout:
#   outputs/autochecks/
#     runX/runX-<label>/   (X = dataset index)
#     validation_report.html
#
# To add a dataset: append a command to DATASETS, use OUTPUT_PLACEHOLDER and
#                   MODEL_PLACEHOLDER as placeholders.
# To add a model:   append to MODELS and add a case in model_label().

set -u

# ── Models ───────────────────────────────────────────────────────────────────
MODELS=(
  "gpt-4o"
  # "gpt-4o-mini"
  # "gpt-5.1"
  # "qwen3-coder-next:latest"
  # "qwen3-coder-careful:latest"
)

model_label() {
  local m
  m="$1"
  case "$m" in
    "gpt-4o")                     echo "4o" ;;
    "gpt-4o-mini")                echo "4o-mini" ;;
    "gpt-5.1")                    echo "5-1" ;;
    "qwen3-coder-next:latest")    echo "qwen3-next" ;;
    "qwen3-coder-careful:latest") echo "qwen3-careful" ;;
    *) echo "$m" | sed 's/[^a-zA-Z0-9]/-/g' | sed 's/--*/-/g' | sed 's/-$//' ;;
  esac
}

# ── Dataset registry ──────────────────────────────────────────────────────────
# Use OUTPUT_PLACEHOLDER for --output and MODEL_PLACEHOLDER for --model.
# Use single quotes for --describe text. Position = run number.

DATASETS=(
  "autobidsify full --input datasets/1-FRESH-Motor-snirf --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --nsubjects 10 --modality nirs --id-strategy numeric --describe 'MNE-BIDS dataset. Appelhoff et al. (2019). License: CC0'"

  "autobidsify full --input datasets/2-CamCAN-no-sidecar --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --nsubjects 30 --modality mri --id-strategy semantic --describe 'Cambridge Centre for Ageing and Neuroscience CamCAN. Phase 2 Arm 1 CC700: MRI T1 653, T2 653, DWI 642. T1 MPRAGE TR 2250ms TE 2.99ms 1mm iso. Shafto et al. 2014. BMC Neurology 14:204. Here we only use 30 subjects.'"

  "autobidsify full --input datasets/3-Visible-Human-dcm --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --modality mri --nsubjects 2 --id-strategy numeric --describe 'NLM Visible Human Project. Complete anatomically detailed 3D representations of human male and female body. Public-domain CT and MRI from one male cadaver and one female cadaver. VHM released 1994, VHF 1995. CT axial scans at 1mm intervals, 512x512 pixel, 12 bits per pixel.'"

  "autobidsify full --input datasets/4-openfnirs-parkinson-snirf --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --modality nirs --nsubjects 40 --id-strategy semantic --describe 'fNIRS dataset Guevara et al 2023 DOI 10.5281/zenodo.7966830. 20 PD patients and 20 healthy controls. Three tasks: resting state file 1_resting_seg_1.snirf task rest, finger tapping file 2_finger_tapping.snirf task fingertapping, walking file 3_walking.snirf task walking. CC BY 4.0.'"

  "autobidsify full --input datasets/5-RSFC-harvard_dataverse-nirs --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --modality nirs --nsubjects 13 --id-strategy numeric --describe 'fNIRS RSFC in Tinnitus. San Juan et al 2017. Harvard Dataverse DOI 10.7910/DVN/ZNZZBV. PLoS ONE 12(6): e0179150.'"

  "autobidsify full --input datasets/6-figshare-9783755-mat --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --modality nirs --nsubjects 30 --id-strategy numeric --describe 'Open Access fNIRS Dataset for Classification of Unilateral Finger and Foot Tapping. DOI 10.6084/m9.figshare.9783755. Bak et al 2019. CC BY 4.0.'"

  "autobidsify full --input datasets/7-mental-arithmetic-mat --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --modality nirs --nsubjects 8 --id-strategy numeric --describe 'Mental arithmetic 003-2014 BNCI Horizon 2020. 8 participants, 52 fNIRS channels. Bauernfeind et al 2011. CC BY-ND 4.0.'"

  "autobidsify full --input datasets/8-eeg-mental-arithmetic-edf --output OUTPUT_PLACEHOLDER --model MODEL_PLACEHOLDER --modality eeg --nsubjects 36 --id-strategy numeric --describe 'EEG During Mental Arithmetic Tasks. Published Dec 2018. 23-channel EEG, International 10/20 scheme. Each subject has 2 EDF files: _1 background EEG before task, _2 EEG during arithmetic task. Group G 24 subjects good count, Group B 12 subjects bad count. DOI 10.13026/C2JQ1P. License Open Data Commons Attribution License v1.0. Zyma et al 2019.'"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

strip_ansi() {
  sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g; s/\r//g'
}

html_escape() {
  printf '%s' "$1" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'
}

colorize_line() {
  local line esc class
  line="$1"
  esc=$(html_escape "$line")
  class=""
  if echo "$line" | grep -qi "error" ||
     echo "$line" | grep -qi "fatal" ||
     echo "$line" | grep -qi "failed" ||
     echo "$line" | grep -qi "non-bids" ||
     echo "$line" | grep -qi "non-compliant"; then
    class="error-line"
  elif echo "$line" | grep -qi "warning" ||
       echo "$line" | grep -qi "warn"; then
    class="warn-line"
  elif echo "$line" | grep -qi "compliant" ||
       echo "$line" | grep -qi "complete" ||
       echo "$line" | grep -qi "pass"; then
    class="ok-line"
  fi
  if [ -n "$class" ]; then
    printf '<span class="%s">%s</span>\n' "$class" "$esc"
  else
    printf '%s\n' "$esc"
  fi
}

colorize_log() {
  while IFS= read -r line; do
    colorize_line "$line"
  done
}

extract_name() {
  echo "$1" | grep -oP '(?<=--input\s)\S+' | head -1 | xargs basename
}

extract_field() {
  echo "$1" | grep -oP "(?<=--${2}\s)\S+" | head -1
}

# ── Setup ─────────────────────────────────────────────────────────────────────

RUN_ROOT="outputs/autochecks"
REPORT_FILE="${RUN_ROOT}/validation_report.html"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p "${RUN_ROOT}"

NUM_DATASETS="${#DATASETS[@]}"
NUM_MODELS="${#MODELS[@]}"

echo ""
echo "============================================================"
echo "  AutoBIDSify batch validation runner"
echo "============================================================"
printf "  Timestamp : %s\n" "${TIMESTAMP}"
printf "  Models    : %d\n" "${NUM_MODELS}"
printf "  Datasets  : %d\n" "${NUM_DATASETS}"
printf "  Output    : %s\n\n" "${RUN_ROOT}"

# ── Collect arrays ────────────────────────────────────────────────────────────

ALL_RUN=()
ALL_NAMES=()
ALL_MODELS=()
ALL_MODALITIES=()
ALL_NSUBJECTS=()
ALL_STATUSES=()
ALL_CHECKED=()
ALL_NONCOMPLIANT=()
ALL_PIPELINE_LOGS=()
ALL_VALIDATOR_LOGS=()

DSIDX=0
for CMD_TEMPLATE in "${DATASETS[@]}"; do
  DSIDX=$((DSIDX + 1))
  RUN_TAG="run${DSIDX}"
  DS_NAME=$(extract_name "$CMD_TEMPLATE")
  MODALITY=$(extract_field "$CMD_TEMPLATE" "modality")
  NSUBJECTS=$(extract_field "$CMD_TEMPLATE" "nsubjects")

  for MODEL in "${MODELS[@]}"; do
    LABEL=$(model_label "$MODEL")
    OUT_DIR="${RUN_ROOT}/${RUN_TAG}/${RUN_TAG}-${LABEL}"
    mkdir -p "${OUT_DIR}"

    CMD="${CMD_TEMPLATE/OUTPUT_PLACEHOLDER/${OUT_DIR}}"
    CMD="${CMD/MODEL_PLACEHOLDER/${MODEL}}"

    echo "------------------------------------------------------------"
    printf "  [run%d] %s  |  model: %s\n" "${DSIDX}" "${DS_NAME}" "${MODEL}"

    PIPELINE_LOG=$(eval "$CMD" 2>&1 | strip_ansi) || true
    echo "$PIPELINE_LOG" | tail -5 | sed 's/^/    /'

    VALIDATOR_LOG=$(echo "$PIPELINE_LOG" | \
      awk '/\[7\/7\] Validating BIDS dataset/,/=== Pipeline Complete ===/' | \
      grep -v "^$" | head -80) || true

    CHECKED=$(echo "$VALIDATOR_LOG" | grep -oP 'checked \K[0-9]+' | head -1) || true
    NONCOMPLIANT=$(echo "$VALIDATOR_LOG" | grep -oP '⚠ \K[0-9]+(?= file)' | head -1) || true
    CHECKED="${CHECKED:-?}"
    NONCOMPLIANT="${NONCOMPLIANT:-0}"

    STATUS="PASS"
    if echo "$VALIDATOR_LOG" | grep -q "All.*checked files have BIDS-compliant"; then
      STATUS="PASS"
    elif [ "${NONCOMPLIANT}" != "0" ] && [ "${NONCOMPLIANT}" != "?" ]; then
      STATUS="FAIL"
    elif echo "$PIPELINE_LOG" | grep -qi "traceback" ||
         echo "$PIPELINE_LOG" | grep -qi "fatal"; then
      STATUS="ERROR"
      NONCOMPLIANT="?"
    fi

    printf "    => %s  (checked=%s, non-compliant=%s)\n\n" \
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

# ── Generate HTML report ──────────────────────────────────────────────────────

{
  printf '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
  printf '<title>AutoBIDSify Validation Report</title>\n<style>\n'
  printf 'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:24px 32px;background:#f5f5f5;color:#222;line-height:1.5}\n'
  printf 'h1{color:#1a1a2e;margin-bottom:4px;font-size:1.8em}\n'
  printf '.meta{color:#666;font-size:.9em;margin-bottom:28px}\n'
  printf '.card{background:#fff;border-radius:10px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.09)}\n'
  printf '.card-header{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap}\n'
  printf '.badge{padding:4px 12px;border-radius:12px;font-size:.78em;font-weight:700;white-space:nowrap}\n'
  printf '.badge-pass{background:#d4edda;color:#155724}\n'
  printf '.badge-fail{background:#f8d7da;color:#721c24}\n'
  printf '.badge-error{background:#fff3cd;color:#856404}\n'
  printf 'h2{margin:0;font-size:1.05em;color:#1a1a2e}\n'
  printf '.section-label{font-weight:700;color:#555;font-size:.78em;text-transform:uppercase;letter-spacing:.06em;margin:14px 0 4px}\n'
  printf 'pre{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:5px;padding:10px 14px;font-size:.8em;white-space:pre-wrap;word-break:break-all;margin:0;max-height:600px;overflow-y:auto;font-family:"SFMono-Regular",Consolas,monospace}\n'
  printf '.error-line{color:#c0392b;font-weight:700}\n'
  printf '.warn-line{color:#d35400}\n'
  printf '.ok-line{color:#1e8449}\n'
  printf 'table{width:100%%;border-collapse:collapse;margin-top:12px}\n'
  printf 'thead th{background:#1a1a2e;color:#fff;padding:9px 14px;text-align:left;font-size:.82em;font-weight:600}\n'
  printf 'tbody td{padding:8px 14px;border-bottom:1px solid #eee;font-size:.86em}\n'
  printf 'tbody tr:last-child td{border-bottom:none}\n'
  printf '.pass-row{background:#f0fff4}\n.fail-row{background:#fff5f5}\n.error-row{background:#fffbf0}\n'
  printf 'hr{border:none;border-top:1px solid #ddd;margin:28px 0}\n'
  printf 'code.tag{background:#eef;border-radius:4px;padding:1px 6px;font-size:.85em}\n'
  printf '</style>\n</head>\n<body>\n'

  printf '<h1>AutoBIDSify Validation Report</h1>\n'
  printf '<p class="meta">Generated: %s &nbsp;|&nbsp; Datasets: %d &nbsp;|&nbsp; Runs: %d</p>\n' \
    "${TIMESTAMP}" "${NUM_DATASETS}" "${#ALL_NAMES[@]}"

  printf '<hr>\n<h2 style="margin-bottom:12px">Summary</h2>\n'
  printf '<table><thead><tr>'
  printf '<th>Run</th><th>Dataset</th><th>Model</th><th>Modality</th>'
  printf '<th>Subjects</th><th>Files checked</th><th>Non-compliant</th><th>Status</th>'
  printf '</tr></thead><tbody>\n'

  for i in "${!ALL_NAMES[@]}"; do
    st="${ALL_STATUSES[$i]}"
    rc="pass-row"
    if [ "$st" = "FAIL" ]; then rc="fail-row"; fi
    if [ "$st" = "ERROR" ]; then rc="error-row"; fi
    bc="badge-pass"
    if [ "$st" = "FAIL" ]; then bc="badge-fail"; fi
    if [ "$st" = "ERROR" ]; then bc="badge-error"; fi
    printf '<tr class="%s">' "$rc"
    printf '<td><code class="tag">%s</code></td>' "${ALL_RUN[$i]}"
    printf '<td>%s</td><td>%s</td><td>%s</td><td>%s</td>' \
      "${ALL_NAMES[$i]}" "${ALL_MODELS[$i]}" "${ALL_MODALITIES[$i]}" "${ALL_NSUBJECTS[$i]}"
    printf '<td>%s</td><td>%s</td>' "${ALL_CHECKED[$i]}" "${ALL_NONCOMPLIANT[$i]}"
    printf '<td><span class="badge %s">%s</span></td></tr>\n' "$bc" "$st"
  done

  printf '</tbody></table>\n<hr>\n'
  printf '<h2 style="margin-bottom:16px">Dataset Details</h2>\n'

  for i in "${!ALL_NAMES[@]}"; do
    st="${ALL_STATUSES[$i]}"
    bc="badge-pass"
    if [ "$st" = "FAIL" ]; then bc="badge-fail"; fi
    if [ "$st" = "ERROR" ]; then bc="badge-error"; fi

    printf '<div class="card">\n'
    printf '  <div class="card-header">\n'
    printf '    <span class="badge %s">%s</span>\n' "$bc" "$st"
    printf '    <h2>%s &nbsp;|&nbsp; %s</h2>\n' "${ALL_RUN[$i]}" "${ALL_NAMES[$i]}"
    printf '    <span style="color:#888;font-size:.85em">%s &nbsp;|&nbsp; %s &nbsp;|&nbsp; %s subjects</span>\n' \
      "${ALL_MODELS[$i]}" "${ALL_MODALITIES[$i]}" "${ALL_NSUBJECTS[$i]}"
    printf '  </div>\n'

    printf '  <div class="section-label">Pipeline log</div>\n'
    printf '  <pre>\n'
    echo "${ALL_PIPELINE_LOGS[$i]}" | awk '/\[7\/7\] Validating BIDS dataset/{exit} {print}' | colorize_log
    printf '  </pre>\n'

    printf '  <div class="section-label">BIDS Validator output</div>\n'
    printf '  <pre>\n'
    if [ -n "${ALL_VALIDATOR_LOGS[$i]}" ]; then
      echo "${ALL_VALIDATOR_LOGS[$i]}" | colorize_log
    else
      printf '<span class="warn-line">Validator section not found in pipeline log.</span>\n'
    fi
    printf '  </pre>\n'
    printf '</div>\n'
  done

  printf '</body></html>\n'

} > "${REPORT_FILE}"

echo "============================================================"
printf "  All runs complete.\n"
printf "  Report: %s\n" "${REPORT_FILE}"
echo "============================================================"