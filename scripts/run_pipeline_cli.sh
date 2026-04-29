#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pipeline_cli.sh --input PATH [options]

Required:
  --input PATH                      Input JSONL dataset.

Common options:
  --mode threshold|fixed|random     Segmentation mode. Default: threshold.
  --run-dir PATH                    Output run directory. Default: runs/<mode>-<size>.
  --model NAME_OR_PATH              Model/tokenizer path. Default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B.
  --reference PATH                  Reference train JSONL.
  --gpu-ids LIST                    GPU list, for example 0,1,2,3. Default: 0,1,2,3.
  --limit-rows N                    Smoke-test row limit.
  --backend hf|sglang               Scoring backend for threshold mode. Default: hf.
  --long-sample-policy window|skip  Overlong sample policy. Default: window.
  --min-step-tokens N               Minimum token gap between boundaries. Default: 12.

Mode presets:
  --size short|base|long            fixed: 64/128/256 tokens; random: 64-128/64-256/128-512. Default: base.
  --fixed-tokens N                  Override fixed token interval.
  --random-min N                    Override random minimum segment tokens.
  --random-max N                    Override random maximum segment tokens.
  --random-seed N                   Override random seed. Default: 42.
  --tau-key KEY                     Threshold tau key for full pipeline. Default: q_0.0100.
  --tau-value VALUE                 Threshold tau value for manual override.

Examples:
  bash scripts/run_pipeline_cli.sh --input /data/bs17k.jsonl --mode fixed --size base
  bash scripts/run_pipeline_cli.sh --input /data/bs17k.jsonl --mode random --size long --gpu-ids 0,1
  bash scripts/run_pipeline_cli.sh --input /data/bs17k.jsonl --mode threshold --tau-key q_0.0100
EOF
}

MODE="threshold"
INPUT=""
RUN_DIR=""
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
REFERENCE_TRAIN_JSONL=""
GPU_IDS="0,1,2,3"
BACKEND="hf"
LIMIT_ROWS=""
LONG_SAMPLE_POLICY="window"
SIZE="base"
FIXED_SEGMENT_TOKENS=""
RANDOM_MIN_SEGMENT_TOKENS=""
RANDOM_MAX_SEGMENT_TOKENS=""
RANDOM_SEED="42"
TAU_KEY="q_0.0100"
TAU_VALUE=""
MIN_STEP_TOKENS="${MIN_STEP_TOKENS:-12}"

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "ERROR: ${option} requires a value." >&2
    exit 1
  fi
}

require_positive_int() {
  local option="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${option} must be a positive integer." >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --input)
      require_value "$1" "${2:-}"
      INPUT="$2"
      shift 2
      ;;
    --mode)
      require_value "$1" "${2:-}"
      MODE="$2"
      shift 2
      ;;
    --run-dir)
      require_value "$1" "${2:-}"
      RUN_DIR="$2"
      shift 2
      ;;
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --reference)
      require_value "$1" "${2:-}"
      REFERENCE_TRAIN_JSONL="$2"
      shift 2
      ;;
    --gpu-ids)
      require_value "$1" "${2:-}"
      GPU_IDS="$2"
      shift 2
      ;;
    --backend)
      require_value "$1" "${2:-}"
      BACKEND="$2"
      shift 2
      ;;
    --limit-rows)
      require_value "$1" "${2:-}"
      LIMIT_ROWS="$2"
      shift 2
      ;;
    --long-sample-policy)
      require_value "$1" "${2:-}"
      LONG_SAMPLE_POLICY="$2"
      shift 2
      ;;
    --min-step-tokens)
      require_value "$1" "${2:-}"
      MIN_STEP_TOKENS="$2"
      shift 2
      ;;
    --size)
      require_value "$1" "${2:-}"
      SIZE="$2"
      shift 2
      ;;
    --fixed-tokens)
      require_value "$1" "${2:-}"
      FIXED_SEGMENT_TOKENS="$2"
      shift 2
      ;;
    --random-min)
      require_value "$1" "${2:-}"
      RANDOM_MIN_SEGMENT_TOKENS="$2"
      shift 2
      ;;
    --random-max)
      require_value "$1" "${2:-}"
      RANDOM_MAX_SEGMENT_TOKENS="$2"
      shift 2
      ;;
    --random-seed)
      require_value "$1" "${2:-}"
      RANDOM_SEED="$2"
      shift 2
      ;;
    --tau-key)
      require_value "$1" "${2:-}"
      TAU_KEY="$2"
      shift 2
      ;;
    --tau-value)
      require_value "$1" "${2:-}"
      TAU_VALUE="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT}" ]]; then
  echo "ERROR: --input is required." >&2
  usage >&2
  exit 1
fi

if [[ "${MODE}" != "threshold" && "${MODE}" != "fixed" && "${MODE}" != "random" ]]; then
  echo "ERROR: --mode must be one of: threshold, fixed, random." >&2
  exit 1
fi

if [[ "${SIZE}" != "short" && "${SIZE}" != "base" && "${SIZE}" != "long" ]]; then
  echo "ERROR: --size must be one of: short, base, long." >&2
  exit 1
fi

case "${SIZE}" in
  short)
    DEFAULT_FIXED_SEGMENT_TOKENS="64"
    DEFAULT_RANDOM_MIN_SEGMENT_TOKENS="64"
    DEFAULT_RANDOM_MAX_SEGMENT_TOKENS="128"
    ;;
  base)
    DEFAULT_FIXED_SEGMENT_TOKENS="128"
    DEFAULT_RANDOM_MIN_SEGMENT_TOKENS="64"
    DEFAULT_RANDOM_MAX_SEGMENT_TOKENS="256"
    ;;
  long)
    DEFAULT_FIXED_SEGMENT_TOKENS="256"
    DEFAULT_RANDOM_MIN_SEGMENT_TOKENS="128"
    DEFAULT_RANDOM_MAX_SEGMENT_TOKENS="512"
    ;;
esac

FIXED_SEGMENT_TOKENS="${FIXED_SEGMENT_TOKENS:-${DEFAULT_FIXED_SEGMENT_TOKENS}}"
RANDOM_MIN_SEGMENT_TOKENS="${RANDOM_MIN_SEGMENT_TOKENS:-${DEFAULT_RANDOM_MIN_SEGMENT_TOKENS}}"
RANDOM_MAX_SEGMENT_TOKENS="${RANDOM_MAX_SEGMENT_TOKENS:-${DEFAULT_RANDOM_MAX_SEGMENT_TOKENS}}"

require_positive_int "--fixed-tokens" "${FIXED_SEGMENT_TOKENS}"
require_positive_int "--random-min" "${RANDOM_MIN_SEGMENT_TOKENS}"
require_positive_int "--random-max" "${RANDOM_MAX_SEGMENT_TOKENS}"
require_positive_int "--min-step-tokens" "${MIN_STEP_TOKENS}"

if (( RANDOM_MIN_SEGMENT_TOKENS > RANDOM_MAX_SEGMENT_TOKENS )); then
  echo "ERROR: --random-min must be <= --random-max." >&2
  exit 1
fi
if [[ "${MODE}" == "fixed" ]] && (( FIXED_SEGMENT_TOKENS < MIN_STEP_TOKENS )); then
  echo "ERROR: --fixed-tokens must be >= MIN_STEP_TOKENS (${MIN_STEP_TOKENS})." >&2
  exit 1
fi
if [[ "${MODE}" == "random" ]] && (( RANDOM_MIN_SEGMENT_TOKENS < MIN_STEP_TOKENS )); then
  echo "ERROR: --random-min must be >= MIN_STEP_TOKENS (${MIN_STEP_TOKENS})." >&2
  exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="runs/${MODE}-${SIZE}"
fi

export INPUT
export RUN_DIR
export MODEL
export BACKEND
export SEGMENTATION_MODE="${MODE}"
export TAU_KEY
export TAU_VALUE
export FIXED_SEGMENT_TOKENS
export RANDOM_MIN_SEGMENT_TOKENS
export RANDOM_MAX_SEGMENT_TOKENS
export RANDOM_SEED
export GPU_IDS
export LONG_SAMPLE_POLICY
export MIN_STEP_TOKENS

if [[ -n "${REFERENCE_TRAIN_JSONL}" ]]; then
  export REFERENCE_TRAIN_JSONL
fi
if [[ -n "${LIMIT_ROWS}" ]]; then
  export LIMIT_ROWS
fi

echo "Running semantic pipeline:"
echo "  mode=${SEGMENTATION_MODE}"
echo "  input=${INPUT}"
echo "  run_dir=${RUN_DIR}"
echo "  model=${MODEL}"
echo "  gpu_ids=${GPU_IDS}"
if [[ "${SEGMENTATION_MODE}" == "fixed" ]]; then
  echo "  fixed_segment_tokens=${FIXED_SEGMENT_TOKENS}"
elif [[ "${SEGMENTATION_MODE}" == "random" ]]; then
  echo "  random_min_segment_tokens=${RANDOM_MIN_SEGMENT_TOKENS}"
  echo "  random_max_segment_tokens=${RANDOM_MAX_SEGMENT_TOKENS}"
  echo "  random_seed=${RANDOM_SEED}"
else
  echo "  tau_key=${TAU_KEY}"
  if [[ -n "${TAU_VALUE}" ]]; then
    echo "  tau_value=${TAU_VALUE}"
  fi
fi
echo "  min_step_tokens=${MIN_STEP_TOKENS}"

bash "${SCRIPT_DIR}/run_full_pipeline.sh"
