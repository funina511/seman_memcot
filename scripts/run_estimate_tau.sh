#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${INPUT:-/path/to/bs17k.jsonl}"
RUN_DIR="${RUN_DIR:-runs/bs17k_adaptivestep}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1500}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
GPU_ID="${GPU_ID:-0}"
TAU_QUANTILES="${TAU_QUANTILES:-0.005,0.01,0.015,0.02}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ASSISTANT_WINDOW_SIZE="${ASSISTANT_WINDOW_SIZE:-4096}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
LONG_SAMPLE_POLICY="${LONG_SAMPLE_POLICY:-window}"

LIMIT_ROWS_ARGS=()
if [[ -n "${LIMIT_ROWS}" ]]; then
  LIMIT_ROWS_ARGS+=(--limit_rows "${LIMIT_ROWS}")
fi

EFFECTIVE_SAMPLE_SIZE="${SAMPLE_SIZE}"
if [[ -n "${LIMIT_ROWS}" ]] && (( EFFECTIVE_SAMPLE_SIZE > LIMIT_ROWS )); then
  # Keep smoke-test subsets usable without forcing the operator to tune SAMPLE_SIZE by hand.
  EFFECTIVE_SAMPLE_SIZE="${LIMIT_ROWS}"
fi

mkdir -p "${RUN_DIR}/sample_tau"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${ROOT_DIR}/tools/prepare_sample.py" \
  --input "${INPUT}" \
  --output "${RUN_DIR}/sample_tau/sampled_indices.json" \
  --sample_size "${EFFECTIVE_SAMPLE_SIZE}" \
  --seed "${SEED}" \
  "${LIMIT_ROWS_ARGS[@]}"

ESTIMATE_ARGS=(
  --input "${INPUT}"
  --sampled_indices "${RUN_DIR}/sample_tau/sampled_indices.json"
  --model "${MODEL}"
  --output "${RUN_DIR}/sample_tau/tau_candidates.json"
  --tau_quantiles "${TAU_QUANTILES}"
  --dtype "${DTYPE}"
  --trust_remote_code "${TRUST_REMOTE_CODE}"
  --max_length "${MAX_LENGTH}"
  --batch_size "${BATCH_SIZE}"
  # Keep the scoring pass on a bounded assistant-side window even for long rows.
  --assistant_window_size "${ASSISTANT_WINDOW_SIZE}"
  "${LIMIT_ROWS_ARGS[@]}"
  # `window` keeps overlong rows in tau estimation; `skip` drops them after counting.
  --long_sample_policy "${LONG_SAMPLE_POLICY}"
)

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${ROOT_DIR}/tools/estimate_tau.py" "${ESTIMATE_ARGS[@]}"
