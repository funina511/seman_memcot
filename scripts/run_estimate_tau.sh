#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${INPUT:-/path/to/bs17k.jsonl}"
RUN_DIR="${RUN_DIR:-runs/bs17k_adaptivestep}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BACKEND="${BACKEND:-hf}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1500}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-${GPU_ID}}"
TAU_QUANTILES="${TAU_QUANTILES:-0.005,0.01,0.015,0.02}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ASSISTANT_WINDOW_SIZE="${ASSISTANT_WINDOW_SIZE:-4096}"
ASSISTANT_STRIDE="${ASSISTANT_STRIDE:-1024}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.65}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-1}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
LONG_SAMPLE_POLICY="${LONG_SAMPLE_POLICY:-window}"

export SGLANG_MEM_FRACTION_STATIC SGLANG_CHUNKED_PREFILL_SIZE SGLANG_CUDA_GRAPH_MAX_BS

LIMIT_ROWS_ARGS=()
if [[ -n "${LIMIT_ROWS}" ]]; then
  LIMIT_ROWS_ARGS+=(--limit_rows "${LIMIT_ROWS}")
fi

if [[ "${BACKEND}" == "sglang" ]]; then
  if ! python3 - <<'PY' >/dev/null 2>&1
import sglang
PY
  then
    echo "ERROR: BACKEND=sglang but current python3 cannot import sglang." >&2
    echo "Please activate your sglang environment before running run_estimate_tau.sh." >&2
    exit 1
  fi
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

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
PIDS=()

# Tau estimation fans out one worker per visible GPU in GPU_IDS.
for rank in "${!GPU_ARRAY[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$rank]}" python3 "${ROOT_DIR}/tools/estimate_tau.py" \
    --input "${INPUT}" \
    --sampled_indices "${RUN_DIR}/sample_tau/sampled_indices.json" \
    --model "${MODEL}" \
    --partial_output "${RUN_DIR}/sample_tau/tau_candidates.rank${rank}.json" \
    --backend "${BACKEND}" \
    --rank "${rank}" \
    --world_size "${WORLD_SIZE}" \
    --tau_quantiles "${TAU_QUANTILES}" \
    --dtype "${DTYPE}" \
    --trust_remote_code "${TRUST_REMOTE_CODE}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --assistant_window_size "${ASSISTANT_WINDOW_SIZE}" \
    --assistant_stride "${ASSISTANT_STRIDE}" \
    "${LIMIT_ROWS_ARGS[@]}" \
    --long_sample_policy "${LONG_SAMPLE_POLICY}" &
  PIDS+=("$!")
done

for rank in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$rank]}"; then
    echo "Tau worker ${rank} failed; aborting before merge." >&2
    exit 1
  fi
done

python3 "${ROOT_DIR}/tools/merge_tau_candidates.py" \
  --inputs "${RUN_DIR}"/sample_tau/tau_candidates.rank*.json \
  --output "${RUN_DIR}/sample_tau/tau_candidates.json" \
  --tau_quantiles "${TAU_QUANTILES}"
