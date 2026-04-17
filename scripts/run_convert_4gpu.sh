#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${INPUT:-/path/to/bs17k.jsonl}"
RUN_DIR="${RUN_DIR:-runs/bs17k-seman}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BACKEND="${BACKEND:-hf}"
REFERENCE_TRAIN_JSONL="${REFERENCE_TRAIN_JSONL:-${ROOT_DIR}/../RRcot/data/train/train.jsonl}"
TAU_VALUE="${TAU_VALUE:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_STEP_TOKENS="${MIN_STEP_TOKENS:-12}"
MIN_STEP_CHARS="${MIN_STEP_CHARS:-8}"
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

if [[ -z "${TAU_VALUE}" ]]; then
  echo "ERROR: TAU_VALUE must be set for shard conversion. Pick one from ${RUN_DIR}/sample_tau/tau_candidates.json or use run_full_pipeline.sh." >&2
  exit 1
fi

if [[ "${BACKEND}" == "sglang" ]]; then
  if ! python3 - <<'PY' >/dev/null 2>&1
import sglang
PY
  then
    echo "ERROR: BACKEND=sglang but current python3 cannot import sglang." >&2
    echo "Please activate your sglang environment before running run_convert_4gpu.sh." >&2
    exit 1
  fi
fi

if [[ ! -f "${REFERENCE_TRAIN_JSONL}" ]]; then
  echo "ERROR: reference train jsonl not found: ${REFERENCE_TRAIN_JSONL}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/export" "${RUN_DIR}/progress" "${RUN_DIR}/logs" "${RUN_DIR}/merged"

CONFIG_PATH="${RUN_DIR}/convert_runtime.env"
CONFIG_KEYS=(
  INPUT
  MODEL
  BACKEND
  REFERENCE_TRAIN_JSONL
  DTYPE
  TRUST_REMOTE_CODE
  BATCH_SIZE
  TAU_VALUE
  WORLD_SIZE
  GPU_IDS
  MAX_LENGTH
  MIN_STEP_TOKENS
  MIN_STEP_CHARS
  ASSISTANT_WINDOW_SIZE
  ASSISTANT_STRIDE
  LIMIT_ROWS
  LONG_SAMPLE_POLICY
)

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
REQUESTED_WORLD_SIZE="${WORLD_SIZE}"
# Conversion shard count always follows the GPU list length.
WORLD_SIZE="${#GPUS[@]}"
if (( WORLD_SIZE <= 0 )); then
  echo "ERROR: GPU_IDS produced no usable devices. Please set GPU_IDS like '0,1,2,3'." >&2
  exit 1
fi
if [[ -n "${REQUESTED_WORLD_SIZE}" && "${REQUESTED_WORLD_SIZE}" != "${WORLD_SIZE}" ]]; then
  echo "INFO: WORLD_SIZE=${REQUESTED_WORLD_SIZE} was overridden to ${WORLD_SIZE} based on GPU_IDS='${GPU_IDS}'." >&2
fi

write_config_snapshot() {
  local path="$1"
  {
    echo "# Auto-generated resume guard for shard conversion."
    for key in "${CONFIG_KEYS[@]}"; do
      printf 'SAVED_%s=%q\n' "${key}" "${!key}"
    done
  } > "${path}"
}

if [[ -f "${CONFIG_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
  MISMATCHES=()
  for key in "${CONFIG_KEYS[@]}"; do
    saved_key="SAVED_${key}"
    saved_value="${!saved_key-__MISSING__}"
    if [[ "${!key}" != "${saved_value}" ]]; then
      MISMATCHES+=("${key}: expected '${saved_value}', got '${!key}'")
    fi
  done
  if (( ${#MISMATCHES[@]} > 0 )); then
    printf 'ERROR: resume config mismatch for RUN_DIR=%s\n' "${RUN_DIR}" >&2
    printf '  %s\n' "${MISMATCHES[@]}" >&2
    echo "Use a fresh RUN_DIR or restore the original settings before resuming." >&2
    exit 1
  fi
else
  # Snapshot the conversion knobs so a later resume cannot silently mix settings.
  write_config_snapshot "${CONFIG_PATH}"
fi

PIDS=()

cleanup_workers() {
  for pid in "${PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

# Make Ctrl-C and wrapper shutdown clean up background shard workers as well.
trap cleanup_workers EXIT INT TERM

for ((RANK = 0; RANK < WORLD_SIZE; RANK++)); do
  GPU="${GPUS[$RANK]}"
  CONVERT_ARGS=(
    --input "${INPUT}"
    --model "${MODEL}"
    --reference_train_jsonl "${REFERENCE_TRAIN_JSONL}"
    --tau "${TAU_VALUE}"
    --backend "${BACKEND}"
    --rank "${RANK}"
    --world_size "${WORLD_SIZE}"
    --output "${RUN_DIR}/export/shard_${RANK}.jsonl"
    --progress "${RUN_DIR}/progress/shard_${RANK}.json"
    --dtype "${DTYPE}"
    --trust_remote_code "${TRUST_REMOTE_CODE}"
    --max_length "${MAX_LENGTH}"
    --batch_size "${BATCH_SIZE}"
    --min_step_tokens "${MIN_STEP_TOKENS}"
    --min_step_chars "${MIN_STEP_CHARS}"
    # Match the estimator window so convert and tau estimation see the same context cap.
    --assistant_window_size "${ASSISTANT_WINDOW_SIZE}"
    --assistant_stride "${ASSISTANT_STRIDE}"
    "${LIMIT_ROWS_ARGS[@]}"
    # `window` keeps overlong rows in shard output; `skip` omits them after counting.
    --long_sample_policy "${LONG_SAMPLE_POLICY}"
  )
  CUDA_VISIBLE_DEVICES="${GPU}" nohup python3 "${ROOT_DIR}/tools/convert_shard.py" "${CONVERT_ARGS[@]}" \
    > "${RUN_DIR}/logs/shard_${RANK}.log" 2>&1 &
  PIDS+=("$!")
done

for ((RANK = 0; RANK < WORLD_SIZE; RANK++)); do
  if ! wait "${PIDS[$RANK]}"; then
    # Stop sibling workers before exiting so a partial failure cannot keep mutating RUN_DIR.
    cleanup_workers
    echo "Shard ${RANK} failed; aborting before merge." >&2
    exit 1
  fi
done

trap - EXIT INT TERM

MERGE_INPUTS=()
for ((RANK = 0; RANK < WORLD_SIZE; RANK++)); do
  MERGE_INPUTS+=("${RUN_DIR}/export/shard_${RANK}.jsonl")
done

python3 "${ROOT_DIR}/tools/merge_jsonl.py" \
  --inputs "${MERGE_INPUTS[@]}" \
  --output "${RUN_DIR}/merged/train.jsonl"
