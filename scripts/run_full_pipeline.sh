#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT="${INPUT:-/path/to/bs17k.jsonl}"
RUN_DIR="${RUN_DIR:-runs/bs17k_adaptivestep}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BACKEND="${BACKEND:-hf}"
TAU_KEY="${TAU_KEY:-q_0.0100}"
TAU_VALUE="${TAU_VALUE:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
ASSISTANT_WINDOW_SIZE="${ASSISTANT_WINDOW_SIZE:-4096}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
LONG_SAMPLE_POLICY="${LONG_SAMPLE_POLICY:-window}"

resolve_tau_value() {
  local tau_path="${RUN_DIR}/sample_tau/tau_candidates.json"
  python3 - "${tau_path}" "${TAU_KEY}" <<'PY'
import json
import sys
from pathlib import Path

tau_path = Path(sys.argv[1])
tau_key = sys.argv[2]
if not tau_path.exists():
    raise SystemExit(f"tau candidate file not found: {tau_path}")

data = json.loads(tau_path.read_text(encoding="utf-8"))
if tau_key not in data:
    raise SystemExit(f"tau key {tau_key!r} not found in {tau_path}")

print(data[tau_key])
PY
}

# Full pipeline wrapper: keep the major runtime knobs visible at this top level.
export INPUT RUN_DIR MODEL BACKEND TAU_KEY TAU_VALUE WORLD_SIZE GPU_IDS ASSISTANT_WINDOW_SIZE LIMIT_ROWS LONG_SAMPLE_POLICY
bash "${SCRIPT_DIR}/run_estimate_tau.sh"
if [[ -z "${TAU_VALUE:-}" ]]; then
  # Default to the 1% candidate unless the operator pins TAU_VALUE explicitly.
  export TAU_VALUE
  TAU_VALUE="$(resolve_tau_value)"
fi
bash "${SCRIPT_DIR}/run_convert_4gpu.sh"
