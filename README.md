# semantic_aware

For a deployment-oriented Chinese guide, see [README_zh.md](/home/elysia/code/semantic_aware/README_zh.md).

## What This Does

`semantic_aware/` packages the semantic-aware AdaptiveStep to LightThinker conversion workflow into one operator-facing workspace. The normal flow is:

1. Sample rows and estimate candidate tau thresholds.
2. Launch 4 shard-conversion workers across GPUs.
3. Merge shard outputs into one sorted training JSONL.

The bash entrypoints under `semantic_aware/scripts/` are the intended starting point for routine runs.

## Directory Layout

```text
semantic_aware/
├── README.md
├── scripts/
│   ├── run_estimate_tau.sh
│   ├── run_convert_4gpu.sh
│   └── run_full_pipeline.sh
├── src/semantic_aware/
│   ├── boundary.py
│   ├── exporter.py
│   ├── io_utils.py
│   ├── protected_tokens.py
│   ├── role_extract.py
│   ├── scoring.py
│   └── tau_estimation.py
└── tools/
    ├── convert_shard.py
    ├── estimate_tau.py
    ├── merge_jsonl.py
    └── prepare_sample.py
```

## Quick Start

Run from the repo root so the relative script paths below work as written.

Set your run-specific values, then either execute the two main phases separately or use the full wrapper:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_adaptivestep
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=2000

bash semantic_aware/scripts/run_estimate_tau.sh
export TAU_VALUE=0.557  # Replace with the candidate you choose from tau_candidates.json
bash semantic_aware/scripts/run_convert_4gpu.sh
```

Or run the whole sequence:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_adaptivestep
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export ASSISTANT_WINDOW_SIZE=4096
export TAU_KEY=q_0.0100

bash semantic_aware/scripts/run_full_pipeline.sh
```

If you do not set `TAU_VALUE` yourself, `run_full_pipeline.sh` reads `${RUN_DIR}/sample_tau/tau_candidates.json` and uses `TAU_KEY` to select a candidate automatically. The default `TAU_KEY=q_0.0100` matches the 1% candidate.

## Step 1 Estimate Tau

The tau-estimation wrapper does two things:

1. Samples source indices into `${RUN_DIR}/sample_tau/sampled_indices.json`
2. Scores the sampled rows and writes candidate taus to `${RUN_DIR}/sample_tau/tau_candidates.json`

Minimal example:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_adaptivestep
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export GPU_ID=0
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=2000

bash semantic_aware/scripts/run_estimate_tau.sh
```

If you want to tune the estimation pass further:

```bash
export SAMPLE_SIZE=1500
export SEED=42
export TAU_QUANTILES=0.005,0.01,0.015,0.02
export MAX_LENGTH=8192
export BATCH_SIZE=1
export DTYPE=bfloat16
export TRUST_REMOTE_CODE=1
export LONG_SAMPLE_POLICY=window

bash semantic_aware/scripts/run_estimate_tau.sh
```

`ASSISTANT_WINDOW_SIZE=4096` keeps the scoring pass on a bounded assistant-side sliding window while preserving the full system/question prefix. `LIMIT_ROWS=2000` is a practical smoke-test limit when you want a shorter run without changing the input file. `LONG_SAMPLE_POLICY=window` keeps overlong rows in the output and tau estimate after counting them; `LONG_SAMPLE_POLICY=skip` records them in the run metadata but drops them from the written tau or shard output.

When `LIMIT_ROWS` is smaller than `SAMPLE_SIZE`, `run_estimate_tau.sh` automatically caps the sampling count to the same top-of-file subset so smoke-test runs do not fail with `sample_size > total_rows`.

Inspect `${RUN_DIR}/sample_tau/tau_candidates.json` and choose the tau value you want to use for conversion.

## Step 2 Run 4-GPU Conversion

The conversion wrapper launches one `convert_shard.py` worker per rank, writes per-rank JSONL shards under `${RUN_DIR}/export/`, tracks resume state under `${RUN_DIR}/progress/`, and captures logs under `${RUN_DIR}/logs/`.

`TAU_VALUE` is required for `run_convert_4gpu.sh`. This keeps the two-step workflow from silently falling back to a stale default tau.

Typical 4-GPU run:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_adaptivestep
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export TAU_VALUE=0.557
export WORLD_SIZE=4
export GPU_IDS=0,1,2,3
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=2000

bash semantic_aware/scripts/run_convert_4gpu.sh
```

Each worker writes:

- `${RUN_DIR}/export/shard_<rank>.jsonl`
- `${RUN_DIR}/progress/shard_<rank>.json`
- `${RUN_DIR}/logs/shard_<rank>.log`

When all ranks finish successfully, the script automatically merges them into `${RUN_DIR}/merged/train.jsonl`.

## Step 3 Merge/Inspect Output

`run_convert_4gpu.sh` already performs the merge step after all shard workers succeed, so the main post-run checks are to inspect the merged file and the per-rank progress/log outputs.

Useful checks:

```bash
ls -R "${RUN_DIR}"
head -n 3 "${RUN_DIR}/merged/train.jsonl"
cat "${RUN_DIR}/progress/shard_0.json"
tail -n 50 "${RUN_DIR}/logs/shard_0.log"
```

If you need to rerun just the merge manually, use the tool directly:

```bash
python3 semantic_aware/tools/merge_jsonl.py \
  --inputs "${RUN_DIR}/export/shard_0.jsonl" "${RUN_DIR}/export/shard_1.jsonl" "${RUN_DIR}/export/shard_2.jsonl" "${RUN_DIR}/export/shard_3.jsonl" \
  --output "${RUN_DIR}/merged/train.jsonl"
```

## Common Parameters

These environment variables are the main knobs exposed by the shell wrappers:

- `INPUT`: source JSONL dataset path.
- `RUN_DIR`: output root for sampled indices, tau candidates, shards, logs, progress, and merged JSONL.
- `MODEL`: model name or local checkpoint path passed to the Python tools.
- `GPU_ID`: single GPU used for tau estimation.
- `TAU_VALUE`: selected tau threshold used during shard conversion.
- `WORLD_SIZE`: number of shard workers to launch.
- `GPU_IDS`: comma-separated GPU list aligned with rank order.
- `DTYPE`: torch dtype string, default `bfloat16`.
- `TRUST_REMOTE_CODE`: set `0` to disable remote-code trust.
- `MAX_LENGTH`: threshold recorded for overlong statistics.
- `BATCH_SIZE`: reserved for future batching support; currently kept at `1` by default.
- `MIN_STEP_TOKENS`: minimum token gap between cut boundaries.
- `MIN_STEP_CHARS`: minimum character length for a split segment.
- `ASSISTANT_WINDOW_SIZE`: assistant-side scoring window size forwarded to both Python tools, default `4096`.
- `LIMIT_ROWS`: optional top-of-file cap for smoke tests and quick reruns.
- `LONG_SAMPLE_POLICY`: `window` keeps overlong rows after counting them; `skip` drops them from the written output while still counting them in metadata.
- `TAU_KEY`: full-pipeline helper key used to pick a tau from `${RUN_DIR}/sample_tau/tau_candidates.json` when `TAU_VALUE` is unset. Default `q_0.0100`.

## Resume Behavior

Each conversion rank persists a JSON progress file at `${RUN_DIR}/progress/shard_<rank>.json`. The converter reads `last_source_idx`, `num_written`, `num_skipped`, `num_overlong`, and `finished` from that file before processing more rows. The same wrapper env vars, including `LIMIT_ROWS`, `ASSISTANT_WINDOW_SIZE`, and `LONG_SAMPLE_POLICY`, are passed through on resume runs.

`run_convert_4gpu.sh` also writes `${RUN_DIR}/convert_runtime.env` as a small resume guard. If you rerun with the same `RUN_DIR` but change `TAU_VALUE`, `WORLD_SIZE`, `GPU_IDS`, `DTYPE`, `TRUST_REMOTE_CODE`, `BATCH_SIZE`, `ASSISTANT_WINDOW_SIZE`, `LIMIT_ROWS`, `LONG_SAMPLE_POLICY`, or other key conversion knobs, the wrapper stops early instead of mixing incompatible shard outputs into one merged dataset.

Operationally, that means:

- Re-running `bash semantic_aware/scripts/run_convert_4gpu.sh` with the same `RUN_DIR` resumes each rank from the last recorded `source_idx` instead of rewriting completed work.
- Existing shard JSONL files are appended to, not replaced, so resume runs should keep the same `RUN_DIR` and shard assignment.
- A `finished: true` progress file means that rank has already completed its assigned modulo shard, and `convert_shard.py` will exit early for that shard instead of reloading the model.
- If you need a clean rerun, remove or archive the old `RUN_DIR` first so progress, logs, and shard outputs do not mix with the new run.
