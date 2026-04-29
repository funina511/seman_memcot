# seman_memcot

For a deployment-oriented Chinese guide, see [README_zh.md](README_zh.md).

## What This Does

`seman_memcot/` packages the semantic-aware AdaptiveStep to LightThinker conversion workflow into one operator-facing workspace. The normal flow is:

1. Sample rows and estimate candidate tau thresholds.
2. Launch 4 shard-conversion workers across GPUs.
3. Merge shard outputs into one sorted training JSONL.

The bash entrypoints under `seman_memcot/scripts/` are the intended starting point for routine runs.

## Directory Layout

```text
seman_memcot/
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
export BACKEND=hf
export SEGMENTATION_MODE=threshold
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=2000

bash seman_memcot/scripts/run_estimate_tau.sh
export TAU_VALUE=0.557  # Replace with the candidate you choose from tau_candidates.json
bash seman_memcot/scripts/run_convert_4gpu.sh
```

Or run the whole sequence:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_adaptivestep
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=hf
export WORLD_SIZE=4
export GPU_IDS=0,1,2,3
export SEGMENTATION_MODE=threshold
export ASSISTANT_WINDOW_SIZE=4096
export TAU_KEY=q_0.0100

bash seman_memcot/scripts/run_full_pipeline.sh
```

If you do not set `TAU_VALUE` yourself, `run_full_pipeline.sh` reads `${RUN_DIR}/sample_tau/tau_candidates.json` and uses `TAU_KEY` to select a candidate automatically. The default `TAU_KEY=q_0.0100` matches the 1% candidate. This tau-estimation phase only runs when `SEGMENTATION_MODE=threshold`; fixed and random modes skip it.

For the fastest correctness smoke checks, use the local Hugging Face backend and keep the sample small:

```bash
export BACKEND=hf
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=skip
```

With `ASSISTANT_WINDOW_SIZE=4096` and `ASSISTANT_STRIDE=1024`, each scoring pass reuses a 4096-token assistant window but only scores about 1024 fresh assistant tokens. Save per-token stride for tiny exact comparisons.

For this first HF smoke pass, focus on segmentation correctness rather than throughput:

- protected/control tokens should remain intact
- ordinary words and identifiers such as `length` or `max_length` should not be split internally
- hyphenated compounds such as `step-by-step` should not be cut inside the compound
- compact formula-like fragments such as `x+y=2`, `n->n+1`, and `a/b` should remain meaningful segments

Optional SGLang comparison:

```bash
export BACKEND=sglang
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=skip
export SGLANG_MEM_FRACTION_STATIC=0.65
export SGLANG_CHUNKED_PREFILL_SIZE=2048
export SGLANG_CUDA_GRAPH_MAX_BS=1
```

## Step 1 Estimate Tau

The tau-estimation wrapper does two things:

1. Samples source indices into `${RUN_DIR}/sample_tau/sampled_indices.json`
2. Scores the sampled rows and writes candidate taus to `${RUN_DIR}/sample_tau/tau_candidates.json`

Minimal example:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_adaptivestep
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=hf
export GPU_IDS=0,1
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=2000

bash seman_memcot/scripts/run_estimate_tau.sh
```

`run_estimate_tau.sh` now launches one worker per visible GPU in `GPU_IDS`, merges the per-rank partial JSON files, and writes `${RUN_DIR}/sample_tau/tau_candidates.json`. For a single-GPU run, leave `GPU_IDS` unset or set `GPU_ID=0`.

If you want to tune the estimation pass further:

```bash
export SAMPLE_SIZE=1500
export SEED=42
export TAU_QUANTILES=0.005,0.01,0.015,0.02
export MAX_LENGTH=8192
export BATCH_SIZE=1
export DTYPE=bfloat16
export TRUST_REMOTE_CODE=1
export BACKEND=sglang
export LONG_SAMPLE_POLICY=window

bash seman_memcot/scripts/run_estimate_tau.sh
```

Use `BACKEND=hf` for the default local Hugging Face scoring path. Use `BACKEND=sglang` only when the current `python3` can import `sglang`; the shell wrappers now fail fast with an explicit environment hint if this check fails.

`BACKEND=sglang` targets `sglang==0.4.6.post5`. For that backend, start with:

```bash
export BACKEND=sglang
export SGLANG_MEM_FRACTION_STATIC=0.65
export SGLANG_CHUNKED_PREFILL_SIZE=2048
export SGLANG_CUDA_GRAPH_MAX_BS=1
```

If you still see VRAM spikes, lower `SGLANG_MEM_FRACTION_STATIC` first, then reduce `SGLANG_CHUNKED_PREFILL_SIZE`.

`ASSISTANT_WINDOW_SIZE=4096` keeps the scoring pass on a bounded assistant-side sliding window while preserving the full system/question prefix. `LIMIT_ROWS=2000` is a practical smoke-test limit when you want a shorter run without changing the input file. `LONG_SAMPLE_POLICY=window` keeps overlong rows in the output and tau estimate after counting them; `LONG_SAMPLE_POLICY=skip` records them in the run metadata but drops them from the written tau or shard output.

One subtlety in the current boundary protection: if a low-confidence cut lands inside a word, the boundary layer first tries to relocate to a nearby safe cut instead of keeping the bad internal split. That relocation window is currently an internal default in `boundary.py`, not a shell-exposed tuning knob.

When `LIMIT_ROWS` is smaller than `SAMPLE_SIZE`, `run_estimate_tau.sh` automatically caps the sampling count to the same top-of-file subset so smoke-test runs do not fail with `sample_size > total_rows`.

Inspect `${RUN_DIR}/sample_tau/tau_candidates.json` and choose the tau value you want to use for conversion.

## Step 2 Run 4-GPU Conversion

The conversion wrapper launches one `convert_shard.py` worker per rank, writes per-rank JSONL shards under `${RUN_DIR}/export/`, tracks resume state under `${RUN_DIR}/progress/`, and captures logs under `${RUN_DIR}/logs/`.

`SEGMENTATION_MODE` controls how conversion chooses boundaries:

- `threshold` keeps the semantic-aware confidence behavior and requires `TAU_VALUE`
- `fixed` cuts after every `FIXED_SEGMENT_TOKENS` assistant tokens
- `random` samples deterministic per-row lengths from `RANDOM_MIN_SEGMENT_TOKENS` through `RANDOM_MAX_SEGMENT_TOKENS` using `RANDOM_SEED + source_idx`

`fixed` and `random` still need `MODEL` so the converter can load the model tokenizer and recover assistant token offsets, but they do not load the full scoring backend/model and do not need `TAU_VALUE`. In `run_full_pipeline.sh`, these modes skip tau estimation and go straight to conversion.

For fixed/random modes, the configured segment size is measured from the last accepted boundary, and the configured size itself must be at least `MIN_STEP_TOKENS`. If a target token is unsafe because of protected tokens or word-internal cuts, the segment extends forward until a safe boundary is found.

`TAU_VALUE` is required for `run_convert_4gpu.sh` only when `SEGMENTATION_MODE=threshold`. This keeps the two-step threshold workflow from silently falling back to a stale default tau.

For less environment-variable setup, use the CLI wrapper:

```bash
bash seman_memcot/scripts/run_pipeline_cli.sh \
  --input /data/bs17k.jsonl \
  --mode fixed \
  --size base \
  --gpu-ids 0,1,2,3
```

`--size short|base|long` maps to fixed intervals of `64|128|256` tokens and random ranges of `64-128|64-256|128-512`.
The wrapper also accepts `--min-step-tokens` and rejects invalid fixed/random combinations before entering the pipeline.

Typical 4-GPU run:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_adaptivestep
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=hf
export SEGMENTATION_MODE=threshold
export TAU_VALUE=0.557
export WORLD_SIZE=4
export GPU_IDS=0,1,2,3
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=2000

bash seman_memcot/scripts/run_convert_4gpu.sh
```

Fixed-token conversion example:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_fixed
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export SEGMENTATION_MODE=fixed
export FIXED_SEGMENT_TOKENS=128
export GPU_IDS=0,1,2,3

bash seman_memcot/scripts/run_convert_4gpu.sh
```

Random-token conversion example:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_random
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export SEGMENTATION_MODE=random
export RANDOM_MIN_SEGMENT_TOKENS=64
export RANDOM_MAX_SEGMENT_TOKENS=256
export RANDOM_SEED=42
export GPU_IDS=0,1,2,3

bash seman_memcot/scripts/run_full_pipeline.sh
```

Each worker writes:

- `${RUN_DIR}/export/shard_<rank>.jsonl`
- `${RUN_DIR}/export/shard_<rank>.jsonl.meta.json`
- `${RUN_DIR}/progress/shard_<rank>.json`
- `${RUN_DIR}/logs/shard_<rank>.log`

Shard conversion inherits non-`thoughts_list` fields from `REFERENCE_TRAIN_JSONL` row-by-row, then overwrites only `thoughts_list` with the semantic split output.

That means `convert_shard.py` is not rebuilding every field from the raw input JSONL. It is using `REFERENCE_TRAIN_JSONL` as the source of truth for `system_prompt`, `question`, and `gt_output`, then replacing only `thoughts_list`. When an output looks suspicious, treat row alignment as an assumption to verify alongside the segmentation itself:

- `INPUT` and `REFERENCE_TRAIN_JSONL` should be in the same physical row order
- the reference row’s `gt_output` should be the assistant text you expect to re-segment
- an odd output can come from either boundary selection or an unexpected reference row

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
python3 seman_memcot/tools/merge_jsonl.py \
  --inputs "${RUN_DIR}/export/shard_0.jsonl" "${RUN_DIR}/export/shard_1.jsonl" "${RUN_DIR}/export/shard_2.jsonl" "${RUN_DIR}/export/shard_3.jsonl" \
  --output "${RUN_DIR}/merged/train.jsonl"
```

## Runtime Metadata

The Python tools write runtime metadata sidecars as `*.meta.json` files next to their main outputs. The most useful ones during operator checks are:

- `${RUN_DIR}/sample_tau/tau_candidates.json.meta.json`
- `${RUN_DIR}/export/shard_<rank>.jsonl.meta.json`

These sidecars record fields such as backend, model, windowing controls, row limits, and summary counters so you can confirm a smoke run and a full run used the expected settings.

The profiling sidecars also record `score_seconds_per_token`, `assistant_stride`, and `cuda_max_memory_allocated_mb`, which are the quickest signals for whether a run is compute-bound or memory-bound.

## Common Parameters

These environment variables are the main knobs exposed by the shell wrappers:

- `INPUT`: source JSONL dataset path.
- `RUN_DIR`: output root for sampled indices, tau candidates, shards, logs, progress, and merged JSONL.
- `MODEL`: model name or local checkpoint path passed to the Python tools.
- `REFERENCE_TRAIN_JSONL`: reference LightThinker train JSONL used by conversion; all fields except `thoughts_list` are inherited from this file.
- `BACKEND`: scoring backend passed end-to-end to tau estimation and shard conversion. Common values are `hf` and `sglang`.
- `SEGMENTATION_MODE`: `threshold`, `fixed`, or `random`; default `threshold`.
- `FIXED_SEGMENT_TOKENS`: assistant-token interval for fixed mode; default `128`.
- `RANDOM_MIN_SEGMENT_TOKENS`: minimum random segment length; default `64`.
- `RANDOM_MAX_SEGMENT_TOKENS`: maximum random segment length; default `256`.
- `RANDOM_SEED`: base seed for deterministic per-row random mode; default `42`.
- `GPU_ID`: fallback single GPU used for tau estimation when `GPU_IDS` is unset.
- `GPU_IDS`: comma-separated GPU list aligned with rank order. Tau estimation now runs one worker per visible GPU in this list.
- `TAU_VALUE`: selected tau threshold used during threshold shard conversion.
- `WORLD_SIZE`: number of shard workers to launch.
- `DTYPE`: torch dtype string, default `bfloat16`.
- `TRUST_REMOTE_CODE`: set `0` to disable remote-code trust.
- `MAX_LENGTH`: threshold recorded for overlong statistics.
- `BATCH_SIZE`: reserved for future batching support; currently kept at `1` by default.
- `MIN_STEP_TOKENS`: minimum token gap between cut boundaries.
- `MIN_STEP_CHARS`: minimum character length for a split segment.
- `ASSISTANT_WINDOW_SIZE`: assistant-side scoring window size forwarded to both Python tools, default `4096`.
- `ASSISTANT_STRIDE`: optional assistant-side stride between scoring windows; defaults to one quarter of `ASSISTANT_WINDOW_SIZE`.
- `LIMIT_ROWS`: optional top-of-file cap for smoke tests and quick reruns.
- `LONG_SAMPLE_POLICY`: `window` keeps overlong rows after counting them; `skip` drops them from the written output while still counting them in metadata.
- `TAU_KEY`: full-pipeline helper key used to pick a tau from `${RUN_DIR}/sample_tau/tau_candidates.json` when `TAU_VALUE` is unset. Default `q_0.0100`.

## Resume Behavior

Each conversion rank persists a JSON progress file at `${RUN_DIR}/progress/shard_<rank>.json`. The converter reads `last_source_idx`, `num_written`, `num_skipped`, `num_overlong`, and `finished` from that file before processing more rows. The same wrapper env vars, including `LIMIT_ROWS`, `ASSISTANT_WINDOW_SIZE`, and `LONG_SAMPLE_POLICY`, are passed through on resume runs.

`run_convert_4gpu.sh` also writes `${RUN_DIR}/convert_runtime.env` as a small resume guard. If you rerun with the same `RUN_DIR` but change `SEGMENTATION_MODE`, `TAU_VALUE`, `FIXED_SEGMENT_TOKENS`, random-mode bounds/seed, `BACKEND`, `WORLD_SIZE`, `GPU_IDS`, `DTYPE`, `TRUST_REMOTE_CODE`, `BATCH_SIZE`, `ASSISTANT_WINDOW_SIZE`, `LIMIT_ROWS`, `LONG_SAMPLE_POLICY`, or other key conversion knobs, the wrapper stops early instead of mixing incompatible shard outputs into one merged dataset.

Operationally, that means:

- Re-running `bash seman_memcot/scripts/run_convert_4gpu.sh` with the same `RUN_DIR` resumes each rank from the last recorded `source_idx` instead of rewriting completed work.
- Existing shard JSONL files are appended to, not replaced, so resume runs should keep the same `RUN_DIR` and shard assignment.
- A `finished: true` progress file means that rank has already completed its assigned modulo shard, and `convert_shard.py` will exit early for that shard instead of reloading the model.
- If you need a clean rerun, remove or archive the old `RUN_DIR` first so progress, logs, and shard outputs do not mix with the new run.

## Smoke Test

Practical smoke-test command:

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k_smoke
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=hf
export GPU_IDS=0,1
export WORLD_SIZE=2
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=skip
export TAU_KEY=q_0.0100

bash seman_memcot/scripts/run_full_pipeline.sh
```

Expected key outputs:

- `${RUN_DIR}/sample_tau/tau_candidates.json`
- `${RUN_DIR}/sample_tau/tau_candidates.json.meta.json`
- `${RUN_DIR}/progress/shard_0.json`
- `${RUN_DIR}/logs/shard_0.log`
- `${RUN_DIR}/merged/train.jsonl`

Correctness checklist for this smoke run:

- `thoughts_list` is not overly fragmented
- protected tokens are never broken apart
- internal cuts do not appear inside ordinary identifiers or hyphenated compounds
- short formula-like fragments are still preserved as meaningful segments

Optional SGLang comparison:

```bash
export BACKEND=sglang
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=skip
export SGLANG_MEM_FRACTION_STATIC=0.65
export SGLANG_CHUNKED_PREFILL_SIZE=2048
export SGLANG_CUDA_GRAPH_MAX_BS=1
```
