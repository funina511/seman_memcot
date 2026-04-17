# seman_memcot 中文部署说明

这份文档面向“尽快跑起来”的场景，重点说明如何把 `Bespoke-Stratos-17k` 转成 `LightThinker` 需要的训练 JSONL，并支持：

- `4096` assistant-side 滑窗 teacher forcing
- `LIMIT_ROWS` 小规模 smoke test
- `4` 卡分 shard 转换
- 断点续跑
- 自动读取估计出的 `tau`

如果你想看英文版总说明，可以参考 [README.md](README.md)。

## 目录结构

```text
seman_memcot/
├── README.md
├── README_zh.md
├── scripts/
│   ├── run_estimate_tau.sh
│   ├── run_convert_4gpu.sh
│   └── run_full_pipeline.sh
├── src/semantic_aware/
└── tools/
```

平时最常用的是这 3 个脚本：

- `seman_memcot/scripts/run_estimate_tau.sh`
- `seman_memcot/scripts/run_convert_4gpu.sh`
- `seman_memcot/scripts/run_full_pipeline.sh`

## 你真正需要记住的流程

整个流程可以压缩成两步：

1. 先抽样估计 `tau`
2. 再固定 `tau` 做全量或子集转换

如果你不想手动抄 `tau`，就直接用 `run_full_pipeline.sh`，它会从 `${RUN_DIR}/sample_tau/tau_candidates.json` 里按 `TAU_KEY` 自动选一个候选值。

## 快速开始

建议先做一个前 `200` 条的小规模试跑，确认切分效果正常。

### 方案 A：一步跑完整流程

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k-seman
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=sglang
export WORLD_SIZE=4
export GPU_IDS=0,1,2,3
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=200
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=window
export TAU_KEY=q_0.0200

bash seman_memcot/scripts/run_full_pipeline.sh
```

这条命令会自动完成：

- 抽样并估计 `tau`
- 从 `tau_candidates.json` 里读取 `TAU_KEY` 对应的值
- 启动 shard 转换
- 合并成最终 `train.jsonl`

如果你想做最快的 smoke test，推荐先用本地 Hugging Face 后端，并把样本量压小：

```bash
export BACKEND=hf
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=skip
```

`ASSISTANT_STRIDE=1024` 表示每个窗口最多复用 `4096` 个 assistant token 上下文，但一次评分约 `1024` 个新 token。不要再使用逐 token stride，除非只是做极小样本的精确对照。

### 方案 B：分两步手动跑

如果你想先人工看一下 `tau_candidates.json`，就分开跑：

```bash
export INPUT=/data/bs17k.jsonl
export RUN_DIR=runs/bs17k-seman
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=hf
export GPU_IDS=0,1
export ASSISTANT_WINDOW_SIZE=4096
export LIMIT_ROWS=2000

bash seman_memcot/scripts/run_estimate_tau.sh
```

看完 `${RUN_DIR}/sample_tau/tau_candidates.json` 之后，再选一个值继续：

```bash
export BACKEND=hf
export TAU_VALUE=0.557
export GPU_IDS=0,1,2,3

bash seman_memcot/scripts/run_convert_4gpu.sh
```

注意：`run_convert_4gpu.sh` 现在要求你显式设置 `TAU_VALUE`，不会再偷偷回退到旧默认值。
另外，`run_estimate_tau.sh` 现在会按 `GPU_IDS` 里的可见卡列表，每张 GPU 启动一个 worker 来并行估计 tau；单卡时不设 `GPU_IDS` 或保留 `GPU_ID=0` 即可。

## 常用参数

最常用的是这些：

- `INPUT`：原始 JSONL 数据集路径
- `RUN_DIR`：本次运行的输出目录
- `MODEL`：用于 teacher forcing 打分的模型
- `REFERENCE_TRAIN_JSONL`：参考训练集路径，转换输出会继承该文件对应行的全部字段，仅覆盖 `thoughts_list`
- `BACKEND`：后端类型，常用 `BACKEND=hf` 或 `BACKEND=sglang`
- `ASSISTANT_WINDOW_SIZE`：assistant 侧滑窗大小，默认 `4096`
- `ASSISTANT_STRIDE`：assistant 侧滑窗步长，可选；默认是 `ASSISTANT_WINDOW_SIZE` 的四分之一
- `LIMIT_ROWS`：只处理输入前 N 条，适合 smoke test
- `TAU_KEY`：完整流程脚本用于自动选取 `tau` 的键，默认 `q_0.0100`
- `TAU_VALUE`：手动转换时使用的具体阈值
- `WORLD_SIZE`：兼容保留参数；转换阶段会自动按 `GPU_IDS` 数量对齐 shard 数
- `GPU_IDS`：转换与 tau 估计都会按这个列表并行；例如 `0,1,2,3`
- `LONG_SAMPLE_POLICY`：超长样本策略，`window` 或 `skip`

`BACKEND=hf` 适合默认的本地 Hugging Face 打分路径。`BACKEND=sglang` 需要在当前 `python3` 环境可导入 `sglang`；shell 脚本会先做前置检查，不满足时会直接失败并提示切换到 sglang 环境。

如果你要和 SGLang 做可选对照比较，`BACKEND=sglang` 这条路径按 `sglang==0.4.6.post5` 的 `Engine.generate` / prompt logprob 方式设计。推荐先这样设置：

```bash
export BACKEND=sglang
export SGLANG_MEM_FRACTION_STATIC=0.65
export SGLANG_CHUNKED_PREFILL_SIZE=2048
export SGLANG_CUDA_GRAPH_MAX_BS=1
```

如果还是出现显存跳变，先把 `SGLANG_MEM_FRACTION_STATIC` 再调低，再考虑减小 `SGLANG_CHUNKED_PREFILL_SIZE`。

## 推荐部署方式

### 1. 先做 200 条 smoke test

```bash
export INPUT=bs17k.jsonl
export RUN_DIR=runs/bs17k-seman
export MODEL=model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=hf
export GPU_IDS=0,1,2,3
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=skip
export TAU_KEY=q_0.0200
bash seman_memcot/scripts/run_full_pipeline.sh
```

这一步主要看：

- `thoughts_list` 是否切得过碎
- protected token 是否被切坏
- `tau` 选得是否过粗或过细

### 2. 再做全量运行

确认 smoke test 没问题后，把 `LIMIT_ROWS` 去掉：

```bash
unset LIMIT_ROWS
bash seman_memcot/scripts/run_full_pipeline.sh
```

## 关于 LIMIT_ROWS 的一个细节

`LIMIT_ROWS` 现在不仅影响转换阶段，也会影响抽样估计 `tau` 的阶段。

也就是说，如果你设置：

```bash
export LIMIT_ROWS=2000
```

那么：

- `prepare_sample.py` 只会在前 `2000` 条里抽样
- `estimate_tau.py` 只会在前 `2000` 条里统计候选 `tau`
- `convert_shard.py` 也只会转换前 `2000` 条

另外，如果 `LIMIT_ROWS < SAMPLE_SIZE`，`run_estimate_tau.sh` 会自动把实际抽样数压到 `LIMIT_ROWS`，避免出现 `sample_size > total_rows` 的报错。

## 断点续跑

转换阶段支持断点续跑，关键文件在：

- `${RUN_DIR}/progress/shard_<rank>.json`
- `${RUN_DIR}/export/shard_<rank>.jsonl`
- `${RUN_DIR}/convert_runtime.env`

其中：

- `progress/shard_<rank>.json` 记录每个 shard 的当前位置
- `convert_runtime.env` 记录这次运行的关键参数，避免你下次用不同参数续跑时把结果混在一起
- 转换脚本会按 `GPU_IDS` 自动设置 `WORLD_SIZE`，因此同一个 `RUN_DIR` 续跑时不需要手动对齐 `WORLD_SIZE`

如果你用同一个 `RUN_DIR` 重新执行：

```bash
bash seman_memcot/scripts/run_convert_4gpu.sh
```

脚本会先检查参数是否和上次一致；如果不一致，会直接报错，让你换新的 `RUN_DIR` 或恢复原参数。
这里的保护项现在也包含 `BACKEND`，避免同一个输出目录里混入不同后端打分结果。

另外，转换阶段还会做 progress 与 shard 输出的一致性保护：

- 如果 `progress/shard_<rank>.json` 缺失，但 `export/shard_<rank>.jsonl` 已有内容，会直接失败（防止重复 append）
- 如果 `num_written` 与 shard 实际行数不一致，会直接失败
- 如果 progress 文件 JSON 损坏，会先重命名为 `.corrupt` 备份，然后直接失败

出现以上报错时，先修复或删除对应 shard 的 `progress` 与 `export` 文件，再重新执行该 `RUN_DIR`。

## 常见输出位置

估计 `tau` 后的结果：

- `${RUN_DIR}/sample_tau/sampled_indices.json`
- `${RUN_DIR}/sample_tau/tau_candidates.json`
- `${RUN_DIR}/sample_tau/tau_candidates.json.meta.json`

转换阶段的结果：

- `${RUN_DIR}/export/shard_<rank>.jsonl`
- `${RUN_DIR}/export/shard_<rank>.jsonl.meta.json`
- `${RUN_DIR}/progress/shard_<rank>.json`
- `${RUN_DIR}/logs/shard_<rank>.log`
- `${RUN_DIR}/merged/train.jsonl`

转换输出记录采用“参考继承”模式：会从 `REFERENCE_TRAIN_JSONL` 对应位置继承除 `thoughts_list` 外的字段，再写入当前语义切分得到的 `thoughts_list`。

这些 `*.meta.json` sidecar 会记录 backend、model、窗口参数、limit_rows、计数统计等运行时元数据，方便确认 smoke test 和正式跑的配置一致。

每个 `*.meta.json` 里还会写入 `score_seconds_per_token`、`assistant_stride`、`cuda_max_memory_allocated_mb`。如果 `200` 条 smoke test 还是很慢，先对比 HF 和 SGLang 的 `score_seconds_per_token`，再看是不是有很多超长样本或异常窗口数。

## 出问题时先看哪里

最先看这几个文件：

```bash
cat "${RUN_DIR}/sample_tau/tau_candidates.json"
cat "${RUN_DIR}/progress/shard_0.json"
tail -n 50 "${RUN_DIR}/logs/shard_0.log"
head -n 3 "${RUN_DIR}/merged/train.jsonl"
```

## Smoke Test 命令

建议先用一个很小的子集做 smoke test：

```bash
export INPUT=bs17k.jsonl
export RUN_DIR=runs/bs17k-smoke
export MODEL=model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=hf
export GPU_IDS=0,1,2,3
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=window
export TAU_KEY=q_0.0200

bash seman_memcot/scripts/run_full_pipeline.sh
```

预期关键输出：

- `${RUN_DIR}/sample_tau/tau_candidates.json`
- `${RUN_DIR}/sample_tau/tau_candidates.json.meta.json`
- `${RUN_DIR}/progress/shard_0.json`
- `${RUN_DIR}/logs/shard_0.log`
- `${RUN_DIR}/merged/train.jsonl`

可选：SGLang 对照比较

```bash
export INPUT=bs17k.jsonl
export RUN_DIR=runs/bs17k_smoke
export MODEL=model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=sglang
export GPU_IDS=4,5
export LIMIT_ROWS=200
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=window
export SGLANG_MEM_FRACTION_STATIC=0.72
export SGLANG_CHUNKED_PREFILL_SIZE=2048
export SGLANG_CUDA_GRAPH_MAX_BS=4
export TAU_KEY=q_0.0200

bash seman_memcot/scripts/run_full_pipeline.sh
```

export INPUT=bs17k.jsonl
export RUN_DIR=runs/bs17k_seman
export MODEL=model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export BACKEND=sglang
export GPU_IDS=4,5
unset LIMIT_ROWS
export ASSISTANT_WINDOW_SIZE=4096
export ASSISTANT_STRIDE=1024
export LONG_SAMPLE_POLICY=window
export SGLANG_MEM_FRACTION_STATIC=0.72
export SGLANG_CHUNKED_PREFILL_SIZE=2048
export SGLANG_CUDA_GRAPH_MAX_BS=4
export TAU_KEY=q_0.0200
bash seman_memcot/scripts/run_full_pipeline.sh
## 当前实现状态

这套 `seman_memcot` 流程目前已经覆盖：

- assistant-side `4096` 滑窗打分
- 低质量碎片段合并/过滤
- protected token 保护
- `LIMIT_ROWS` 小规模试跑
- shell 脚本全流程入口
- 断点续跑和参数一致性保护

还需要你在目标服务器上补做的一步是：在项目根目录执行一次完整 pytest；如果服务器上还没有 `pytest`，先安装它再跑。

```bash
pytest tests -v
```

如果你是在一个依赖不完整的精简环境里先做预检查，至少先跑 `compileall`、`--help`、shell 语法检查和几条针对性的 smoke check；但在正式跑服务器任务前，仍然建议补一次完整 `pytest`。
