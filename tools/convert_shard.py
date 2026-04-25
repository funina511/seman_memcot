#!/usr/bin/env python3
"""Convert one dataset shard into LightThinker training JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic_aware.boundary import (
    pick_boundaries,
    pick_fixed_token_boundaries,
    pick_random_token_boundaries,
    split_text_by_token_boundaries,
)
from semantic_aware.exporter import build_output_record_from_reference
from semantic_aware.io_utils import (
    append_jsonl,
    iter_jsonl,
    load_progress,
    save_progress,
    write_runtime_metadata,
)
from semantic_aware.protected_tokens import build_cuttable_mask, find_protected_spans
from semantic_aware.scoring import count_scoring_windows, load_tokenizer, tokenize_prompt_and_assistant
from semantic_aware.scoring_backends import init_scoring_backend


def _count_output_rows(path):
    row_count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row_count += 1
    return row_count


def _validate_resume_consistency(*, progress, progress_exists, output_rows, output_path, progress_path, args):
    if not progress_exists:
        if output_rows > 0:
            raise RuntimeError(
                "Progress file is missing but shard output already exists. "
                f"output={output_path} has {output_rows} rows while progress={progress_path} is absent. "
                "Refusing to append because this can duplicate data. "
                "Please restore/remove both files for this shard, then rerun."
            )
        return

    expected_written = progress.get("num_written")
    if expected_written is None:
        raise RuntimeError(
            f"Progress file {progress_path} is missing required key 'num_written'. "
            "Please repair/remove this shard progress before resuming."
        )
    if expected_written != output_rows:
        raise RuntimeError(
            "Progress/output mismatch detected for shard resume: "
            f"progress num_written={expected_written}, output rows={output_rows}. "
            f"progress={progress_path}, output={output_path}. "
            "Refusing to append to avoid duplicate samples."
        )

    saved_rank = progress.get("rank")
    if saved_rank is not None and saved_rank != args.rank:
        raise RuntimeError(
            f"Progress rank mismatch: saved rank={saved_rank}, current rank={args.rank}. "
            f"progress={progress_path}."
        )
    saved_world_size = progress.get("world_size")
    if saved_world_size is not None and saved_world_size != args.world_size:
        raise RuntimeError(
            f"Progress world_size mismatch: saved world_size={saved_world_size}, "
            f"current world_size={args.world_size}. progress={progress_path}."
        )

    rows_seen = progress.get("rows_seen")
    if rows_seen is not None and rows_seen < expected_written:
        raise RuntimeError(
            f"Progress rows_seen={rows_seen} is smaller than num_written={expected_written}. "
            f"progress={progress_path}."
        )
    rows_scored = progress.get("rows_scored")
    if (
        getattr(args, "segmentation_mode", "threshold") == "threshold"
        and rows_scored is not None
        and rows_scored < expected_written
    ):
        raise RuntimeError(
            f"Progress rows_scored={rows_scored} is smaller than num_written={expected_written}. "
            f"progress={progress_path}."
        )

    if progress.get("finished") and progress.get("last_source_idx", -1) < 0 and expected_written > 0:
        raise RuntimeError(
            f"Progress is marked finished but last_source_idx is invalid. progress={progress_path}."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--tau", default=None, type=float, help="Chosen tau value for threshold segmentation")
    parser.add_argument(
        "--segmentation_mode",
        default="threshold",
        choices=["threshold", "fixed", "random"],
        help="Segmentation mode: threshold scores model confidences; fixed/random use tokenizer-only token counts",
    )
    parser.add_argument(
        "--fixed_segment_tokens",
        default=128,
        type=int,
        help="Assistant-token interval for fixed segmentation",
    )
    parser.add_argument(
        "--random_min_segment_tokens",
        default=64,
        type=int,
        help="Minimum assistant-token segment size for random segmentation",
    )
    parser.add_argument(
        "--random_max_segment_tokens",
        default=256,
        type=int,
        help="Maximum assistant-token segment size for random segmentation",
    )
    parser.add_argument(
        "--random_seed",
        default=42,
        type=int,
        help="Base seed for deterministic per-row random segmentation",
    )
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "sglang"],
        help="Scoring backend name",
    )
    parser.add_argument("--rank", required=True, type=int, help="Current shard rank")
    parser.add_argument("--world_size", required=True, type=int, help="Total shard count")
    parser.add_argument("--output", required=True, help="Output JSONL path for this shard")
    parser.add_argument("--progress", required=True, help="Progress JSON path for this shard")
    parser.add_argument(
        "--reference_train_jsonl",
        default=str(PROJECT_ROOT.parent / "RRcot" / "data" / "train" / "train.jsonl"),
        help="Reference LightThinker train.jsonl; all fields except thoughts_list are inherited",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype name")
    parser.add_argument("--trust_remote_code", default=1, type=int, help="Pass 0 to disable trust_remote_code")
    parser.add_argument("--max_length", default=8192, type=int, help="Length threshold used for overlong statistics")
    parser.add_argument("--batch_size", default=1, type=int, help="Reserved for future batching support")
    parser.add_argument("--min_step_tokens", default=12, type=int, help="Minimum token distance between boundaries")
    parser.add_argument("--min_step_chars", default=8, type=int, help="Minimum segment character length")
    parser.add_argument(
        "--assistant_window_size",
        default=4096,
        type=int,
        help="Maximum assistant tokens scored per teacher-forcing window",
    )
    parser.add_argument(
        "--assistant_stride",
        default=None,
        type=int,
        help="Optional stride between scoring windows; defaults to one quarter of the window size",
    )
    parser.add_argument(
        "--limit_rows",
        default=None,
        type=int,
        help="Optional cap on the number of input rows iterated from the top of the dataset",
    )
    parser.add_argument(
        "--long_sample_policy",
        default="window",
        choices=["window", "skip"],
        help="Whether already-scored over-max_length rows stay in output or are dropped after counting them",
    )
    args = parser.parse_args()
    _apply_segmentation_defaults(args)
    _validate_segmentation_args(args, parser.error)
    return args


def _apply_segmentation_defaults(args):
    defaults = {
        "segmentation_mode": "threshold",
        "fixed_segment_tokens": 128,
        "random_min_segment_tokens": 64,
        "random_max_segment_tokens": 256,
        "random_seed": 42,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def _validate_segmentation_args(args, error):
    if args.segmentation_mode == "threshold" and args.tau is None:
        error("--tau is required when --segmentation_mode=threshold")
    if args.fixed_segment_tokens <= 0:
        error("--fixed_segment_tokens must be positive")
    if args.random_min_segment_tokens <= 0:
        error("--random_min_segment_tokens must be positive")
    if args.random_max_segment_tokens <= 0:
        error("--random_max_segment_tokens must be positive")
    if args.random_min_segment_tokens > args.random_max_segment_tokens:
        error("--random_min_segment_tokens must be <= --random_max_segment_tokens")


def _cuda_peak_memory_mb():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)


def main():
    args = parse_args()
    _apply_segmentation_defaults(args)
    _validate_segmentation_args(
        args,
        lambda message: (_ for _ in ()).throw(ValueError(message)),
    )
    output_path = Path(args.output)
    progress_path = Path(args.progress)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=True)
    if not Path(args.reference_train_jsonl).exists():
        raise ValueError(f"reference_train_jsonl not found: {args.reference_train_jsonl}")

    progress_exists = progress_path.exists()
    progress = load_progress(args.progress)
    output_rows = _count_output_rows(args.output)
    _validate_resume_consistency(
        progress=progress,
        progress_exists=progress_exists,
        output_rows=output_rows,
        output_path=output_path,
        progress_path=progress_path,
        args=args,
    )
    if progress.get("finished"):
        # A finished shard should be a cheap no-op on resume.
        return
    last_source_idx = progress.get("last_source_idx", -1)

    if args.segmentation_mode == "threshold":
        scoring_backend, tokenizer = init_scoring_backend(
            backend_name=args.backend,
            model_name=args.model,
            dtype=args.dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
    else:
        scoring_backend = None
        tokenizer = load_tokenizer(
            args.model,
            trust_remote_code=bool(args.trust_remote_code),
        )
    reference_iter = iter_jsonl(args.reference_train_jsonl, limit_rows=args.limit_rows)

    written = progress.get("num_written", 0)
    skipped = progress.get("num_skipped", 0)
    overlong = progress.get("num_overlong", 0)
    rows_seen = progress.get("rows_seen", 0)
    rows_scored = progress.get("rows_scored", 0)
    token_count = progress.get("token_count", 0)
    window_count = progress.get("window_count", 0)
    scored_token_count = progress.get("scored_token_count", 0)
    score_seconds_total = progress.get("score_seconds_total", 0.0)
    current_last_source_idx = last_source_idx

    def persist_progress(*, finished):
        save_progress(
            args.progress,
            {
                "rank": args.rank,
                "world_size": args.world_size,
                "last_source_idx": current_last_source_idx,
                "num_written": written,
                "num_skipped": skipped,
                "num_overlong": overlong,
                "rows_seen": rows_seen,
                "rows_scored": rows_scored,
                "token_count": token_count,
                "score_seconds_total": score_seconds_total,
                "window_count": window_count,
                "scored_token_count": scored_token_count,
                "finished": finished,
            },
        )

    try:
        import torch
    except ImportError:
        pass
    else:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    for source_idx, _ in enumerate(iter_jsonl(args.input, limit_rows=args.limit_rows)):
        try:
            reference_record = next(reference_iter)
        except StopIteration as error:
            raise ValueError(
                "reference_train_jsonl has fewer rows than input; "
                f"missing row for source_idx={source_idx}."
            ) from error

        if source_idx % args.world_size != args.rank:
            continue
        if source_idx <= last_source_idx:
            continue
        rows_seen += 1
        current_last_source_idx = source_idx

        reference_source_idx = reference_record.get("source_idx")
        if reference_source_idx is not None and reference_source_idx != source_idx:
            raise ValueError(
                "source_idx mismatch between input and reference_train_jsonl: "
                f"source_idx={source_idx}, reference_source_idx={reference_source_idx}"
            )

        system_prompt = reference_record.get("system_prompt", "") or ""
        question = reference_record.get("question", "") or ""
        assistant = reference_record.get("gt_output", "") or ""
        if not question or not assistant:
            skipped += 1
            persist_progress(finished=False)
            continue

        if args.segmentation_mode in {"fixed", "random"}:
            _, token_ids, offsets = tokenize_prompt_and_assistant(
                tokenizer,
                system_prompt=system_prompt,
                question=question,
                assistant_text=assistant,
            )
            confidences = []
        else:
            token_ids = offsets = confidences = None

        if args.long_sample_policy == "skip":
            # Skip overlong rows before model scoring to avoid avoidable OOM spikes.
            if token_ids is None:
                try:
                    _, preview_assistant_ids, _ = tokenize_prompt_and_assistant(
                        tokenizer,
                        system_prompt=system_prompt,
                        question=question,
                        assistant_text=assistant,
                    )
                except Exception:
                    preview_assistant_ids = None
            else:
                preview_assistant_ids = token_ids
            if preview_assistant_ids is not None and len(preview_assistant_ids) > args.max_length:
                overlong += 1
                persist_progress(finished=False)
                continue

        if args.segmentation_mode == "threshold":
            started = time.perf_counter()
            token_ids, offsets, confidences = scoring_backend.score_assistant_tokens(
                system_prompt=system_prompt,
                question=question,
                assistant_text=assistant,
                assistant_window_size=args.assistant_window_size,
                assistant_stride=args.assistant_stride,
            )
            score_seconds_total += time.perf_counter() - started
            rows_scored += 1
        token_count += len(token_ids)
        if args.segmentation_mode == "threshold":
            window_count += count_scoring_windows(
                prefix_ids=[],
                assistant_ids=token_ids,
                assistant_window_size=args.assistant_window_size,
                assistant_stride=args.assistant_stride,
            )
        scored_token_count += len(confidences)
        if len(token_ids) > args.max_length:
            overlong += 1
            # `skip` drops already-scored overlong rows from output while keeping their counts.
            if args.long_sample_policy == "skip":
                persist_progress(finished=False)
                continue
        protected_spans = find_protected_spans(assistant)
        cuttable_mask = build_cuttable_mask(
            token_ids=token_ids,
            offsets=offsets,
            special_ids=getattr(tokenizer, "all_special_ids", []),
            protected_spans=protected_spans,
        )
        if args.segmentation_mode == "threshold":
            boundaries = pick_boundaries(
                confidences=confidences,
                cuttable_mask=cuttable_mask,
                tau=args.tau,
                min_step_tokens=args.min_step_tokens,
                text=assistant,
                offsets=offsets,
            )
        elif args.segmentation_mode == "fixed":
            boundaries = pick_fixed_token_boundaries(
                token_count=len(token_ids),
                cuttable_mask=cuttable_mask,
                segment_tokens=args.fixed_segment_tokens,
                min_step_tokens=args.min_step_tokens,
                text=assistant,
                offsets=offsets,
            )
        else:
            boundaries = pick_random_token_boundaries(
                token_count=len(token_ids),
                cuttable_mask=cuttable_mask,
                min_segment_tokens=args.random_min_segment_tokens,
                max_segment_tokens=args.random_max_segment_tokens,
                min_step_tokens=args.min_step_tokens,
                source_idx=source_idx,
                random_seed=args.random_seed,
                text=assistant,
                offsets=offsets,
            )
        thoughts_list = split_text_by_token_boundaries(
            text=assistant,
            offsets=offsets,
            boundaries=boundaries,
            min_step_chars=args.min_step_chars,
        )
        append_jsonl(
            args.output,
            build_output_record_from_reference(
                reference_record=reference_record,
                thoughts_list=thoughts_list,
                source_idx=source_idx,
            ),
        )

        written += 1
        persist_progress(finished=False)

    persist_progress(finished=True)
    score_seconds_avg = score_seconds_total / rows_scored if rows_scored else 0.0
    write_runtime_metadata(
        args.output,
        {
            "backend": args.backend,
            "model": args.model,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "limit_rows": args.limit_rows,
            "tau": args.tau,
            "rank": args.rank,
            "world_size": args.world_size,
            "rows_seen": rows_seen,
            "rows_scored": rows_scored,
            "rows_written": written,
            "rows_skipped": skipped,
            "token_count": token_count,
            "window_count": window_count,
            "scored_token_count": scored_token_count,
            "num_overlong": overlong,
            "score_seconds_total": round(score_seconds_total, 6),
            "score_seconds_avg": round(score_seconds_avg, 6),
            "score_seconds_per_token": round(score_seconds_total / token_count, 8) if token_count else 0.0,
            "cuda_max_memory_allocated_mb": _cuda_peak_memory_mb(),
            "assistant_window_size": args.assistant_window_size,
            "assistant_stride": args.assistant_stride,
            "long_sample_policy": args.long_sample_policy,
            "segmentation_mode": args.segmentation_mode,
            "fixed_segment_tokens": args.fixed_segment_tokens,
            "random_min_segment_tokens": args.random_min_segment_tokens,
            "random_max_segment_tokens": args.random_max_segment_tokens,
            "random_seed": args.random_seed,
        },
    )


if __name__ == "__main__":
    main()
