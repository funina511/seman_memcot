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

from semantic_aware.boundary import pick_boundaries, split_text_by_token_boundaries
from semantic_aware.exporter import build_output_record
from semantic_aware.io_utils import (
    append_jsonl,
    iter_jsonl,
    load_progress,
    save_progress,
    write_runtime_metadata,
)
from semantic_aware.protected_tokens import build_cuttable_mask, find_protected_spans
from semantic_aware.role_extract import extract_roles
from semantic_aware.scoring import load_model_and_tokenizer
from semantic_aware.scoring_backends import get_scoring_backend


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--tau", required=True, type=float, help="Chosen tau value")
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
    return parser.parse_args()


def main():
    args = parse_args()
    progress = load_progress(args.progress)
    if progress.get("finished"):
        # A finished shard should be a cheap no-op on resume.
        return
    last_source_idx = progress.get("last_source_idx", -1)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    scoring_backend = get_scoring_backend(
        backend_name=args.backend,
        model=model,
        tokenizer=tokenizer,
    )

    written = progress.get("num_written", 0)
    skipped = progress.get("num_skipped", 0)
    overlong = progress.get("num_overlong", 0)
    rows_seen = 0
    rows_scored = 0
    token_count = 0
    score_seconds_total = 0.0

    for source_idx, obj in enumerate(iter_jsonl(args.input, limit_rows=args.limit_rows)):
        if source_idx % args.world_size != args.rank:
            continue
        if source_idx <= last_source_idx:
            continue
        rows_seen += 1

        system_prompt, question, assistant = extract_roles(obj)
        if not question or not assistant:
            skipped += 1
            progress = {
                "rank": args.rank,
                "world_size": args.world_size,
                "last_source_idx": source_idx,
                "num_written": written,
                "num_skipped": skipped,
                "num_overlong": overlong,
                "finished": False,
            }
            save_progress(args.progress, progress)
            continue

        started = time.perf_counter()
        token_ids, offsets, confidences = scoring_backend.score_assistant_tokens(
            system_prompt=system_prompt,
            question=question,
            assistant_text=assistant,
            assistant_window_size=args.assistant_window_size,
        )
        score_seconds_total += time.perf_counter() - started
        rows_scored += 1
        token_count += len(token_ids)
        if len(token_ids) > args.max_length:
            overlong += 1
            # `skip` drops already-scored overlong rows from output while keeping their counts.
            if args.long_sample_policy == "skip":
                progress = {
                    "rank": args.rank,
                    "world_size": args.world_size,
                    "last_source_idx": source_idx,
                    "num_written": written,
                    "num_skipped": skipped,
                    "num_overlong": overlong,
                    "finished": False,
                }
                save_progress(args.progress, progress)
                continue
        protected_spans = find_protected_spans(assistant)
        cuttable_mask = build_cuttable_mask(
            token_ids=token_ids,
            offsets=offsets,
            special_ids=getattr(tokenizer, "all_special_ids", []),
            protected_spans=protected_spans,
        )
        boundaries = pick_boundaries(
            confidences=confidences,
            cuttable_mask=cuttable_mask,
            tau=args.tau,
            min_step_tokens=args.min_step_tokens,
        )
        thoughts_list = split_text_by_token_boundaries(
            text=assistant,
            offsets=offsets,
            boundaries=boundaries,
            min_step_chars=args.min_step_chars,
        )
        append_jsonl(
            args.output,
            build_output_record(
                source_idx=source_idx,
                system_prompt=system_prompt,
                question=question,
                gt_output=assistant,
                thoughts_list=thoughts_list,
            ),
        )

        written += 1
        progress = {
            "rank": args.rank,
            "world_size": args.world_size,
            "last_source_idx": source_idx,
            "num_written": written,
            "num_skipped": skipped,
            "num_overlong": overlong,
            "finished": False,
        }
        save_progress(args.progress, progress)

    progress = {
        "rank": args.rank,
        "world_size": args.world_size,
        "last_source_idx": progress.get("last_source_idx", last_source_idx),
        "num_written": written,
        "num_skipped": skipped,
        "num_overlong": overlong,
        "finished": True,
    }
    save_progress(args.progress, progress)
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
            "num_overlong": overlong,
            "score_seconds_total": round(score_seconds_total, 6),
            "score_seconds_avg": round(score_seconds_avg, 6),
            "assistant_window_size": args.assistant_window_size,
            "long_sample_policy": args.long_sample_policy,
        },
    )


if __name__ == "__main__":
    main()
