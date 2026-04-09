#!/usr/bin/env python3
"""Estimate tau candidates from a sampled subset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic_aware.io_utils import iter_jsonl, read_json, write_json, write_runtime_metadata
from semantic_aware.protected_tokens import build_cuttable_mask, find_protected_spans
from semantic_aware.role_extract import extract_roles
from semantic_aware.scoring import load_model_and_tokenizer
from semantic_aware.scoring_backends import get_scoring_backend
from semantic_aware.tau_estimation import compute_quantile, estimate_tau_from_records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--sampled_indices", required=True, help="JSON list of sampled source indices")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--output", default=None, help="Output JSON path for tau candidates")
    parser.add_argument(
        "--partial_output",
        default=None,
        help="Optional per-rank partial JSON path used before merge_tau_candidates.py",
    )
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "sglang"],
        help="Scoring backend name",
    )
    parser.add_argument("--rank", default=0, type=int, help="Current worker rank for sampled-index partitioning")
    parser.add_argument("--world_size", default=1, type=int, help="Total tau worker count")
    parser.add_argument(
        "--tau_quantiles",
        default="0.005,0.01,0.015,0.02",
        help="Comma-separated quantiles to estimate",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype name")
    parser.add_argument("--trust_remote_code", default=1, type=int, help="Pass 0 to disable trust_remote_code")
    parser.add_argument("--max_length", default=8192, type=int, help="Length threshold recorded for future checks")
    parser.add_argument("--batch_size", default=1, type=int, help="Reserved for future batching support")
    parser.add_argument("--max_samples", default=None, type=int, help="Optional cap for sampled rows")
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
        help="Whether already-scored over-max_length rows stay in tau estimation or are dropped after counting them",
    )
    return parser.parse_args()


def _select_rank_indices(sampled_indices, *, rank, world_size):
    sorted_indices = sorted(sampled_indices)
    return {
        source_idx
        for position, source_idx in enumerate(sorted_indices)
        if position % world_size == rank
    }


def _build_runtime_metadata(
    *,
    args,
    rows_seen,
    rows_scored,
    sample_count,
    token_count,
    overlong,
    score_seconds_total,
):
    score_seconds_avg = score_seconds_total / rows_scored if rows_scored else 0.0
    return {
        "backend": args.backend,
        "model": args.model,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "limit_rows": args.limit_rows,
        "rank": args.rank,
        "world_size": args.world_size,
        "rows_seen": rows_seen,
        "rows_scored": rows_scored,
        "sample_count": sample_count,
        "token_count": token_count,
        "num_overlong": overlong,
        "score_seconds_total": round(score_seconds_total, 6),
        "score_seconds_avg": round(score_seconds_avg, 6),
        "assistant_window_size": args.assistant_window_size,
        "long_sample_policy": args.long_sample_policy,
    }


def main():
    args = parse_args()
    if args.output is None and args.partial_output is None:
        raise ValueError("At least one of --output or --partial_output must be provided.")
    if args.world_size <= 0:
        raise ValueError("world_size must be positive.")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size).")

    quantiles = [float(item) for item in args.tau_quantiles.split(",") if item]
    sampled_indices = read_json(args.sampled_indices)
    if args.max_samples is not None:
        sampled_indices = sorted(sampled_indices)[: args.max_samples]
    assigned_indices = _select_rank_indices(
        sampled_indices,
        rank=args.rank,
        world_size=args.world_size,
    )

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

    records = []
    usable_confidences = []
    source_indices = []
    overlong = 0
    rows_seen = 0
    rows_scored = 0
    token_count = 0
    score_seconds_total = 0.0

    for source_idx, obj in enumerate(iter_jsonl(args.input, limit_rows=args.limit_rows)):
        if source_idx not in assigned_indices:
            continue
        rows_seen += 1
        system_prompt, question, assistant = extract_roles(obj)
        if not question or not assistant:
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
            # `skip` drops already-scored overlong rows from tau estimation while keeping their counts.
            if args.long_sample_policy == "skip":
                continue
        protected_spans = find_protected_spans(assistant)
        cuttable_mask = build_cuttable_mask(
            token_ids=token_ids,
            offsets=offsets,
            special_ids=getattr(tokenizer, "all_special_ids", []),
            protected_spans=protected_spans,
        )
        source_indices.append(source_idx)
        usable_confidences.extend(
            confidence
            for confidence, is_cuttable in zip(confidences, cuttable_mask)
            if is_cuttable
        )
        records.append(
            {
                "source_idx": source_idx,
                "confidences": confidences,
                "cuttable_mask": cuttable_mask,
            }
        )

    runtime_metadata = _build_runtime_metadata(
        args=args,
        rows_seen=rows_seen,
        rows_scored=rows_scored,
        sample_count=len(records),
        token_count=token_count,
        overlong=overlong,
        score_seconds_total=score_seconds_total,
    )

    if args.partial_output:
        partial_payload = {
            "source_indices": source_indices,
            "usable_confidences": usable_confidences,
            "_meta": {
                **runtime_metadata,
            },
        }
        write_json(args.partial_output, partial_payload)
        write_runtime_metadata(args.partial_output, runtime_metadata)

    if args.output:
        result = estimate_tau_from_records(records, quantiles) if records else {
            f"q_{quantile:.4f}": None for quantile in quantiles
        }
        result["_meta"] = {
            **runtime_metadata,
        }
        write_json(args.output, result)
        write_runtime_metadata(args.output, runtime_metadata)


if __name__ == "__main__":
    main()
