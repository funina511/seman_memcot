#!/usr/bin/env python3
"""Estimate tau candidates from a sampled subset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic_aware.io_utils import iter_jsonl, read_json, write_json
from semantic_aware.protected_tokens import build_cuttable_mask, find_protected_spans
from semantic_aware.role_extract import extract_roles
from semantic_aware.scoring import load_model_and_tokenizer, score_assistant_tokens
from semantic_aware.tau_estimation import estimate_tau_from_records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--sampled_indices", required=True, help="JSON list of sampled source indices")
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument("--output", required=True, help="Output JSON path for tau candidates")
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


def main():
    args = parse_args()
    quantiles = [float(item) for item in args.tau_quantiles.split(",") if item]
    sampled_indices = set(read_json(args.sampled_indices))
    if args.max_samples is not None:
        sampled_indices = set(sorted(sampled_indices)[: args.max_samples])

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    records = []
    overlong = 0
    for source_idx, obj in enumerate(iter_jsonl(args.input, limit_rows=args.limit_rows)):
        if source_idx not in sampled_indices:
            continue
        system_prompt, question, assistant = extract_roles(obj)
        if not question or not assistant:
            continue
        token_ids, offsets, confidences = score_assistant_tokens(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            question=question,
            assistant_text=assistant,
            assistant_window_size=args.assistant_window_size,
        )
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
        records.append(
            {
                "source_idx": source_idx,
                "confidences": confidences,
                "cuttable_mask": cuttable_mask,
            }
        )

    result = estimate_tau_from_records(records, quantiles)
    result["_meta"] = {
        "model": args.model,
        "sample_count": len(records),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "assistant_window_size": args.assistant_window_size,
        "limit_rows": args.limit_rows,
        "long_sample_policy": args.long_sample_policy,
        "num_overlong": overlong,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
