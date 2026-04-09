#!/usr/bin/env python3
"""Merge per-rank tau partials into one tau_candidates.json."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic_aware.io_utils import read_json, write_json, write_runtime_metadata
from semantic_aware.tau_estimation import compute_quantile


MERGE_META_KEYS = [
    "backend",
    "model",
    "max_length",
    "batch_size",
    "assistant_window_size",
    "long_sample_policy",
    "limit_rows",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Per-rank partial JSON paths")
    parser.add_argument("--output", required=True, help="Output JSON path for merged tau candidates")
    parser.add_argument(
        "--tau_quantiles",
        default="0.005,0.01,0.015,0.02",
        help="Comma-separated quantiles to estimate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    quantiles = [float(item) for item in args.tau_quantiles.split(",") if item]
    usable_confidences = []
    source_indices = []
    score_seconds_total = 0.0
    rows_seen = 0
    rows_scored = 0
    token_count = 0
    merged_config = None

    for input_path in args.inputs:
        payload = read_json(input_path)
        usable_confidences.extend(payload.get("usable_confidences", []))
        source_indices.extend(payload.get("source_indices", []))
        meta = payload.get("_meta", {})
        current_config = {key: meta.get(key) for key in MERGE_META_KEYS}
        if merged_config is None:
            merged_config = current_config
        elif current_config != merged_config:
            mismatches = [
                f"{key}: expected {merged_config[key]!r}, got {current_config[key]!r}"
                for key in MERGE_META_KEYS
                if current_config[key] != merged_config[key]
            ]
            raise ValueError("Mismatched tau partial configs: " + "; ".join(mismatches))
        score_seconds_total += meta.get("score_seconds_total", 0.0)
        rows_seen += meta.get("rows_seen", 0)
        rows_scored += meta.get("rows_scored", 0)
        token_count += meta.get("token_count", 0)
    result = (
        {
            f"q_{quantile:.4f}": compute_quantile(usable_confidences, quantile)
            for quantile in quantiles
        }
        if usable_confidences
        else {f"q_{quantile:.4f}": None for quantile in quantiles}
    )
    result["_meta"] = {
        **(merged_config or {key: None for key in MERGE_META_KEYS}),
        "partial_count": len(args.inputs),
        "sample_count": len(set(source_indices)),
        "rows_seen": rows_seen,
        "rows_scored": rows_scored,
        "token_count": token_count,
        "score_seconds_total": round(score_seconds_total, 6),
    }
    write_json(args.output, result)
    write_runtime_metadata(args.output, result["_meta"])


if __name__ == "__main__":
    main()
