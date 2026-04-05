#!/usr/bin/env python3
"""Sample source indices for tau estimation."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic_aware.io_utils import iter_jsonl, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--sample_size", required=True, type=int, help="Number of rows to sample")
    parser.add_argument("--seed", default=42, type=int, help="Sampling seed")
    parser.add_argument(
        "--limit_rows",
        default=None,
        type=int,
        help="Optional cap on the number of input rows considered from the top of the dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Keep tau sampling aligned with any smoke-test subset used downstream.
    total_rows = sum(1 for _ in iter_jsonl(args.input, limit_rows=args.limit_rows))
    if args.sample_size > total_rows:
        raise ValueError(f"sample_size={args.sample_size} exceeds total_rows={total_rows}")

    rng = random.Random(args.seed)
    sampled = sorted(rng.sample(range(total_rows), args.sample_size))
    write_json(args.output, sampled)


if __name__ == "__main__":
    main()
