#!/usr/bin/env python3
"""Merge shard outputs into one sorted JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semantic_aware.io_utils import iter_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input shard JSONL files")
    parser.add_argument("--output", required=True, help="Merged JSONL path")
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for path in args.inputs:
        records.extend(iter_jsonl(path))
    records.sort(key=lambda item: item["source_idx"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
