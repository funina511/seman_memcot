"""JSONL and progress IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys


def iter_jsonl(path, limit_rows=None):
    """Yield decoded JSON objects from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        yielded_rows = 0
        for line in handle:
            if limit_rows is not None and yielded_rows >= limit_rows:
                break
            line = line.strip()
            if line:
                yielded_rows += 1
                yield json.loads(line)


def append_jsonl(path, obj):
    """Append one JSON object to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path, obj):
    """Write JSON data to disk."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def read_json(path):
    """Read JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_progress(path):
    """Load progress metadata if present, otherwise return an empty record."""
    progress_path = Path(path)
    if not progress_path.exists():
        return {}
    try:
        return read_json(progress_path)
    except json.JSONDecodeError:
        recovery_path = progress_path.with_name(f"{progress_path.name}.corrupt")
        suffix = 1
        while recovery_path.exists():
            recovery_path = progress_path.with_name(f"{progress_path.name}.corrupt.{suffix}")
            suffix += 1
        progress_path.replace(recovery_path)
        print(
            f"Warning: progress file {progress_path} was corrupt and was moved to {recovery_path}; resuming from empty progress.",
            file=sys.stderr,
        )
        return {}


def save_progress(path, progress):
    """Persist progress metadata."""
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=progress_path.parent,
        delete=False,
    ) as handle:
        json.dump(progress, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(progress_path)
