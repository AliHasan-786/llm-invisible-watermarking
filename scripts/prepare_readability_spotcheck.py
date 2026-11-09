#!/usr/bin/env python3
"""Prepare the preregistered blinded readability queue and separate key."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.article50_analysis import readability_queue  # noqa: E402
from pipeline.article50 import read_jsonl, write_jsonl_atomic  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.completions.parent / "readability"
    queue, key = readability_queue(read_jsonl(args.completions))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_dir / "blinded_queue.jsonl", queue)
    write_jsonl_atomic(output_dir / "answer_key.jsonl", key)
    print(f"Prepared {len(queue)} pairs; {sum(len(row['reviewers']) == 2 for row in queue)} overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
