#!/usr/bin/env python3
"""Produce paired SynthID-minus-Kirchenbauer TPR differences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import (  # noqa: E402
    calibration_summary,
    paired_tpr_difference,
)
from pipeline.article50 import read_jsonl, write_json_atomic  # noqa: E402


def _calibration(path: Path):
    return [row for row in read_jsonl(path) if row["split"] == "calibration"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthid-scores", type=Path, required=True)
    parser.add_argument("--kirchenbauer-scores", type=Path, required=True)
    parser.add_argument("--synthid-calibration", type=Path)
    parser.add_argument("--kirchenbauer-calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    synthid_rows = read_jsonl(args.synthid_scores)
    kirchenbauer_rows = read_jsonl(args.kirchenbauer_scores)
    synthid_calibration = _calibration(
        args.synthid_calibration or args.synthid_scores
    )
    kirchenbauer_calibration = _calibration(
        args.kirchenbauer_calibration or args.kirchenbauer_scores
    )
    synthid_threshold = calibration_summary(
        synthid_calibration, n_bootstrap=args.bootstrap
    )["threshold"]
    kirchenbauer_threshold = calibration_summary(
        kirchenbauer_calibration, n_bootstrap=args.bootstrap
    )["threshold"]

    conditions = sorted(
        {row.get("condition", "clean") for row in synthid_rows}
        & {row.get("condition", "clean") for row in kirchenbauer_rows}
    )
    comparisons = []
    for condition in conditions:
        synthid_condition = [
            row
            for row in synthid_rows
            if row.get("condition", "clean") == condition
        ]
        kirchenbauer_condition = [
            row
            for row in kirchenbauer_rows
            if row.get("condition", "clean") == condition
        ]
        result = paired_tpr_difference(
            synthid_condition,
            kirchenbauer_condition,
            synthid_threshold=float(synthid_threshold),
            kirchenbauer_threshold=float(kirchenbauer_threshold),
            n_bootstrap=args.bootstrap,
        )
        result.update(
            {
                "condition": condition,
                "synthid_threshold": synthid_threshold,
                "kirchenbauer_threshold": kirchenbauer_threshold,
            }
        )
        comparisons.append(result)
    write_json_atomic(args.output, comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
