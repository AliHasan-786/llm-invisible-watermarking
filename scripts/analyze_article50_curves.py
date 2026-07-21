#!/usr/bin/env python3
"""Build ROC/DET and secondary operating-point artifacts from score caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.article50_analysis import (  # noqa: E402
    operating_point_supported,
    roc_det_curve,
)
from evaluation.metrics import calibrate_threshold, realized_fpr  # noqa: E402
from pipeline.article50 import read_jsonl, write_json_atomic  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("--calibration-scores", type=Path)
    parser.add_argument("--positive-scheme", choices=("kirchenbauer", "synthid"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.scores)
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    if not calibration_rows and args.calibration_scores:
        calibration_rows = [
            row
            for row in read_jsonl(args.calibration_scores)
            if row["split"] == "calibration"
        ]
    calibration = [float(row["score"]) for row in calibration_rows]
    if not calibration:
        raise ValueError("calibration scores are required")
    heldout = [row for row in rows if row["split"] == "heldout"]
    conditions = sorted({row.get("condition", "clean") for row in heldout})
    artifacts = []
    for condition in conditions:
        condition_rows = [row for row in heldout if row.get("condition", "clean") == condition]
        positive = [
            float(row["score"])
            for row in condition_rows
            if row["scheme"] == args.positive_scheme
        ]
        controls = [
            float(row["score"])
            for row in condition_rows
            if row["scheme"] == "control"
        ]
        curve = roc_det_curve(positive, controls)
        points = []
        for target in (0.001, 0.01, 0.05):
            supported = operating_point_supported(len(calibration), target)
            point = {"target_fpr": target, "supported": supported}
            if supported:
                threshold = calibrate_threshold(calibration, target)
                point.update(
                    {
                        "threshold": threshold,
                        "realized_calibration_fpr": realized_fpr(calibration, threshold),
                        "heldout_tpr": sum(score > threshold for score in positive) / len(positive),
                        "heldout_fpr": sum(score > threshold for score in controls) / len(controls),
                    }
                )
            points.append(point)
        artifacts.append(
            {
                "condition": condition,
                "positive_scheme": args.positive_scheme,
                "curve": curve,
                "operating_points": points,
            }
        )
    write_json_atomic(args.output, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
