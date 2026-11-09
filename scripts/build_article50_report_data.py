#!/usr/bin/env python3
"""Build the traceable headline table from confirmatory cache artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.article50 import write_json_atomic  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root: Path):
    rows = []
    sources = []
    for path in sorted(root.glob("**/headline_by_condition.json")):
        if "pilot" in path.parts:
            raise ValueError(f"pilot artifact cannot enter report data: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            if row.get("result_scope") != "confirmatory":
                raise ValueError(f"non-confirmatory result in {path}")
            rows.append({**row, "source_artifact": str(path.relative_to(REPO))})
        sources.append({"path": str(path.relative_to(REPO)), "sha256": digest(path)})
    return rows, sources


def markdown(rows):
    lines = [
        "| Detector | Scheme | Condition | Calibration FPR (95% CI) | Held-out FPR | TPR (95% CI) | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cal_ci = row["realized_calibration_fpr_ci_95"]
        tpr_ci = row["heldout_tpr_ci_95"]
        lines.append(
            "| {detector} | {scheme} | {condition} | {cal:.1%} "
            "[{cal_low:.1%}, {cal_high:.1%}] | {fpr:.1%} | {tpr:.1%} "
            "[{tpr_low:.1%}, {tpr_high:.1%}] | {n} |".format(
                detector=row["detector"],
                scheme=row["scheme"],
                condition=row["condition"],
                cal=row["realized_calibration_fpr"],
                cal_low=cal_ci[0],
                cal_high=cal_ci[1],
                fpr=row["heldout_fpr"],
                tpr=row["heldout_tpr"],
                tpr_low=tpr_ci[0],
                tpr_high=tpr_ci[1],
                n=row["complete_case_prompts"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=REPO / "results/article50/confirmatory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "results/article50/report"
    )
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    rows, sources = collect(args.root)
    if not rows and not args.allow_empty:
        raise SystemExit("no confirmatory headline artifacts found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / "headline.json",
        {"rows": rows, "sources": sources, "row_count": len(rows)},
    )
    (args.output_dir / "headline.md").write_text(markdown(rows), encoding="utf-8")
    print(f"Built {len(rows)} report rows from {len(sources)} verified artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
