#!/usr/bin/env python3
"""Score Article 50 completions and emit clean held-out headline metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.metrics import headline_detection_summary  # noqa: E402
from pipeline.article50 import read_jsonl, write_json_atomic, write_jsonl_atomic  # noqa: E402
from watermark.detector import WatermarkDetector  # noqa: E402
from watermark.synthid import SynthIDWeightedMeanDetector  # noqa: E402


def score_rows(
    rows,
    detector_name: str,
    eos_token_id: int | None,
    vocab_size: int | None,
):
    if detector_name == "kirchenbauer":
        recorded_sizes = {int(row["tokenizer_vocab_size"]) for row in rows if "tokenizer_vocab_size" in row}
        if vocab_size is None:
            if len(recorded_sizes) != 1:
                raise ValueError(
                    "Kirchenbauer scoring requires one recorded tokenizer_vocab_size "
                    "or an explicit --vocab-size"
                )
            vocab_size = recorded_sizes.pop()
        detector = WatermarkDetector(vocab_size=vocab_size, gamma=0.5, seed=42)
    elif detector_name == "synthid":
        detector = SynthIDWeightedMeanDetector(device="cpu")
    else:
        raise ValueError(detector_name)

    scored = []
    failures = []
    for row in rows:
        try:
            if detector_name == "kirchenbauer":
                result = detector.score_sequence(row["token_ids"])
                score = result.z_score
                scored_tokens = result.total_tokens
            else:
                result = detector.score_sequence(
                    row["token_ids"],
                    eos_token_id=eos_token_id,
                )
                score = result.score
                scored_tokens = result.scored_ngrams
            scored.append(
                {
                    "prompt_id": row["prompt_id"],
                    "split": row["split"],
                    "scheme": row["scheme"],
                    "replicate": row["replicate"],
                    "detector": detector_name,
                    "score": score,
                    "scored_tokens": scored_tokens,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "prompt_id": row["prompt_id"],
                    "split": row["split"],
                    "scheme": row["scheme"],
                    "replicate": row["replicate"],
                    "detector": detector_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return scored, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completions", type=Path)
    parser.add_argument("--detector", choices=("kirchenbauer", "synthid"), required=True)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    output_dir = args.output_dir or args.completions.parent / f"scores_{args.detector}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.completions)
    scored, failures = score_rows(
        rows,
        args.detector,
        args.eos_token_id,
        args.vocab_size,
    )
    write_jsonl_atomic(output_dir / "scores.jsonl", scored)
    write_jsonl_atomic(output_dir / "score_failures.jsonl", failures)

    calibration = [row for row in scored if row["split"] == "calibration"]
    heldout = [row for row in scored if row["split"] == "heldout"]
    positive_scheme = "kirchenbauer" if args.detector == "kirchenbauer" else "synthid"
    headline = headline_detection_summary(
        calibration,
        heldout,
        positive_scheme=positive_scheme,
        n_bootstrap=args.bootstrap,
    )
    headline.update(
        {
            "detector": args.detector,
            "score_failures": len(failures),
            "result_scope": "pilot"
            if "pilot" in args.completions.parts
            else "confirmatory",
        }
    )
    write_json_atomic(output_dir / "headline_clean.json", headline)
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
