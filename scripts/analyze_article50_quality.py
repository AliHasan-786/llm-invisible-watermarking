#!/usr/bin/env python3
"""Compute pinned GPT-2 perplexity and paired H5 quality summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.article50_analysis import GPT2_REFERENCE, paired_quality_summary  # noqa: E402
from pipeline.article50 import read_jsonl, write_json_atomic, write_jsonl_atomic  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output_dir = args.output_dir or args.completions.parent / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(args.completions) if row["split"] == "heldout"]

    # Load the pinned reference directly here; the legacy helper accepts only a name.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(**GPT2_REFERENCE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(**GPT2_REFERENCE).to(args.device)
    model.eval()
    scored = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        encoded = tokenizer(
            [row["completion"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(args.device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, :-1, :].contiguous()
        labels = encoded["input_ids"][:, 1:].contiguous()
        mask = encoded["attention_mask"][:, 1:].float()
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape(labels.shape)
        perplexities = torch.exp((losses * mask).sum(1) / mask.sum(1).clamp(min=1))
        for row, perplexity in zip(batch, perplexities.detach().cpu().tolist()):
            scored.append(
                {
                    "prompt_id": row["prompt_id"],
                    "source": row["source"],
                    "split": row["split"],
                    "scheme": row["scheme"],
                    "n_tokens": row["n_tokens"],
                    "perplexity": perplexity,
                    "reference_model": GPT2_REFERENCE,
                }
            )
    write_jsonl_atomic(output_dir / "perplexity.jsonl", scored)
    summary = paired_quality_summary(scored)
    write_json_atomic(
        output_dir / "quality_summary.json",
        {"reference_model": GPT2_REFERENCE, "schemes": summary},
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
