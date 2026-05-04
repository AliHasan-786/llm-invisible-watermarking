"""
Length-stratified detectability curves.

Truncates every sequence to token count N, re-calibrates the detection threshold
at each length bin, and records TPR. TPR should rise monotonically with length.

No generation required — pure post-hoc analysis of an existing corpus.

Usage:
    python scripts/eval_length.py
    python scripts/eval_length.py --model google/gemma-2-9b-it

Output:
    results/length_curves_{model_slug}.json

Runtime: ~10 min.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from evaluation.metrics import compute_tpr_at_fpr


LENGTH_BINS = [25, 50, 75, 100, 125, 150, 175, 200]


def model_slug(model_name: str) -> str:
    name = model_name.lower().split("/")[-1]
    for key in ("llama", "gemma", "mistral", "falcon", "phi"):
        if key in name:
            return key
    return name.split("-")[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--corpus", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    slug = model_slug(args.model)

    corpus_path = args.corpus or f"results/corpus_{slug}_d2.jsonl"
    output_path = args.output or f"results/length_curves_{slug}.json"

    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found. Run generate_corpus.py first.")
        sys.exit(1)

    print(f"Model:   {args.model}")
    print(f"Corpus:  {corpus_path}")
    corpus = load_corpus(corpus_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    detector = WatermarkDetector(vocab_size=len(tokenizer), gamma=args.gamma, seed=args.seed)

    wm_items  = [x for x in corpus if x["watermarked"]]
    uwm_items = [x for x in corpus if not x["watermarked"]]
    print(f"  {len(wm_items)} watermarked, {len(uwm_items)} unwatermarked\n")

    results = []
    for n_tok in LENGTH_BINS:
        wm_at  = [x for x in wm_items  if len(x["token_ids"]) >= n_tok]
        uwm_at = [x for x in uwm_items if len(x["token_ids"]) >= n_tok]

        if len(wm_at) < 10 or len(uwm_at) < 10:
            print(f"  n_tokens={n_tok:4d}  SKIPPED (n_wm={len(wm_at)}, n_uwm={len(uwm_at)})")
            continue

        wm_z  = [detector.score_sequence(x["token_ids"][:n_tok]).z_score for x in wm_at]
        uwm_z = [detector.score_sequence(x["token_ids"][:n_tok]).z_score for x in uwm_at]
        threshold, tpr, fpr = compute_tpr_at_fpr(wm_z, uwm_z, args.target_fpr)

        print(f"  n_tokens={n_tok:4d}  TPR={tpr:.3f}  threshold={threshold:.3f}  "
              f"n_wm={len(wm_at)}  n_uwm={len(uwm_at)}")
        results.append({
            "n_tokens": n_tok,
            "tpr": float(tpr),
            "fpr": float(fpr),
            "threshold": float(threshold),
            "n_wm": len(wm_at),
            "n_uwm": len(uwm_at),
            "calibration": "per_length",
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved length curves → {output_path}")
    print("Expected: TPR rises monotonically, approaching 1.0 at >=150 tokens.")


if __name__ == "__main__":
    main()
