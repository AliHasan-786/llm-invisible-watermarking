"""
Detection + text-quality evaluation on a watermarked corpus.

Computes z-scores for every sample, calibrates the detection threshold at 1% FPR
on the unwatermarked half, reports TPR, and scores text quality via GPT-2 perplexity.

Usage:
    # LLaMA (primary)
    python scripts/eval_detection.py

    # Gemma 2 9B (secondary)
    python scripts/eval_detection.py --model google/gemma-2-9b-it

    # Explicit paths
    python scripts/eval_detection.py --model meta-llama/Llama-3.1-8B-Instruct \
        --corpus results/corpus_llama_d2.jsonl

Output:
    results/detection_{model_slug}_summary.json   ← calibrated threshold + TPR + perplexity
    results/detection_{model_slug}_zscores.npz    ← raw z-score arrays for figure generation

Runtime: ~15-30 min (GPT-2 perplexity scoring is the slow step; no GPU needed for detection).
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from evaluation.metrics import compute_z_scores, compute_perplexity


def model_slug(model_name: str) -> str:
    name = model_name.lower().split("/")[-1]
    for key in ("llama", "gemma", "mistral", "falcon", "phi"):
        if key in name:
            return key
    return name.split("-")[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--corpus", default=None,
                   help="Corpus JSONL path (auto-derived from model slug if omitted)")
    p.add_argument("--output-summary", default=None)
    p.add_argument("--output-zscores", default=None)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--min-tokens", type=int, default=150,
                   help="Min tokens for long-completion TPR (acceptance criterion: >0.90)")
    return p.parse_args()


def main():
    args = parse_args()
    slug = model_slug(args.model)

    corpus_path   = args.corpus         or f"results/corpus_{slug}_d2.jsonl"
    summary_path  = args.output_summary or f"results/detection_{slug}_summary.json"
    zscores_path  = args.output_zscores or f"results/detection_{slug}_zscores.npz"

    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found. Run generate_corpus.py first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")
    print(f"Corpus:  {corpus_path}")

    corpus    = load_corpus(corpus_path)
    wm_items  = [x for x in corpus if x["watermarked"]]
    uwm_items = [x for x in corpus if not x["watermarked"]]
    print(f"  {len(wm_items)} watermarked, {len(uwm_items)} unwatermarked")

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  Vocab size: {vocab_size:,}")

    detector = WatermarkDetector(vocab_size=vocab_size, gamma=args.gamma, seed=args.seed)

    print("Computing z-scores...")
    wm_z, uwm_z = compute_z_scores(corpus, detector, tokenizer)

    threshold  = detector.calibrate_threshold(uwm_z, target_fpr=args.target_fpr)
    tpr_all    = sum(z > threshold for z in wm_z)  / len(wm_z)  if wm_z  else 0.0
    fpr_actual = sum(z > threshold for z in uwm_z) / len(uwm_z) if uwm_z else 0.0
    long_wm_z  = [z for item, z in zip(wm_items, wm_z) if item["n_tokens"] >= args.min_tokens]
    tpr_long   = sum(z > threshold for z in long_wm_z) / len(long_wm_z) if long_wm_z else 0.0

    print(f"  Calibrated z-threshold:              {threshold:.3f}")
    print(f"  TPR @ 1% FPR (all lengths):          {tpr_all:.1%}")
    print(f"  TPR @ 1% FPR (>={args.min_tokens}t): {tpr_long:.1%}  [target: >0.90]")
    print(f"  Actual FPR:                          {fpr_actual:.2%}")

    if tpr_long < 0.90:
        print("  WARNING: TPR < 0.90 - verify watermark seed matches between processor and detector.")

    print("\nScoring text quality with GPT-2...")
    wm_ppl  = compute_perplexity([x["completion"] for x in wm_items],  device=device)
    uwm_ppl = compute_perplexity([x["completion"] for x in uwm_items], device=device)
    mean_wm_ppl  = float(np.mean(wm_ppl))
    mean_uwm_ppl = float(np.mean(uwm_ppl))
    ppl_ratio    = mean_wm_ppl / mean_uwm_ppl if mean_uwm_ppl > 0 else float("nan")

    print(f"  Watermarked:   mean={mean_wm_ppl:.2f}  median={float(np.median(wm_ppl)):.2f}")
    print(f"  Unwatermarked: mean={mean_uwm_ppl:.2f}  median={float(np.median(uwm_ppl)):.2f}")
    print(f"  Ratio (wm/uwm): {ppl_ratio:.3f}")

    summary = {
        "model": args.model,
        "delta": 2.0,
        "gamma": args.gamma,
        "seed": args.seed,
        "n_watermarked": len(wm_items),
        "n_unwatermarked": len(uwm_items),
        "calibrated_z_threshold": float(threshold),
        "tpr_at_1pct_fpr_all": float(tpr_all),
        f"tpr_at_1pct_fpr_ge{args.min_tokens}tok": float(tpr_long),
        "n_long_wm": len(long_wm_z),
        "actual_fpr": float(fpr_actual),
        "mean_ppl_wm": mean_wm_ppl,
        "median_ppl_wm": float(np.median(wm_ppl)),
        "mean_ppl_uwm": mean_uwm_ppl,
        "median_ppl_uwm": float(np.median(uwm_ppl)),
        "ppl_ratio_wm_over_uwm": ppl_ratio,
    }

    os.makedirs(os.path.dirname(summary_path) if os.path.dirname(summary_path) else ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary   → {summary_path}")

    np.savez(zscores_path,
             wm_z=np.array(wm_z), uwm_z=np.array(uwm_z),
             wm_ppl=np.array(wm_ppl), uwm_ppl=np.array(uwm_ppl))
    print(f"Saved z-scores  → {zscores_path}")


if __name__ == "__main__":
    main()
