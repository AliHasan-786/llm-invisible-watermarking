"""
Robustness sweep: word substitution, token insertion/deletion, optional LLM paraphrase.

Reads the calibrated threshold from the detection summary (run eval_detection.py first),
then tests whether the watermark survives adversarial post-processing.

Usage:
    # Basic (no LLM paraphrase — fast)
    python scripts/eval_robustness.py

    # With LLM paraphrase attack (loads the model — ~60 min extra)
    python scripts/eval_robustness.py --with-paraphrase

    # Gemma 2 secondary (basic only)
    python scripts/eval_robustness.py --model google/gemma-2-9b-it

Output:
    results/robustness_{model_slug}.json
    results/robustness_{model_slug}_zscores.npz

Runtime: ~10-30 min without paraphrase; ~90 min with --with-paraphrase.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from evaluation.robustness import evaluate_robustness


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
    p.add_argument("--detection-summary", default=None,
                   help="JSON summary from eval_detection.py (provides calibrated threshold)")
    p.add_argument("--output", default=None)
    p.add_argument("--output-zscores", default=None)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-paraphrase", action="store_true",
                   help="Include LLM paraphrase attack (loads the full model, slow)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantization for the paraphraser (required on free Colab T4)")
    return p.parse_args()


def main():
    args = parse_args()
    slug = model_slug(args.model)

    corpus_path  = args.corpus            or f"results/corpus_{slug}_d2.jsonl"
    summary_path = args.detection_summary or f"results/detection_{slug}_summary.json"
    output_path  = args.output            or f"results/robustness_{slug}.json"
    zscores_path = args.output_zscores    or f"results/robustness_{slug}_zscores.npz"

    for path in (corpus_path, summary_path):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found.")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")

    with open(summary_path) as f:
        det_summary = json.load(f)
    calibrated_threshold = det_summary["calibrated_z_threshold"]
    print(f"Calibrated threshold: {calibrated_threshold:.3f}  (from {summary_path})")

    corpus    = load_corpus(corpus_path)
    wm_items  = [x for x in corpus if x["watermarked"]]
    print(f"Corpus: {len(wm_items)} watermarked samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    detector   = WatermarkDetector(vocab_size=vocab_size, gamma=args.gamma, seed=args.seed,
                                   z_threshold=calibrated_threshold)

    paraphraser_model = None
    if args.with_paraphrase:
        print(f"Loading {args.model} as paraphraser (this loads the full model)...")
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            paraphraser_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            paraphraser_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
            )
        paraphraser_model.eval()

    print("Running robustness evaluation...")
    raw_results = evaluate_robustness(
        corpus=corpus,
        detector=detector,
        tokenizer=tokenizer,
        substitution_rates=[0.05, 0.10, 0.15, 0.20],
        paraphraser_model=paraphraser_model,
        paraphraser_tokenizer=tokenizer if paraphraser_model else None,
        device=device,
    )

    print(f"\n{'Condition':<35} {'n':>5} {'mean_z':>8} {'median_z':>9} {'TPR@1%FPR':>11}")
    print("-" * 70)

    summary_out = {}
    all_z_arrays = {}
    for condition, detection_results in raw_results.items():
        z_vals = np.array([r.z_score for r in detection_results])
        tpr = float(np.mean(z_vals > calibrated_threshold))
        summary_out[condition] = {
            "n": len(detection_results),
            "mean_z": float(np.mean(z_vals)),
            "median_z": float(np.median(z_vals)),
            "tpr_at_1pct_fpr": tpr,
        }
        all_z_arrays[condition] = z_vals
        print(f"{condition:<35} {len(detection_results):>5} {np.mean(z_vals):>8.2f} "
              f"{np.median(z_vals):>9.2f} {tpr:>10.1%}")

    print("\nExpected: baseline > sub_5pct > sub_10pct > ... > llm_paraphrase")

    with open(output_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved robustness summary → {output_path}")

    np.savez(zscores_path, **all_z_arrays)
    print(f"Saved raw z-scores      → {zscores_path}")


if __name__ == "__main__":
    main()
