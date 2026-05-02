"""
Phase 3: Detection + perplexity evaluation on the headline Gemma corpus.

Runtime: ~15-30 min (perplexity scoring with GPT-2).
Input:   results/corpus_gemma3_d2.jsonl
Output:  results/headline_gemma_summary.json
         results/headline_gemma_zscores.npz
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from evaluation.metrics import compute_z_scores, compute_perplexity

CORPUS_PATH = "results/corpus_gemma3_d2.jsonl"
SUMMARY_PATH = "results/headline_gemma_summary.json"
ZSCORES_PATH = "results/headline_gemma_zscores.npz"

MODEL_NAME = "google/gemma-3-4b-it"
GAMMA = 0.5
SEED = 42
TARGET_FPR = 0.01
MIN_TOKENS_FOR_TPR = 150  # acceptance criterion: TPR > 0.90 for completions >= 150 tokens

if not os.path.exists(CORPUS_PATH):
    print(f"ERROR: {CORPUS_PATH} not found. Run scripts/run_headline_gemma.py first.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading corpus...")
corpus = load_corpus(CORPUS_PATH)
wm_items = [x for x in corpus if x["watermarked"]]
uwm_items = [x for x in corpus if not x["watermarked"]]
print(f"  {len(wm_items)} watermarked, {len(uwm_items)} unwatermarked")

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
print(f"  Vocab size: {vocab_size:,}")

detector = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)

print("Computing z-scores...")
wm_z, uwm_z = compute_z_scores(corpus, detector, tokenizer)

calibrated_threshold = detector.calibrate_threshold(uwm_z, target_fpr=TARGET_FPR)

# Use the calibrated threshold as the single source of truth
tpr_all = sum(z > calibrated_threshold for z in wm_z) / len(wm_z) if wm_z else 0.0
fpr_actual = sum(z > calibrated_threshold for z in uwm_z) / len(uwm_z) if uwm_z else 0.0

# TPR for completions >= 150 tokens
long_wm_z = [z for item, z in zip(wm_items, wm_z) if item["n_tokens"] >= MIN_TOKENS_FOR_TPR]
tpr_long = sum(z > calibrated_threshold for z in long_wm_z) / len(long_wm_z) if long_wm_z else 0.0

print(f"  Calibrated z-threshold: {calibrated_threshold:.3f}")
print(f"  TPR @ 1% FPR (all lengths): {tpr_all:.1%}")
print(f"  TPR @ 1% FPR (>={MIN_TOKENS_FOR_TPR} tokens): {tpr_long:.1%}  [acceptance: >0.90]")
print(f"  Actual FPR: {fpr_actual:.2%}")

if tpr_long < 0.90:
    print("  WARNING: TPR < 0.90 for long completions — check seed/hash alignment between processor and detector.")

print("\nComputing perplexity with GPT-2...")
wm_texts = [x["completion"] for x in wm_items]
uwm_texts = [x["completion"] for x in uwm_items]

wm_ppl = compute_perplexity(wm_texts, model_name="gpt2", device=device)
uwm_ppl = compute_perplexity(uwm_texts, model_name="gpt2", device=device)

mean_wm_ppl = float(np.mean(wm_ppl))
median_wm_ppl = float(np.median(wm_ppl))
mean_uwm_ppl = float(np.mean(uwm_ppl))
median_uwm_ppl = float(np.median(uwm_ppl))
ppl_ratio = mean_wm_ppl / mean_uwm_ppl if mean_uwm_ppl > 0 else float("nan")

print(f"  Watermarked perplexity:   mean={mean_wm_ppl:.2f}, median={median_wm_ppl:.2f}")
print(f"  Unwatermarked perplexity: mean={mean_uwm_ppl:.2f}, median={median_uwm_ppl:.2f}")
print(f"  Perplexity ratio (wm/uwm): {ppl_ratio:.3f}")

summary = {
    "model": MODEL_NAME,
    "delta": 2.0,
    "gamma": GAMMA,
    "seed": SEED,
    "n_watermarked": len(wm_items),
    "n_unwatermarked": len(uwm_items),
    "calibrated_z_threshold": float(calibrated_threshold),
    "tpr_at_1pct_fpr_all": float(tpr_all),
    f"tpr_at_1pct_fpr_ge{MIN_TOKENS_FOR_TPR}tok": float(tpr_long),
    "n_long_wm": len(long_wm_z),
    "actual_fpr": float(fpr_actual),
    "mean_ppl_wm": mean_wm_ppl,
    "median_ppl_wm": median_wm_ppl,
    "mean_ppl_uwm": mean_uwm_ppl,
    "median_ppl_uwm": median_uwm_ppl,
    "ppl_ratio_wm_over_uwm": ppl_ratio,
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to {SUMMARY_PATH}")

np.savez(
    ZSCORES_PATH,
    wm_z=np.array(wm_z),
    uwm_z=np.array(uwm_z),
    wm_ppl=np.array(wm_ppl),
    uwm_ppl=np.array(uwm_ppl),
)
print(f"Saved z-scores and perplexities to {ZSCORES_PATH}")
