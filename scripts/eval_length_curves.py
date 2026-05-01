"""
Phase 4: Length-stratified detectability curves.

Re-calibrates threshold per length bin (honest: avoids leaking the full-corpus
threshold into truncated evaluations). TPR vs n_tokens should rise monotonically.

Runtime: ~10 min (pure post-hoc analysis, no generation).
Input:   results/corpus_gemma3_d2.jsonl
         results/headline_gemma_summary.json  (for calibrated threshold reference)
Output:  results/length_curves_gemma.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformers import AutoTokenizer

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from evaluation.metrics import compute_tpr_at_fpr

CORPUS_PATH = "results/corpus_gemma3_d2.jsonl"
SUMMARY_PATH = "results/headline_gemma_summary.json"
OUTPUT_PATH = "results/length_curves_gemma.json"

MODEL_NAME = "google/gemma-3-4b-it"
GAMMA = 0.5
SEED = 42
TARGET_FPR = 0.01
LENGTH_BINS = [25, 50, 75, 100, 125, 150, 175, 200]

if not os.path.exists(CORPUS_PATH):
    print(f"ERROR: {CORPUS_PATH} not found.")
    sys.exit(1)

print("Loading corpus and tokenizer...")
corpus = load_corpus(CORPUS_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vocab_size = len(tokenizer)
detector = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)

wm_items = [x for x in corpus if x["watermarked"]]
uwm_items = [x for x in corpus if not x["watermarked"]]
print(f"  {len(wm_items)} watermarked, {len(uwm_items)} unwatermarked")

# Pre-compute full-length z-scores for unwatermarked (for calibration)
print("Computing unwatermarked z-scores (full length, for per-bin calibration)...")
uwm_z_full = [detector.score_sequence(item["token_ids"]).z_score for item in uwm_items]

length_curve = []
for n_tok in LENGTH_BINS:
    # Truncate watermarked token_ids to n_tok and recompute z-scores
    wm_z_truncated = []
    for item in wm_items:
        truncated = item["token_ids"][:n_tok]
        r = detector.score_sequence(truncated)
        wm_z_truncated.append(r.z_score)

    # Also truncate unwatermarked to n_tok for per-length calibration
    uwm_z_truncated = []
    for item in uwm_items:
        truncated = item["token_ids"][:n_tok]
        r = detector.score_sequence(truncated)
        uwm_z_truncated.append(r.z_score)

    threshold, tpr, fpr = compute_tpr_at_fpr(wm_z_truncated, uwm_z_truncated, TARGET_FPR)
    n_valid = sum(1 for item in wm_items if len(item["token_ids"]) >= n_tok)

    print(f"  n_tokens={n_tok:4d}  TPR={tpr:.3f}  threshold={threshold:.3f}  n_full_length={n_valid}")
    length_curve.append({
        "n_tokens": n_tok,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "threshold": float(threshold),
        "n_wm_with_full_length": n_valid,
        "calibration": "per_length",
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(length_curve, f, indent=2)
print(f"\nSaved length curves to {OUTPUT_PATH}")
print("Expected: TPR rises monotonically with n_tokens, approaching 1.0 at >=150 tokens.")
