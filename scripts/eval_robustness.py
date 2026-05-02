"""
Phase 5: Robustness sweep on the headline Gemma corpus.

Conditions: word substitution (5/10/15/20%), token deletion 10%,
            token insertion 10%, LLM paraphrase (same Gemma-3 4B model).

Runtime: ~30-90 min (paraphrasing is the slow step).
Input:   results/corpus_gemma3_d2.jsonl
         results/headline_gemma_summary.json
Output:  results/robustness_gemma.json
         results/robustness_gemma_zscores.npz
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from evaluation.robustness import evaluate_robustness

CORPUS_PATH = "results/corpus_gemma3_d2.jsonl"
SUMMARY_PATH = "results/headline_gemma_summary.json"
OUTPUT_PATH = "results/robustness_gemma.json"
ZSCORES_PATH = "results/robustness_gemma_zscores.npz"

MODEL_NAME = "google/gemma-3-4b-it"
GAMMA = 0.5
SEED = 42

if not os.path.exists(CORPUS_PATH):
    print(f"ERROR: {CORPUS_PATH} not found.")
    sys.exit(1)
if not os.path.exists(SUMMARY_PATH):
    print(f"ERROR: {SUMMARY_PATH} not found. Run eval_headline_gemma.py first.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

with open(SUMMARY_PATH) as f:
    summary = json.load(f)
calibrated_threshold = summary["calibrated_z_threshold"]
print(f"Calibrated threshold from Phase 3: {calibrated_threshold:.3f}")

print("Loading corpus...")
corpus = load_corpus(CORPUS_PATH)
wm_items = [x for x in corpus if x["watermarked"]]
print(f"  {len(wm_items)} watermarked samples")

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

detector = WatermarkDetector(
    vocab_size=vocab_size, gamma=GAMMA, seed=SEED, z_threshold=calibrated_threshold
)

print(f"Loading model {MODEL_NAME} for LLM paraphrasing...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
model.eval()

print("Running robustness evaluation (including LLM paraphrase)...")
raw_results = evaluate_robustness(
    corpus=corpus,
    detector=detector,
    tokenizer=tokenizer,
    substitution_rates=[0.05, 0.10, 0.15, 0.20],
    paraphraser_model=model,
    paraphraser_tokenizer=tokenizer,
    device=device,
)

# Summarize and save
print(f"\n{'Condition':<35} {'n':>5} {'mean_z':>8} {'median_z':>9} {'TPR@1%FPR':>11}")
print("-" * 70)

summary_results = {}
all_z_arrays = {}

for condition, detection_results in raw_results.items():
    z_vals = np.array([r.z_score for r in detection_results])
    tpr = float(np.mean(z_vals > calibrated_threshold))
    summary_results[condition] = {
        "n": len(detection_results),
        "mean_z": float(np.mean(z_vals)),
        "median_z": float(np.median(z_vals)),
        "tpr_at_1pct_fpr": tpr,
    }
    all_z_arrays[condition] = z_vals
    print(f"{condition:<35} {len(detection_results):>5} {np.mean(z_vals):>8.2f} {np.median(z_vals):>9.2f} {tpr:>10.1%}")

print("\nExpected ordering: baseline > sub_5pct > sub_10pct > ... > llm_paraphrase")
print("Insertion should hurt more than deletion (Kirchenbauer's expected result).")

with open(OUTPUT_PATH, "w") as f:
    json.dump(summary_results, f, indent=2)
print(f"\nSaved robustness summary to {OUTPUT_PATH}")

np.savez(ZSCORES_PATH, **all_z_arrays)
print(f"Saved raw z-score arrays to {ZSCORES_PATH}")
