"""
Phase 7: LLaMA 3.1 8B replication — detectability + word substitution only.

If LLaMA access is denied on HuggingFace, this script exits cleanly and
writes results/SKIPPED.md with the reason.

Runtime: ~1-2 hours.
Output:  results/corpus_llama_d2.jsonl
         results/llama_replication_summary.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer

from pipeline.generate import CorpusGenerator, load_corpus
from watermark.detector import WatermarkDetector
from evaluation.metrics import compute_z_scores, compute_tpr_at_fpr
from evaluation.robustness import evaluate_robustness

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA_SUMMARY = "results/headline_gemma_summary.json"
CORPUS_PATH = "results/corpus_llama_d2.jsonl"
SUMMARY_PATH = "results/llama_replication_summary.json"
SKIPPED_PATH = "results/SKIPPED.md"

DELTA = 2.0
GAMMA = 0.5
SEED = 42
N_PROMPTS = 200  # ~67 per dataset
TARGET_FPR = 0.01
MIN_TOKENS = 150

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load Gemma calibrated threshold for consistent comparison
calibrated_threshold = None
if os.path.exists(GEMMA_SUMMARY):
    with open(GEMMA_SUMMARY) as f:
        gemma_summ = json.load(f)
    calibrated_threshold = gemma_summ["calibrated_z_threshold"]
    print(f"Using Gemma calibrated threshold: {calibrated_threshold:.3f}")

# Try loading LLaMA tokenizer first to catch access errors early
try:
    print(f"Checking access to {LLAMA_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  Access confirmed.")
except Exception as e:
    msg = (
        f"LLaMA access denied or unavailable: {e}\n\n"
        "Phase 7 skipped. Report claim downgraded from 'two model families' to "
        "'Gemma-3 4B (LLaMA replication left to future work due to access/compute constraints)'."
    )
    print(f"\nWARNING: {msg}")
    os.makedirs("results", exist_ok=True)
    with open(SKIPPED_PATH, "a") as f:
        f.write(f"## Phase 7 — LLaMA Replication\n\n{msg}\n\n")
    sys.exit(0)

# Generate corpus
n_per_dataset = N_PROMPTS // 3
print(f"\nGenerating {N_PROMPTS} prompts ({n_per_dataset} per dataset) with {LLAMA_MODEL}...")
try:
    gen = CorpusGenerator(
        model_name=LLAMA_MODEL,
        delta=DELTA,
        gamma=GAMMA,
        seed=SEED,
        max_new_tokens=200,
    )
    corpus = gen.generate_corpus(
        n_per_dataset=n_per_dataset,
        output_path=CORPUS_PATH,
        resume=True,
    )
except Exception as e:
    msg = f"LLaMA generation failed: {e}"
    print(f"\nERROR: {msg}")
    with open(SKIPPED_PATH, "a") as f:
        f.write(f"## Phase 7 — LLaMA Replication\n\n{msg}\n\n")
    sys.exit(1)

wm_items = [x for x in corpus if x["watermarked"]]
uwm_items = [x for x in corpus if not x["watermarked"]]
print(f"  {len(wm_items)} watermarked, {len(uwm_items)} unwatermarked")

vocab_size = len(tokenizer)
detector = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)

wm_z, uwm_z = compute_z_scores(corpus, detector, tokenizer)

# Calibrate on this corpus's unwatermarked samples
threshold_llama, tpr_all, fpr_actual = compute_tpr_at_fpr(wm_z, uwm_z, TARGET_FPR)
long_wm_z = [z for item, z in zip(wm_items, wm_z) if item["n_tokens"] >= MIN_TOKENS]
tpr_long = sum(z > threshold_llama for z in long_wm_z) / len(long_wm_z) if long_wm_z else 0.0

print(f"  LLaMA calibrated threshold: {threshold_llama:.3f}")
print(f"  TPR @ 1% FPR (all):         {tpr_all:.1%}")
print(f"  TPR @ 1% FPR (>={MIN_TOKENS}tok): {tpr_long:.1%}  [target: >0.90]")

# Word substitution at 10%
rob = evaluate_robustness(corpus, detector, tokenizer, substitution_rates=[0.10])
sub10_results = rob["word_sub_10pct"]
tpr_sub10 = sum(r.z_score > threshold_llama for r in sub10_results) / len(sub10_results)
print(f"  TPR after 10% word sub:     {tpr_sub10:.1%}")

summary = {
    "model": LLAMA_MODEL,
    "delta": DELTA,
    "gamma": GAMMA,
    "seed": SEED,
    "n_watermarked": len(wm_items),
    "n_unwatermarked": len(uwm_items),
    "calibrated_z_threshold": float(threshold_llama),
    "tpr_at_1pct_fpr_all": float(tpr_all),
    f"tpr_at_1pct_fpr_ge{MIN_TOKENS}tok": float(tpr_long),
    "tpr_word_sub_10pct": float(tpr_sub10),
    "actual_fpr": float(fpr_actual),
}

with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved LLaMA replication summary to {SUMMARY_PATH}")
