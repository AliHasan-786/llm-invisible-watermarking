"""
Phase 6: Delta sweep — quality vs. detectability tradeoff.

For each delta in [0.5, 1.0, 2.0, 4.0, 8.0], generate 100 prompts per dataset
using the SAME prompt set (deterministic sampling from the headline prompts), then
compute TPR and GPT-2 perplexity. The d=2.0 corpus is reused from Phase 2.

Runtime: ~2-3 hours (run overnight).
Output:  results/corpus_gemma3_d{delta}.jsonl  (for each delta != 2.0)
         results/delta_sweep_gemma.json
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pipeline.generate import CorpusGenerator, load_corpus
from watermark.detector import WatermarkDetector
from evaluation.metrics import compute_z_scores, compute_tpr_at_fpr, compute_perplexity

DELTAS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_PER_DATASET = 100
MODEL_NAME = "google/gemma-3-4b-it"
GAMMA = 0.5
SEED = 42
TARGET_FPR = 0.01
HEADLINE_CORPUS = "results/corpus_gemma3_d2.jsonl"
OUTPUT_PATH = "results/delta_sweep_gemma.json"

if not os.path.exists(HEADLINE_CORPUS):
    print(f"ERROR: {HEADLINE_CORPUS} not found. Run run_headline_gemma.py first.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Build deterministic prompt list from the headline corpus (100 per dataset, same across all deltas)
random.seed(SEED)
print("Sampling fixed prompt set from headline corpus...")
headline = load_corpus(HEADLINE_CORPUS)
# Collect one entry per (source, prompt) — deduplicate
seen_prompts: dict[str, list] = {}
for item in headline:
    src = item["source"]
    if src not in seen_prompts:
        seen_prompts[src] = []
    if item["prompt"] not in [p["prompt"] for p in seen_prompts[src]]:
        seen_prompts[src].append({"source": src, "prompt": item["prompt"]})

fixed_prompts = []
for src, items in seen_prompts.items():
    random.shuffle(items)
    fixed_prompts.extend(items[:N_PER_DATASET])
print(f"Fixed prompt set: {len(fixed_prompts)} prompts across {len(seen_prompts)} datasets")

# Load tokenizer once (reused for detection across all deltas)
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
vocab_size = len(tokenizer)

# Load GPT-2 reference model once for perplexity
print("Loading GPT-2 reference for perplexity scoring...")
from transformers import AutoTokenizer as HFTok, AutoModelForCausalLM as HFCLM
gpt2_tokenizer = HFTok.from_pretrained("gpt2")
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = HFCLM.from_pretrained("gpt2").to(device)
gpt2_model.eval()

# Pre-compute unwatermarked z-scores from d=2.0 corpus for calibration
# (unwatermarked distribution doesn't depend on delta — reuse it)
print("Pre-computing unwatermarked z-scores from d=2.0 for calibration...")
d2_corpus = load_corpus(HEADLINE_CORPUS)
detector_ref = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)
_, uwm_z_ref = compute_z_scores(d2_corpus, detector_ref, tokenizer)
uwm_z_ref_arr = sorted(uwm_z_ref)
n_uwm = len(uwm_z_ref_arr)
idx_calib = int((1 - TARGET_FPR) * n_uwm)
calibrated_threshold = uwm_z_ref_arr[min(idx_calib, n_uwm - 1)]
print(f"  Shared calibrated threshold: {calibrated_threshold:.3f}")

sweep_results = []

for delta in DELTAS:
    corpus_path = f"results/corpus_gemma3_d{delta}.jsonl"
    print(f"\n{'='*60}")
    print(f"Delta = {delta}")

    # For d=2.0, reuse the existing corpus
    if delta == 2.0 and os.path.exists(HEADLINE_CORPUS):
        print(f"  Reusing existing corpus from {HEADLINE_CORPUS}")
        corpus_path = HEADLINE_CORPUS
        corpus = load_corpus(corpus_path)
        # Filter down to N_PER_DATASET prompts per dataset
        fixed_prompt_set = set(p["prompt"] for p in fixed_prompts)
        corpus = [x for x in corpus if x["prompt"] in fixed_prompt_set]
    else:
        # Generate corpus for this delta
        gen = CorpusGenerator(
            model_name=MODEL_NAME,
            delta=delta,
            gamma=GAMMA,
            seed=SEED,
            max_new_tokens=200,
        )
        # Override the generator's prompts by monkey-patching _load_prompts
        def _fixed_load_prompts(n_per_dataset=None, _fp=fixed_prompts):
            return list(_fp)
        gen._load_prompts = _fixed_load_prompts

        corpus = gen.generate_corpus(
            n_per_dataset=N_PER_DATASET,
            output_path=corpus_path,
            resume=True,
        )
        del gen  # free GPU memory before perplexity step
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    wm_items = [x for x in corpus if x["watermarked"]]
    uwm_items = [x for x in corpus if not x["watermarked"]]
    print(f"  Corpus: {len(wm_items)} wm, {len(uwm_items)} uwm")

    # Detection with this delta's watermark processor (detector uses same gamma/seed)
    detector_d = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)
    wm_z_d, _ = compute_z_scores(corpus, detector_d, tokenizer)
    tpr = sum(z > calibrated_threshold for z in wm_z_d) / len(wm_z_d) if wm_z_d else 0.0
    print(f"  TPR @ 1% FPR: {tpr:.3f}")

    # Perplexity with GPT-2
    print("  Computing perplexity...")
    wm_texts = [x["completion"] for x in wm_items[:50]]
    uwm_texts = [x["completion"] for x in uwm_items[:50]]

    # Use compute_perplexity with already-loaded gpt2 (pass model/tokenizer directly via inline loop)
    def _ppl_batch(texts, max_length=512, batch_size=8):
        import torch
        import torch.nn as nn
        perplexities = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = gpt2_tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length
            ).to(device)
            with torch.no_grad():
                out = gpt2_model(**enc)
                logits = out.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = enc["input_ids"][:, 1:].contiguous()
                mask = enc["attention_mask"][:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                tok_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_logits.size(0), -1)
                masked = tok_loss * mask.float()
                n_tok = mask.float().sum(dim=1).clamp(min=1)
                ppl = torch.exp(masked.sum(dim=1) / n_tok).cpu().tolist()
                perplexities.extend(ppl)
        return perplexities

    wm_ppl = _ppl_batch(wm_texts)
    uwm_ppl = _ppl_batch(uwm_texts)
    mean_wm_ppl = float(np.mean(wm_ppl))
    mean_uwm_ppl = float(np.mean(uwm_ppl))
    ppl_ratio = mean_wm_ppl / mean_uwm_ppl if mean_uwm_ppl > 0 else float("nan")
    print(f"  PPL wm={mean_wm_ppl:.2f}  uwm={mean_uwm_ppl:.2f}  ratio={ppl_ratio:.3f}")

    sweep_results.append({
        "delta": delta,
        "tpr": float(tpr),
        "mean_ppl_wm": mean_wm_ppl,
        "mean_ppl_uwm": mean_uwm_ppl,
        "ppl_ratio": ppl_ratio,
        "n_wm": len(wm_items),
        "n_uwm": len(uwm_items),
        "calibrated_threshold": float(calibrated_threshold),
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(sweep_results, f, indent=2)
print(f"\nSaved delta sweep results to {OUTPUT_PATH}")
print("Expected: TPR and PPL both rise monotonically with delta. Knee ~delta=2.0.")
