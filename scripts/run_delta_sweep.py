"""
Phase 6: Delta sweep — quality vs. detectability tradeoff.

For each delta in [0.5, 1.0, 2.0, 4.0, 8.0], generate 100 prompts per dataset
using the SAME prompt set (deterministic sampling from the headline prompts), then
compute TPR and GPT-2 perplexity. The d=2.0 corpus is reused from Phase 2.

Two-pass design (avoids T4 OOM):
  Pass 1: load Gemma once, generate corpora for all deltas, release.
  Pass 2: load GPT-2 once, score perplexity + detection for all deltas.

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

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from watermark.logits_processor import WatermarkLogitsProcessor
from evaluation.metrics import compute_z_scores

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

# Pre-compute calibrated threshold from d=2.0 unwatermarked z-scores
# (unwatermarked distribution doesn't depend on delta — reuse it)
print("Pre-computing unwatermarked z-scores from d=2.0 for calibration...")
detector_ref = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)
_, uwm_z_ref = compute_z_scores(headline, detector_ref, tokenizer)
uwm_z_ref_arr = sorted(uwm_z_ref)
n_uwm = len(uwm_z_ref_arr)
idx_calib = int((1 - TARGET_FPR) * n_uwm)
calibrated_threshold = uwm_z_ref_arr[min(idx_calib, n_uwm - 1)]
print(f"  Shared calibrated threshold: {calibrated_threshold:.3f}")


# ─── Pass 1: Generation across all deltas (Gemma only) ──────────────────────

print("\n=== PASS 1: Generation across all deltas ===")
print("Loading Gemma-3 4B once (will be reused for all deltas)...")
gemma_tokenizer = tokenizer  # already loaded above
gemma_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)
gemma_model.eval()


def generate_for_delta(delta, fixed_prompts):
    corpus_path = f"results/corpus_gemma3_d{delta}.jsonl"
    if delta == 2.0 and os.path.exists(HEADLINE_CORPUS):
        print(f"  delta={delta}: reusing {HEADLINE_CORPUS}")
        return HEADLINE_CORPUS

    print(f"  delta={delta}: generating to {corpus_path}")
    processor = WatermarkLogitsProcessor(
        vocab_size=vocab_size, delta=delta, gamma=GAMMA, seed=SEED,
    )

    completed_keys = set()
    existing = []
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.append(item)
                    completed_keys.add((item["source"], item["prompt"], item["watermarked"]))
        print(f"    Resuming: {len(existing)} existing samples")

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    file_mode = "a" if existing else "w"
    with open(corpus_path, file_mode) as f:
        for i, prompt_item in enumerate(fixed_prompts):
            for wm in [True, False]:
                key = (prompt_item["source"], prompt_item["prompt"], wm)
                if key in completed_keys:
                    continue
                try:
                    inputs = gemma_tokenizer(
                        prompt_item["prompt"], return_tensors="pt",
                        truncation=True, max_length=512,
                    ).to(device)
                    gen_kwargs = dict(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                        pad_token_id=gemma_tokenizer.pad_token_id,
                    )
                    if wm:
                        gen_kwargs["logits_processor"] = [processor]
                    with torch.no_grad():
                        output_ids = gemma_model.generate(**gen_kwargs)
                    prompt_len = inputs["input_ids"].shape[1]
                    completion_ids = output_ids[0, prompt_len:].tolist()
                    completion_text = gemma_tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    )
                    record = {
                        "prompt": prompt_item["prompt"],
                        "completion": completion_text,
                        "token_ids": completion_ids,
                        "n_tokens": len(completion_ids),
                        "watermarked": wm,
                        "source": prompt_item["source"],
                        "model": MODEL_NAME,
                        "delta": delta,
                        "gamma": GAMMA,
                        "seed": SEED,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"    Error on prompt {i} (wm={wm}): {e}")
            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{len(fixed_prompts)}] prompts done")
    return corpus_path


corpus_paths = {}
for delta in DELTAS:
    corpus_paths[delta] = generate_for_delta(delta, fixed_prompts)

# Release Gemma before loading GPT-2
print("\nReleasing Gemma to free GPU memory...")
del gemma_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ─── Pass 2: Detection + perplexity (GPT-2 loaded, Gemma released) ──────────

print("\n=== PASS 2: Evaluation across all deltas ===")
print("Loading GPT-2 for perplexity scoring...")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
if gpt2_tokenizer.pad_token is None:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
gpt2_model.eval()


def ppl_batch(texts, max_length=512, batch_size=8):
    import torch.nn as nn
    perplexities = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = gpt2_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
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


sweep_results = []
for delta in DELTAS:
    print(f"\n--- delta={delta} ---")
    corpus = load_corpus(corpus_paths[delta])
    # If we reused the headline corpus for d=2.0, filter to fixed prompt set
    if delta == 2.0:
        fixed_prompt_set = set(p["prompt"] for p in fixed_prompts)
        corpus = [x for x in corpus if x["prompt"] in fixed_prompt_set]

    wm_items = [x for x in corpus if x["watermarked"]]
    uwm_items = [x for x in corpus if not x["watermarked"]]
    print(f"  {len(wm_items)} wm, {len(uwm_items)} uwm")

    detector_d = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)
    wm_z_d, _ = compute_z_scores(corpus, detector_d, tokenizer)
    tpr = sum(z > calibrated_threshold for z in wm_z_d) / len(wm_z_d) if wm_z_d else 0.0
    print(f"  TPR @ 1% FPR: {tpr:.3f}")

    wm_texts = [x["completion"] for x in wm_items[:50]]
    uwm_texts = [x["completion"] for x in uwm_items[:50]]
    wm_ppl = ppl_batch(wm_texts)
    uwm_ppl = ppl_batch(uwm_texts)
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
