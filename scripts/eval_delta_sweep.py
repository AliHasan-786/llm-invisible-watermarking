"""
Delta sweep: quality vs. detectability tradeoff.

Generates corpora at δ ∈ {0.5, 1.0, 2.0, 4.0, 8.0} using a fixed subset of
prompts drawn from the headline corpus, then scores TPR and GPT-2 perplexity.
The d=2.0 corpus is reused from generate_corpus.py (not regenerated).

Two-pass design to avoid GPU OOM on T4:
  Pass 1 — load the LLM once, generate all delta corpora, then release.
  Pass 2 — load GPT-2 once, score perplexity for all deltas.

Usage:
    python scripts/eval_delta_sweep.py
    python scripts/eval_delta_sweep.py --model google/gemma-2-9b-it

Output:
    results/delta_sweep_{model_slug}.json
    results/corpus_{model_slug}_d{delta}.jsonl  (for delta != 2.0)

Runtime: ~2-3 hours.
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from pipeline.generate import load_corpus
from watermark.detector import WatermarkDetector
from watermark.logits_processor import WatermarkLogitsProcessor
from evaluation.metrics import compute_z_scores


DELTAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def model_slug(model_name: str) -> str:
    name = model_name.lower().split("/")[-1]
    for key in ("llama", "gemma", "mistral", "falcon", "phi"):
        if key in name:
            return key
    return name.split("-")[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--headline-corpus", default=None,
                   help="d=2.0 corpus to reuse and sample prompts from (auto-derived if omitted)")
    p.add_argument("--output", default=None)
    p.add_argument("--n-per-dataset", type=int, default=100,
                   help="Prompts per dataset in the fixed sweep subset")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fpr", type=float, default=0.01)
    return p.parse_args()


def ppl_batch(texts, model, tokenizer, device, max_length=512, batch_size=8):
    import torch.nn as nn
    perplexities = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**enc)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = enc["input_ids"][:, 1:].contiguous()
            mask         = enc["attention_mask"][:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            tok_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), -1)
            masked = tok_loss * mask.float()
            n_tok  = mask.float().sum(dim=1).clamp(min=1)
            ppl    = torch.exp(masked.sum(dim=1) / n_tok).cpu().tolist()
            perplexities.extend(ppl)
    return perplexities


def generate_delta_corpus(delta, fixed_prompts, llm, tokenizer, vocab_size,
                           gamma, seed, slug, device, headline_corpus_path, model_name):
    corpus_path = f"results/corpus_{slug}_d{delta}.jsonl"
    if delta == 2.0 and os.path.exists(headline_corpus_path):
        print(f"  delta={delta}: reusing {headline_corpus_path}")
        return headline_corpus_path

    print(f"  delta={delta}: generating → {corpus_path}")
    processor = WatermarkLogitsProcessor(vocab_size=vocab_size, delta=delta, gamma=gamma, seed=seed)

    completed_keys: set = set()
    existing: list = []
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing.append(item)
                    completed_keys.add((item["source"], item["prompt"], item["watermarked"]))
        print(f"    Resuming: {len(existing)} existing samples")

    os.makedirs("results", exist_ok=True)
    with open(corpus_path, "a" if existing else "w") as f:
        for i, prompt_item in enumerate(fixed_prompts):
            for wm in [True, False]:
                key = (prompt_item["source"], prompt_item["prompt"], wm)
                if key in completed_keys:
                    continue
                try:
                    inputs = tokenizer(
                        prompt_item["prompt"], return_tensors="pt",
                        truncation=True, max_length=512,
                    ).to(device)
                    gen_kwargs = dict(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    if wm:
                        gen_kwargs["logits_processor"] = [processor]
                    with torch.no_grad():
                        output_ids = llm.generate(**gen_kwargs)
                    prompt_len     = inputs["input_ids"].shape[1]
                    completion_ids = output_ids[0, prompt_len:].tolist()
                    record = {
                        "prompt":      prompt_item["prompt"],
                        "completion":  tokenizer.decode(completion_ids, skip_special_tokens=True),
                        "token_ids":   completion_ids,
                        "n_tokens":    len(completion_ids),
                        "watermarked": wm,
                        "source":      prompt_item["source"],
                        "model":       model_name,
                        "delta": delta, "gamma": gamma, "seed": seed,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"    Error prompt {i} (wm={wm}): {e}")
            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{len(fixed_prompts)}] done")
    return corpus_path


def main():
    global args
    args = parse_args()
    slug = model_slug(args.model)

    headline_corpus = args.headline_corpus or f"results/corpus_{slug}_d2.jsonl"
    output_path     = args.output          or f"results/delta_sweep_{slug}.json"

    if not os.path.exists(headline_corpus):
        print(f"ERROR: {headline_corpus} not found. Run generate_corpus.py first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Model: {args.model}")

    random.seed(args.seed)
    print("Sampling fixed prompt set from headline corpus...")
    headline      = load_corpus(headline_corpus)
    seen_prompts: dict = {}
    for item in headline:
        src = item["source"]
        if src not in seen_prompts:
            seen_prompts[src] = []
        if item["prompt"] not in [p["prompt"] for p in seen_prompts[src]]:
            seen_prompts[src].append({"source": src, "prompt": item["prompt"]})

    fixed_prompts = []
    for src, items in seen_prompts.items():
        random.shuffle(items)
        fixed_prompts.extend(items[:args.n_per_dataset])
    print(f"Fixed prompt set: {len(fixed_prompts)} prompts across {len(seen_prompts)} datasets")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Calibrate threshold once from the d=2.0 unwatermarked z-scores
    # (unwatermarked distribution is independent of delta — safe to reuse)
    print("Calibrating threshold from d=2.0 corpus...")
    det_ref  = WatermarkDetector(vocab_size=vocab_size, gamma=args.gamma, seed=args.seed)
    _, uwm_z = compute_z_scores(headline, det_ref, tokenizer)
    uwm_sorted = sorted(uwm_z)
    idx = int((1 - args.target_fpr) * len(uwm_sorted))
    calibrated_threshold = uwm_sorted[min(idx, len(uwm_sorted) - 1)]
    print(f"  Shared calibrated threshold: {calibrated_threshold:.3f}")

    # ── Pass 1: generation (LLM loaded once) ─────────────────────────────────
    print(f"\n=== PASS 1: Generation ===")
    print(f"Loading {args.model}...")
    llm = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    llm.eval()

    corpus_paths = {}
    for delta in DELTAS:
        corpus_paths[delta] = generate_delta_corpus(
            delta, fixed_prompts, llm, tokenizer, vocab_size,
            args.gamma, args.seed, slug, device, headline_corpus, args.model
        )

    print("\nReleasing LLM from GPU memory...")
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Pass 2: evaluation (GPT-2 loaded once) ───────────────────────────────
    print(f"\n=== PASS 2: Evaluation ===")
    print("Loading GPT-2 for perplexity scoring...")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    if gpt2_tok.pad_token is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt2.eval()

    fixed_prompt_set = set(p["prompt"] for p in fixed_prompts)
    sweep_results = []

    for delta in DELTAS:
        print(f"\n--- delta={delta} ---")
        corpus = load_corpus(corpus_paths[delta])
        if delta == 2.0:
            corpus = [x for x in corpus if x["prompt"] in fixed_prompt_set]

        wm_items  = [x for x in corpus if x["watermarked"]]
        uwm_items = [x for x in corpus if not x["watermarked"]]
        print(f"  {len(wm_items)} wm, {len(uwm_items)} uwm")

        det   = WatermarkDetector(vocab_size=vocab_size, gamma=args.gamma, seed=args.seed)
        wm_z, _ = compute_z_scores(corpus, det, tokenizer)
        tpr   = sum(z > calibrated_threshold for z in wm_z) / len(wm_z) if wm_z else 0.0

        wm_ppl  = ppl_batch([x["completion"] for x in wm_items[:50]],  gpt2, gpt2_tok, device)
        uwm_ppl = ppl_batch([x["completion"] for x in uwm_items[:50]], gpt2, gpt2_tok, device)
        mean_wm_ppl  = float(np.mean(wm_ppl))
        mean_uwm_ppl = float(np.mean(uwm_ppl))
        ppl_ratio    = mean_wm_ppl / mean_uwm_ppl if mean_uwm_ppl > 0 else float("nan")
        print(f"  TPR={tpr:.3f}  PPL wm={mean_wm_ppl:.2f}  uwm={mean_uwm_ppl:.2f}  ratio={ppl_ratio:.3f}")

        sweep_results.append({
            "delta":                float(delta),
            "tpr":                  float(tpr),
            "mean_ppl_wm":          mean_wm_ppl,
            "mean_ppl_uwm":         mean_uwm_ppl,
            "ppl_ratio":            ppl_ratio,
            "n_wm":                 len(wm_items),
            "n_uwm":                len(uwm_items),
            "calibrated_threshold": float(calibrated_threshold),
        })

    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved delta sweep → {output_path}")
    print("Expected: TPR and PPL both rise monotonically with delta. Knee at ~delta=2.0.")


if __name__ == "__main__":
    main()
