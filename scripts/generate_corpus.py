"""
Generate a watermarked + unwatermarked corpus for any causal LLM.

Draws 150 prompts per dataset (CNN/DailyMail, WritingPrompts, TriviaQA) and
produces paired watermarked / unwatermarked completions. Fully resumable.

Usage:
    # LLaMA (primary)
    python scripts/generate_corpus.py

    # Gemma 2 9B (secondary)
    python scripts/generate_corpus.py --model google/gemma-2-9b-it

    # Custom delta or smaller run
    python scripts/generate_corpus.py --model google/gemma-2-9b-it --n-per-dataset 67 --delta 2.0

Output:
    results/corpus_{model_slug}_d{delta}.jsonl

Runtime: ~2-4 hours on T4/A100. Re-run to resume from last checkpoint.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pipeline.generate import CorpusGenerator


def model_slug(model_name: str) -> str:
    """Convert e.g. 'meta-llama/Llama-3.1-8B-Instruct' → 'llama'."""
    name = model_name.lower().split("/")[-1]
    for key in ("llama", "gemma", "mistral", "falcon", "phi"):
        if key in name:
            return key
    return name.split("-")[0]


def parse_args():
    p = argparse.ArgumentParser(description="Generate watermarked corpus for a causal LLM.")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="HuggingFace model ID (default: LLaMA 3.1 8B Instruct)")
    p.add_argument("--n-per-dataset", type=int, default=150,
                   help="Prompts per dataset (CNN, WritingPrompts, TriviaQA)")
    p.add_argument("--delta", type=float, default=2.0,
                   help="Green-list logit bias δ")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Green-list vocabulary fraction γ")
    p.add_argument("--seed", type=int, default=42,
                   help="Watermark secret key (must match detector at eval time)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--output", default=None,
                   help="Output JSONL path. Auto-derived from model + delta if omitted.")
    return p.parse_args()


def main():
    args = parse_args()
    slug = model_slug(args.model)
    delta_str = str(args.delta).rstrip("0").rstrip(".")
    output = args.output or f"results/corpus_{slug}_d{delta_str}.jsonl"

    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Generation will be very slow on CPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Model:          {args.model}")
    print(f"Output:         {output}")
    print(f"δ={args.delta}  γ={args.gamma}  seed={args.seed}  n_per_dataset={args.n_per_dataset}")

    gen = CorpusGenerator(
        model_name=args.model,
        delta=args.delta,
        gamma=args.gamma,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    gen.generate_corpus(
        n_per_dataset=args.n_per_dataset,
        output_path=output,
        resume=True,
    )
    print(f"\nCorpus saved to {output}")


if __name__ == "__main__":
    main()
