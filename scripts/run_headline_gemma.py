"""
Phase 2: Generate headline corpus with Gemma-3 4B at delta=2.0.

Runtime: ~2-4 hours on T4/A100. Fully resumable — re-run to continue.
Output:  results/corpus_gemma3_d2.jsonl (~900 lines)
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from pipeline.generate import CorpusGenerator

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

if not torch.cuda.is_available():
    print("WARNING: No GPU detected. Generation will be extremely slow on CPU.")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

generator = CorpusGenerator(
    model_name="google/gemma-3-4b-it",
    delta=2.0,
    gamma=0.5,
    seed=42,
    max_new_tokens=200,
)

generator.generate_corpus(
    n_per_dataset=150,
    output_path="results/corpus_gemma3_d2.jsonl",
    resume=True,
)
