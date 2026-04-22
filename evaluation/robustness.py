"""
Robustness evaluation: adversarial modifications to watermarked text.

Tests whether detection survives:
  1. Random word substitutions (synonym or random replacement)
  2. Token insertion / deletion
  3. Paraphrasing (placeholder — requires an LLM paraphraser)
"""

import random
import re
from typing import List, Optional


def apply_word_substitution(
    text: str,
    substitution_rate: float = 0.1,
    seed: Optional[int] = None,
) -> str:
    """
    Replace a random fraction of words with random words from the same vocabulary.
    This simulates a naive adversary trying to destroy the watermark.

    Args:
        text: Input text string.
        substitution_rate: Fraction of words to replace (e.g. 0.1 = 10%).
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return text

    n_to_replace = max(1, int(len(words) * substitution_rate))
    indices = rng.sample(range(len(words)), min(n_to_replace, len(words)))

    # Simple random replacement: swap with a word from elsewhere in the text
    # (a real adversary would use synonyms, but this tests robustness)
    word_pool = [w for w in words if len(w) > 3]
    if not word_pool:
        word_pool = words

    for idx in indices:
        words[idx] = rng.choice(word_pool)

    return " ".join(words)


def apply_token_insertion_deletion(
    token_ids: List[int],
    vocab_size: int,
    modification_rate: float = 0.1,
    mode: str = "both",
    seed: Optional[int] = None,
) -> List[int]:
    """
    Insert random tokens and/or delete tokens from a sequence.

    Args:
        token_ids: List of integer token IDs.
        vocab_size: Vocabulary size (for sampling random insertion tokens).
        modification_rate: Fraction of tokens to insert or delete.
        mode: "insert", "delete", or "both".
        seed: Random seed.
    """
    rng = random.Random(seed)
    ids = list(token_ids)

    n_ops = max(1, int(len(ids) * modification_rate))

    if mode in ("delete", "both"):
        for _ in range(n_ops):
            if len(ids) > 1:
                idx = rng.randint(0, len(ids) - 1)
                ids.pop(idx)

    if mode in ("insert", "both"):
        for _ in range(n_ops):
            idx = rng.randint(0, len(ids))
            ids.insert(idx, rng.randint(0, vocab_size - 1))

    return ids


def apply_paraphrase_placeholder(text: str) -> str:
    """
    Placeholder for LLM-based paraphrasing.
    In the actual experiment, replace this with a call to a paraphrasing model
    (e.g., prompt another LLM: 'Rewrite this text in different words: {text}').
    """
    raise NotImplementedError(
        "Paraphrasing requires an external LLM call. "
        "Use pipeline/generate.py with a paraphrase prompt to implement this."
    )


def evaluate_robustness(
    corpus: List[dict],
    detector,
    tokenizer,
    substitution_rates: List[float] = [0.05, 0.10, 0.15, 0.20],
) -> dict:
    """
    Run robustness experiments on a watermarked corpus.

    Returns dict mapping experiment_name -> list of DetectionResult objects.
    """
    results = {}

    wm_items = [item for item in corpus if item["watermarked"]]

    # Baseline: no modification
    baseline_results = []
    for item in wm_items:
        r = detector.score_sequence(item["token_ids"])
        baseline_results.append(r)
    results["baseline"] = baseline_results

    # Word substitution at various rates
    for rate in substitution_rates:
        key = f"word_sub_{int(rate*100)}pct"
        rate_results = []
        for item in wm_items:
            modified_text = apply_word_substitution(item["completion"], substitution_rate=rate)
            modified_ids = tokenizer.encode(modified_text, add_special_tokens=False)
            r = detector.score_sequence(modified_ids)
            rate_results.append(r)
        results[key] = rate_results

    # Token deletion at 10%
    del_results = []
    for item in wm_items:
        modified_ids = apply_token_insertion_deletion(
            item["token_ids"], vocab_size=tokenizer.vocab_size, modification_rate=0.10, mode="delete"
        )
        r = detector.score_sequence(modified_ids)
        del_results.append(r)
    results["token_deletion_10pct"] = del_results

    # Token insertion at 10%
    ins_results = []
    for item in wm_items:
        modified_ids = apply_token_insertion_deletion(
            item["token_ids"], vocab_size=tokenizer.vocab_size, modification_rate=0.10, mode="insert"
        )
        r = detector.score_sequence(modified_ids)
        ins_results.append(r)
    results["token_insertion_10pct"] = ins_results

    return results
