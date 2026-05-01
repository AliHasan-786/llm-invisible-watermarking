"""
Robustness evaluation: adversarial modifications to watermarked text.

Tests whether detection survives:
  1. Random word substitutions (synonym or random replacement)
  2. Token insertion / deletion
  3. LLM-based paraphrasing
"""

import random
from typing import List, Optional

import torch


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


def apply_llm_paraphrase(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 4,
) -> List[str]:
    """
    Paraphrase texts using an LLM. Critically, no LogitsProcessor is applied —
    the paraphraser must NOT re-embed the watermark.

    Args:
        texts: List of text strings to paraphrase.
        model: A loaded HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        device: "cuda" or "cpu".
        batch_size: How many texts to process at once.
    Returns:
        List of paraphrased strings, same length as input.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            prompt = (
                "Rewrite the following passage in different words while preserving meaning. "
                "Return ONLY the rewrite, no preamble.\n\n"
                f"Passage:\n{text}\n\nRewrite:"
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            rewrite_ids = output_ids[0, prompt_len:]
            rewrite_text = tokenizer.decode(rewrite_ids, skip_special_tokens=True)
            results.append(rewrite_text.strip())
    return results


def evaluate_robustness(
    corpus: List[dict],
    detector,
    tokenizer,
    substitution_rates: List[float] = [0.05, 0.10, 0.15, 0.20],
    paraphraser_model=None,
    paraphraser_tokenizer=None,
    device: str = "cpu",
) -> dict:
    """
    Run robustness experiments on a watermarked corpus.

    Args:
        corpus: List of sample dicts with 'completion', 'token_ids', 'watermarked'.
        detector: WatermarkDetector instance.
        tokenizer: Tokenizer matching the generation model.
        substitution_rates: Word substitution rates to evaluate.
        paraphraser_model: If provided, runs LLM paraphrase condition.
        paraphraser_tokenizer: Required when paraphraser_model is given.
        device: Device string for paraphraser ("cuda" or "cpu").
    Returns:
        Dict mapping experiment_name -> list of DetectionResult objects.
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

    # LLM paraphrase (only if paraphraser provided)
    if paraphraser_model is not None and paraphraser_tokenizer is not None:
        print("Running LLM paraphrase condition...")
        wm_texts = [item["completion"] for item in wm_items]
        paraphrased = apply_llm_paraphrase(
            wm_texts, paraphraser_model, paraphraser_tokenizer, device=device
        )
        para_results = []
        for para_text in paraphrased:
            para_ids = paraphraser_tokenizer.encode(para_text, add_special_tokens=False)
            r = detector.score_sequence(para_ids)
            para_results.append(r)
        results["llm_paraphrase"] = para_results

    return results


if __name__ == "__main__":
    # Sanity check: paraphrase two short strings and print side by side.
    # Requires a model to be loaded — uses a tiny local model for quick verification.
    import sys
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    print(f"Loading {model_name} for paraphrase sanity check...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    mdl.eval()

    test_inputs = [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Machine learning models are trained on large datasets to recognize patterns.",
    ]
    rewrites = apply_llm_paraphrase(test_inputs, mdl, tok, device=device)
    for orig, rw in zip(test_inputs, rewrites):
        print(f"\nINPUT:  {orig}")
        print(f"OUTPUT: {rw}")
