"""
Evaluation metrics: perplexity, detection statistics, TPR/FPR curves.
"""

import math
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_perplexity(
    texts: List[str],
    model_name: str = "gpt2",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
) -> List[float]:
    """
    Compute perplexity of each text under a reference model (not the generation model).
    Lower perplexity = higher quality / more natural text.

    Use GPT-2 or a small independent model as the scorer so it's not biased
    by the watermarked model's distribution.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    perplexities = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            # outputs.loss is mean NLL over all tokens in the batch
            # We want per-sample perplexity, so compute token-level NLL manually
            logits = outputs.logits  # (B, T, V)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = encodings["input_ids"][:, 1:].contiguous()
            attention_mask = encodings["attention_mask"][:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), -1)

            # Mask padding tokens
            masked_loss = token_loss * attention_mask.float()
            n_tokens = attention_mask.float().sum(dim=1)
            mean_loss = masked_loss.sum(dim=1) / n_tokens.clamp(min=1)
            ppl = torch.exp(mean_loss).cpu().tolist()
            perplexities.extend(ppl)

    return perplexities


def compute_z_scores(
    corpus: List[dict],
    detector,
    tokenizer,
) -> Tuple[List[float], List[float]]:
    """
    Given a corpus (list of dicts with 'token_ids' and 'watermarked'),
    return (watermarked_z_scores, unwatermarked_z_scores).
    """
    wm_z, uwm_z = [], []
    for item in corpus:
        token_ids = item["token_ids"]
        if not token_ids:
            continue
        result = detector.score_sequence(token_ids)
        if item["watermarked"]:
            wm_z.append(result.z_score)
        else:
            uwm_z.append(result.z_score)
    return wm_z, uwm_z


def compute_tpr_at_fpr(
    wm_z_scores: List[float],
    uwm_z_scores: List[float],
    target_fpr: float = 0.01,
) -> Tuple[float, float]:
    """
    Given watermarked and unwatermarked z-score lists, return (threshold, TPR)
    at the specified FPR level.
    """
    uwm_sorted = sorted(uwm_z_scores)
    n = len(uwm_sorted)
    idx = int((1 - target_fpr) * n)
    threshold = uwm_sorted[min(idx, n - 1)]

    tpr = sum(z > threshold for z in wm_z_scores) / len(wm_z_scores) if wm_z_scores else 0.0
    fpr_actual = sum(z > threshold for z in uwm_z_scores) / len(uwm_z_scores) if uwm_z_scores else 0.0
    return threshold, tpr, fpr_actual


def roc_curve_data(
    wm_z_scores: List[float],
    uwm_z_scores: List[float],
    n_thresholds: int = 200,
) -> Tuple[List[float], List[float]]:
    """Return (fpr_list, tpr_list) for plotting an ROC curve."""
    all_z = sorted(wm_z_scores + uwm_z_scores, reverse=True)
    thresholds = np.linspace(min(all_z), max(all_z), n_thresholds)
    fprs, tprs = [], []
    for t in thresholds:
        tpr = sum(z > t for z in wm_z_scores) / max(len(wm_z_scores), 1)
        fpr = sum(z > t for z in uwm_z_scores) / max(len(uwm_z_scores), 1)
        tprs.append(tpr)
        fprs.append(fpr)
    return fprs, tprs
