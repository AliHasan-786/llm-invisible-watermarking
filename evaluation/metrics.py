"""
Evaluation metrics: perplexity, detection statistics, TPR/FPR curves.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Mapping, Sequence, Tuple

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
    import gc
    wm_z, uwm_z = [], []
    for i, item in enumerate(corpus):
        token_ids = item["token_ids"]
        if not token_ids:
            continue
        result = detector.score_sequence(token_ids)
        if item["watermarked"]:
            wm_z.append(result.z_score)
        else:
            uwm_z.append(result.z_score)
        if i % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
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
    threshold = calibrate_threshold(uwm_z_scores, target_fpr)

    tpr = sum(z > threshold for z in wm_z_scores) / len(wm_z_scores) if wm_z_scores else 0.0
    fpr_actual = sum(z > threshold for z in uwm_z_scores) / len(uwm_z_scores) if uwm_z_scores else 0.0
    return threshold, tpr, fpr_actual


def calibrate_threshold(
    control_scores: Sequence[float],
    target_fpr: float = 0.01,
) -> float:
    """Largest-power observed threshold whose strict-`>` FPR is within target.

    Ties count as missed. Choosing the smallest observed threshold satisfying
    the constraint therefore maximizes detections without exceeding the
    empirical target.
    """
    if not control_scores:
        raise ValueError("at least one calibration control score is required")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    values = np.asarray(control_scores, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("control scores must be finite")
    allowed_exceedances = math.floor(target_fpr * len(values))
    descending = np.sort(values)[::-1]
    return float(descending[allowed_exceedances])


def realized_fpr(scores: Sequence[float], threshold: float) -> float:
    if not scores:
        raise ValueError("at least one score is required")
    return float(np.mean(np.asarray(scores, dtype=float) > threshold))


def clustered_calibration_fpr_ci(
    scores_by_prompt: Mapping[str, Sequence[float]],
    threshold: float,
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap realized FPR while retaining each prompt's control triplet."""
    if not scores_by_prompt:
        raise ValueError("at least one calibration prompt is required")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    clusters = []
    for prompt_identifier, values in sorted(scores_by_prompt.items()):
        array = np.asarray(values, dtype=float)
        if len(array) != 3:
            raise ValueError(
                f"calibration prompt {prompt_identifier} has {len(array)} controls; expected 3"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("calibration scores must be finite")
        clusters.append(array > threshold)
    detections = np.stack(clusters)
    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n_clusters, size=n_clusters)
        estimates[index] = detections[sample].mean()
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return float(low), float(high)


def calibration_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    target_fpr: float = 0.01,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Return threshold and amended realized-calibration-FPR estimate."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["prompt_id"])].append(float(row["score"]))
    all_scores = [value for values in grouped.values() for value in values]
    threshold = calibrate_threshold(all_scores, target_fpr)
    ci = clustered_calibration_fpr_ci(
        grouped,
        threshold,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "target_fpr": target_fpr,
        "threshold": threshold,
        "realized_calibration_fpr": realized_fpr(all_scores, threshold),
        "realized_calibration_fpr_ci_95": list(ci),
        "n_calibration_prompts": len(grouped),
        "n_calibration_controls": len(all_scores),
        "bootstrap_resamples": n_bootstrap,
        "bootstrap_unit": "prompt_cluster_with_three_controls",
        "ties": "not_detected",
    }


def bootstrap_proportion_ci(
    detections: Sequence[bool],
    *,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a held-out prompt-level proportion."""
    values = np.asarray(detections, dtype=float)
    if len(values) == 0:
        raise ValueError("at least one held-out observation is required")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    estimates = samples.mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return float(low), float(high)


def headline_detection_summary(
    calibration_rows: Sequence[Mapping[str, object]],
    heldout_rows: Sequence[Mapping[str, object]],
    *,
    positive_scheme: str,
    target_fpr: float = 0.01,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Build the clean held-out headline row for one detector/scheme pair."""
    calibration = calibration_summary(
        calibration_rows,
        target_fpr=target_fpr,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    threshold = float(calibration["threshold"])
    positive_by_prompt = {
        str(row["prompt_id"]): float(row["score"])
        for row in heldout_rows
        if row["scheme"] == positive_scheme
    }
    controls_by_prompt = {
        str(row["prompt_id"]): float(row["score"])
        for row in heldout_rows
        if row["scheme"] == "control"
    }
    complete_prompt_ids = sorted(set(positive_by_prompt) & set(controls_by_prompt))
    if not complete_prompt_ids:
        raise ValueError("held-out positive and control scores are both required")
    positive = [positive_by_prompt[prompt] for prompt in complete_prompt_ids]
    controls = [controls_by_prompt[prompt] for prompt in complete_prompt_ids]
    positive_hits = [score > threshold for score in positive]
    control_hits = [score > threshold for score in controls]
    available_positive_hits = [
        score > threshold for score in positive_by_prompt.values()
    ]
    available_control_hits = [
        score > threshold for score in controls_by_prompt.values()
    ]
    return {
        **calibration,
        "scheme": positive_scheme,
        "condition": "clean",
        "heldout_tpr": float(np.mean(positive_hits)),
        "heldout_tpr_ci_95": list(
            bootstrap_proportion_ci(
                positive_hits,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        ),
        "heldout_fpr": float(np.mean(control_hits)),
        "heldout_fpr_ci_95": list(
            bootstrap_proportion_ci(
                control_hits,
                n_bootstrap=n_bootstrap,
                seed=seed + 1,
            )
        ),
        "n_heldout_positive": len(positive),
        "n_heldout_controls": len(controls),
        "complete_case_prompts": len(complete_prompt_ids),
        "all_available_sensitivity": {
            "heldout_tpr": float(np.mean(available_positive_hits)),
            "heldout_fpr": float(np.mean(available_control_hits)),
            "n_heldout_positive": len(available_positive_hits),
            "n_heldout_controls": len(available_control_hits),
        },
    }


def paired_tpr_difference(
    synthid_rows: Sequence[Mapping[str, object]],
    kirchenbauer_rows: Sequence[Mapping[str, object]],
    *,
    synthid_threshold: float,
    kirchenbauer_threshold: float,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    """Paired SynthID-minus-Kirchenbauer TPR difference by prompt."""
    synthid = {
        str(row["prompt_id"]): float(row["score"])
        for row in synthid_rows
        if row["scheme"] == "synthid"
    }
    kirchenbauer = {
        str(row["prompt_id"]): float(row["score"])
        for row in kirchenbauer_rows
        if row["scheme"] == "kirchenbauer"
    }
    prompt_ids = sorted(set(synthid) & set(kirchenbauer))
    if not prompt_ids:
        raise ValueError("no paired SynthID and Kirchenbauer prompts")
    differences = np.asarray(
        [
            float(synthid[prompt] > synthid_threshold)
            - float(kirchenbauer[prompt] > kirchenbauer_threshold)
            for prompt in prompt_ids
        ]
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(n_bootstrap, len(differences)), replace=True)
    estimates = samples.mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return {
        "synthid_minus_kirchenbauer_tpr": float(differences.mean()),
        "ci_95": [float(low), float(high)],
        "n_paired_prompts": len(prompt_ids),
        "bootstrap_resamples": n_bootstrap,
        "ties": "not_detected",
    }


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
