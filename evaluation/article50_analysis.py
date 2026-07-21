"""Result-agnostic curve, quality, and readability helpers."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


GPT2_REFERENCE = {
    "model_id": "openai-community/gpt2",
    "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
}


def roc_det_curve(
    positive_scores: Sequence[float],
    control_scores: Sequence[float],
) -> dict[str, object]:
    """Return exact empirical ROC/DET points and trapezoidal AUROC."""
    positive = np.asarray(positive_scores, dtype=float)
    controls = np.asarray(control_scores, dtype=float)
    if len(positive) == 0 or len(controls) == 0:
        raise ValueError("positive and control scores are required")
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(controls)):
        raise ValueError("scores must be finite")
    thresholds = np.concatenate(
        ([np.inf], np.sort(np.unique(np.concatenate((positive, controls))))[::-1], [-np.inf])
    )
    fpr = np.asarray([(controls > threshold).mean() for threshold in thresholds])
    tpr = np.asarray([(positive > threshold).mean() for threshold in thresholds])
    order = np.argsort(fpr, kind="stable")
    auc = float(np.trapz(tpr[order], fpr[order]))
    return {
        "thresholds": [None if not np.isfinite(value) else float(value) for value in thresholds],
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "fnr": (1 - tpr).tolist(),
        "auroc": auc,
        "n_positive": len(positive),
        "n_control": len(controls),
        "ties": "not_detected",
    }


def operating_point_supported(n_controls: int, target_fpr: float) -> bool:
    """Whether at least one false positive is resolvable below the target."""
    return math.floor(n_controls * target_fpr) >= 1


def paired_quality_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    ratio_ceiling: float = 1.15,
) -> list[dict[str, object]]:
    """Summarize paired scheme/control perplexity and length ratios."""
    by_scheme: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if row.get("split") == "heldout":
            by_scheme[str(row["scheme"])][str(row["prompt_id"])] = row
    controls = by_scheme.get("control", {})
    summaries = []
    for scheme in ("kirchenbauer", "synthid"):
        pairs = sorted(set(controls) & set(by_scheme.get(scheme, {})))
        if not pairs:
            continue
        ppl_ratios = np.asarray(
            [
                float(by_scheme[scheme][prompt]["perplexity"])
                / float(controls[prompt]["perplexity"])
                for prompt in pairs
            ]
        )
        length_ratios = np.asarray(
            [
                int(by_scheme[scheme][prompt]["n_tokens"])
                / max(1, int(controls[prompt]["n_tokens"]))
                for prompt in pairs
            ]
        )
        summaries.append(
            {
                "scheme": scheme,
                "n_pairs": len(pairs),
                "mean_paired_perplexity_ratio": float(ppl_ratios.mean()),
                "median_paired_perplexity_ratio": float(np.median(ppl_ratios)),
                "mean_length_ratio": float(length_ratios.mean()),
                "h5_ceiling": ratio_ceiling,
                "h5_pass": bool(ppl_ratios.mean() <= ratio_ceiling),
            }
        )
    return summaries


def _blind_hash(prompt_id: str, scheme: str) -> str:
    return hashlib.sha256(f"readability:42:{prompt_id}:{scheme}".encode()).hexdigest()


def readability_queue(
    rows: Sequence[Mapping[str, object]],
    *,
    pairs_per_model: int = 60,
    overlap: int = 20,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Create a balanced, blinded 60-pair queue and separate answer key."""
    heldout: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if row.get("split") == "heldout":
            heldout[str(row["prompt_id"])][str(row["scheme"])] = row
    candidates = []
    for prompt_id, schemes in heldout.items():
        for scheme in ("kirchenbauer", "synthid"):
            if "control" in schemes and scheme in schemes:
                candidates.append(
                    {
                        "prompt_id": prompt_id,
                        "source": str(schemes[scheme]["source"]),
                        "scheme": scheme,
                        "control": schemes["control"],
                        "watermarked": schemes[scheme],
                        "sort_hash": _blind_hash(prompt_id, scheme),
                    }
                )
    strata: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for candidate in candidates:
        strata[(candidate["source"], candidate["scheme"])].append(candidate)
    if len(strata) != 6:
        raise ValueError("readability queue requires all three sources and both schemes")
    per_stratum = pairs_per_model // len(strata)
    if per_stratum * len(strata) != pairs_per_model:
        raise ValueError("pairs_per_model must divide evenly across six strata")
    selected = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda row: row["sort_hash"])
        if len(ordered) < per_stratum:
            raise ValueError(f"insufficient complete pairs in stratum {key}")
        selected.extend(ordered[:per_stratum])
    selected.sort(key=lambda row: row["sort_hash"])
    secondary_ids = {
        row["sort_hash"]
        for row in sorted(selected, key=lambda row: row["sort_hash"] + "secondary")[:overlap]
    }
    queue = []
    key_rows = []
    for index, item in enumerate(selected, start=1):
        rng = random.Random(int(item["sort_hash"][:16], 16))
        watermarked_first = bool(rng.getrandbits(1))
        first = item["watermarked"] if watermarked_first else item["control"]
        second = item["control"] if watermarked_first else item["watermarked"]
        item_id = f"readability-{index:03d}"
        queue.append(
            {
                "item_id": item_id,
                "source": item["source"],
                "text_a": first["completion"],
                "text_b": second["completion"],
                "reviewers": ["primary", "secondary"]
                if item["sort_hash"] in secondary_ids
                else ["primary"],
                "rubric": {
                    "fluency_a": "integer 1-5",
                    "fluency_b": "integer 1-5",
                    "coherence_a": "integer 1-5",
                    "coherence_b": "integer 1-5",
                    "task_fulfillment_a": "integer 1-5",
                    "task_fulfillment_b": "integer 1-5",
                    "watermark_guess": "A, B, or tie",
                    "rationale": "one sentence",
                },
            }
        )
        key_rows.append(
            {
                "item_id": item_id,
                "prompt_id": item["prompt_id"],
                "scheme": item["scheme"],
                "watermarked_side": "A" if watermarked_first else "B",
            }
        )
    return queue, key_rows
