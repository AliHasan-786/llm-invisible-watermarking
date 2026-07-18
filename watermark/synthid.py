"""SynthID-Text generation configuration and untrained weighted-mean detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pipeline.article50 import SYNTHID_CONFIG


@dataclass(frozen=True)
class SynthIDScore:
    score: float
    scored_ngrams: int


def generation_config():
    """Construct the pinned Transformers SynthID generation configuration."""
    from transformers import SynthIDTextWatermarkingConfig

    return SynthIDTextWatermarkingConfig(**SYNTHID_CONFIG)


def weighted_mean(g_values, mask, weights=None):
    """Torch implementation of google-deepmind/synthid-text weighted_mean_score."""
    import torch

    if g_values.ndim != 3 or mask.ndim != 2:
        raise ValueError("expected g_values [batch, sequence, depth] and mask [batch, sequence]")
    if g_values.shape[:2] != mask.shape:
        raise ValueError("g_values and mask sequence dimensions must match")
    depth = g_values.shape[-1]
    if weights is None:
        weights = torch.linspace(10.0, 1.0, depth, device=g_values.device)
    else:
        weights = torch.as_tensor(weights, dtype=torch.float32, device=g_values.device)
    if weights.shape != (depth,) or torch.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("weights must be non-negative with one value per depth")
    weights = weights * (depth / weights.sum())
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("cannot score a sequence with no unmasked n-grams")
    weighted = g_values.to(torch.float32) * weights.reshape(1, 1, depth)
    return (weighted * mask.unsqueeze(2)).sum(dim=(1, 2)) / (depth * counts)


class SynthIDWeightedMeanDetector:
    """Score token sequences with the pinned open-source SynthID hash function."""

    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import SynthIDTextWatermarkLogitsProcessor

        self.device = torch.device(device)
        self.processor = SynthIDTextWatermarkLogitsProcessor(
            **SYNTHID_CONFIG,
            device=self.device,
        )

    def score_sequence(
        self,
        token_ids: Sequence[int],
        *,
        eos_token_id: int | None = None,
    ) -> SynthIDScore:
        import torch

        ids = torch.as_tensor([list(token_ids)], dtype=torch.long, device=self.device)
        if ids.shape[1] < SYNTHID_CONFIG["ngram_len"]:
            raise ValueError(
                f"SynthID requires at least {SYNTHID_CONFIG['ngram_len']} tokens"
            )
        g_values = self.processor.compute_g_values(ids)
        repetition_mask = self.processor.compute_context_repetition_mask(ids)
        mask = repetition_mask
        if eos_token_id is not None:
            eos_mask = self.processor.compute_eos_token_mask(ids, eos_token_id)
            eos_mask = eos_mask[:, SYNTHID_CONFIG["ngram_len"] - 1 :]
            mask = mask * eos_mask
        score = weighted_mean(g_values, mask)
        return SynthIDScore(
            score=float(score[0].detach().cpu()),
            scored_ngrams=int(mask[0].sum().detach().cpu()),
        )
