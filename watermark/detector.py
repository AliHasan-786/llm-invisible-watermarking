"""
Watermark detection via one-sided z-test (Kirchenbauer et al., 2023).

For a candidate text, we recompute the green/red partition for each token
using the same secret key and count how many tokens landed on the green list.
Under the null hypothesis (no watermark), ~gamma fraction are green by chance.
A statistically significant excess triggers detection.
"""

import hashlib
import math
from dataclasses import dataclass
from typing import List

import torch
import scipy.stats as stats


@dataclass
class DetectionResult:
    z_score: float
    p_value: float
    green_count: int
    total_tokens: int
    green_fraction: float
    is_watermarked: bool


class WatermarkDetector:
    def __init__(self, vocab_size: int, gamma: float = 0.5, seed: int = 42, z_threshold: float = 4.0):
        """
        Args:
            vocab_size: Must match the processor used during generation.
            gamma: Must match the processor's gamma.
            seed: Must match the processor's seed.
            z_threshold: z-score cutoff for declaring watermark present.
                         z=4.0 corresponds to ~1-in-30000 false positive rate.
                         Calibrate empirically to hit exactly 1% FPR on your corpus.
        """
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.seed = seed
        self.z_threshold = z_threshold
        self._greenlist_cache: dict[int, set] = {}

    def _get_greenlist_set(self, prev_token_id: int) -> set:
        if prev_token_id in self._greenlist_cache:
            return self._greenlist_cache[prev_token_id]

        key = f"{self.seed}:{prev_token_id}"
        hash_int = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        rng = torch.Generator()
        rng.manual_seed(hash_int % (2 ** 32))

        green_list_size = int(self.vocab_size * self.gamma)
        perm = torch.randperm(self.vocab_size, generator=rng)
        greenlist = set(perm[:green_list_size].tolist())

        self._greenlist_cache[prev_token_id] = greenlist
        return greenlist

    def score_sequence(self, token_ids: List[int]) -> DetectionResult:
        """
        Score a token sequence and return detection statistics.

        Args:
            token_ids: List of integer token IDs (output of tokenizer.encode).
                       Needs at least 2 tokens to score anything.
        """
        if len(token_ids) < 2:
            return DetectionResult(
                z_score=0.0, p_value=1.0, green_count=0,
                total_tokens=0, green_fraction=0.0, is_watermarked=False
            )

        green_count = 0
        total = 0  # increments only for valid (in-vocab) tokens

        for i in range(1, len(token_ids)):
            prev_token = token_ids[i - 1]
            cur_token = token_ids[i]
            # Skip tokens outside the (tokenizer) vocab — consistent with processor's clamp.
            # In practice the model never emits such tokens, but guard defensively.
            if prev_token >= self.vocab_size or cur_token >= self.vocab_size:
                continue
            greenlist = self._get_greenlist_set(prev_token)
            if cur_token in greenlist:
                green_count += 1
            total += 1

        if total == 0:
            return DetectionResult(
                z_score=0.0, p_value=1.0, green_count=0,
                total_tokens=0, green_fraction=0.0, is_watermarked=False
            )

        expected = self.gamma * total
        std = math.sqrt(total * self.gamma * (1 - self.gamma))
        z = (green_count - expected) / std if std > 0 else 0.0
        p_value = 1 - stats.norm.cdf(z)

        return DetectionResult(
            z_score=z,
            p_value=p_value,
            green_count=green_count,
            total_tokens=total,
            green_fraction=green_count / total if total > 0 else 0.0,
            is_watermarked=z > self.z_threshold,
        )

    def calibrate_threshold(self, unwatermarked_z_scores: List[float], target_fpr: float = 0.01) -> float:
        """
        Given z-scores from unwatermarked text, find the threshold that achieves target FPR.
        Call this on your control corpus, then set self.z_threshold to the result.
        """
        unwatermarked_z_scores_sorted = sorted(unwatermarked_z_scores)
        idx = int((1 - target_fpr) * len(unwatermarked_z_scores_sorted))
        threshold = unwatermarked_z_scores_sorted[min(idx, len(unwatermarked_z_scores_sorted) - 1)]
        self.z_threshold = threshold
        return threshold
