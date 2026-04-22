"""
Watermark injection via logit manipulation (Kirchenbauer et al., 2023).

At each decoding step, the previous token seeds a hash that partitions the
vocabulary into a green list (fraction gamma) and red list. A fixed bias
delta is added to green-list logits before softmax, nudging the model to
prefer green tokens without changing model weights.
"""

import hashlib
import torch
from transformers import LogitsProcessor


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, delta: float = 2.0, gamma: float = 0.5, seed: int = 42):
        """
        Args:
            vocab_size: Size of the tokenizer vocabulary.
            delta: Bias added to green-list logits (higher = stronger watermark, lower quality).
            gamma: Fraction of vocabulary assigned to the green list (0 < gamma < 1).
            seed: Secret key for the hash function; must match detector.
        """
        self.vocab_size = vocab_size
        self.delta = delta
        self.gamma = gamma
        self.seed = seed
        self._greenlist_cache: dict[int, torch.Tensor] = {}

    def _get_greenlist_ids(self, prev_token_id: int) -> torch.Tensor:
        if prev_token_id in self._greenlist_cache:
            return self._greenlist_cache[prev_token_id]

        key = f"{self.seed}:{prev_token_id}"
        hash_int = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        rng = torch.Generator()
        rng.manual_seed(hash_int % (2 ** 32))

        green_list_size = int(self.vocab_size * self.gamma)
        perm = torch.randperm(self.vocab_size, generator=rng)
        greenlist = perm[:green_list_size]

        self._greenlist_cache[prev_token_id] = greenlist
        return greenlist

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            prev_token_id = input_ids[batch_idx, -1].item()
            greenlist_ids = self._get_greenlist_ids(prev_token_id)
            scores[batch_idx, greenlist_ids] += self.delta
        return scores
