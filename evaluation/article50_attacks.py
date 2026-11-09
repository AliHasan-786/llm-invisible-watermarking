"""Frozen attack definitions and deterministic transformations for Article 50."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


PARAPHRASER_REVISIONS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    },
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    },
}

TRANSLATOR = {
    "model_id": "facebook/nllb-200-distilled-600M",
    "revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
}

PARAPHRASE_PROMPT = (
    "Rewrite the passage in different words while preserving its meaning and "
    "level of detail. Return only the rewrite.\n\n{passage}"
)
SUMMARY_PROMPT = (
    "Summarize the passage's factual and semantic content in at most 80 tokens. "
    "Return only the summary.\n\n{passage}"
)
REGENERATE_PROMPT = (
    "Using only the summary below, write a new passage with the same purpose and "
    "approximately {target_tokens} tokens. Return only the passage.\n\n{summary}"
)

STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her hers him his "
    "i if in into is it its me my nor not of on or our ours she so than that the "
    "their theirs them then there these they this those to too us was we were what "
    "when where which who will with you your yours".split()
)


@dataclass(frozen=True)
class AttackSpec:
    condition: str
    family: str
    parameters: dict[str, object]
    model_backed: bool = False


def attack_specs() -> tuple[AttackSpec, ...]:
    specs: list[AttackSpec] = []
    for rate in (0.05, 0.10, 0.20):
        percent = int(rate * 100)
        specs.extend(
            [
                AttackSpec(f"word_substitution_{percent}", "word_substitution", {"rate": rate}),
                AttackSpec(f"token_insertion_{percent}", "token_insertion", {"rate": rate}),
                AttackSpec(f"token_deletion_{percent}", "token_deletion", {"rate": rate}),
            ]
        )
    for length in (25, 50, 75, 100, 125, 150, 175, 200):
        specs.append(AttackSpec(f"prefix_{length}", "prefix", {"length": length}))
    for length in (50, 100):
        for location in ("start", "middle", "end"):
            specs.append(
                AttackSpec(
                    f"span_{location}_{length}",
                    "span",
                    {"length": length, "location": location},
                )
            )
    specs.extend(
        [
            AttackSpec("paraphrase_qwen_1pass", "paraphrase", {"models": ["qwen"]}, True),
            AttackSpec("paraphrase_mistral_1pass", "paraphrase", {"models": ["mistral"]}, True),
            AttackSpec("paraphrase_qwen_2pass", "paraphrase", {"models": ["qwen", "qwen"]}, True),
            AttackSpec(
                "paraphrase_mistral_2pass",
                "paraphrase",
                {"models": ["mistral", "mistral"]},
                True,
            ),
            AttackSpec(
                "paraphrase_qwen_mistral",
                "paraphrase",
                {"models": ["qwen", "mistral"]},
                True,
            ),
            AttackSpec(
                "paraphrase_mistral_qwen",
                "paraphrase",
                {"models": ["mistral", "qwen"]},
                True,
            ),
            AttackSpec("backtranslation_french", "backtranslation", {"pivot": "fra_Latn"}, True),
            AttackSpec("backtranslation_german", "backtranslation", {"pivot": "deu_Latn"}, True),
            AttackSpec("laundering_qwen", "laundering", {"model": "qwen"}, True),
            AttackSpec("laundering_mistral", "laundering", {"model": "mistral"}, True),
        ]
    )
    return tuple(specs)


def attack_manifest() -> list[dict[str, object]]:
    return [asdict(spec) for spec in attack_specs()]


def attack_seed(prompt_id: str, condition: str, pass_index: int = 0) -> int:
    digest = hashlib.sha256(
        f"attack:42:{prompt_id}:{condition}:{pass_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def attack_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return str(row["prompt_id"]), str(row["scheme"]), str(row["condition"])


def _operation_count(length: int, rate: float) -> int:
    if length == 0:
        return 0
    return min(length, max(1, math.floor(length * rate)))


def word_substitution(text: str, rate: float, seed: int) -> str:
    """Replace sampled words with other non-stopwords from the same completion."""
    words = text.split()
    count = _operation_count(len(words), rate)
    if count == 0:
        return text
    pool = [word for word in words if word.lower().strip(".,!?;:\"'()") not in STOPWORDS]
    if not pool:
        pool = list(words)
    rng = random.Random(seed)
    indices = rng.sample(range(len(words)), count)
    for index in indices:
        alternatives = [word for word in pool if word != words[index]]
        if alternatives:
            words[index] = rng.choice(alternatives)
    return " ".join(words)


def token_edit(
    token_ids: Sequence[int],
    *,
    rate: float,
    mode: str,
    seed: int,
    valid_non_special_ids: Sequence[int] | None = None,
) -> list[int]:
    """Insert or delete at indices sampled without replacement."""
    ids = list(token_ids)
    count = _operation_count(len(ids), rate)
    if count == 0:
        return ids
    rng = random.Random(seed)
    if mode == "delete":
        removed = set(rng.sample(range(len(ids)), count))
        return [token for index, token in enumerate(ids) if index not in removed]
    if mode != "insert":
        raise ValueError("mode must be insert or delete")
    if not valid_non_special_ids:
        raise ValueError("token insertion requires valid non-special token IDs")
    positions = sorted(rng.sample(range(len(ids) + 1), min(count, len(ids) + 1)))
    inserts = {position: rng.choice(valid_non_special_ids) for position in positions}
    result = []
    for position in range(len(ids) + 1):
        if position in inserts:
            result.append(inserts[position])
        if position < len(ids):
            result.append(ids[position])
    return result


def prefix(token_ids: Sequence[int], length: int) -> list[int] | None:
    return list(token_ids[:length]) if len(token_ids) >= length else None


def span(token_ids: Sequence[int], length: int, location: str) -> list[int] | None:
    if len(token_ids) < length:
        return None
    if location == "start":
        start = 0
    elif location == "middle":
        start = (len(token_ids) - length) // 2
    elif location == "end":
        start = len(token_ids) - length
    else:
        raise ValueError("location must be start, middle, or end")
    return list(token_ids[start : start + length])


def deterministic_attack_rows(
    clean_rows: Iterable[Mapping[str, object]],
    *,
    valid_non_special_ids: Sequence[int],
    encode,
    decode,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply all non-model attacks, returning outputs and ineligible conditions."""
    outputs = []
    ineligible = []
    specs = [spec for spec in attack_specs() if not spec.model_backed]
    for clean in clean_rows:
        if clean.get("split") != "heldout":
            continue
        for spec in specs:
            seed = attack_seed(str(clean["prompt_id"]), spec.condition)
            token_ids = list(clean["token_ids"])
            attacked_text = None
            attacked_ids = None
            if spec.family == "word_substitution":
                attacked_text = word_substitution(
                    str(clean["completion"]), float(spec.parameters["rate"]), seed
                )
                attacked_ids = list(encode(attacked_text))
            elif spec.family == "token_insertion":
                attacked_ids = token_edit(
                    token_ids,
                    rate=float(spec.parameters["rate"]),
                    mode="insert",
                    seed=seed,
                    valid_non_special_ids=valid_non_special_ids,
                )
            elif spec.family == "token_deletion":
                attacked_ids = token_edit(
                    token_ids,
                    rate=float(spec.parameters["rate"]),
                    mode="delete",
                    seed=seed,
                )
            elif spec.family == "prefix":
                attacked_ids = prefix(token_ids, int(spec.parameters["length"]))
            elif spec.family == "span":
                attacked_ids = span(
                    token_ids,
                    int(spec.parameters["length"]),
                    str(spec.parameters["location"]),
                )
            if attacked_ids is None:
                ineligible.append(
                    {
                        "prompt_id": clean["prompt_id"],
                        "scheme": clean["scheme"],
                        "condition": spec.condition,
                        "reason": "source_completion_too_short",
                    }
                )
                continue
            if attacked_text is None:
                attacked_text = str(decode(attacked_ids))
            outputs.append(
                {
                    "prompt_id": clean["prompt_id"],
                    "source": clean["source"],
                    "scheme": clean["scheme"],
                    "split": "heldout",
                    "condition": spec.condition,
                    "attack_family": spec.family,
                    "attack_parameters": spec.parameters,
                    "attack_seed": seed,
                    "completion": attacked_text,
                    "token_ids": attacked_ids,
                    "n_tokens": len(attacked_ids),
                    "source_n_tokens": len(token_ids),
                    "length_retention": len(attacked_ids) / max(1, len(token_ids)),
                    "tokenizer_vocab_size": clean.get("tokenizer_vocab_size"),
                    "eos_token_id": clean.get("eos_token_id"),
                }
            )
    return outputs, ineligible
