"""Frozen Article 50 study protocol helpers.

This module contains no model-loading code. It makes the prompt split, seed
derivation, generation plan, and run metadata independently testable before
gated model access or GPU compute is used.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


GLOBAL_SEED = 42
CALIBRATION_REPLICATES = 3
SCHEMES = ("control", "kirchenbauer", "synthid")

MODEL_REVISIONS = {
    "gemma": {
        "model_id": "google/gemma-2-9b-it",
        "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
}

SYNTHID_CONFIG = {
    "keys": [654, 400, 836, 123, 340, 443, 597, 160, 57],
    "ngram_len": 5,
    "sampling_table_seed": 0,
    "sampling_table_size": 65536,
    "context_history_size": 1024,
}

GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_new_tokens": 200,
    "do_sample": True,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_id(source: str, source_index: int, prompt: str) -> str:
    """Return a stable prompt identifier tied to source row and frozen text."""
    payload = {
        "source": source,
        "source_index": int(source_index),
        "prompt": prompt,
    }
    return sha256_text(_canonical_json(payload))


def derived_seed(
    prompt_identifier: str,
    replicate_index: int = 0,
    global_seed: int = GLOBAL_SEED,
) -> int:
    """Derive a deterministic torch-compatible seed from frozen identifiers."""
    if replicate_index < 0:
        raise ValueError("replicate_index must be non-negative")
    digest = hashlib.sha256(
        f"{prompt_identifier}:{global_seed}:{replicate_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def assign_splits(prompts: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Assign an exact, deterministic half split within each prompt source.

    Hash-ordering avoids dependence on input ordering. For an odd source count,
    the calibration split receives the smaller half.
    """
    by_source: dict[str, list[str]] = {}
    for item in prompts:
        identifier = str(item["prompt_id"])
        source = str(item["source"])
        by_source.setdefault(source, []).append(identifier)

    assignments: dict[str, str] = {}
    for source, identifiers in by_source.items():
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"duplicate prompt_id in source {source}")
        ordered = sorted(
            identifiers,
            key=lambda identifier: sha256_text(f"split:{GLOBAL_SEED}:{identifier}"),
        )
        calibration_count = len(ordered) // 2
        for identifier in ordered[:calibration_count]:
            assignments[identifier] = "calibration"
        for identifier in ordered[calibration_count:]:
            assignments[identifier] = "heldout"
    return assignments


def build_generation_plan(
    prompts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build the preregistered calibration and held-out generation plan."""
    assignments = assign_splits(prompts)
    plan: list[dict[str, object]] = []
    for item in sorted(prompts, key=lambda row: str(row["prompt_id"])):
        identifier = str(item["prompt_id"])
        split = assignments[identifier]
        base = {
            "prompt_id": identifier,
            "source": str(item["source"]),
            "source_index": int(item["source_index"]),
            "prompt": str(item["prompt"]),
            "split": split,
        }
        if split == "calibration":
            for replicate in range(CALIBRATION_REPLICATES):
                plan.append(
                    {
                        **base,
                        "scheme": "control",
                        "replicate": replicate,
                        "seed": derived_seed(identifier, replicate),
                    }
                )
        else:
            seed = derived_seed(identifier, 0)
            for scheme in SCHEMES:
                plan.append(
                    {
                        **base,
                        "scheme": scheme,
                        "replicate": 0,
                        "seed": seed,
                    }
                )
    return plan


def generation_key(item: Mapping[str, object]) -> tuple[str, str, str, int]:
    return (
        str(item["prompt_id"]),
        str(item["split"]),
        str(item["scheme"]),
        int(item["replicate"]),
    )


def write_json_atomic(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class RunManifest:
    protocol: str
    pilot: bool
    model_alias: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    global_seed: int
    calibration_replicates: int
    generation: dict[str, object]
    synthid: dict[str, object]
    git_commit: str | None
    git_dirty: bool | None
    requirements_sha256: str
    python: str
    platform: str
    created_at_utc: str


def create_run_manifest(
    repo: str | Path,
    model_alias: str,
    *,
    pilot: bool,
) -> RunManifest:
    repo = Path(repo).resolve()
    if model_alias not in MODEL_REVISIONS:
        raise ValueError(f"unknown model alias: {model_alias}")
    pinned = MODEL_REVISIONS[model_alias]
    requirements = repo / "requirements.txt"
    status = _git_value(repo, "status", "--porcelain")
    return RunManifest(
        protocol="PREREGISTRATION.md@approved-2026-07-18",
        pilot=pilot,
        model_alias=model_alias,
        model_id=pinned["model_id"],
        model_revision=pinned["revision"],
        tokenizer_revision=pinned["revision"],
        global_seed=GLOBAL_SEED,
        calibration_replicates=CALIBRATION_REPLICATES,
        generation=dict(GENERATION_CONFIG),
        synthid=dict(SYNTHID_CONFIG),
        git_commit=_git_value(repo, "rev-parse", "HEAD"),
        git_dirty=None if status is None else bool(status),
        requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
        python=platform.python_version(),
        platform=platform.platform(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def manifest_dict(manifest: RunManifest) -> dict[str, object]:
    return asdict(manifest)
