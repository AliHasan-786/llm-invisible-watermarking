"""Verify cached baseline artifacts and reproduce the published headline table.

This command intentionally performs no model downloads. It verifies every
cached file against its SHA-256 digest, validates corpus metadata, and
recomputes detection/robustness summary statistics from cached z-score arrays.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "results"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
TOLERANCE = 1e-12


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(name: str):
    with (CACHE_DIR / name).open() as handle:
        return json.load(handle)


def assert_close(label: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=TOLERANCE):
        fail(f"{label}: expected {expected}, got {actual}")


def verify_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        fail(f"Missing cache manifest: {MANIFEST_PATH}")
    manifest = load_json("manifest.json")
    for record in manifest["files"]:
        path = CACHE_DIR / record["path"]
        if not path.is_file():
            fail(f"Missing cached artifact: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            fail(f"SHA-256 mismatch for {path.name}: {actual}")
    return manifest


def verify_corpus(name: str, expected_rows: int, expected_model: str, expected_delta: float) -> None:
    rows = 0
    with (CACHE_DIR / name).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            rows += 1
            if item.get("model") != expected_model:
                fail(f"{name}:{line_number}: unexpected model {item.get('model')}")
            if item.get("delta") != expected_delta:
                fail(f"{name}:{line_number}: unexpected delta {item.get('delta')}")
            if item.get("gamma") != 0.5 or item.get("seed") != 42:
                fail(f"{name}:{line_number}: unexpected watermark configuration")
            if item.get("n_tokens") != len(item.get("token_ids", [])):
                fail(f"{name}:{line_number}: n_tokens does not match token_ids")
    if rows != expected_rows:
        fail(f"{name}: expected {expected_rows} rows, got {rows}")


def verify_detection(prefix: str) -> dict:
    summary = load_json(f"detection_{prefix}_summary.json")
    arrays = np.load(CACHE_DIR / f"detection_{prefix}_zscores.npz")
    wm_z = arrays["wm_z"]
    uwm_z = arrays["uwm_z"]
    threshold = summary["calibrated_z_threshold"]

    if len(wm_z) != summary["n_watermarked"] or len(uwm_z) != summary["n_unwatermarked"]:
        fail(f"{prefix}: z-score array lengths do not match summary")
    assert_close(
        f"{prefix} TPR",
        float(np.mean(wm_z > threshold)),
        summary["tpr_at_1pct_fpr_all"],
    )
    assert_close(
        f"{prefix} FPR",
        float(np.mean(uwm_z > threshold)),
        summary["actual_fpr"],
    )
    return summary


def verify_robustness(prefix: str, threshold: float) -> dict:
    summary = load_json(f"robustness_{prefix}.json")
    arrays = np.load(CACHE_DIR / f"robustness_{prefix}_zscores.npz")
    if set(summary) != set(arrays.files):
        fail(f"{prefix}: robustness conditions differ between JSON and NPZ")
    for condition, expected in summary.items():
        values = arrays[condition]
        if len(values) != expected["n"]:
            fail(f"{prefix}/{condition}: unexpected sample count")
        assert_close(f"{prefix}/{condition} mean", float(np.mean(values)), expected["mean_z"])
        assert_close(f"{prefix}/{condition} median", float(np.median(values)), expected["median_z"])
        assert_close(
            f"{prefix}/{condition} TPR",
            float(np.mean(values > threshold)),
            expected["tpr_at_1pct_fpr"],
        )
    return summary


def main() -> int:
    manifest = verify_manifest()
    verify_corpus("corpus_gemma_d2.jsonl", 402, "google/gemma-2-9b-it", 2.0)
    verify_corpus(
        "corpus_llama_d2.jsonl",
        197,
        "meta-llama/Llama-3.1-8B-Instruct",
        2.0,
    )
    gemma = verify_detection("gemma")
    llama = verify_detection("llama")
    verify_robustness("gemma", gemma["calibrated_z_threshold"])
    llama_robustness = verify_robustness("llama", llama["calibrated_z_threshold"])

    print(f"Verified {len(manifest['files'])} cached artifacts against SHA-256.")
    print()
    print("Original Kirchenbauer baseline")
    print("model                         n_wm  n_ctrl  TPR@1%FPR  PPL ratio")
    for summary in (gemma, llama):
        print(
            f"{summary['model']:<29} "
            f"{summary['n_watermarked']:>4}  "
            f"{summary['n_unwatermarked']:>6}  "
            f"{summary['tpr_at_1pct_fpr_all']:>9.1%}  "
            f"{summary['ppl_ratio_wm_over_uwm']:>9.3f}"
        )
    print()
    paraphrase = llama_robustness["llm_paraphrase"]
    print(
        "Llama LLM-paraphrase TPR: "
        f"{paraphrase['tpr_at_1pct_fpr']:.1%} "
        f"(n={paraphrase['n']}; reproduced from cached z-scores)"
    )
    print("The paraphrased texts themselves are not cached, so the attack must still be rerun.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
