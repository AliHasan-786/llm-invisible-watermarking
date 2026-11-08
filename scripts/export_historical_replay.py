"""Export a deterministic, fail-closed browser replay from preserved evidence.

This command performs no inference. It verifies every file in the historical
manifest, checks that cached summaries agree with score arrays, and only then
writes the replay contract used by the local site.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "site" / "data" / "historical_replay.json"
TOLERANCE = 1e-12


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def assert_close(label: str, actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=TOLERANCE):
        fail(f"{label}: expected {expected}, got {actual}")


def verify_manifest(results: Path) -> Dict[str, Any]:
    manifest_path = results / "manifest.json"
    if not manifest_path.is_file():
        fail(f"Missing cache manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        fail("Unsupported or malformed cache manifest")
    for record in manifest["files"]:
        path = results / record["path"]
        if not path.is_file():
            fail(f"Missing cached artifact: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            fail(f"SHA-256 mismatch for {path.name}: {actual}")
    return manifest


def finite_values(label: str, values: np.ndarray) -> List[float]:
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        fail(f"{label}: expected a finite one-dimensional score array")
    return [float(value) for value in values]


def build_model(results: Path, key: str) -> Dict[str, Any]:
    detection = load_json(results / f"detection_{key}_summary.json")
    robustness = load_json(results / f"robustness_{key}.json")
    lengths = load_json(results / f"length_curves_{key}.json")
    detection_arrays = np.load(results / f"detection_{key}_zscores.npz")
    robustness_arrays = np.load(results / f"robustness_{key}_zscores.npz")

    wm_z = detection_arrays["wm_z"]
    control_z = detection_arrays["uwm_z"]
    threshold = float(detection["calibrated_z_threshold"])
    if len(wm_z) != detection["n_watermarked"] or len(control_z) != detection["n_unwatermarked"]:
        fail(f"{key}: detection array lengths do not match summary")
    assert_close(
        f"{key} clean TPR",
        float(np.mean(wm_z > threshold)),
        float(detection["tpr_at_1pct_fpr_all"]),
    )
    assert_close(
        f"{key} clean FPR",
        float(np.mean(control_z > threshold)),
        float(detection["actual_fpr"]),
    )

    if set(robustness) != set(robustness_arrays.files):
        fail(f"{key}: robustness JSON and score-array conditions differ")
    edit_cells = []
    for condition, summary in robustness.items():
        scores = robustness_arrays[condition]
        if len(scores) != summary["n"]:
            fail(f"{key}/{condition}: robustness sample count mismatch")
        assert_close(f"{key}/{condition} mean", float(np.mean(scores)), float(summary["mean_z"]))
        assert_close(f"{key}/{condition} median", float(np.median(scores)), float(summary["median_z"]))
        assert_close(
            f"{key}/{condition} TPR",
            float(np.mean(scores > threshold)),
            float(summary["tpr_at_1pct_fpr"]),
        )
        edit_cells.append(
            {
                "condition": condition,
                "status": "measured_historical",
                "n": int(summary["n"]),
                "meanZ": float(summary["mean_z"]),
                "medianZ": float(summary["median_z"]),
                "tprAtHistoricalThreshold": float(summary["tpr_at_1pct_fpr"]),
                "scores": finite_values(f"{key}/{condition}", scores),
                "transformedTextCached": False,
                "transformationReconstructable": condition != "llm_paraphrase",
            }
        )
    if "llm_paraphrase" not in robustness:
        edit_cells.append(
            {
                "condition": "llm_paraphrase",
                "status": "not_run",
                "n": None,
                "meanZ": None,
                "medianZ": None,
                "tprAtHistoricalThreshold": None,
                "scores": None,
                "transformedTextCached": False,
                "transformationReconstructable": False,
            }
        )

    length_cells = []
    for cell in lengths:
        required = {"n_tokens", "tpr", "fpr", "threshold", "n_wm", "n_uwm", "calibration"}
        if set(cell) != required or cell["calibration"] != "per_length":
            fail(f"{key}: malformed historical length cell")
        length_cells.append(
            {
                "tokens": int(cell["n_tokens"]),
                "status": "measured_historical",
                "threshold": float(cell["threshold"]),
                "tpr": float(cell["tpr"]),
                "fpr": float(cell["fpr"]),
                "nWatermarked": int(cell["n_wm"]),
                "nControl": int(cell["n_uwm"]),
                "calibration": "per_length_same_cache",
            }
        )

    return {
        "modelId": detection["model"],
        "scheme": "kirchenbauer_greenlist",
        "configuration": {
            "delta": float(detection["delta"]),
            "gamma": float(detection["gamma"]),
            "seed": int(detection["seed"]),
        },
        "clean": {
            "status": "measured_historical",
            "threshold": threshold,
            "thresholdRule": "historical_empirical_upper_tail",
            "calibrationLimitation": (
                "The preserved coursework cache does not record a calibration/held-out split. "
                "This is not a confirmatory calibrated-FPR estimate."
            ),
            "nWatermarked": int(detection["n_watermarked"]),
            "nControl": int(detection["n_unwatermarked"]),
            "tpr": float(detection["tpr_at_1pct_fpr_all"]),
            "realizedFpr": float(detection["actual_fpr"]),
            "pplRatio": float(detection["ppl_ratio_wm_over_uwm"]),
            "watermarkedScores": finite_values(f"{key}/watermarked", wm_z),
            "controlScores": finite_values(f"{key}/control", control_z),
        },
        "lengthCells": length_cells,
        "editCells": edit_cells,
    }


def build_contract(results: Path) -> Dict[str, Any]:
    manifest = verify_manifest(results)
    models = {key: build_model(results, key) for key in ("gemma", "llama")}
    return {
        "schemaVersion": 1,
        "asOf": "2026-11-08",
        "status": "historical_coursework_replay",
        "scope": (
            "Local replay of preserved Kirchenbauer detector evidence. No model inference, "
            "live detection, SynthID result, or Article 50 compliance conclusion."
        ),
        "provenance": {
            "manifestPath": "results/manifest.json",
            "manifestSha256": sha256(results / "manifest.json"),
            "verifiedArtifactCount": len(manifest["files"]),
            "source": manifest["source"],
            "rawTextIncluded": False,
            "rights": (
                "Numeric derivatives of repository-owned coursework caches only. Upstream "
                "dataset terms continue to govern prompts and completions; none are duplicated here."
            ),
        },
        "models": models,
        "synthId": {
            "status": "not_run",
            "threshold": None,
            "scores": None,
            "lengthCells": None,
            "editCells": None,
            "reason": "No preserved SynthID generation or detector output exists.",
        },
        "confirmatoryArticle50": {
            "status": "not_run",
            "calibratedFpr": None,
            "results": None,
            "reason": "The preregistered pilot and confirmatory runs have not executed.",
        },
    }


def serialize(contract: Dict[str, Any]) -> bytes:
    return (json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_atomic(output: Path, content: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(output.parent), delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(str(temporary), str(output))
    output.chmod(0o644)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if the tracked export is stale")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    content = serialize(build_contract(args.results))
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != content:
            fail(f"Replay export is missing or stale: {args.output}")
        print(f"Verified deterministic replay export: {args.output}")
        return 0
    write_atomic(args.output, content)
    print(f"Wrote verified replay export: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
