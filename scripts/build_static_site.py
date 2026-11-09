"""Build the allowlisted public static package for the evidence replay.

The repository contains raw coursework corpora, notebooks, and research code.
They are intentionally excluded. This builder copies only reviewed frontend
assets, the numeric replay derivative, its license notice, and linked methods
documents after verifying every source hash and the replay's manifest binding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist"
RESULTS = ROOT / "results"
PUBLIC_FILES = {
    "site/index.html": "index.html",
    "site/styles.css": "styles.css",
    "site/app.js": "app.js",
    "site/replay_contract.js": "replay_contract.js",
    "site/data/historical_replay.json": "historical-replay.json",
    "assets/text-provenance-lab-thumbnail.svg": "assets/text-provenance-lab-thumbnail.svg",
    "assets/ASSET_LICENSES.md": "assets/ASSET_LICENSES.md",
    "REPORT.md": "REPORT.md",
    "PREREGISTRATION.md": "PREREGISTRATION.md",
    "DATA_PROVENANCE.md": "DATA_PROVENANCE.md",
    "docs/HISTORICAL_REPLAY_CONTRACT.md": "docs/HISTORICAL_REPLAY_CONTRACT.md",
}
FORBIDDEN_SUFFIXES = {".ipynb", ".jsonl", ".npz", ".py", ".pyc", ".pyo", ".env"}
FORBIDDEN_PARTS = {"api", "backend", "functions", "notebooks", "pipeline", "results", "tests", "watermark"}
FORBIDDEN_REPLAY_KEYS = {"completion", "completions", "generated_text", "prompt", "prompts", "text", "token_ids", "tokenids"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_numeric_derivative() -> None:
    tracked = ROOT / "site" / "data" / "historical_replay.json"
    manifest_path = RESULTS / "manifest.json"
    if not tracked.is_file() or not manifest_path.is_file():
        fail("Tracked replay derivative or source manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or len(manifest.get("files", [])) != 17:
        fail("Unexpected source manifest artifact count")
    for record in manifest["files"]:
        source = RESULTS / record["path"]
        if not source.is_file() or sha256_path(source) != record["sha256"]:
            fail(f"Source artifact is missing or corrupt: {record['path']}")
    replay = json.loads(tracked.read_text(encoding="utf-8"))
    if replay.get("provenance", {}).get("manifestSha256") != sha256_path(manifest_path):
        fail("Replay derivative does not reference the current source manifest")
    if replay.get("buildVerification", {}).get("status") != "verified_by_exporter":
        fail("Replay derivative lacks exporter verification status")


def reject_raw_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_REPLAY_KEYS:
                fail(f"Forbidden raw-text field in replay derivative: {path}.{key}")
            reject_raw_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_raw_keys(child, f"{path}[{index}]")


def build(output: Path) -> Dict[str, Any]:
    verify_numeric_derivative()
    if output == ROOT or output == output.parent:
        fail("Refusing to replace the repository or filesystem root")
    if output.exists():
        if not output.is_dir():
            fail(f"Static output exists and is not a directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    inventory = []
    for source_name, destination_name in sorted(PUBLIC_FILES.items(), key=lambda item: item[1]):
        source = ROOT / source_name
        destination = output / destination_name
        if not source.is_file():
            fail(f"Missing allowlisted public source: {source_name}")
        content = source.read_bytes()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        inventory.append(
            {
                "path": destination_name,
                "sha256": sha256_bytes(content),
                "sizeBytes": len(content),
                "source": source_name,
            }
        )

    replay = json.loads((output / "historical-replay.json").read_text(encoding="utf-8"))
    reject_raw_keys(replay)

    manifest = {
        "schemaVersion": 1,
        "package": "text-provenance-lab-static",
        "artifactCount": len(inventory),
        "files": inventory,
        "exclusions": [
            "raw coursework corpora and prompts/completions",
            "NPZ source score caches",
            "notebooks and generated outputs",
            "Python research/evaluation/runtime code",
            "backend functions, credentials, and environment files",
        ],
    }
    manifest_content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output / "_build_manifest.json").write_bytes(manifest_content)

    expected = set(PUBLIC_FILES.values()) | {"_build_manifest.json"}
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    if actual != expected:
        fail(f"Unexpected static package inventory: {sorted(actual ^ expected)}")
    for relative in actual:
        path = Path(relative)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(part.lower() in FORBIDDEN_PARTS for part in path.parts):
            fail(f"Forbidden file in static package: {relative}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build(args.output.resolve())
    print(f"Built {manifest['artifactCount']} allowlisted public artifacts in {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
