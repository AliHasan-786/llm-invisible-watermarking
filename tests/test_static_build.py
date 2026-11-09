import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_static_site.py"
EXPECTED_FILES = {
    "DATA_PROVENANCE.md",
    "PREREGISTRATION.md",
    "REPORT.md",
    "_build_manifest.json",
    "app.js",
    "assets/ASSET_LICENSES.md",
    "assets/text-provenance-lab-thumbnail.svg",
    "docs/HISTORICAL_REPLAY_CONTRACT.md",
    "historical-replay.json",
    "index.html",
    "replay_contract.js",
    "styles.css",
}


class LocalAssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, _tag, attrs):
        values = dict(attrs)
        for name in ("href", "src"):
            value = values.get(name)
            if value:
                self.urls.append(value)
        if values.get("property") == "og:image" and values.get("content"):
            self.urls.append(values["content"])


def test_static_package_is_exact_allowlist_without_raw_or_backend_files(tmp_path):
    output = tmp_path / "public"
    result = subprocess.run(
        ["python", str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    assert files == EXPECTED_FILES
    assert not any(path.suffix in {".jsonl", ".npz", ".ipynb", ".py", ".env"} for path in output.rglob("*"))
    assert not any(part in {"api", "backend", "functions", "pipeline", "results", "tests"} for path in output.rglob("*") for part in path.parts)

    manifest = json.loads((output / "_build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifactCount"] == 11
    assert {item["path"] for item in manifest["files"]} == EXPECTED_FILES - {"_build_manifest.json"}
    replay = json.loads((output / "historical-replay.json").read_text(encoding="utf-8"))
    serialized = json.dumps(replay).lower()
    for forbidden in ('"prompt":', '"completion":', '"token_ids":', '"text":'):
        assert forbidden not in serialized

    parser = LocalAssetParser()
    parser.feed((output / "index.html").read_text(encoding="utf-8"))
    for url in parser.urls:
        parsed = urlparse(url)
        if parsed.scheme or url.startswith(("#", "data:")):
            continue
        target = output / parsed.path.lstrip("/")
        assert target.is_file(), f"broken packaged link: {url}"
    assert (output / "historical-replay.json").is_file()

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output.rglob("*")
        if path.is_file()
    )
    assert not re.search(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", public_text)
    assert not re.search(r"(?i)(?:api[_-]?key|secret|access[_-]?token)\s*[:=]\s*['\"][^'\"]+", public_text)


def test_vercel_builds_only_the_allowlisted_output_directory():
    config = json.loads((REPO / "vercel.json").read_text(encoding="utf-8"))
    assert config["installCommand"] == "true"
    assert config["buildCommand"] == "python3 scripts/build_static_site.py"
    assert config["outputDirectory"] == "dist"
    assert "rewrites" not in config
