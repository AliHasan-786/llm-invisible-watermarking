import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"


def test_site_has_required_interactions_and_honest_scope():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for required in (
        "Text Provenance Lab",
        "Reset demo sample",
        "Paraphrase proxy",
        "Translation proxy",
        "Cut to half",
        "no SynthID or confirmatory result exists yet",
        "not the cached detector",
        "does not pronounce legal compliance",
        "data-historical-replay",
        "The deterministic exporter—not this browser—verified",
        "https://ali-hasan-ai.vercel.app",
        "https://github.com/AliHasan-786/llm-invisible-watermarking",
    ):
        assert required in html


def test_site_design_constraints_and_mobile_breakpoints():
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    assert "gradient" not in css.lower()
    assert "@media (max-width: 480px)" in css
    assert "border-radius: 2px" in css
    assert "Georgia" in css
    assert ".table-wrap:focus-visible" in css
    assert ".rate-tick {" in css and "left: var(--value)" in css
    assert ".rate-label {" in css and "left: clamp(45px, var(--label-value)" in css
    assert ".score-dot.control" in css and "rotate(45deg)" in css
    assert ".score-dot.marked" in css and "border-radius: 50%" in css


def test_production_markup_links_accessible_tables_and_exact_rate_ticks():
    app = (SITE / "app.js").read_text(encoding="utf-8")
    assert 'role="region"' in app
    assert 'tabindex="0"' in app
    assert "ratePercent(cell.tpr)" in app
    assert "ratePercent(cell.fpr)" in app
    assert 'class="rate-tick tpr"' in app
    assert 'class="rate-tick fpr"' in app
    assert "Baseline is the unmodified historical completion" in app
    assert '"/historical-replay.json"' in app


def test_javascript_syntax_when_node_is_available():
    for path in (SITE / "app.js", SITE / "replay_contract.js"):
        result = subprocess.run(
            ["node", "--check", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_browser_contract_loading_and_malformed_states():
    result = subprocess.run(
        ["node", "--test", str(REPO / "tests" / "js" / "replay_contract.test.js")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_vercel_config_uses_allowlisted_static_build():
    config = json.loads((REPO / "vercel.json").read_text(encoding="utf-8"))
    assert config["installCommand"] == "true"
    assert config["buildCommand"] == "python3 scripts/build_static_site.py"
    assert config["outputDirectory"] == "dist"
    assert "rewrites" not in config


def test_portfolio_metadata_and_licensed_thumbnail():
    metadata = json.loads((REPO / "portfolio.json").read_text(encoding="utf-8"))
    replay = json.loads((SITE / "data" / "historical_replay.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "watermarking"
    assert metadata["title"] == "Text Provenance Lab"
    assert metadata["evidenceStage"] == "coursework"
    assert metadata["primaryMetric"] is None
    assert metadata["status"] == "production"
    assert metadata["demoUrl"] == "https://text-provenance-lab.vercel.app"
    assert metadata["writeupUrl"].endswith("/blob/main/REPORT.md")
    assert metadata["release"] == {
        "reviewedSourceCommit": "e794de39b9100dc51bdbff398f0a35b986abcd4b",
        "sourceMergeCommit": "27f3f89ec2368f324ef24c3a2fcfd92c3214ca10",
        "metadataRevision": "git:containing-commit",
        "checkedAt": "2026-11-09",
        "publicationStatus": "production-live",
        "productionUrl": "https://text-provenance-lab.vercel.app",
        "productionDeploymentId": "dpl_4DHxhKGFWBjyw3Q9cPM2nM8Wxga6",
    }
    assert len(metadata["evidence"]) == 3
    evidence = {item["id"]: item for item in metadata["evidence"]}
    assert evidence["historical-gemma-clean-tpr"]["value"] == replay["models"]["gemma"]["clean"]["tpr"]
    assert evidence["historical-llama-clean-tpr"]["value"] == replay["models"]["llama"]["clean"]["tpr"]
    llama_paraphrase = next(
        cell for cell in replay["models"]["llama"]["editCells"]
        if cell["condition"] == "llm_paraphrase"
    )
    assert evidence["historical-llama-paraphrase-tpr"]["value"] == llama_paraphrase["tprAtHistoricalThreshold"]
    thumbnail = REPO / metadata["thumbnail"]
    ET.parse(thumbnail)
    licenses = (REPO / "assets" / "ASSET_LICENSES.md").read_text(encoding="utf-8")
    assert "Creative Commons Attribution 4.0" in licenses


def test_public_release_workflow_is_dependency_free_and_honestly_scoped():
    workflow = (REPO / ".github" / "workflows" / "public-release.yml").read_text(encoding="utf-8")
    assert "public release offline smoke" in workflow
    assert "python scripts/build_static_site.py" in workflow
    assert "node --test tests/js/replay_contract.test.js" in workflow
    assert "pip install" not in workflow
    assert "npm install" not in workflow
    assert "requirements.txt" not in workflow
    assert "make check" not in workflow
