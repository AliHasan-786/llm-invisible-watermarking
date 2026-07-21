import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"


def test_site_has_required_interactions_and_honest_scope():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for required in (
        "Generate marked sample",
        "Paraphrase proxy",
        "Translation proxy",
        "Cut to half",
        "SynthID extension pending gated compute",
        "not the token-level benchmark",
        "does not pronounce legal compliance",
    ):
        assert required in html


def test_site_design_constraints_and_mobile_breakpoints():
    css = (SITE / "styles.css").read_text(encoding="utf-8")
    assert "gradient" not in css.lower()
    assert "@media (max-width: 480px)" in css
    assert "border-radius: 2px" in css
    assert "Georgia" in css


def test_javascript_syntax_when_node_is_available():
    result = subprocess.run(
        ["node", "--check", str(SITE / "app.js")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_vercel_config_routes_static_site():
    config = json.loads((REPO / "vercel.json").read_text(encoding="utf-8"))
    rewrites = {item["source"]: item["destination"] for item in config["rewrites"]}
    assert rewrites["/"] == "/site/index.html"
    assert rewrites["/app.js"] == "/site/app.js"
