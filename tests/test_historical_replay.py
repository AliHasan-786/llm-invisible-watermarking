import json
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "export_historical_replay.py"
RESULTS = REPO / "results"
TRACKED_EXPORT = REPO / "site" / "data" / "historical_replay.json"


def run_export(results, output, check=False):
    command = ["python", str(SCRIPT), "--results", str(results), "--output", str(output)]
    if check:
        command.append("--check")
    return subprocess.run(command, capture_output=True, text=True)


def test_tracked_replay_is_deterministic_and_fail_closed():
    result = run_export(RESULTS, TRACKED_EXPORT, check=True)
    assert result.returncode == 0, result.stdout + result.stderr
    contract = json.loads(TRACKED_EXPORT.read_text(encoding="utf-8"))
    assert contract["provenance"]["verifiedArtifactCount"] == 17
    assert contract["provenance"]["rawTextIncluded"] is False
    assert contract["synthId"]["status"] == "not_run"
    assert contract["synthId"]["scores"] is None
    assert contract["confirmatoryArticle50"]["calibratedFpr"] is None
    assert contract["models"]["gemma"]["clean"]["tpr"] == 0.900497512437811
    assert contract["models"]["llama"]["clean"]["tpr"] == 0.9797979797979798
    gemma_paraphrase = next(
        cell for cell in contract["models"]["gemma"]["editCells"]
        if cell["condition"] == "llm_paraphrase"
    )
    assert gemma_paraphrase["status"] == "not_run"
    assert gemma_paraphrase["tprAtHistoricalThreshold"] is None


def test_missing_input_writes_no_output(tmp_path):
    copied = tmp_path / "results"
    shutil.copytree(RESULTS, copied)
    (copied / "detection_gemma_summary.json").unlink()
    output = tmp_path / "replay.json"
    result = run_export(copied, output)
    assert result.returncode == 1
    assert "Missing cached artifact" in result.stdout
    assert not output.exists()


def test_corrupt_input_writes_no_output(tmp_path):
    copied = tmp_path / "results"
    shutil.copytree(RESULTS, copied)
    with (copied / "robustness_llama.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    output = tmp_path / "replay.json"
    result = run_export(copied, output)
    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stdout
    assert not output.exists()
