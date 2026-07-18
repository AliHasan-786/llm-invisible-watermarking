import math

import pytest
import torch

from evaluation.metrics import (
    calibrate_threshold,
    calibration_summary,
    clustered_calibration_fpr_ci,
    headline_detection_summary,
    realized_fpr,
)
from pipeline.article50 import (
    build_generation_plan,
    derived_seed,
    generation_key,
    prompt_id,
)
from watermark.synthid import weighted_mean


def _prompts(n_per_source=4):
    rows = []
    for source in ("cnn_dailymail", "writing_prompts", "trivia_qa"):
        for index in range(n_per_source):
            prompt = f"{source} prompt {index}"
            rows.append(
                {
                    "source": source,
                    "source_index": index,
                    "prompt": prompt,
                    "prompt_id": prompt_id(source, index, prompt),
                }
            )
    return rows


def test_generation_plan_triples_calibration_and_matches_heldout_seeds():
    plan = build_generation_plan(_prompts())
    calibration = [row for row in plan if row["split"] == "calibration"]
    heldout = [row for row in plan if row["split"] == "heldout"]

    assert len(calibration) == 6 * 3
    assert len(heldout) == 6 * 3
    assert {row["scheme"] for row in calibration} == {"control"}

    for prompt_identifier in {row["prompt_id"] for row in calibration}:
        rows = [row for row in calibration if row["prompt_id"] == prompt_identifier]
        assert {row["replicate"] for row in rows} == {0, 1, 2}
        assert len({row["seed"] for row in rows}) == 3

    for prompt_identifier in {row["prompt_id"] for row in heldout}:
        rows = [row for row in heldout if row["prompt_id"] == prompt_identifier]
        assert {row["scheme"] for row in rows} == {
            "control",
            "kirchenbauer",
            "synthid",
        }
        assert len({row["seed"] for row in rows}) == 1

    assert len({generation_key(row) for row in plan}) == len(plan)


def test_plan_is_input_order_independent_and_seed_is_stable():
    prompts = _prompts()
    forward = build_generation_plan(prompts)
    reverse = build_generation_plan(list(reversed(prompts)))
    assert forward == reverse
    assert derived_seed(prompts[0]["prompt_id"], 0) == derived_seed(
        prompts[0]["prompt_id"], 0
    )
    assert derived_seed(prompts[0]["prompt_id"], 0) != derived_seed(
        prompts[0]["prompt_id"], 1
    )


def test_threshold_obeys_strict_greater_than_and_ties_are_missed():
    scores = list(range(100))
    threshold = calibrate_threshold(scores, target_fpr=0.01)
    assert threshold == 98
    assert realized_fpr(scores, threshold) == 0.01

    tied = [0.0] * 99 + [1.0]
    threshold = calibrate_threshold(tied, target_fpr=0.01)
    assert threshold == 0.0
    assert realized_fpr(tied, threshold) == 0.01


def test_clustered_bootstrap_requires_control_triplets_and_is_reproducible():
    grouped = {
        "a": [0.0, 0.0, 1.0],
        "b": [0.0, 0.0, 0.0],
        "c": [0.0, 0.0, 0.0],
    }
    first = clustered_calibration_fpr_ci(grouped, 0.5, n_bootstrap=200, seed=7)
    second = clustered_calibration_fpr_ci(grouped, 0.5, n_bootstrap=200, seed=7)
    assert first == second
    assert 0 <= first[0] <= first[1] <= 1
    with pytest.raises(ValueError, match="expected 3"):
        clustered_calibration_fpr_ci({"a": [0.0, 1.0]}, 0.5)


def test_calibration_summary_reports_amended_counts():
    rows = [
        {"prompt_id": f"p{prompt}", "score": float(prompt * 3 + replicate)}
        for prompt in range(4)
        for replicate in range(3)
    ]
    summary = calibration_summary(rows, n_bootstrap=100)
    assert summary["n_calibration_prompts"] == 4
    assert summary["n_calibration_controls"] == 12
    assert summary["bootstrap_unit"] == "prompt_cluster_with_three_controls"
    assert summary["ties"] == "not_detected"


def test_weighted_mean_matches_reference_formula():
    g_values = torch.tensor(
        [
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ]
    )
    mask = torch.tensor([[1, 0, 1]])
    weights = torch.tensor([3.0, 2.0, 1.0])
    actual = weighted_mean(g_values, mask, weights)[0].item()
    normalized = weights * (3 / weights.sum())
    expected = (
        (g_values[0, 0] * normalized).sum()
        + (g_values[0, 2] * normalized).sum()
    ) / (3 * 2)
    assert math.isclose(actual, expected.item())


def test_weighted_mean_rejects_empty_mask():
    with pytest.raises(ValueError, match="no unmasked"):
        weighted_mean(torch.ones((1, 2, 3)), torch.zeros((1, 2)))


def test_headline_summary_uses_calibration_only_and_reports_heldout_fpr():
    calibration = [
        {"prompt_id": f"c{prompt}", "score": score}
        for prompt, scores in enumerate(([0.0, 0.1, 0.2], [0.3, 0.4, 1.0]))
        for score in scores
    ]
    heldout = [
        {"prompt_id": "h1", "scheme": "control", "score": 0.1},
        {"prompt_id": "h2", "scheme": "control", "score": 2.0},
        {"prompt_id": "h1", "scheme": "synthid", "score": 1.5},
        {"prompt_id": "h2", "scheme": "synthid", "score": 0.2},
    ]
    summary = headline_detection_summary(
        calibration,
        heldout,
        positive_scheme="synthid",
        n_bootstrap=100,
    )
    assert summary["n_calibration_controls"] == 6
    assert summary["n_heldout_positive"] == 2
    assert summary["n_heldout_controls"] == 2
    assert summary["heldout_tpr"] == 0.5
    assert summary["heldout_fpr"] == 0.5
