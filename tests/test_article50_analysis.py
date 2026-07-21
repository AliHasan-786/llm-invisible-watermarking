from evaluation.article50_analysis import (
    operating_point_supported,
    paired_quality_summary,
    readability_queue,
    roc_det_curve,
)


def test_roc_det_perfect_separation_and_support_rule():
    curve = roc_det_curve([2.0, 3.0], [0.0, 1.0])
    assert curve["auroc"] == 1.0
    assert curve["fnr"] == [1 - value for value in curve["tpr"]]
    assert operating_point_supported(100, 0.01)
    assert not operating_point_supported(99, 0.01)


def test_paired_quality_summary_evaluates_fixed_h5_ceiling():
    rows = []
    for prompt, control_ppl, wm_ppl in (("a", 10, 11), ("b", 20, 24)):
        rows.extend(
            [
                {
                    "prompt_id": prompt,
                    "split": "heldout",
                    "scheme": "control",
                    "perplexity": control_ppl,
                    "n_tokens": 100,
                },
                {
                    "prompt_id": prompt,
                    "split": "heldout",
                    "scheme": "synthid",
                    "perplexity": wm_ppl,
                    "n_tokens": 100,
                },
            ]
        )
    summary = paired_quality_summary(rows)[0]
    assert summary["scheme"] == "synthid"
    assert summary["mean_paired_perplexity_ratio"] == 1.15
    assert summary["h5_pass"]


def test_readability_queue_is_balanced_blinded_and_has_20_overlap():
    rows = []
    for source in ("cnn_dailymail", "writing_prompts", "trivia_qa"):
        for index in range(10):
            prompt = f"{source}-{index}"
            for scheme in ("control", "kirchenbauer", "synthid"):
                rows.append(
                    {
                        "prompt_id": prompt,
                        "source": source,
                        "split": "heldout",
                        "scheme": scheme,
                        "completion": f"{scheme} text for {prompt}",
                    }
                )
    queue, key = readability_queue(rows)
    assert len(queue) == len(key) == 60
    assert sum(row["reviewers"] == ["primary", "secondary"] for row in queue) == 20
    assert {row["watermarked_side"] for row in key} == {"A", "B"}
    assert all("scheme" not in row for row in queue)
