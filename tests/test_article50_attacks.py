from evaluation.article50_attacks import (
    attack_seed,
    attack_specs,
    deterministic_attack_rows,
    prefix,
    span,
    token_edit,
    word_substitution,
)


def test_attack_matrix_matches_preregistration():
    specs = attack_specs()
    conditions = {spec.condition for spec in specs}
    assert len(specs) == 33
    assert len(conditions) == len(specs)
    assert sum(spec.model_backed for spec in specs) == 10
    assert "paraphrase_qwen_mistral" in conditions
    assert "backtranslation_german" in conditions
    assert "laundering_mistral" in conditions


def test_token_edits_are_deterministic_and_without_replacement():
    source = list(range(20))
    deleted = token_edit(source, rate=0.20, mode="delete", seed=7)
    assert deleted == token_edit(source, rate=0.20, mode="delete", seed=7)
    assert len(deleted) == 16
    inserted = token_edit(
        source,
        rate=0.20,
        mode="insert",
        seed=7,
        valid_non_special_ids=[90, 91],
    )
    assert len(inserted) == 24
    assert all(token in source or token in (90, 91) for token in inserted)


def test_word_substitution_uses_source_words_and_stable_seed():
    text = "alpha beta gamma delta epsilon zeta"
    attacked = word_substitution(text, 0.20, 42)
    assert attacked == word_substitution(text, 0.20, 42)
    assert len(attacked.split()) == len(text.split())
    assert set(attacked.split()) <= set(text.split())


def test_prefix_and_spans_require_enough_tokens():
    ids = list(range(10))
    assert prefix(ids, 5) == [0, 1, 2, 3, 4]
    assert prefix(ids, 11) is None
    assert span(ids, 4, "start") == [0, 1, 2, 3]
    assert span(ids, 4, "middle") == [3, 4, 5, 6]
    assert span(ids, 4, "end") == [6, 7, 8, 9]


def test_deterministic_runner_records_ineligible_lengths_and_lineage():
    clean = [
        {
            "prompt_id": "p1",
            "source": "trivia_qa",
            "scheme": "synthid",
            "split": "heldout",
            "completion": "one two three four five six",
            "token_ids": list(range(30)),
            "tokenizer_vocab_size": 100,
            "eos_token_id": 99,
        }
    ]
    outputs, ineligible = deterministic_attack_rows(
        clean,
        valid_non_special_ids=list(range(90)),
        encode=lambda text: list(range(len(text.split()))),
        decode=lambda ids: " ".join(map(str, ids)),
    )
    assert len(outputs) + len(ineligible) == 23
    assert {row["reason"] for row in ineligible} == {"source_completion_too_short"}
    assert all(row["attack_seed"] == attack_seed("p1", row["condition"]) for row in outputs)
    assert all(row["tokenizer_vocab_size"] == 100 for row in outputs)
