# Data and result provenance

## Scope

This card covers the preserved CS 5788 baseline and the planned Article 50 evaluation. It records lineage; it does not relicense upstream datasets or model outputs.

## Prompt sources

| Dataset | Upstream identifier | Split in code | Use |
| --- | --- | --- | --- |
| CNN/DailyMail | `cnn_dailymail`, config `3.0.0` | `test` | summarization prompts |
| WritingPrompts | `euclaise/writingprompts` | `test` | open-ended continuation prompts |
| TriviaQA | `trivia_qa`, config `rc` | `validation` | question-answer prompts |

`pipeline/generate.py` stores the prompt text, generated completion, completion token IDs, source name, model ID, watermark parameters, and sample length. The baseline used seed 42, `gamma=0.5`, `delta=2.0` for headline runs, temperature 1.0, top-p 0.95, and at most 200 new tokens.

The source datasets can contain names and text about real people because they are public news, story, and question-answer corpora. The study does not enrich those records, infer sensitive traits, or make person-level claims. Upstream licenses and dataset cards control reuse.

## Preserved baseline lineage

The canonical clone was missing its ignored result files. On 2026-07-18 the complete baseline cache was recovered from:

```text
/Users/alihasan/Downloads/Learning Materials/AI & Generative Models/Gen Models Final/results/
```

An independently saved duplicate under `Gemma_Results/results/` matched the Gemma artifacts byte-for-byte. The repository copies are listed in `results/baseline/manifest.json`, which records a SHA-256 digest for every file.

The preserved cache contains:

- five Gemma delta-sweep corpora (402 rows each);
- one Llama headline corpus (197 rows);
- detection summaries and raw z-score/perplexity arrays for both models;
- length-curve summaries for both models;
- robustness summaries and raw z-score arrays for both models.

The original LLM-paraphrase output is not present. The 27.3% TPR value appears in the final report, but no raw paraphrased text or z-score array survived. It is therefore excluded from `make reproduce-baseline` and is a required replication target.

## Planned evaluation lineage

Each new run will write:

- an immutable configuration record with git commit, model ID and revision, package lock hash, hardware, random seeds, and start/end timestamps;
- prompt IDs and hashes so raw upstream prompt text need not be duplicated unnecessarily;
- generated text and token IDs for every scheme/control pair;
- attack outputs as separate immutable records linked to the source generation;
- detector scores, calibration/test split membership, thresholds, and metrics;
- SHA-256 manifests for all cached shards.

No result is deleted because it is negative. Failed or excluded samples remain in the run ledger with a reason code.

## Known limitations

- The source prompts are English-language even when an attack temporarily translates them.
- Only two target model families are planned.
- The SynthID condition evaluates the open-source implementation, not Google's production Gemini deployment.
- Attacks are author-designed and author-run.
- Dataset versions were not pinned by revision in the original coursework cache; P1 must pin exact dataset revisions before generation.
