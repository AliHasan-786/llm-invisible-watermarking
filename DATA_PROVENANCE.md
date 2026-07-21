# Data and result provenance

## Scope

This card covers the preserved CS 5788 baseline and the planned Article 50 evaluation. It records lineage; it does not relicense upstream datasets or model outputs.

## Prompt sources

| Dataset | Upstream identifier | Split in code | Use |
| --- | --- | --- | --- |
| CNN/DailyMail | `abisee/cnn_dailymail`, config `3.0.0`, revision `96df5e686bee6baa90b8bee7c28b81fa3fa6223d` | `test` | summarization prompts |
| WritingPrompts | `euclaise/writingprompts`, revision `35f0aa359452ba8147b34d925684fccee26679cc` | `test` | open-ended continuation prompts |
| TriviaQA | `mandarjoshi/trivia_qa`, config `rc`, revision `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` | `validation` | question-answer prompts |

`pipeline/generate.py` stores the prompt text, generated completion, completion token IDs, source name, model ID, watermark parameters, and sample length. The baseline used seed 42, `gamma=0.5`, `delta=2.0` for headline runs, temperature 1.0, top-p 0.95, and at most 200 new tokens.

The source datasets can contain names and text about real people because they are public news, story, and question-answer corpora. The study does not enrich those records, infer sensitive traits, or make person-level claims. Upstream licenses and dataset cards control reuse.

## Preserved baseline lineage

The local clone initially had a stale tracking ref and ignored result stubs, but the current GitHub `main` already contains the complete baseline cache. After fetching the force-updated remote history, those committed root artifacts became the source of truth. An independently saved local copy under `Gemma_Results/results/` matches the Gemma artifacts byte-for-byte. The committed files are listed in `results/manifest.json`, which records a SHA-256 digest for every baseline artifact.

The preserved cache contains:

- five Gemma delta-sweep corpora (402 rows each);
- one Llama headline corpus (197 rows);
- detection summaries and raw z-score/perplexity arrays for both models;
- length-curve summaries for both models;
- robustness summaries and raw z-score arrays for both models.

The Llama robustness summary and z-score array include the LLM-paraphrase condition and reproduce the report's 27.3% TPR exactly. The paraphrased text itself is not present, so the detector score can be audited but the complete attack cannot be replayed. It remains a required fresh replication target.

## Planned evaluation lineage

Each new run will write:

- an immutable configuration record with git commit, model ID and revision, package lock hash, hardware, random seeds, and start/end timestamps;
- prompt IDs and hashes so raw upstream prompt text need not be duplicated unnecessarily;
- generated text and token IDs for every scheme/control pair;
- attack outputs as separate immutable records linked to the source generation;
- detector scores, calibration/test split membership, thresholds, and metrics;
- SHA-256 manifests for all cached shards.

No result is deleted because it is negative. Failed or excluded samples remain in the run ledger with a reason code.

The P1 prompt freezer resolves the public datasets at the immutable revisions
above before sampling. Its metadata records the source row index, constructed
prompt, token count, exclusion totals, target tokenizer revision, and prompt
file digest. No prompt is selected or removed using completion quality or a
detector score.

## Known limitations

- The source prompts are English-language even when an attack temporarily translates them.
- Only two target model families are planned.
- The SynthID condition evaluates the open-source implementation, not Google's production Gemini deployment.
- Attacks are author-designed and author-run.
- Dataset revisions were not pinned in the original coursework cache. The P1 extension pins them, but that does not retroactively version the baseline prompt download.
