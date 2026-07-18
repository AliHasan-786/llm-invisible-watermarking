# Preregistration: watermark robustness under Article 50

**Status:** proposed and frozen for gate review on 2026-07-18. No SynthID result generation may begin until this protocol is approved. The registration commit is the first git commit containing this file and can be resolved with:

```bash
git log --reverse --format=%H -- PREREGISTRATION.md | head -1
```

## 1. Research question and scope

At a detector threshold calibrated to no more than 1% empirical false positives, how often do two text-watermark schemes remain detectable after realistic transformations?

The schemes are:

1. the repository's Kirchenbauer-style green-list logit-bias watermark;
2. SynthID-Text using Hugging Face Transformers' open-source generation implementation;
3. an unwatermarked control generated with otherwise matched settings.

The study evaluates technical detectability. It does not determine legal compliance, audit Google's production deployment, or treat the EU AI Act as defining a numeric pass threshold.

## 2. Policy framing fixed before results

Article 50(2) of Regulation (EU) 2024/1689 says providers of systems generating synthetic text shall ensure outputs are marked in a machine-readable format and detectable as artificially generated or manipulated, with technical solutions that are effective, interoperable, robust, and reliable as far as technically feasible. The relevant transparency obligations become applicable on August 2, 2026.

The report may say that a measured system did or did not meet this study's declared operating point under a named condition. It may not say that a vendor is compliant or non-compliant, provide legal advice, or equate this benchmark with the law.

Primary sources:

- [Regulation (EU) 2024/1689, Article 50 and Article 113](https://eur-lex.europa.eu/eli/reg/2024/1689/oj?locale=en)
- [European Commission Code of Practice on Transparency of AI-Generated Content](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)
- [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4)
- [Google SynthID-Text developer documentation](https://ai.google.dev/responsible/docs/safeguards/synthid)
- [Hugging Face SynthID generation/detection API](https://huggingface.co/docs/transformers/internal/generation_utils)

## 3. Confirmatory hypotheses

- **H1 (baseline replication):** The preserved Kirchenbauer LLM-paraphrase result will reproduce within ±5 percentage points of 27.3% TPR at the declared 1% FPR protocol.
- **H2 (one-pass paraphrase):** SynthID-Text will retain a higher TPR than the Kirchenbauer baseline after one paraphrase pass, pooled across prompts.
- **H3 (two-pass paraphrase):** Both watermarks will fall below the study's predeclared "usable" reference point of 80% TPR at no more than 1% empirical FPR after two paraphrase passes.
- **H4 (length):** TPR will increase with scored token length for both schemes.
- **H5 (quality):** At the selected watermark configurations, the mean reference-model perplexity ratio relative to paired unwatermarked generations will be no greater than 1.15.

The 80% reference point is an evaluation convention chosen before results, not a statutory or industry compliance threshold. Results will also be shown against 90% and 95% reference lines so the conclusion is not driven by one convention.

## 4. Target models and software

Run order:

1. `google/gemma-2-9b-it`
2. `meta-llama/Llama-3.1-8B-Instruct`

Before the first generation, the run configuration must record each exact model revision and tokenizer revision. A model substitution is a protocol deviation and requires a new decision brief before running.

Pinned environment: `requirements.txt`. SynthID generation uses `transformers==4.57.6` and `SynthIDTextWatermarkingConfig` with the documented nine keys `[654, 400, 836, 123, 340, 443, 597, 160, 57]`, `ngram_len=5`, `sampling_table_seed=0`, `sampling_table_size=65536`, and `context_history_size=1024`. The primary SynthID score is the untrained weighted-mean detector from the official `google-deepmind/synthid-text` reference implementation, with thresholds calibrated empirically by length. A Bayesian-detector result may be exploratory only unless its training set and weights are separately preregistered before use.

Kirchenbauer configuration: `gamma=0.5`, `delta=2.0`, key seed 42. This preserves the course baseline.

Generation settings for all three conditions: temperature 1.0, top-p 0.95, maximum 200 new tokens. A deterministic per-prompt sampling seed is derived from SHA-256 of the prompt ID plus the global seed 42 and reused across conditions.

## 5. Prompt corpus and splits

Sources are CNN/DailyMail `3.0.0` test, WritingPrompts test, and TriviaQA `rc` validation. Sample 150 unique prompts per source with seed 42 after applying only these exclusions:

- missing or empty prompt field;
- prompt cannot be tokenized;
- prompt exceeds the configured source-specific truncation limit after standard preprocessing.

The resulting prompt list is frozen before generation and reused across schemes and target models. Every failed generation remains in the run ledger. No prompt is removed based on output quality or detector score.

For each target model, split prompts by a deterministic hash:

- 50% calibration controls;
- 50% held-out evaluation.

Only unwatermarked outputs in the calibration split set detector thresholds. All reported confirmatory TPR/FPR values use the held-out split. Dataset-stratified results are secondary; the pooled held-out result is primary.

## 6. Fixed attack suite

Every attack is applied to the same clean held-out generations. Randomized attacks use per-sample seeds recorded in the cache.

### 6.1 LLM paraphrase

Paraphrasers:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

Prompt: "Rewrite the passage in different words while preserving its meaning and level of detail. Return only the rewrite." Decode with temperature 0.8, top-p 0.95, maximum output length equal to the source completion's token count capped at 250.

Conditions:

- one pass with each paraphraser;
- two passes using the same model twice;
- two passes in both cross-model orders.

The second pass receives only the first rewrite. No watermark processor is active during paraphrasing.

### 6.2 Back-translation

Use `facebook/nllb-200-distilled-600M` with pivots French (`fra_Latn`) and German (`deu_Latn`), translating English→pivot→English. Decode deterministically with beam size 4. Record translation failures without replacement.

### 6.3 Token/word edits

- word substitution at 5%, 10%, and 20% of whitespace-delimited words;
- token insertion at 5%, 10%, and 20% of completion tokens;
- token deletion at 5%, 10%, and 20% of completion tokens.

Indices are sampled without replacement. Inserted token IDs are sampled uniformly from valid non-special tokenizer IDs. Word substitution draws from other non-stopword words in the same completion, matching the existing harness. The exact random seed is stored per output.

### 6.4 Truncation and partial-text detection

Score prefixes at 25, 50, 75, 100, 125, 150, 175, and 200 tokens when available. Also score contiguous start, middle, and end spans of 50 and 100 tokens. A sample contributes only to a length condition it is long enough to support.

### 6.5 Cross-model laundering

For each completion:

1. use the laundering model to produce a factual/semantic summary capped at 80 tokens;
2. in a fresh context, ask it to regenerate a passage matching the original purpose and approximate length from that summary.

Run separately with both paraphraser models. Neither step uses a watermark processor.

## 7. Metrics and decision rules

Primary metric: TPR on held-out watermarked outputs at the detector threshold selected on calibration controls to yield the largest attainable empirical FPR not exceeding 1%. Because finite control samples make exactly 1% impossible in some strata, every table reports target FPR, realized calibration FPR, held-out FPR, threshold, and sample counts.

Secondary metrics:

- full ROC and DET curves;
- area under ROC;
- TPR at 0.1%, 1%, and 5% target FPR where sample size supports it;
- TPR versus scored token length;
- paired change in detector score after attack;
- reference perplexity under `openai-community/gpt2`, reported as watermarked/control ratio;
- completion length and attack length-retention ratio.

Uncertainty:

- 95% percentile bootstrap confidence intervals with 10,000 prompt-level paired resamples;
- paired bootstrap confidence intervals for SynthID-minus-Kirchenbauer TPR differences;
- no null-hypothesis significance claim when a confidence interval or sample size is insufficient.

Ties at the threshold are classified as not detected. Missing/failed outputs are reported by condition. The primary complete-case table includes only prompt pairs present in all compared schemes; an all-available sensitivity table is also reported.

## 8. Readability spot check

Draw 60 prompt-matched pairs per target model, balanced across the three prompt sources and blinded/randomized between watermarked and control text. An independent model context scores fluency, coherence, and task fulfillment on a 1–5 rubric and guesses which output is watermarked. A second independent context reviews 20 overlapping pairs; agreement is reported. This is an authored/model-assisted spot check, not a human-subject study or an external benchmark. Ali may optionally inspect examples, but the pipeline does not require his hours.

## 9. Planned tables and figures

- headline TPR@1%FPR table: scheme × attack, one panel per target model;
- ROC and DET curves for clean, one-pass paraphrase, two-pass paraphrase, and laundering;
- TPR-versus-length curves;
- paired detector-score degradation distributions;
- quality table with perplexity, length retention, and readability spot-check results;
- failure ledger and protocol-deviation appendix.

## 10. Exclusions, deviations, and stopping

No condition may be dropped for poor performance. Compute or access failures are reported as missing, with cached partial outputs preserved.

Allowed implementation fixes without re-registration:

- a bug discovered by a test before any affected result is inspected;
- resumability, logging, serialization, or performance changes that do not alter generated tokens or scores.

Anything that changes a model, prompt sample, attack instruction, budget, detector, metric, split, or hypothesis requires:

1. stopping affected runs;
2. a dated deviation entry;
3. a new `DECISION_BRIEF.md`;
4. approval before resuming.

The Gemma run stops only when all planned prompt-condition pairs have either a cached output or a recorded failure. The Llama run follows the same rule. There is no early stopping based on effect size.

## 11. Compute plan

Recommended first choice: a no-cost Colab or Kaggle GPU with resumable cache shards written every 10 prompt-condition pairs. Paid compute is a fallback only after a timed 20-prompt pilot estimates runtime and cost. Any paid rental requires Ali's explicit approval before credentials or money are used.

The pilot is an engineering check only. Its outputs are quarantined from confirmatory results unless the exact frozen configuration was used and the preregistration gate had already been approved.

## 12. Limitations written before results

- Two target models cannot establish ecosystem-wide behavior.
- Source prompts are English only.
- The open-source SynthID implementation is not Google's production Gemini deployment.
- The attack suite is designed and run by the study author and may omit stronger adaptive attacks.
- Empirical 1% FPR calibration is noisy with hundreds of controls; very low FPR claims require much larger control sets.
- Perplexity and a small blinded spot check are incomplete measures of text quality.
- Results measure detectability under this protocol and do not determine legal compliance.
