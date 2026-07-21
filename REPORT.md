# Watermark Under Article 50

## An independent robustness evaluation of open-source text watermarking

**Report status, 20 July 2026:** methods and preserved-coursework baseline
complete; preregistered SynthID experiment awaiting gated model access and GPU
execution. Blank extension tables are deliberate. They must be populated only
by `scripts/build_article50_report_data.py` from verified run caches.

## Abstract

Text watermarking changes token sampling so a detector can later recover a
machine-readable statistical signal. This study asks a narrow operational
question: at a detector threshold calibrated to no more than 1% empirical
false positives, how often do open-source Kirchenbauer-style and SynthID-Text
watermarks remain detectable after editing, paraphrasing, translation,
truncation, and cross-model laundering?

The protocol was frozen before any SynthID result. It fixes two target models,
three prompt sources, exact model revisions, matched sampling seeds, a separate
calibration split, three control completions per calibration prompt, 33 attack
conditions, strict threshold ties, and prompt-level uncertainty. The preserved
coursework baseline reproduces 90.0% clean TPR for Gemma 2 9B and 98.0% for
Llama 3.1 8B at its declared 1% FPR procedure. A cached Llama paraphrase score
array reproduces 27.3% TPR, but the underlying paraphrased text is absent, so
that result remains historical evidence rather than a newly validated claim.
No SynthID result is reported yet.

This benchmark measures technical detectability under a declared protocol. It
does not decide whether a provider complies with Article 50, evaluate Google's
production Gemini watermark, or provide legal advice.

## 1. Policy question

Article 50(2) of Regulation (EU) 2024/1689 requires providers of systems that
generate synthetic text, audio, image, or video content to ensure outputs are
marked in a machine-readable format and detectable as artificially generated
or manipulated, with solutions that are effective, interoperable, robust, and
reliable as far as technically feasible. Under Article 113, the relevant rules
apply from **2 August 2026**. The European Commission published its final
voluntary Code of Practice on marking and labelling AI-generated content on 10
June 2026 and Article 50 transparency guidelines on 20 July 2026.

Those sources do not turn this study's 80%, 90%, or 95% TPR reference lines
into legal thresholds. They motivate a measurement problem: “detectable” is
not informative without a false-positive cost, content length, transformation,
model, and detector.

Primary policy sources:

- [Regulation (EU) 2024/1689, Articles 50 and 113](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689)
- [Commission guidelines on Article 50 transparency obligations, 20 July 2026](https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems)
- [Final Code of Practice announcement, 10 June 2026](https://digital-strategy.ec.europa.eu/en/news/commission-publishes-code-practice-marking-and-labelling-ai-generated-content)

## 2. Systems under test

### 2.1 Kirchenbauer-style green-list watermark

At each generation step, a keyed hash of the previous token partitions the
vocabulary into green and red lists. The generator adds a fixed logit bias to
green tokens. Detection reconstructs the partitions and converts excess green
hits into a one-sided z-score. This repository's frozen configuration uses
`gamma=0.5`, `delta=2.0`, and key seed 42.

### 2.2 Open-source SynthID-Text

SynthID-Text uses Tournament sampling: pseudorandom g-values influence token
selection across multiple watermarking depths. The primary detector here is
the official untrained weighted-mean score, with repeated contexts and EOS
positions masked. The generator is Transformers 4.57.6's open-source
implementation with the nine documented keys and n-gram length five.

The [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4)
describes a production system and reports large-scale quality and detectability
evaluations. This repository tests public code and declared settings. Its
findings cannot be transferred to Google's proprietary production deployment.

### 2.3 Unwatermarked control

Controls use the same target model, prompt, decoding settings, and matched seed
without a watermark processor. Calibration controls and held-out controls have
separate roles; no held-out observation selects a detector threshold.

## 3. Preregistered design

The binding protocol is `PREREGISTRATION.md`, frozen in commit `0869e39` before
any SynthID result.

- **Models:** pinned Gemma 2 9B Instruct first, then Llama 3.1 8B Instruct.
- **Prompts:** 150 each from CNN/DailyMail, WritingPrompts, and TriviaQA.
- **Split:** deterministic, source-stratified 50/50 calibration and held-out.
- **Calibration amendment:** three unwatermarked completions per calibration
  prompt using distinct SHA-256-derived seeds.
- **Held-out generation:** matched control, Kirchenbauer, and SynthID outputs.
- **Primary metric:** held-out TPR at the most powerful observed threshold with
  empirical calibration FPR no greater than 1%; scores tied with the threshold
  count as not detected.
- **Uncertainty:** 10,000 prompt-level bootstrap resamples; calibration-FPR
  resamples retain each prompt's three-control cluster.
- **Quality:** GPT-2 reference perplexity ratio, output length, and a blinded
  model-assisted readability spot check.

## 4. Attack suite

Every eligible clean held-out output receives the same fixed attack matrix.

| Family | Conditions |
| --- | --- |
| Word substitution | 5%, 10%, 20% |
| Token insertion | 5%, 10%, 20% |
| Token deletion | 5%, 10%, 20% |
| Prefix detection | 25–200 tokens in 25-token steps |
| Partial-text detection | start/middle/end spans of 50 and 100 tokens |
| LLM paraphrase | Qwen and Mistral; one pass, same-model two pass, both cross-model orders |
| Back-translation | English→French→English; English→German→English |
| Cross-model laundering | summarize then regenerate with Qwen or Mistral |

Short outputs are marked ineligible for unsupported length conditions rather
than replaced. Model or translation failures remain in an append-only ledger.

## 5. Preserved baseline evidence

These results predate the extension preregistration and are retained as the
coursework baseline. `make reproduce-baseline` verifies their cached files and
recomputes the values below.

| Generation model | Watermarked / control | Clean TPR at declared 1% FPR | Perplexity ratio | Artifact |
| --- | ---: | ---: | ---: | --- |
| `google/gemma-2-9b-it` | 201 / 201 | 90.0% | 1.095 | `results/detection_gemma_summary.json` |
| `meta-llama/Llama-3.1-8B-Instruct` | 99 / 98 | 98.0% | 1.014 | `results/detection_llama_summary.json` |

The cached Llama robustness score array yields 27.3% TPR after the historical
LLM-paraphrase condition (`n=99`). Because its rewritten texts did not survive,
H1 requires a fresh attack run within ±5 percentage points before this project
promotes paraphrase collapse as a new result.

## 6. Preregistered extension results

**Not run.** The following table must not be hand-edited. The report-data build
will generate it from verified Gemma and Llama caches after the personal
credential/compute gates and all stopping conditions are satisfied.

| Target model | Scheme | Condition | Calibration FPR (95% CI) | Held-out FPR | TPR (95% CI) | n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Pending | Pending | Pending | — | — | — | — |

Hypotheses H1–H5 remain unresolved. H5's perplexity-ratio ceiling is 1.15 and
will be reported as failed if the observed ratio exceeds it.

## 7. Interpretation rules

The report may state that a tested implementation did or did not meet this
study's declared operating point under a named condition. It may not state that
a vendor is legally compliant or non-compliant. A low TPR after an attack means
the tested detector lost evidence under that transformation; it does not show
that all provenance mechanisms fail, nor that watermarking has no deployment
value. Conversely, strong clean detection does not establish robustness.

## 8. Limitations fixed before results

1. Two target models cannot establish ecosystem-wide behavior.
2. Prompts are English only.
3. The tested open-source SynthID variant is not Google's production Gemini
   deployment.
4. The author-run suite may omit stronger or adaptive attacks.
5. Hundreds of clustered controls remain a coarse basis for very-low-FPR
   claims even after tripling completions.
6. Perplexity and a small model-assisted spot check incompletely measure text
   quality.
7. The study measures detectability and does not determine legal compliance.

## 9. Reproduction and provenance

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make check
```

New runs store exact model/tokenizer revisions, generation settings, seeds,
package-lock hash, Git commit, hardware metadata, completions, failures, attack
lineage, detector scores, thresholds, and bootstrap settings. Pilot artifacts
are isolated from `confirmatory/`. No public number may be copied from the
pilot tree.
