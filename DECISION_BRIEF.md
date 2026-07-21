# Decision brief: P1 engineering complete — execution gate

**Date:** 2026-07-20

**Branch:** `codex/watermark-p0`

**Recommended call:** P1's frozen execution harness is ready. Proceed with the
already-approved 20-prompt no-cost pilot when Ali is available to complete the
personal Hugging Face credential cell. Do not authorize paid compute, model
substitution, quantization, merge, publication, or deployment through this
decision.

## What is complete

- Exact source-stratified split, matched held-out seeds, and three distinct
  calibration-control seeds per prompt.
- Immutable model, tokenizer, and public-dataset revisions.
- Transformers 4.57.6 SynthID generation plus official untrained weighted-mean
  detection semantics.
- All 33 frozen attack conditions: 23 deterministic edits/length conditions
  and 10 model-backed paraphrase, back-translation, and laundering conditions.
- One-model-at-a-time attack loading for constrained hosted GPUs.
- Clean and condition-level thresholds, prompt-clustered calibration-FPR CIs,
  held-out complete-case/all-available results, and paired scheme differences.
- Resumable completion/failure/attack ledgers and a report-data builder that
  rejects pilot or non-confirmatory artifacts.
- A Colab notebook that checks the preregistration freeze, pauses for masked
  credential entry, runs the pilot, verifies ledger completeness, and bundles
  the result directory.

No Gemma, Llama, SynthID, or new attack result was produced.

## Verification

| Evidence | Result |
| --- | --- |
| P1 detector/protocol/attack tests | 20 passed |
| New Python entry points | compile successfully |
| Frozen attack manifest | 33 unique conditions: 23 deterministic, 10 model-backed |
| Baseline cache check | 17/17 artifact hashes and preserved values reproduce |
| Local preflight | stops as designed: no CUDA and no authenticated gated-model session |
| Colab notebook | 15 cells; valid notebook JSON and Python syntax |
| Dataset packaging | all three public repositories verified as Parquet and pinned at immutable upstream revisions |

## Remaining P1 work

The pilot and full Gemma suite require two resources Codex cannot silently
cross:

1. Ali enters a read-only Hugging Face token that already has access to the
   pinned Gemma repository.
2. A no-cost hosted CUDA runtime loads the exact frozen configuration.

If the runtime requires quantization or any model/configuration change, stop
and issue a deviation brief. The pilot is engineering evidence only and never
enters confirmatory tables.

## Personal action requested

Open `output/jupyter-notebook/article50_gemma_pilot.ipynb` in Colab, select a
GPU runtime, and personally complete the masked Hugging Face login cell. After
that step Codex can use computer control to run and monitor the remaining
cells, collect the artifact ZIP, and continue automatically.
