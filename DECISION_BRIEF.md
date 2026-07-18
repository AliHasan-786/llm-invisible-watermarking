# Decision brief: P1 implementation checkpoint — compute gate

**Date:** 2026-07-18

**Branch:** `codex/watermark-p0`

**Decision requested:** authorize only the already-approved personal credential
entry in a no-cost Colab/Kaggle CUDA runtime when Ali is ready. No paid compute,
merge, publication, or deployment authorization is requested.

## Outcome

The approved calibration amendment is frozen and the P1 execution path is
implemented. No new model output or SynthID result exists. The local pilot
preflight stops for two independent expected reasons:

1. no Hugging Face session is authenticated for the gated pinned Gemma model;
2. the local Apple Silicon machine has no CUDA GPU.

The runner made no login attempt, used no credential, and incurred no compute
cost. The next safe action requiring Ali is to enter an authorized Hugging Face
token in a no-cost hosted GPU session and run the commands in
`PILOT_RUNBOOK.md`.

## Implemented protocol controls

| Frozen requirement | Implementation evidence |
| --- | --- |
| deterministic 50/50 source-stratified split | `pipeline/article50.py::assign_splits` |
| distinct derived seeds | SHA-256 of prompt ID, global seed 42, and replicate index |
| three calibration controls per prompt | `build_generation_plan`; tested counts and uniqueness |
| matched held-out conditions | one seed reused for control, Kirchenbauer, and SynthID |
| exact pinned models/revisions | `MODEL_REVISIONS` and immutable run manifest |
| exact SynthID generation config | Transformers 4.57.6 adapter with nine fixed keys and n-gram length 5 |
| official primary detector | weighted mean with 10→1 depth weights, repeated-context mask, and EOS mask |
| strict threshold rule | largest-power observed threshold with `score > threshold`; ties missed |
| amended calibration uncertainty | 10,000 prompt-cluster resamples retaining each triplet |
| held-out reporting | clean TPR/FPR, separate bootstrap intervals, thresholds, counts, and realized calibration FPR |
| failure/resume rules | stable generation keys; append-only completion and failure ledgers |
| pilot quarantine | separate `results/article50/pilot/` tree and explicit result scope |

## Verification

| Check | Result |
| --- | --- |
| protocol and detector tests | 14 passed |
| Python compilation | passed for new runner, scorer, protocol, metrics, and SynthID modules |
| local Gemma preflight | blocked as designed: no authenticated gated-model access; no CUDA |
| full local check | `make PYTHON=python check` passed: 14 tests, 17/17 cache hashes, and all preserved headline values reproduced |
| new SynthID results | none |

## Risks retained

1. The pilot has not tested model-loading memory, runtime, or the full
   Transformers generation path on CUDA.
2. A no-cost runtime may not have enough memory for Gemma 2 9B at the frozen
   loading precision. Quantization or model substitution is not silently
   allowed; either would trigger the deviation protocol.
3. The pilot is an engineering artifact, not confirmatory evidence.
4. The 1% tail remains statistically coarse even after tripling controls; the
   realized-FPR interval must remain adjacent to every headline TPR.
5. H5 remains fixed at 1.15 and is reported as failed if the observed ratio
   exceeds it.

## Gate boundary

Proceed only with a no-cost pilot using the exact pinned revision and current
frozen configuration. Stop and write a deviation brief if the hosted runtime
cannot load it unchanged. Paid compute, credential rotation, merging,
publication, and deployment remain separate gates.
