# Closest-work and contribution review

**Review date:** 2026-11-08

**Status:** positioning amendment only; the frozen July protocol, model revisions,
seeds, calibration split, hypotheses, and primary metrics are unchanged.

## Decision

The July framing that this could be the first independent SynthID robustness
evaluation is no longer supportable. By November 2026, the three candidate
works below substantially overlap the planned task, edit, length, and
controlled-false-positive evaluation. The project remains useful if it narrows
its contribution to a transparent, matched empirical audit of **threshold
transfer**:

> When a detector threshold is selected once on clean calibration controls,
> how do held-out false-positive and true-positive behavior transfer across
> prompt source/task, output length, and edit-induced distribution shift,
> compared with thresholds recalibrated for each condition?

This is not a claim that calibration shift is newly discovered. The useful
artifact would be the paired, prompt-matched implementation, immutable lineage,
and explicit comparison between an operationally frozen threshold and
condition-specific recalibration across two watermark schemes.

## Primary-methods comparison

| Work | Verified scope, venue, date | Direct overlap | Remaining distinction |
| --- | --- | --- | --- |
| [EntroBench](https://aclanthology.org/2026.findings-acl.2089/) | Findings of ACL 2026, July 2026. Seven tasks, three entropy levels, eight watermark methods including SynthID-Text; practical operations include summarization, bullet conversion, and code extraction. Primary reporting uses TPR at 1% FPR. | Multi-task evaluation, practical transformations, quality, controlled FPR, and SynthID are already covered. Its operation-specific results recalibrate on unwatermarked text produced by the same operation. | This project can contrast that condition-specific approach with one clean threshold frozen before edits, on matched prompts and two fixed schemes. |
| [The Provenance Horizon](https://aclanthology.org/2026.emnlp-main.1813/) | EMNLP 2026 main proceedings, October 2026. Theory and experiments on short regulated text under edit channels; includes green-list and SynthID-Text, nominal 1% thresholds, multiple lengths, and light/LLM paraphrase conditions. | Length, editing, realized FPR drift, SynthID, and policy-relevant short-text failure are already covered. | The planned study has broader prompt/task matching and can expose the operational difference between retaining one clean threshold and recalibrating by condition. It must not claim the general phenomenon is new. |
| [Watermarking: from Impossibility to Auditable Compliance](https://arxiv.org/abs/2610.06317) | arXiv preprint submitted 2026-10-05; no peer-reviewed venue verified. Proposes description-length robustness profiles and conformal calibration with separate false-attribution, false-exclusion, and inconclusive outcomes. | It explicitly treats calibration as cell-specific across model, language, domain, length, and edit family, and warns against transferring guarantees across mismatched cells. | A concrete open, matched replay could operationalize the warning, but cannot claim conceptual priority or validated compliance. |

## Dated secondary-analysis amendment

If confirmatory outputs are eventually generated, add the following **secondary,
non-preregistered analysis** without modifying or replacing any July primary
hypothesis or table:

1. Apply the single clean calibration threshold to every held-out task, length,
   and edit condition.
2. Separately estimate a condition-specific threshold using only that
   condition's unwatermarked calibration controls when the frozen generation
   design actually supplies them.
3. Report realized held-out FPR, TPR, and prompt-paired differences for both
   policies, with prompt-clustered intervals and the available sample count.
4. Label missing calibration cells `not_run`; do not infer them from
   watermarked examples or substitute a theoretical normal threshold.
5. Describe the analysis as exploratory/secondary and cite the three works
   above. A material deviation in generation, attacks, or calibration data
   requires a new dated protocol decision before execution.

The historical coursework cache cannot answer this question: it does not
record a calibration/held-out split, task-stratified score membership, or
edited controls. Its new browser artifact is therefore labeled a historical
replay only.

## Claims retired

- “First independent robustness evaluation of SynthID-Text.”
- “Almost no independent third-party robustness evaluation exists.”
- “Near-industry default” or “adopted by OpenAI” without dated primary evidence.
- Any suggestion that the historical caches establish calibrated held-out FPR,
  current SynthID performance, or legal compliance.
