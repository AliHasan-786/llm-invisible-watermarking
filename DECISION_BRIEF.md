# Decision brief: P0 preregistration gate

**Date:** 2026-07-18

**Branch:** `codex/watermark-p0`

**Decision requested:** approve or reject the frozen P1/P2 evaluation protocol in `PREREGISTRATION.md`. Approval authorizes implementation and a no-cost 20-prompt engineering pilot; it does **not** authorize paid compute, credential entry/rotation, publication, or claims about vendors.

## Recommendation

**Approve the protocol and no-cost pilot.** It directly covers the spec's fixed attack suite, avoids changing methods after seeing SynthID results, uses held-out controls, pins model/software revisions, and keeps Article 50 framing as measurement rather than legal opinion.

Do not authorize paid compute yet. Use the pilot to produce a runtime/cost estimate, then return with a separate money decision if free Colab/Kaggle capacity is insufficient.

Because this preregistration names vendors/models and will govern public-facing results, send this brief plus `PREREGISTRATION.md` to Claude Opus 4.8 for independent review before the yes/no call.

## What was done

1. Recovered the complete course baseline cache from the canonical project materials and verified an independently saved Gemma copy was byte-identical.
2. Committed the preregistration scaffold before importing any baseline result artifact:
   - preregistration commit: `faa3038227b2868cdee678ea742d063a153c18af`
   - baseline cache commit: `66fb4ce433eb87a7e9e4a31acfc58c23fc32f2aa`
   - no SynthID result exists.
3. Added a SHA-256 manifest for 17 cached artifacts and an offline one-command verifier.
4. Added six detector-math tests and fixed two issues those tests exposed:
   - detector-only imports no longer require Transformers;
   - finite-sample threshold calibration now respects the requested maximum empirical FPR under the detector's strict comparison rule.
5. Deleted the ignored zero-byte Gemma corpus, empty length-curve stub, stale skip note, and stale ignored figures from the clone.
6. Added a pinned environment, data/provenance card, README, private explainer, portfolio-entry draft with a publication hold, and a staged CI workflow.
7. Corrected the temporal framing: Article 50's relevant obligations become applicable on August 2, 2026; they have not yet taken effect as of this brief.

## Evidence

| Evidence | Path or command | Result |
| --- | --- | --- |
| Frozen protocol | `PREREGISTRATION.md` | schemes, hypotheses, attacks, budgets, models/revisions, metrics, exclusions, stopping rule, and compute plan fixed |
| Baseline lineage | `DATA_PROVENANCE.md` | cache source, data fields, real-person-content note, limitations, and future run ledger defined |
| Cache integrity | `results/baseline/manifest.json` | 17/17 files match SHA-256 |
| Offline reproduction | `python3 scripts/reproduce_baseline.py` | Gemma 90.0% TPR; Llama 98.0% TPR; exact match to cached summaries |
| Detector tests | `python3 -m unittest discover -s tests -v` | 6/6 passed |
| Full local check | `make check` | passed |
| Dependency resolution | disposable venv + `pip install --dry-run -r requirements.txt` | resolved with no version conflict |
| Secrets scan | credential-pattern scan over tracked/untracked text files | no matches |
| CI | `docs/ci/quality.yml` | staged; not activated because workflow-scope credentials are a separate gate |
| Ali study guide | `EXPLAINER.md` | architecture, decisions, current limits, and 10 interview questions |

## P0 status against the spec

| P0 item | Status | Evidence |
| --- | --- | --- |
| README | complete | `README.md` |
| detector-math tests | complete | `tests/test_detector.py` |
| repo cleanup / empty stubs | complete | clean result layout under `results/baseline/` |
| preregistered attacks, budgets, metrics, models, hypotheses | complete, awaiting approval | `PREREGISTRATION.md` |
| compute plan | complete, awaiting any later money decision | preregistration §11 |
| P0 gate | **open** | this brief |

P1 and later phases have not started.

## Acceptance-criteria status

- **One-command rerun from caches:** met for the preserved course baseline through `make reproduce-baseline`; not yet applicable to new SynthID results.
- **Every number traces to a cached run with pinned model IDs:** met for the displayed Gemma/Llama headline values. The original report's 27.3% LLM-paraphrase TPR lacks a surviving raw cache and is explicitly quarantined as report-only.
- **Preregistration commit predates result commits:** met for this extension. The first preregistration commit precedes the baseline-cache recovery commit, and no SynthID result exists.
- **Interactive works on mobile:** not started; P2.
- **Original paraphrase collapse reproduced within ±5 percentage points:** not met; required before new claims ship.
- **Required limitations:** written before new results in `PREREGISTRATION.md`, `README.md`, and `DATA_PROVENANCE.md`.

## Uncertainty and risk

1. **Missing paraphrase cache:** the 27.3% figure is supported by the final PDF but not a surviving raw output. H1 makes replication mandatory.
2. **Low-FPR resolution:** roughly hundreds of controls make 1% FPR estimates coarse. The protocol reports realized FPR and bootstrap intervals and avoids stronger low-FPR claims.
3. **Gated target models:** Gemma and Llama require manually approved Hugging Face access. Codex will not request, enter, or rotate credentials without Ali.
4. **Implementation scope:** the study evaluates open-source SynthID-Text, not Google's production Gemini deployment.
5. **Author-run attacks:** the attack suite may miss adaptive strategies; this remains a headline limitation.
6. **CI is staged, not active:** activation requires a workflow-scoped GitHub credential and should happen only in an authorized session.

## Call options

- **YES:** approve `PREREGISTRATION.md` and authorize P1 implementation plus a no-cost 20-prompt pilot. Paid compute and credentials remain separate gates.
- **NO:** identify the protocol section to revise. No SynthID run will start.
