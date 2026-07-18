# Decision brief: P0 preregistration gate — approved

**Date:** 2026-07-18

**Branch:** `codex/watermark-p0`

**Decision:** approved on 2026-07-18 with one pre-freeze amendment: generate three unwatermarked calibration completions per calibration prompt using distinct derived seeds, and report realized calibration FPR with a prompt-clustered bootstrap confidence interval.

## Authority now granted

Proceed with P1 implementation and the no-cost 20-prompt engineering pilot. The pilot remains quarantined from confirmatory results unless it uses the exact frozen configuration.

Paid compute, credential entry or rotation, merging, publication, and deployment remain separate gates. H5's 1.15 perplexity-ratio ceiling remains unchanged and must be reported as failed if the observed value exceeds it.

## What was done

1. Fetched and inspected the force-updated canonical GitHub history, which already contains the complete course baseline cache, report sources, figures, and current evaluation scripts.
2. Verified an independently saved Gemma copy is byte-identical and added a SHA-256 manifest over the authoritative committed root caches.
3. Committed the preregistration after all pre-existing coursework results and before any SynthID result:
   - preregistration commit: `bfe4481`
   - no SynthID result exists.
4. Added a SHA-256 manifest for 17 cached artifacts and an offline one-command verifier.
5. Added six detector-math tests and fixed two issues those tests exposed:
   - detector-only imports no longer require Transformers;
   - finite-sample threshold calibration now respects the requested maximum empirical FPR under the detector's strict comparison rule.
6. Preserved the newer remote cleanup, including its removal of the stale skip note that contradicted shipped Llama results.
7. Added a pinned environment, data/provenance card, README, private explainer, portfolio-entry draft with a publication hold, and a staged CI workflow.
8. Corrected the temporal framing: Article 50's relevant obligations become applicable on August 2, 2026; they have not yet taken effect as of this brief.

## Evidence

| Evidence | Path or command | Result |
| --- | --- | --- |
| Frozen protocol | `PREREGISTRATION.md` | schemes, hypotheses, attacks, budgets, models/revisions, metrics, exclusions, stopping rule, and compute plan fixed |
| Baseline lineage | `DATA_PROVENANCE.md` | remote history, cache source, data fields, real-person-content note, limitations, and future run ledger defined |
| Cache integrity | `results/manifest.json` | 17/17 files match SHA-256 |
| Offline reproduction | `python3 scripts/reproduce_baseline.py` | Gemma 90.0% TPR; Llama 98.0% TPR; Llama paraphrase 27.3%; exact matches to cached arrays |
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
| repo cleanup / empty stubs | complete | current remote `main` cleanup preserved; authoritative artifacts remain under `results/` |
| preregistered attacks, budgets, metrics, models, hypotheses | complete and approved with calibration amendment | `PREREGISTRATION.md` |
| compute plan | complete, awaiting any later money decision | preregistration §11 |
| P0 gate | **closed: approved** | this brief |

P1 is authorized. P2 and later phase gates remain unchanged.

## Acceptance-criteria status

- **One-command rerun from caches:** met for the preserved course baseline through `make reproduce-baseline`; not yet applicable to new SynthID results.
- **Every number traces to a cached run with pinned model IDs:** met for the displayed Gemma/Llama headline values and the 27.3% LLM-paraphrase score. The paraphrased text itself is not cached.
- **Preregistration commit predates result commits:** met for the extension's new results. Coursework results necessarily predate this extension's preregistration; no SynthID result exists.
- **Interactive works on mobile:** not started; P2.
- **Original paraphrase collapse reproduced within ±5 percentage points:** not met; required before new claims ship.
- **Required limitations:** written before new results in `PREREGISTRATION.md`, `README.md`, and `DATA_PROVENANCE.md`.

## Uncertainty and risk

1. **Partial paraphrase cache:** the 27.3% figure is exactly supported by cached z-scores, but the paraphrased texts needed to replay the attack are absent. H1 makes fresh replication mandatory.
2. **Low-FPR resolution:** three calibration completions per calibration prompt triple the tail sample without changing the prompt population. The protocol reports realized FPR with a prompt-clustered bootstrap interval and avoids stronger low-FPR claims.
3. **Gated target models:** Gemma and Llama require manually approved Hugging Face access. Codex will not request, enter, or rotate credentials without Ali.
4. **Implementation scope:** the study evaluates open-source SynthID-Text, not Google's production Gemini deployment.
5. **Author-run attacks:** the attack suite may miss adaptive strategies; this remains a headline limitation.
6. **CI is staged, not active:** activation requires a workflow-scoped GitHub credential and should happen only in an authorized session.

## Next gate

If the no-cost pilot cannot run without new credential entry or paid compute, stop with the measured blocker and request only that specific authorization.
