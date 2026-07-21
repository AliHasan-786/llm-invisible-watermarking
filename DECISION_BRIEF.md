# Decision brief: P1 harness + pre-result P2 artifacts

**Date:** 2026-07-20

**Branch:** `codex/watermark-p0`

**Latest completed phase commit:** `f6dfad5` — frozen P1 attack and analysis
harness.

## Outcome

P1's non-compute implementation is complete and pushed. P2's curve, quality,
readability, report, mobile interactive, portfolio draft, and deployment
configuration are also built and locally verified without inventing an
extension result. The project
is not complete: no pinned-model pilot, confirmatory SynthID run, fresh H1
paraphrase replication, quality analysis, merge, or deployment has occurred.

## P2 artifacts

- `REPORT.md`: paper-structured methods report with verified baseline values,
  current Commission sources, interpretation rules, limitations, and a blocked
  extension table.
- `site/`: warm-paper editorial interactive with the preserved baseline chart,
  live paste/edit score, four transformation controls, before/after panels,
  and study-design walkthrough.
- `vercel.json`: static routing prepared but not deployed.
- `PORTFOLIO_ENTRY_DRAFT.md`: updated draft with an explicit publication hold.
- `scripts/analyze_article50_curves.py`: ROC/DET and supported secondary
  operating points.
- `scripts/analyze_article50_quality.py`: pinned GPT-2 paired perplexity,
  length, and fixed H5 decision.
- `scripts/prepare_readability_spotcheck.py`: balanced 60-pair blinded queue
  with 20 second-context overlaps.

The browser lab labels its word-level score as a mechanics demonstrator. It
does not claim to run the benchmark tokenizer/detector, execute an LLM or real
translation, report SynthID performance, or determine legal compliance.

## Verification

| Check | Result |
| --- | --- |
| Full test suite | 27 passed |
| Baseline reproduction | 17/17 hashes; 90.0%, 98.0%, and historical 27.3% reproduced |
| Report-data builder | rejects pilot/non-confirmatory artifacts; empty extension table is not committed as a result stub |
| JavaScript | `node --check` passed |
| Desktop browser | live score and all transformation controls work |
| Mobile browser | 390×844, zero horizontal overflow, four usable attack controls |
| Browser console | zero current errors or warnings |
| Translation proxy smoke test | score changed 9.75→0.72 and before/after copy updated |
| Public claims | no SynthID result; all displayed numbers trace to preserved cache artifacts |

## Current manual gates

1. **Credentials:** Colab is open at Google's sign-in screen. Ali must sign in,
   then personally enter a masked read-only Hugging Face token in the notebook.
2. **Compute deviation:** if a no-cost GPU cannot load the exact frozen Gemma
   configuration, stop. Quantization, substitution, or paid compute requires a
   new decision.
3. **Merge:** draft PR #3 remains unmerged.
4. **Publication/deployment:** the site and portfolio copy remain local/draft.
   Deploying or publishing under Ali's name requires his explicit go.

## Recommended next sequence

1. Complete the two sign-ins and run the 20-prompt pilot.
2. If unchanged configuration succeeds, run the full Gemma suite and populate
   its headline table from caches.
3. Run the Llama suite, fresh H1 replication, attacks, and quality analysis.
4. Re-run report/site checks with real confirmatory tables.
5. Submit the compliance-framing and publication gate for independent review,
   then ask Ali for merge/deploy authorization.
