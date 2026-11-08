# Decision brief: November readiness and contribution gate

**Date:** 2026-11-08

**Branch:** `codex/watermark-p0`

**Gate:** coordinator review; no merge, deployment, credentials, or compute

## Outcome

The existing repository remains the correct home for the work. It already owns
the original implementation, frozen preregistration, 17 immutable baseline
artifacts, full non-compute harness, report, and draft site. Creating a second
repository would split lineage and weaken the preregistration/result history.

The branch now adds a dated current-literature review and a deterministic local
historical-evidence replay. It does not add a current experimental result.

## Contribution decision

EntroBench (Findings of ACL 2026) and *The Provenance Horizon* (EMNLP 2026)
already cover substantial parts of task/edit/length/controlled-FPR SynthID
evaluation. The October 2026 arXiv preprint *Watermarking: from Impossibility to
Auditable Compliance* also foregrounds calibration-cell mismatch.

Retire the old first-evaluation and unsupported industry-adoption framing. The
remaining useful question is a matched empirical audit of one clean threshold
under task, length, and edit distribution shift versus condition-specific
recalibration. This is recorded as a secondary November amendment and leaves
all July frozen choices intact.

## Evidence replay delivered

- `scripts/export_historical_replay.py` verifies every manifest hash before
  loading evidence and checks summaries against raw score arrays.
- `site/data/historical_replay.json` contains only preserved numeric evidence:
  clean score distributions, measured length cells, and measured edit cells.
- Gemma's absent LLM-paraphrase cell, all SynthID cells, and all confirmatory
  cells are `not_run` with null values.
- Llama's historical paraphrase score series is present, but rewritten text is
  explicitly missing.
- Historical thresholds are labeled unheld-out/uncalibrated; length thresholds
  are labeled per-length. The site never promotes these to current findings.
- No prompt, completion, token sequence, or legacy rewrite is duplicated.
- Tests cover deterministic/stale output, missing inputs, corrupt inputs,
  manifest provenance, null semantics, static routing, and JavaScript syntax.

## Phase and acceptance status

| Requirement | Status |
| --- | --- |
| P0 hygiene, tests, preregistration, provenance | Complete and frozen |
| Calibration amendment: three controls per calibration prompt | Implemented in protocol/harness; no run executed |
| P1 SynthID integration and full Gemma suite | Harness complete; pilot and results not run |
| P2 Llama/attacks/quality/report/interactive | Non-compute implementation complete; confirmatory results and deployment not run |
| One-command rerun from caches | Historical baseline and replay verified by `make check` |
| Every public number traces to an artifact | Historical replay does; blocked current cells remain null |
| Fresh paraphrase reproduction within ±5 pp | Not run; 27.3% remains historical scores-only evidence |
| Mobile interaction | Implemented; current gate requires local verification only |
| Pinned confirmatory model results | Not run |

## Manual gates remaining

No user action is required for this review gate. Future execution still needs:

1. credentials supplied personally in the gated notebook environment;
2. a no-cost GPU capable of the exact frozen revisions, or a separately
   approved compute deviation;
3. independent review of compliance framing and the November contribution
   amendment before publication;
4. explicit merge and deployment authorization.

Do not substitute or download models, run inference, spend money, merge, or
deploy from this gate.
