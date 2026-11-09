# Decision brief: historical replay release candidate

**Date:** 2026-11-09

**Assignment:** `watermark-replay-release-hardening-1`

**Branch:** `codex/watermark-p0`

**Gate:** coordinator release review; pilot and scientific execution remain stopped

## Recommended call

Approve the historical replay as a standalone release candidate for coordinator
design/framing review. Do not promote it as a completed SynthID or confirmatory
Article 50 evaluation.

## Public experience

- The first screen is now **Text Provenance Lab**, with completed historical
  evidence separated from the frozen, unexecuted comparison protocol.
- Visitors can inspect clean score distributions, eight length cells per model,
  and measured edit cells using a fixed -4 to 10 z-score axis and fixed 0–100%
  rate scale. No values are silently clipped or rescaled between models.
- Every view states denominators and has an accessible numeric table.
- On narrow screens, each wide table is a labeled, keyboard-focusable region
  with a visible focus ring and arrow-key horizontal scrolling. The page itself
  does not overflow at 320 CSS pixels.
- Clean-score controls use diamond marks and watermarked scores use circles, so
  the distinction does not depend on color. Rate ticks retain their exact
  geometry at 0% and other edge values while only labels are inset.
- `not_run` means the experiment did not run. `unavailable` means a measured
  score survives but its transformed text does not. The two states are never
  conflated.
- The word-level browser mechanics toy is visually and verbally separated from
  the real cached-evidence replay.
- Source, methods, provenance, and portfolio-return links are visible.

## Integrity and failure behavior

- Contract version 2 declares fixed display scales and build verification.
- The deterministic exporter verifies all 17 manifest hashes and checks
  summaries against score arrays before writing atomically.
- The browser validates the complete contract; it no longer treats a copied
  artifact count as verification and does not claim to re-hash source files.
- Missing/corrupt exporter inputs and stale output fail closed.
- Browser contract tests import the production validator. They require positive
  integer counts and the exact unique model/length/edit cells, recompute clean
  and edit rates and summaries from score arrays using strict
  `score > threshold`, and reject contradictions, malformed arrays, fabricated
  not-run values, out-of-axis scores, malformed JSON, HTTP errors, and network
  failures. The UI shows no fallback headline metrics.

## Integration package

- Stable portfolio ID: `watermarking`.
- Display title: `Text Provenance Lab`; frozen protocol/report titles remain
  unchanged internally.
- `portfolio.json` distinguishes release state from coursework evidence stage
  and traces each thumbnail/headline value to manifest-backed artifacts.
- `assets/text-provenance-lab-thumbnail.svg` is representative of actual
  historical values and explicitly marks the paraphrase value scores-only.
- The self-authored asset is licensed CC BY 4.0 in
  `assets/ASSET_LICENSES.md`.
- `scripts/build_static_site.py` emits a deterministic 12-file package: 11
  allowlisted public artifacts plus a build manifest. It verifies all 17 source
  hashes and the replay's manifest binding before copying; `make check` verifies
  byte-for-byte replay freshness.
- Raw corpus JSONL, prompts/completions, notebooks, result caches, credentials,
  Python source, and backend/function code are absent from the public package.
- `vercel.json` publishes only the generated `dist/` output. No deployment was
  performed.

## Verification evidence

- `make PYTHON=python check`: 35 tests passed; all 17 cached hashes and the
  deterministic replay were verified.
- Production browser validator: 10 Node tests passed, including contradictory
  rate/summary cases and exact 0%, 1%, and 100% geometry conversion.
- Exact-package browser QA at 320 × 844: no page overflow; table keyboard focus,
  visible outline, and arrow-key scroll confirmed; zero-rate tick remained at
  the origin; missing replay returned the unavailable state with no fallback;
  normal reload produced zero console errors or warnings.
- Exact static inventory and local-link tests passed.

## Review requirements by concern

### Scientific execution — still binding

1. Do not run the pilot or confirmatory suite without the exact frozen model
   revisions, personally supplied credentials, and an appropriate no-cost GPU.
2. Model substitution, quantization, paid compute, or protocol/calibration
   deviation requires a new scientific decision.
3. Fresh H1 paraphrase reproduction within ±5 percentage points remains
   required before a new result claim.
4. The November contribution amendment and compliance framing require
   independent scientific/legal review before claims from the planned study
   are published.

### Release framing — coordinator-owned

1. Review first-screen wording, scale legibility, keyboard/mobile behavior,
   thumbnail, metadata, and source/return links.
2. Confirm that the portfolio snapshot promotes only the coursework replay and
   does not imply SynthID results.
3. Merge/deployment are routine coordinator actions after this review; no
   additional user-permission hold is asserted here.

No reviewer automation was invoked. No credentials, model/provider access,
inference, paid compute, merge, or deployment occurred in this assignment.
