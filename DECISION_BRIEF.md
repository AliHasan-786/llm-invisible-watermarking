# Decision brief: historical length evidence instrument

**Date:** 2026-11-09

**Assignment:** `watermark-presentation-visual-1`

**Base:** accepted closeout on `main` at
`e33773b8fd67818b9a55e21f46a246e8de7c1813`

**Gate:** visual and interaction review only; no deployment, merge, or scientific execution

## Recommended call

Approve this feature branch for presentation review. The candidate turns the
eight preserved length cells for each model into one direct, fixed-scale signal
instrument while retaining the exact numeric table. It does not add a result,
smooth a curve, alter a source cache, or imply that the planned SynthID study
ran.

## What changed

- The Length cells view now compares the 16 saved TPR measurements on a fixed
  0 to 100% scale. Straight segments connect only measured cells.
- Gemma uses a solid line and circular markers. Llama uses a dashed line and
  square markers, so the comparison does not depend on color.
- The existing model and measured-cell controls move a focus rail and update
  the exact TPR, FPR, threshold, watermarked count, and control count.
- Direct endpoint labels identify the two series without a detached legend.
- The figure links to both reviewed JSON sources at commit `e794de3` and says
  that the values are historical measurements, not a live detector run.
- The chart remains a named, keyboard-focusable horizontal scroll region at
  320px. The full numeric table remains the non-visual alternative.
- Visitor copy now addresses the reader directly, and null states say `Not run`
  or `Not set` rather than relying on an em dash.

## Evidence boundary

The instrument reads the existing deterministic browser contract. It does not
change the 17 verified source artifacts, their hashes, the replay manifest, the
frozen protocol, or the exact 12-file public allowlist. The separate word-level
mechanics demo remains labeled illustrative and is not presented as detector
output. Saved score-only evidence, missing transformed text, and unrun study
cells retain their distinct states.

## Design rationale and credits

- Anatomy informed the instrument framing: one physical-feeling focus rail,
  exact readouts, and a strict split between mechanism and evidence.
- UI/UX Pro Max informed direct series labels, redundant line and marker
  encoding, visible keyboard focus, table fallback, narrow-screen scrolling,
  and reduced-motion behavior.
- 3Dviz Pro Max informed the decision not to add unsupported depth or a heavy
  renderer. A flat evidence plot is the truthful representation for these
  two-dimensional saved measurements.
- The graph animation guidance informed the use of real values, plain labels,
  straight segments, and a single 180ms selected-point cue. Motion is removed
  when reduced motion is requested.

Representative captures:

- `output/playwright/watermark-length-instrument-desktop.png`
- `output/playwright/watermark-length-instrument-mobile.png`

## Review evidence

The exact built package was served locally and checked in a real Chromium
browser. The desktop and 320px views were inspected. Model and length selection
updated to the saved values. At 320px the page itself remained 320px wide while
the named chart region scrolled independently from 244px to its 680px content
width. Keyboard focus produced a visible outline, Arrow Right moved the chart
40px, reduced motion set the selected-point animation to `none`, and the browser
console had zero errors or warnings. At 200% root text scaling the document
still matched the 320px viewport and kept the chart overflow inside its named
scroll region.

The full release gate passed before review handoff:

- `make PYTHON=python check`: 37 tests passed, all 17 cached artifacts matched
  their SHA-256 digests, baseline reproduction passed, and replay freshness
  passed.
- `node --test tests/js/replay_contract.test.js`: all 10 browser-contract tests
  passed.
- `python scripts/build_static_site.py --output <candidate-dir>`: 11 allowlisted
  artifacts plus the build manifest were produced.
- `node --check site/app.js`, the exact public inventory check, secret scans,
  and `git diff --check` passed.

## Remaining gates

- Coordinator or reviewer approval is required before merge.
- Deployment and promotion are explicitly out of scope for this assignment.
- Any pilot, confirmatory run, provider/model access, credentials, paid compute,
  or new public scientific claim remains separately gated.
