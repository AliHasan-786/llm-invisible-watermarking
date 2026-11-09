# Historical evidence replay contract

`site/data/historical_replay.json` is a deterministic browser-facing export of
the 17 artifacts listed in `results/manifest.json`. It is a replay, not a live
detector and not a new experiment.

Contract version 2 declares one fixed score axis for both models and every
clean/edit view, plus one fixed 0–100% rate axis. The exporter derives whole-z
bounds that contain every preserved score and refuses to write if any value
falls outside them. The interface prints the bounds and provides numeric tables
with denominators as an alternative to each visual.

## Guarantees

- The exporter verifies all manifest SHA-256 hashes before constructing output.
- Detection and robustness summaries are checked against preserved score arrays.
- No prompt, completion, rewritten passage, or token sequence is duplicated.
- No transformed attack text survives in the cache. Deterministic edit attacks
  can be reconstructed from the source corpus; the LLM paraphrase cannot.
- Measured cells use `status: measured_historical`.
- Missing cells use `status: not_run` and numeric evidence is `null`, never zero.
- A transformed-text artifact for a measured score cell is `unavailable`; an
  experiment that never ran is `not_run`. The interface presents these states
  separately. The unmodified baseline uses `not_applicable` because it has no
  transformation output.
- SynthID and the preregistered Article 50 experiment remain `not_run`.
- A missing, corrupt, inconsistent, or stale source causes a nonzero exit and no
  partial output.
- `buildVerification.status` records that the exporter performed hash and
  summary/array verification. The browser validates the complete versioned
  contract; it does not claim to re-hash the source artifacts.

Historical clean thresholds were estimated from the preserved coursework
controls without a recorded calibration/held-out split. Length cells used a
separate threshold at each length. These values therefore cannot be presented
as confirmatory calibrated-FPR evidence.

## Commands

```bash
python scripts/export_historical_replay.py
python scripts/export_historical_replay.py --check
```

The second command verifies that the tracked browser export is byte-for-byte
current. `make check` includes that verification, malformed-contract tests,
out-of-axis refusal, null-state enforcement, and network/HTTP/JSON loading
failure tests.
