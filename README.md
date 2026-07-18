# Watermark Under Article 50

An evaluation harness for measuring how reliably text watermarks remain detectable after realistic editing and laundering attacks. The study compares a green-list logit-bias watermark, the open-source SynthID-Text implementation, and an unwatermarked control at a fixed false-positive operating point.

The policy question is deliberately narrow: when Article 50 asks providers to make synthetic text machine-readable and detectable, what detection rate remains at 1% false positives after common transformations? This repository produces measurements for that conversation; it does not offer legal advice or claim that a single benchmark establishes compliance.

## Status

Phase P0 is at the preregistration gate. The original coursework baseline has been recovered and made reproducible from local caches. The SynthID comparison has **not** been run, and no result is claimed for it yet.

Article 50's relevant transparency obligations become applicable on **August 2, 2026**. Article 50(2) calls for machine-readable marking and for technical solutions that are effective, interoperable, robust, and reliable as far as technically feasible. See the [official regulation](https://eur-lex.europa.eu/eli/reg/2024/1689/oj?locale=en) and the European Commission's [Code of Practice page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content).

## Reproduce the preserved baseline

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make check
```

`make check` runs detector-math tests, verifies every cached artifact against its SHA-256 digest, validates corpus metadata, and recomputes headline detection and robustness statistics from cached z-score arrays. It downloads no models.

The CI workflow is staged at `docs/ci/quality.yml` because the repository token may not have GitHub Actions workflow scope. Activate it from an appropriately scoped session with:

```bash
mkdir -p .github/workflows
cp docs/ci/quality.yml .github/workflows/quality.yml
git add .github/workflows/quality.yml
```

Preserved results from the original Kirchenbauer-style implementation:

| Generation model | Watermarked / control | TPR at 1% FPR | Perplexity ratio |
| --- | ---: | ---: | ---: |
| `google/gemma-2-9b-it` | 201 / 201 | 90.0% | 1.095 |
| `meta-llama/Llama-3.1-8B-Instruct` | 99 / 98 | 98.0% | 1.014 |

These values trace to `results/baseline/detection_*_summary.json` and the corresponding `.npz` arrays. The original report's 27.3% TPR after an LLM paraphrase is retained as a report claim but has no surviving raw paraphrase cache; it must be reproduced within ±5 percentage points before any new public claim ships.

## Study design

The fixed protocol is in [`PREREGISTRATION.md`](PREREGISTRATION.md). In brief:

- schemes: Kirchenbauer baseline, open-source SynthID-Text, no-watermark control;
- target models: Gemma 2 9B Instruct first, then Llama 3.1 8B Instruct;
- attacks: two paraphrasers at one and two passes, French/German back-translation, fixed-budget substitutions and insertions, truncation/partial text, and regenerate-from-summary laundering;
- primary metric: TPR at a threshold calibrated to at most 1% empirical FPR on a held-out control split;
- quality: perplexity ratio plus a blinded readability spot check;
- reporting: full ROC/DET curves, length curves, confidence intervals, negative results, and protocol deviations.

## Repository map

```text
watermark/                  Kirchenbauer injector and detector
pipeline/                   Prompt loading and generation
evaluation/                 Metrics and attack helpers
scripts/reproduce_baseline.py
                            Offline cache verifier and table reproduction
results/baseline/           Preserved, hashed coursework artifacts
tests/                      Detector-math tests
docs/ci/quality.yml         Staged GitHub Actions workflow
PREREGISTRATION.md          Frozen protocol awaiting approval
DATA_PROVENANCE.md          Dataset and cache lineage
EXPLAINER.md                Architecture and interview preparation for Ali
DECISION_BRIEF.md           Current gate evidence and recommendation
```

## Methods honesty and limitations

The baseline is an authored evaluation, not an external benchmark. The expanded study is limited to two open-weight models, English source prompts, an open-source SynthID implementation rather than Google's production deployment, and attacks designed and run by the study author. Threshold estimates at 1% FPR are coarse with hundreds rather than tens of thousands of controls; confidence intervals and realized FPR are therefore reported alongside TPR.

## Skills demonstrated

This project demonstrates experimental design, statistical calibration, reproducible ML pipelines, adversarial evaluation, model integration, data provenance, test engineering, and policy-facing technical communication. The planned interactive explains detector behavior to non-technical visitors without turning the project into dashboard chrome.

## Original course project

Ali Hasan and Ammar Syed built the original implementation for Cornell Tech CS 5788. The current extension preserves that baseline and adds the preregistered independent evaluation layer. Source datasets and model weights remain governed by their upstream terms.
