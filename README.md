# Watermark Under Article 50

An evaluation harness for measuring how reliably text watermarks remain detectable after realistic editing and laundering attacks. The study compares a green-list logit-bias watermark, the open-source SynthID-Text implementation, and an unwatermarked control at a fixed false-positive operating point.

The policy question is deliberately narrow: when Article 50 asks providers to make synthetic text machine-readable and detectable, what detection rate remains at 1% false positives after common transformations? This repository produces measurements for that conversation; it does not offer legal advice or claim that a single benchmark establishes compliance.

## Status

Phase P0 is complete. P1's execution code is complete: the pinned SynthID adapter, immutable prompt revisions, tripled calibration controls, 33-condition attack matrix, detector scoring, paired comparison, cache-to-report builder, resumable ledgers, and one-click quarantined pilot notebook are implemented and tested. P2's ROC/DET, fixed operating-point, pinned quality, blinded readability, report, and mobile-interactive pipelines are also built, but remain undeployed and contain no SynthID result. The pilot is blocked locally because this machine has no CUDA GPU and is not logged in to the gated Hugging Face model repository. No login, paid compute, merge, or deployment was attempted.

Article 50's relevant transparency obligations become applicable on **August 2, 2026**. Article 50(2) calls for machine-readable marking and for technical solutions that are effective, interoperable, robust, and reliable as far as technically feasible. See the [official regulation](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689), the Commission's [20 July 2026 implementation guidelines](https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems), and its [final Code of Practice announcement](https://digital-strategy.ec.europa.eu/en/news/commission-publishes-code-practice-marking-and-labelling-ai-generated-content).

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

These values trace to `results/detection_*_summary.json` and the corresponding `.npz` arrays. The original report's 27.3% TPR after an LLM paraphrase is exactly reproducible from `results/robustness_llama_zscores.npz`; the paraphrased texts themselves are not cached, so the attack must still be rerun within ±5 percentage points before any new public claim ships.

## Study design

The fixed protocol is in [`PREREGISTRATION.md`](PREREGISTRATION.md). In brief:

- schemes: Kirchenbauer baseline, open-source SynthID-Text, no-watermark control;
- target models: Gemma 2 9B Instruct first, then Llama 3.1 8B Instruct;
- attacks: two paraphrasers at one and two passes, French/German back-translation, fixed-budget substitutions and insertions, truncation/partial text, and regenerate-from-summary laundering;
- primary metric: TPR at a threshold calibrated to at most 1% empirical FPR on a held-out control split;
- quality: perplexity ratio plus a blinded readability spot check;
- reporting: full ROC/DET curves, length curves, confidence intervals, negative results, and protocol deviations.

## Run the frozen P1 pipeline

The runner fails closed when gated access or a CUDA runtime is absent:

```bash
python scripts/run_article50.py preflight --model gemma
python scripts/run_article50.py freeze-prompts --model gemma
python scripts/run_article50.py generate --model gemma --pilot-prompts 20
```

See `PILOT_RUNBOOK.md` for the no-cost Colab/Kaggle handoff. Pilot artifacts
remain under `results/article50/pilot/`; the confirmatory path is separate.
After generation, each detector is scored independently:

```bash
python scripts/score_article50.py results/article50/pilot/gemma/completions.jsonl --detector kirchenbauer
python scripts/score_article50.py results/article50/pilot/gemma/completions.jsonl --detector synthid
python scripts/run_article50_attacks.py results/article50/pilot/gemma/completions.jsonl --model gemma --mode deterministic
```

Each clean headline artifact records the target FPR, threshold, realized
calibration FPR and its prompt-clustered bootstrap interval, held-out TPR/FPR
and intervals, sample counts, and strict tie rule.

[Open the one-click Gemma pilot in Colab](https://colab.research.google.com/github/AliHasan-786/llm-invisible-watermarking/blob/codex/watermark-p0/output/jupyter-notebook/article50_gemma_pilot.ipynb). Ali must complete the masked Hugging Face login cell; the notebook handles the remaining no-cost pilot workflow and bundles the artifacts.

After confirmatory scoring, build the secondary curves, quality table, and
blinded readability queue:

```bash
python scripts/analyze_article50_curves.py SCORES.jsonl --positive-scheme synthid --output curves.json
python scripts/analyze_article50_quality.py COMPLETIONS.jsonl --device cuda
python scripts/prepare_readability_spotcheck.py COMPLETIONS.jsonl
```

The quality scorer pins `openai-community/gpt2` at an immutable revision and
reports the predeclared H5 ceiling whether it passes or fails. The readability
queue balances three sources × two watermark schemes and assigns 20 of 60
pairs to a second independent review context.

## Preview the interactive

```bash
python -m http.server 8000
# open http://localhost:8000/site/
```

The browser lab uses a disclosed word-level mechanics demonstrator, not the
benchmark tokenizer or detector. Its chart contains only preserved coursework
values. `REPORT.md` is similarly result-safe: extension tables stay pending
until the report-data builder finds confirmatory artifacts.

## Repository map

```text
watermark/                  Kirchenbauer injector and detector
pipeline/                   Prompt loading and generation
evaluation/                 Metrics and attack helpers
scripts/run_article50.py    Frozen prompt, plan, manifest, and generation runner
scripts/run_article50_attacks.py
                            Frozen deterministic and model-backed attack matrix
scripts/score_article50.py  Detector scoring and condition headline artifacts
scripts/compare_article50_schemes.py
                            Paired SynthID-minus-Kirchenbauer intervals
scripts/build_article50_report_data.py
                            Confirmatory-cache to report-table builder
scripts/analyze_article50_curves.py
                            ROC/DET and supported operating-point artifacts
scripts/analyze_article50_quality.py
                            Pinned GPT-2 paired perplexity/length analysis
scripts/prepare_readability_spotcheck.py
                            Balanced blinded 60-pair review queue
scripts/reproduce_baseline.py
                            Offline cache verifier and table reproduction
results/                    Preserved coursework artifacts and SHA-256 manifest
tests/                      Detector-math tests
PILOT_RUNBOOK.md            Quarantined no-cost GPU pilot handoff
output/jupyter-notebook/    Colab-ready gated-compute pilot
REPORT.md                   Paper-structured report; extension results pending
site/                       Mobile editorial mechanics lab
docs/ci/quality.yml         Staged GitHub Actions workflow
PREREGISTRATION.md          Approved and frozen protocol
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
