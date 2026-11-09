# No-cost Gemma pilot runbook

**Status:** prepared, but locally blocked by the approved credential and compute
gates. Pilot artifacts are quarantined under `results/article50/pilot/` and
must not be copied into the confirmatory result tree.

## Preconditions

- Use a no-cost Colab or Kaggle CUDA runtime.
- Clone this repository and check out the draft PR branch.
- Confirm the checked-out preregistration freeze commit and leave the worktree
  clean.
- Ali enters a Hugging Face token with access to the exact pinned Gemma
  revision. Do not place the token in a notebook cell, command history, repo
  file, or run artifact.

## Commands

Preferred: open
[`output/jupyter-notebook/article50_gemma_pilot.ipynb`](output/jupyter-notebook/article50_gemma_pilot.ipynb)
in Colab. It verifies the preregistration freeze, pauses for Ali's masked
Hugging Face login, executes the commands below, checks plan completeness, and
creates a ZIP for download.

Manual equivalent:

```bash
python -m pip install -r requirements.txt
python scripts/run_article50.py preflight --model gemma
python scripts/run_article50.py freeze-prompts --model gemma
python scripts/run_article50.py generate --model gemma --pilot-prompts 20
python scripts/score_article50.py results/article50/pilot/gemma/completions.jsonl --detector kirchenbauer
python scripts/score_article50.py results/article50/pilot/gemma/completions.jsonl --detector synthid
python scripts/run_article50_attacks.py results/article50/pilot/gemma/completions.jsonl --model gemma --mode deterministic
```

The pilot plans 20 prompts using deterministic, source-balanced selection.
Calibration prompts receive three distinct control seeds; held-out prompts
receive matched control, Kirchenbauer, and SynthID generations.

Copy the complete `results/article50/pilot/gemma/` directory back as one unit.
The required evidence is:

- `run_manifest.json`;
- `generation_plan.jsonl` and its summary;
- `completions.jsonl`;
- `failures.jsonl`, including an empty file when there are no failures.
- clean detector scores/headline diagnostics and deterministic attack caches.

## Stop conditions

Stop without substitution if the pinned model revision cannot be loaded, the
runtime has no CUDA GPU, a dependency differs from `requirements.txt`, or the
20-prompt plan does not finish with an output or recorded failure for every
planned generation. Paid compute, changed credentials, model substitution,
merging, deployment, and publication remain separate gates.
