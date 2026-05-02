# Project Handoff — for Ammar

Hey Ammar — this is a status doc to bring you fully up to speed on the watermarking project. I've been working on this asynchronously over the last several days. Read this first, then jump in.

## TL;DR

The implementation is done. The report and slides are drafted. We hit a real obstacle running on Colab (Gemma-3 doesn't work as a text-only model in current HuggingFace tooling), and we're pivoting to **LLaMA 3.1 8B as primary + Mistral 7B as secondary**. Gemma stays in the report as a documented compatibility-failure analysis.

Everything you need is in this repo. The active patch and runbook are in `docs/CLAUDE_CODE_PIVOT_LLAMA_MISTRAL.md` and `docs/COLAB_RUNBOOK_LLAMA_MISTRAL.md`. **Tell your Claude Code to read those files first** — they're the most current source of truth.

## What's been done

- Full implementation of the Kirchenbauer green-list logit-bias watermark as a HuggingFace `LogitsProcessor` (`watermark/logits_processor.py`)
- Detector with z-test and threshold calibration (`watermark/detector.py`)
- Corpus generation pipeline (`pipeline/generate.py`) with resume logic
- Robustness module with word substitution, token insertion/deletion, and same-model paraphrase (`evaluation/robustness.py`)
- All 7 experiment scripts in `scripts/` — already bug-fixed (commit history)
- LLaMA 3.1 8B corpus already generated: `results/corpus_llama_d2.jsonl` (~400 lines, watermarked + control)
- 4-page CVPR-style report draft in `docs/main.tex` and `docs/report_draft.md` — Method/Intro/Related Work fully written, results sections scaffolded with `\result{X}` placeholders ready to fill in
- 5-slide deck (`docs/watermarking_3min.pptx`) and 3-min speaker script (`docs/SPEAKER_SCRIPT.md`)

## What's NOT done (where you can help)

### Currently active (Ali might be doing this now)
- Apply the Mistral pivot patch (`docs/CLAUDE_CODE_PIVOT_LLAMA_MISTRAL.md`) — repoints eval scripts at LLaMA, adds Mistral as second model, adds Mistral eval script
- Run the full Colab pipeline (`docs/COLAB_RUNBOOK_LLAMA_MISTRAL.md`) — ~5 hours of mostly unattended GPU time

### Things you can grab anytime

1. **Hand-draw Figure 1** (the method schematic). The report has a placeholder `\fbox{}` in §3.3. We need a two-panel diagram: generation flow (prompt → LLM → SHA-256 → green/red split → +δ → softmax) and detection flow (text → re-derive partition → count greens → z-test). Best done in TikZ inside `main.tex`. Ali can give you TikZ source if you ask.

2. **Fill in the report's `\result{X}` and `\todo{}` placeholders** once the Colab runs finish. Search `main.tex` for those markers; each is keyed to a specific JSON file in `results/`. Mechanical work, ~30 minutes.

3. **Write the Abstract and Conclusion** (currently `\todo{}` blocks at top and bottom of `main.tex`). They depend on having headline numbers, so do this after the runs finish.

4. **Add the Gemma compatibility note** as a new §3.5 in the report. Draft text is in `docs/CLAUDE_CODE_PIVOT_TO_LLAMA.md` toward the bottom under "Report changes." Just a paragraph, but it's the "high-quality analysis of failures" credit on the rubric.

5. **Re-run `demo.ipynb` end-to-end** with real outputs after the Colab runs finish. Currently every cell has `execution_count: None`. The notebook needs to be exported as a 4-page PDF and appended to the report PDF for Gradescope submission.

6. **Update slide values** in `docs/slides_build.js` once results land. Search for `placeholders` — there are two clearly marked blocks (one per chart). Then run `node build.js` to regenerate the .pptx.

7. **Update the speaker script numbers** in `docs/SPEAKER_SCRIPT.md`. There's a swap-table at the bottom showing which JSON keys map to which `[X percent]` markers in the script.

## Project context (in case you want a refresher)

The project re-implements Kirchenbauer et al. (2023) "A Watermark for Large Language Models" and evaluates it on modern instruction-tuned models. At each decoding step, a SHA-256 hash of the prior token deterministically partitions the vocabulary into a "green list" and "red list," and we add a small bias δ (=2.0) to green-list logits before sampling. Detection counts green-list hits and runs a one-sided z-test. We measure detectability (TPR @ 1% FPR), robustness (word substitution, insertion/deletion, paraphrase), and quality (perplexity vs unwatermarked, swept across δ).

## Why we're not using Gemma anymore

Gemma-3 4B loads as `Gemma3ForConditionalGeneration` — a multimodal vision-language class — under current HuggingFace transformers. Text-only generation produces NaN logits, crashing `torch.multinomial` with a CUDA assert. This is an environment compatibility issue, not a watermark bug (the same code works fine on LLaMA). Fixing it cleanly would require either a transformers upgrade with uncertain results, or constructing dummy image inputs. Within our timeline, the right move is to make LLaMA primary, add Mistral 7B (ungated, distinct architecture) as the second model for cross-family generalization, and document Gemma as a compatibility failure in §3.5 of the report.

The proposal commits to two model families. LLaMA + Mistral satisfies that — the Kirchenbauer scheme generalizing across LLaMA and Mistral is a stronger result than originally planned, since the original paper only tested OPT and Llama-1.

## Critical files to read before doing anything

1. `docs/CLAUDE_CODE_PIVOT_LLAMA_MISTRAL.md` — the active patch
2. `docs/COLAB_RUNBOOK_LLAMA_MISTRAL.md` — the active runbook
3. `docs/main.tex` — the report draft (search for `\todo{` and `\result{` to see what's still needed)
4. `proj_proposal.pdf` (root) — the original commitments we made
5. `project-guidelines.pdf` (root) — the grading rubric

## Coordination

- Use the issues tab or just message Ali on Slack/text to coordinate
- The repo is the source of truth — push small, push often
- **Don't run experiments on Colab unless you've coordinated with Ali first** — we're sharing GPU credits and there's no point doing duplicate runs

Welcome to the project. Most of the heavy lifting is done; what's left is execution.
