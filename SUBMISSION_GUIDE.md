# Submission guide: what's done, what's left, how to ship

Today is **2026-05-04**. Project is due **2026-05-12**. Both Gemma 2 9B and LLaMA 3.1 8B evaluations are complete and integrated; the only remaining work is **assembly** (Overleaf compile → combine PDFs → zip the repo) and the **in-class talk**.

---

## 1. What's already done in the repo

| Deliverable | Location | Status |
|---|---|---|
| 4-page CVPR-style report (LaTeX) | `docs/main.tex` + `docs/references.bib` | Real Gemma + LLaMA numbers in every section, including §3.7 cross-architecture with Fig. 6 |
| TikZ Figure 1 (method schematic) | embedded inline in `main.tex` | ready |
| Figures 2-5 (Gemma headline plots) | `figures/fig{2..5}*.pdf` | ready |
| Figure 6 (cross-family TPR vs length) | `figures/fig6_cross_model.pdf` | ready (built from cached length curves; no HF auth needed locally) |
| Figure-1 PNG variant for slides | `figures/fig1_method.png` | ready |
| 4-page demo notebook (with executed outputs) | `demo.ipynb` | ready, summary mentions both models |
| Demo notebook PDF | `demo.pdf` | rendered, **exactly 4 pages** |
| 5-slide pptx | `docs/watermarking_3min.pptx` | 16:9, real Gemma + LLaMA numbers paired in slide 3 |
| 3-min speaker script | `docs/SPEAKER_SCRIPT.md` | LLaMA take-away updated with real 98% TPR |
| LLaMA evaluation Colab notebook | `colab_llama_eval_only.ipynb` | already run; kept for reproducibility |
| Gemma corpora & JSON results | `results/corpus_gemma_d*.jsonl`, `results/*_gemma*.json` | from Ammar's run |
| LLaMA JSON results | `results/*_llama_*.json`, `results/corpus_llama_d2.jsonl` | from your Colab run earlier today |

### Headline numbers in the report

| Metric | Gemma 2 9B | LLaMA 3.1 8B |
|---|---|---|
| TPR @ 1% FPR (all lengths) | 90.0% | **98.0%** |
| TPR @ 1% FPR (≥150 tok) | 98.7% | 98.0% |
| GPT-2 PPL ratio (wm/uwm) | 1.09 | **1.01** |
| Robustness at 20% word sub | 75.1% | **89.9%** |
| Robustness at 10% del / ins | 86.1% / 86.1% | **98.0% / 97.0%** |

LLaMA outperforms Gemma on every metric except the ≥150-token TPR (where they're statistically tied). Nothing in the report is a placeholder.

---

## 2. Steps to ship (the only thing left)

### 2.1  Compile the report on Overleaf

LaTeX is **not** installed on this Mac. Easiest path is Overleaf (free):

1. Open <https://www.overleaf.com/> → **New Project → Templates → CVPR 2024** (or any prior CVPR template; they all ship `cvpr.sty` and `ieee_fullname.bst`).
2. Once the template loads, **delete the existing `main.tex`** and **upload `docs/main.tex`** from this repo in its place.
3. **Upload `docs/references.bib`** next to it.
4. **Upload these five figure PDFs into a `figures/` folder** in the Overleaf file tree (drag-drop the whole `figures/` directory works):
   - `fig2_zscore_hist.pdf`
   - `fig3_length_curve.pdf`
   - `fig4_robustness.pdf`
   - `fig5_delta_tradeoff.pdf`
   - `fig6_cross_model.pdf`

   Figure 1 is inline TikZ, so no upload needed.

5. **Recompile** (top-left → Recompile). Expected output: **4 pages of body + 1 page of references = 5 pages total**, with page numbers at the bottom.

   *Hiccups:* "Citation undefined" on first compile is normal; bibtex needs two passes, so just click Recompile again. "File `cvpr.sty' not found" means you started from a blank doc, so recreate the project from the CVPR template.

6. Top-right **Download → PDF** as `report.pdf` and save it into the local repo at `~/Downloads/GM/llm-invisible-watermarking/report.pdf`.

### 2.2  Combine `report.pdf` + `demo.pdf` for Gradescope

```bash
cd ~/Downloads/GM/llm-invisible-watermarking
python3 - <<'PY'
from PyPDF2 import PdfReader, PdfWriter
w = PdfWriter()
for fp in ["report.pdf", "demo.pdf"]:
    for p in PdfReader(fp).pages:
        w.add_page(p)
with open("submission.pdf", "wb") as f:
    w.write(f)
print("wrote submission.pdf,", sum(1 for _ in PdfReader("submission.pdf").pages), "pages total")
PY
```

`submission.pdf` is what goes to Gradescope under the "written report + Jupyter notebook" upload (single file, per the course instructions).

### 2.3  Zip the code repository

Course also wants the code repo as a separate zip:

```bash
cd ~/Downloads/GM
zip -r llm-invisible-watermarking.zip llm-invisible-watermarking \
    -x 'llm-invisible-watermarking/.git/*' \
    -x 'llm-invisible-watermarking/__pycache__/*' \
    -x '*/__pycache__/*' \
    -x 'llm-invisible-watermarking/demo.html' \
    -x 'llm-invisible-watermarking/results/corpus_gemma_d*.jsonl'
ls -lh llm-invisible-watermarking.zip
```

The exclude line drops 5 MB of Gemma corpus JSONL but keeps the small JSON summaries; final zip lands ~600 KB. If you'd rather include the corpora, delete the last `-x` line and the final zip will be ~6 MB.

### 2.4  Upload both files to Gradescope

- "Final Project Report" submission → upload `submission.pdf`
- "Final Project Code" submission (or whatever the second slot is called) → upload `llm-invisible-watermarking.zip`

---

## 3. Slides for the in-class talk

`docs/watermarking_3min.pptx` is already a 16:9, 5-slide deck with the real numbers for both models. Open it in Keynote or PowerPoint to spot-check.

Then:
- Run through `docs/SPEAKER_SCRIPT.md` once on a stopwatch, targeting 3:00 sharp.
- The pace cheat-sheet at the bottom of the script tells you which line to skip if you're behind.
- The course says you don't have to submit slides, but they want them on the projector for the live talk. AirDrop / USB the pptx onto whichever laptop is driving the room.

---

## 4. Verification checklist before submitting

```bash
cd ~/Downloads/GM/llm-invisible-watermarking
echo "--- corpora ---"          && ls -la results/corpus_*.jsonl
echo "--- gemma summaries ---"  && ls -la results/*gemma*.json
echo "--- llama summaries ---"  && ls -la results/*llama*.json
echo "--- figures ---"          && ls -la figures/*.pdf
echo "--- demo PDF ---"         && python3 -c "from PyPDF2 import PdfReader; print('demo:', len(PdfReader('demo.pdf').pages),'pages')"
[ -f report.pdf ] && python3 -c "from PyPDF2 import PdfReader; print('report:', len(PdfReader('report.pdf').pages),'pages')"
[ -f submission.pdf ] && python3 -c "from PyPDF2 import PdfReader; print('submission:', len(PdfReader('submission.pdf').pages),'pages')"
```

Expected:
- `demo.pdf` = **4 pages** (course limit)
- `report.pdf` = ~5 pages (4 body + ~1 references; course limit is 4 pages excluding refs)
- `submission.pdf` = report + demo combined, ~9 pages
- 6 figures (`fig2`-`fig6`) in `figures/`
- Both Gemma and LLaMA JSON summaries present

If `report.pdf` is more than 4 body pages, ping me with which subsection overflowed and I'll trim it.

---

## 5. Optional follow-ups (not blocking submission)

- **Push results back to GitHub** so Ammar can pull. The repo `.gitignore` excludes `results/` and `figures/`; if you want to share the LLaMA numbers, drop the `results/*` and `figures/*` lines from `.gitignore` and commit, **or** zip and email/Slack them.
- **Re-run `scripts/build_slides.py`** if you tweak the JSON results. The build script reads from `results/*.json` and re-emits `docs/watermarking_3min.pptx` from scratch, so any edits to numbers stay in sync automatically.
- **Re-render `demo.pdf`** if you edit `demo.ipynb`:

  ```bash
  cd ~/Downloads/GM/llm-invisible-watermarking
  /Users/ali.hasan/Library/Python/3.9/bin/jupyter nbconvert --to notebook --execute demo.ipynb --output demo.ipynb --ExecutePreprocessor.timeout=180
  /Users/ali.hasan/Library/Python/3.9/bin/jupyter nbconvert --to html demo.ipynb --output demo.html
  python3 scripts/print_notebook_pdf.py "$(pwd)/demo.html" "$(pwd)/demo.pdf"
  ```

- **Re-run the LLaMA Colab eval** if for any reason you want fresh numbers (e.g. a different δ): `colab_llama_eval_only.ipynb` is still in the repo, runs in ~45 min on a free T4, and produces a results zip you unzip into `results/` exactly as before.
