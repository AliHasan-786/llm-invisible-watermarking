"""
Phase 8: Generate all final figures as PDF.

CVPR style: 9pt font, vector PDF, no figure titles (caption goes in LaTeX),
bbox_inches='tight'.

Input:  results/headline_gemma_zscores.npz
        results/length_curves_gemma.json
        results/robustness_gemma.json
        results/delta_sweep_gemma.json
        results/llama_replication_summary.json  (optional)
Output: figures/fig{1-6}.pdf  +  figures/fig1_method.txt
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
matplotlib.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MISSING = []  # collect missing result files

# ─── Figure 1: Method schematic (hand-drawn placeholder) ────────────────────

FIG1_TXT = """\
fig1_method.pdf — HAND-DRAWN IN TIKZ OR INKSCAPE
=================================================

Left panel: Watermark Injection
  - Prompt → LLM (at each decoding step):
      previous token id
        → SHA-256(seed:prev_token_id) → integer → RNG seed
        → torch.randperm(vocab_size)[:gamma*V] → green list
        → add +delta to green-list logits
        → softmax → sample next token
  - Repeat until max_new_tokens reached.
  - Output: watermarked text (indistinguishable to human reader).

Right panel: Watermark Detection
  - Candidate text → tokenize → token id sequence
  - For each token i (i >= 1):
      use token[i-1] as prev → recompute green list (same seed)
      check if token[i] ∈ green list → count green hits
  - z = (green_count - gamma*T) / sqrt(T*gamma*(1-gamma))
  - One-sided z-test: z > threshold → WATERMARKED

Caption idea (for LaTeX):
  "Overview of the Kirchenbauer et al. (2023) watermarking scheme.
   (Left) During generation, a hash of the preceding token deterministically
   partitions the vocabulary into a green list (fraction γ) and a red list;
   a bias δ is added to green-list logits before sampling.
   (Right) Detection re-derives the partition and applies a z-test to count
   the excess of green-list tokens."

TODO: Draw this figure in TikZ (recommended) or Inkscape, export as PDF,
      and include as \\includegraphics{figures/fig1_method.pdf} in the report.
"""

with open("figures/fig1_method.txt", "w") as f:
    f.write(FIG1_TXT)
print("Wrote figures/fig1_method.txt  (hand-draw this in TikZ/Inkscape)")


# ─── Figure 2: Z-score histograms ───────────────────────────────────────────

ZSCORES_PATH = "results/headline_gemma_zscores.npz"
SUMMARY_PATH = "results/headline_gemma_summary.json"

if os.path.exists(ZSCORES_PATH) and os.path.exists(SUMMARY_PATH):
    data = np.load(ZSCORES_PATH)
    wm_z = data["wm_z"]
    uwm_z = data["uwm_z"]
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    threshold = summary["calibrated_z_threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

    ax = axes[0]
    bins = np.linspace(min(uwm_z.min(), wm_z.min()) - 0.5, max(uwm_z.max(), wm_z.max()) + 0.5, 45)
    ax.hist(uwm_z, bins=bins, alpha=0.65, label="Unwatermarked", color="#4878CF", density=True)
    ax.hist(wm_z,  bins=bins, alpha=0.65, label="Watermarked",   color="#D65F5F", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("z-score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")

    ax2 = axes[1]
    # TPR vs FPR (ROC)
    from evaluation.metrics import roc_curve_data
    fprs, tprs = roc_curve_data(list(wm_z), list(uwm_z))
    ax2.plot(fprs, tprs, color="#D65F5F", linewidth=1.5, label="Gemma-3 4B")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_xlim(0, 0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("figures/fig2_zscore_hist.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig2_zscore_hist.pdf")
else:
    MISSING.append("fig2 — missing headline_gemma_zscores.npz or headline_gemma_summary.json")
    print(f"SKIP fig2: {MISSING[-1]}")


# ─── Figure 3: TPR vs token length ──────────────────────────────────────────

LENGTH_PATH = "results/length_curves_gemma.json"

if os.path.exists(LENGTH_PATH):
    with open(LENGTH_PATH) as f:
        length_data = json.load(f)

    n_tokens = [d["n_tokens"] for d in length_data]
    tprs = [d["tpr"] for d in length_data]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(n_tokens, tprs, "o-", color="#D65F5F", linewidth=1.5, markersize=4)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8, label="0.95 reference")
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig("figures/fig3_length_curve.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig3_length_curve.pdf")
else:
    MISSING.append("fig3 — missing length_curves_gemma.json")
    print(f"SKIP fig3: {MISSING[-1]}")


# ─── Figure 4: Robustness bar chart ─────────────────────────────────────────

ROB_PATH = "results/robustness_gemma.json"

if os.path.exists(ROB_PATH):
    with open(ROB_PATH) as f:
        rob_data = json.load(f)

    condition_order = [
        "baseline",
        "word_sub_5pct",
        "word_sub_10pct",
        "word_sub_15pct",
        "word_sub_20pct",
        "token_deletion_10pct",
        "token_insertion_10pct",
        "llm_paraphrase",
    ]
    condition_labels = [
        "Baseline",
        "Sub 5%",
        "Sub 10%",
        "Sub 15%",
        "Sub 20%",
        "Delete 10%",
        "Insert 10%",
        "LLM Para.",
    ]

    tprs = []
    labels = []
    for cond, lbl in zip(condition_order, condition_labels):
        if cond in rob_data:
            tprs.append(rob_data[cond]["tpr_at_1pct_fpr"])
            labels.append(lbl)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, tprs, color="#4878CF", alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("figures/fig4_robustness.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig4_robustness.pdf")
else:
    MISSING.append("fig4 — missing robustness_gemma.json")
    print(f"SKIP fig4: {MISSING[-1]}")


# ─── Figure 5: Delta tradeoff (twin-axis) ───────────────────────────────────

DELTA_PATH = "results/delta_sweep_gemma.json"

if os.path.exists(DELTA_PATH):
    with open(DELTA_PATH) as f:
        delta_data = json.load(f)

    deltas = [d["delta"] for d in delta_data]
    tprs_d = [d["tpr"] for d in delta_data]
    ppls_d = [d["mean_ppl_wm"] for d in delta_data]

    fig, ax1 = plt.subplots(figsize=(4.0, 2.8))
    color1, color2 = "#D65F5F", "#4878CF"

    ax1.plot(deltas, tprs_d, "o-", color=color1, linewidth=1.5, markersize=4, label="TPR")
    ax1.set_xlabel("Logit bias δ")
    ax1.set_ylabel("TPR @ 1% FPR", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(deltas, ppls_d, "s--", color=color2, linewidth=1.5, markersize=4, label="Perplexity")
    ax2.set_ylabel("Perplexity (GPT-2)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="center right", fontsize=7.5)

    plt.tight_layout()
    plt.savefig("figures/fig5_delta_tradeoff.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig5_delta_tradeoff.pdf")
else:
    MISSING.append("fig5 — missing delta_sweep_gemma.json")
    print(f"SKIP fig5: {MISSING[-1]}")


# ─── Figure 6: LLaMA comparison (optional) ──────────────────────────────────

LLAMA_SUMM = "results/llama_replication_summary.json"
LLAMA_CORPUS = "results/corpus_llama_d2.jsonl"

if (os.path.exists(LLAMA_SUMM) and os.path.exists(LENGTH_PATH) and
        os.path.exists(LLAMA_CORPUS)):
    from pipeline.generate import load_corpus
    from watermark.detector import WatermarkDetector
    from evaluation.metrics import compute_tpr_at_fpr
    from transformers import AutoTokenizer

    with open(LLAMA_SUMM) as f:
        llama_summ = json.load(f)
    llama_threshold = llama_summ["calibrated_z_threshold"]

    with open(LENGTH_PATH) as f:
        gemma_curve = json.load(f)

    print("Computing LLaMA length curve for fig6...")
    llama_corpus = load_corpus(LLAMA_CORPUS)
    llama_tok = AutoTokenizer.from_pretrained(llama_summ["model"])
    if llama_tok.pad_token is None:
        llama_tok.pad_token = llama_tok.eos_token
    llama_det = WatermarkDetector(
        vocab_size=len(llama_tok), gamma=llama_summ["gamma"], seed=llama_summ["seed"]
    )
    llama_wm = [x for x in llama_corpus if x["watermarked"]]
    llama_uwm = [x for x in llama_corpus if not x["watermarked"]]
    length_bins = [d["n_tokens"] for d in gemma_curve]
    llama_tprs = []
    for n_tok in length_bins:
        wm_z = [llama_det.score_sequence(x["token_ids"][:n_tok]).z_score for x in llama_wm]
        uwm_z = [llama_det.score_sequence(x["token_ids"][:n_tok]).z_score for x in llama_uwm]
        _, tpr, _ = compute_tpr_at_fpr(wm_z, uwm_z, 0.01)
        llama_tprs.append(tpr)

    gemma_tprs = [d["tpr"] for d in gemma_curve]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(length_bins, gemma_tprs, "o-", color="#D65F5F", linewidth=1.5, markersize=4, label="Gemma-3 4B")
    ax.plot(length_bins, llama_tprs, "s--", color="#4878CF", linewidth=1.5, markersize=4, label="LLaMA-3.1 8B")
    ax.axhline(0.95, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig("figures/fig6_llama_compare.pdf", bbox_inches="tight")
    plt.close()
    print("Saved figures/fig6_llama_compare.pdf")
else:
    print("SKIP fig6 — LLaMA replication results not yet available (Phase 7 not run).")

# ─── Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
if MISSING:
    print("MISSING (run corresponding experiment first):")
    for m in MISSING:
        print(f"  - {m}")
else:
    print("All figures generated successfully.")
