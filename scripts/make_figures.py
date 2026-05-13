"""
Generate all final figures as PDF.

Reads results files for the primary and (optionally) secondary model and
produces 6 figures in CVPR style (9pt font, vector PDF).

Usage:
    # Default: LLaMA primary, Gemma 2 secondary
    python scripts/make_figures.py

    # Explicit models
    python scripts/make_figures.py \
        --primary-model meta-llama/Llama-3.1-8B-Instruct \
        --secondary-model google/gemma-2-9b-it

Output: figures/fig{1-6}.pdf  +  figures/fig1_method.txt
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_slug(model_name: str) -> str:
    name = model_name.lower().split("/")[-1]
    for key in ("llama", "gemma", "mistral", "falcon", "phi"):
        if key in name:
            return key
    return name.split("-")[0]


def model_label(model_name: str) -> str:
    """Human-readable label for figure legends."""
    name = model_name.lower()
    if "llama-3.1-8b" in name:  return "LLaMA 3.1 8B"
    if "llama"  in name:        return "LLaMA"
    if "gemma-2-9b" in name:    return "Gemma 2 9B"
    if "gemma-2-2b" in name:    return "Gemma 2 2B"
    if "gemma"  in name:        return "Gemma"
    if "mistral" in name:       return "Mistral 7B"
    return model_name.split("/")[-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--primary-model",   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--secondary-model", default="google/gemma-2-9b-it",
                   help="Set to empty string to skip cross-model figure")
    return p.parse_args()


def main():
    args = parse_args()
    p_slug = model_slug(args.primary_model)
    p_label = model_label(args.primary_model)
    s_slug  = model_slug(args.secondary_model) if args.secondary_model else None
    s_label = model_label(args.secondary_model) if args.secondary_model else None

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

    MISSING = []

    # ── Figure 1: Method schematic (hand-drawn placeholder) ──────────────────

    FIG1_TXT = """\
fig1_method.pdf - HAND-DRAWN IN TIKZ OR INKSCAPE
=================================================

Left panel: Watermark Injection
  Prompt → LLM (at each decoding step):
    previous token id
      → SHA-256(seed:prev_token_id) → integer → RNG seed
      → torch.randperm(vocab_size)[:gamma*V] → green list
      → add +delta to green-list logits
      → softmax → sample next token
  Repeat until max_new_tokens reached.
  Output: watermarked text (indistinguishable to human reader).

Right panel: Watermark Detection
  Candidate text → tokenize → token id sequence
  For each token i (i >= 1):
    use token[i-1] as prev → recompute green list (same seed)
    check if token[i] ∈ green list → count green hits
  z = (green_count - gamma*T) / sqrt(T*gamma*(1-gamma))
  One-sided z-test: z > threshold → WATERMARKED

Caption (suggested for LaTeX):
  "Overview of the Kirchenbauer et al. (2023) scheme. (Left) At each generation
   step a hash of the preceding token deterministically partitions the vocabulary
   into a green list (fraction γ) and a red list; bias δ is added to green-list
   logits before sampling. (Right) Detection re-derives the partition and applies
   a z-test to count the excess of green-list tokens."

TODO: Draw in TikZ (recommended) or Inkscape, export as PDF,
      then \\includegraphics{figures/fig1_method.pdf} in the report.
"""
    with open("figures/fig1_method.txt", "w") as f:
        f.write(FIG1_TXT)
    print("Wrote figures/fig1_method.txt  (hand-draw this in TikZ/Inkscape)")


    # ── Figure 2: Z-score histograms + ROC ───────────────────────────────────

    zscores_path = f"results/detection_{p_slug}_zscores.npz"
    summary_path = f"results/detection_{p_slug}_summary.json"

    if os.path.exists(zscores_path) and os.path.exists(summary_path):
        data = np.load(zscores_path)
        wm_z, uwm_z = data["wm_z"], data["uwm_z"]
        with open(summary_path) as f:
            summary = json.load(f)
        threshold = summary["calibrated_z_threshold"]

        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

        ax = axes[0]
        bins = np.linspace(min(uwm_z.min(), wm_z.min()) - 0.5,
                           max(uwm_z.max(), wm_z.max()) + 0.5, 45)
        ax.hist(uwm_z, bins=bins, alpha=0.65, label="Unwatermarked", color="#4878CF", density=True)
        ax.hist(wm_z,  bins=bins, alpha=0.65, label="Watermarked",   color="#D65F5F", density=True)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0,
                   label=f"Threshold = {threshold:.2f}")
        ax.set_xlabel("z-score")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")

        ax2 = axes[1]
        from evaluation.metrics import roc_curve_data
        fprs, tprs = roc_curve_data(list(wm_z), list(uwm_z))
        ax2.plot(fprs, tprs, color="#D65F5F", linewidth=1.5, label=p_label)
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
        MISSING.append(f"fig2 - missing {zscores_path} or {summary_path}")
        print(f"SKIP fig2: {MISSING[-1]}")


    # ── Figure 3: TPR vs token length ────────────────────────────────────────

    length_path = f"results/length_curves_{p_slug}.json"

    if os.path.exists(length_path):
        with open(length_path) as f:
            length_data = json.load(f)

        n_tokens = [d["n_tokens"] for d in length_data]
        tprs     = [d["tpr"]      for d in length_data]

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
        MISSING.append(f"fig3 - missing {length_path}")
        print(f"SKIP fig3: {MISSING[-1]}")


    # ── Figure 4: Robustness bar chart ───────────────────────────────────────

    rob_path = f"results/robustness_{p_slug}.json"

    if os.path.exists(rob_path):
        with open(rob_path) as f:
            rob_data = json.load(f)

        condition_order = [
            "baseline", "word_sub_5pct", "word_sub_10pct", "word_sub_15pct",
            "word_sub_20pct", "token_deletion_10pct", "token_insertion_10pct", "llm_paraphrase",
        ]
        condition_labels = [
            "Baseline", "Sub 5%", "Sub 10%", "Sub 15%",
            "Sub 20%", "Delete 10%", "Insert 10%", "LLM Para.",
        ]

        tprs, labels = [], []
        for cond, lbl in zip(condition_order, condition_labels):
            if cond in rob_data:
                tprs.append(rob_data[cond]["tpr_at_1pct_fpr"])
                labels.append(lbl)

        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        x = np.arange(len(labels))
        ax.bar(x, tprs, color="#4878CF", alpha=0.8, width=0.6)
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
        MISSING.append(f"fig4 - missing {rob_path}")
        print(f"SKIP fig4: {MISSING[-1]}")


    # ── Figure 5: Delta tradeoff (twin-axis) ─────────────────────────────────

    delta_path = f"results/delta_sweep_{p_slug}.json"

    if os.path.exists(delta_path):
        with open(delta_path) as f:
            delta_data = json.load(f)

        deltas  = [d["delta"]       for d in delta_data]
        tprs_d  = [d["tpr"]         for d in delta_data]
        ratios_d = [d["ppl_ratio"]  for d in delta_data]

        fig, ax1 = plt.subplots(figsize=(4.0, 2.8))
        color1, color2 = "#D65F5F", "#4878CF"

        ax1.plot(deltas, tprs_d, "o-",  color=color1, linewidth=1.5, markersize=4, label="TPR")
        ax1.set_xlabel("Logit bias δ")
        ax1.set_ylabel("TPR @ 1% FPR", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0, 1.05)

        ax2 = ax1.twinx()
        ax2.plot(deltas, ratios_d, "s--", color=color2, linewidth=1.5, markersize=4, label="PPL ratio (wm/uwm)")
        ax2.set_ylabel("PPL ratio (wm / uwm)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, lbl1 = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="center right", fontsize=7.5)

        plt.tight_layout()
        plt.savefig("figures/fig5_delta_tradeoff.pdf", bbox_inches="tight")
        plt.close()
        print("Saved figures/fig5_delta_tradeoff.pdf")
    else:
        MISSING.append(f"fig5 - missing {delta_path}")
        print(f"SKIP fig5: {MISSING[-1]}")


    # ── Figure 6: Cross-model comparison (primary vs secondary) ──────────────
    # Reads only the cached length_curves_*.json on each side - no tokenizer
    # download, no HF auth needed locally.

    s_length_path = f"results/length_curves_{s_slug}.json" if s_slug else None
    if s_slug and os.path.exists(length_path) and os.path.exists(s_length_path):
        with open(length_path)   as f: p_curve = json.load(f)
        with open(s_length_path) as f: s_curve = json.load(f)

        p_n = [r["n_tokens"] for r in p_curve]; p_t = [r["tpr"] for r in p_curve]
        s_n = [r["n_tokens"] for r in s_curve]; s_t = [r["tpr"] for r in s_curve]

        fig, ax = plt.subplots(figsize=(4.0, 2.8))
        ax.plot(p_n, p_t, "o-",  color="#D65F5F", linewidth=1.5, markersize=4, label=p_label)
        ax.plot(s_n, s_t, "s--", color="#4878CF", linewidth=1.5, markersize=4, label=s_label)
        ax.axhline(0.95, color="black", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_ylabel("TPR @ 1% FPR")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.savefig("figures/fig6_cross_model.pdf", bbox_inches="tight")
        plt.close()
        print("Saved figures/fig6_cross_model.pdf")
    else:
        if s_slug:
            print(f"SKIP fig6 - need both {length_path} and {s_length_path}")
        else:
            print("SKIP fig6 - no secondary model")


    # ── Summary ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 50)
    if MISSING:
        print("MISSING (run corresponding experiment first):")
        for m in MISSING:
            print(f"  - {m}")
    else:
        print("All figures generated successfully.")


if __name__ == "__main__":
    main()
