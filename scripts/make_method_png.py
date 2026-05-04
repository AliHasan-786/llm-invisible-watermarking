"""Render a small method schematic as PNG for the slide deck (slide 2)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figures/fig1_method.png"

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")

BLUE  = "#cfe0ff"
RED   = "#ffd0d0"
GRAY  = "#e8e8e8"
EDGE  = "#333333"

def box(x, y, w, h, text, color=BLUE, fs=10, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        edgecolor=EDGE, facecolor=color, linewidth=1.2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, wrap=True)

def arrow(x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12, color=EDGE, linewidth=1.2)
    ax.add_patch(a)

# ── INJECTION ROW (top) ────────────────────────────────────────────────
ax.text(5.25, 4.40, "Watermark injection", ha="center", fontsize=12, fontweight="bold")
box(0.10, 3.30, 1.45, 0.7, "Prompt\n$w_{1:n}$", BLUE)
box(1.95, 3.30, 1.55, 0.7, "LLM →\nlogits $\\ell_t$", BLUE)
box(3.90, 3.30, 2.20, 0.7, "$\\tilde\\ell_t = \\ell_t + \\delta\\,\\mathbf{1}[v\\in G_t]$", BLUE)
box(6.50, 3.30, 1.85, 0.7, "softmax →\nsample $w_t$", BLUE)
box(2.50, 2.05, 2.30, 0.6, "SHA-256$(s\\,\\Vert\\,w_{t-1})$", GRAY, fs=9)
box(2.50, 1.30, 2.30, 0.6, "permute → green list $G_t$", GRAY, fs=9)

arrow(1.55, 3.65, 1.95, 3.65)
arrow(3.50, 3.65, 3.90, 3.65)
arrow(6.10, 3.65, 6.50, 3.65)
arrow(2.70, 3.30, 3.40, 2.65)
arrow(3.65, 2.05, 3.65, 1.90)
arrow(4.80, 1.60, 4.95, 3.30)
# feedback
arrow(7.00, 3.30, 6.95, 2.85)
ax.plot([6.95, 1.20], [2.85, 2.85], color=EDGE, linestyle="--", linewidth=0.9)
arrow(1.20, 2.85, 1.20, 3.30)

# ── DETECTION ROW (bottom) ─────────────────────────────────────────────
ax.text(5.25, 0.95, "Watermark detection", ha="center", fontsize=12, fontweight="bold")
box(0.10, 0.10, 1.50, 0.7, "Candidate text", RED)
box(2.00, 0.10, 1.30, 0.7, "tokenize", RED)
box(3.65, 0.10, 2.30, 0.7, "re-derive $G_t$\nfrom $w_{t-1}$", GRAY, fs=9)
box(6.30, 0.10, 1.95, 0.7, "count\n$X{=}\\sum\\mathbf{1}[w_t\\!\\in\\!G_t]$", RED, fs=9)
box(8.55, 0.10, 1.85, 0.7, "$z=\\dfrac{X-\\gamma T}{\\sqrt{T\\gamma(1-\\gamma)}}$", RED, fs=10)

arrow(1.60, 0.45, 2.00, 0.45)
arrow(3.30, 0.45, 3.65, 0.45)
arrow(5.95, 0.45, 6.30, 0.45)
arrow(8.25, 0.45, 8.55, 0.45)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
