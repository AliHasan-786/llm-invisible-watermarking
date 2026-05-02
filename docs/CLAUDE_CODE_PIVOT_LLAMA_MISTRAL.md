# Claude Code Patch — LLaMA primary + Mistral as second model

**Context.** Gemma-3 4B has unrecoverable compatibility issues in our Colab environment (loads as `Gemma3ForConditionalGeneration` multimodal class, produces NaN logits in text-only generation). We're pivoting to **LLaMA 3.1 8B-Instruct as primary** and **Mistral 7B Instruct v0.3 as secondary** (cross-family generalization study). Gemma stays in the report as a documented failure analysis.

**Mistral choice rationale.** Ungated (no approval delay), distinct architecture from LLaMA (different lineage), 7B params fits comfortably on A100. We do *reduced* scope on Mistral: 200 prompts at δ=2.0, headline detectability + word-substitution robustness only. No full δ-sweep, no paraphrase study on Mistral — those stay LLaMA-only. This keeps total compute under 5 hours.

**Existing artifacts.** `results/corpus_llama_d2.jsonl` already exists from Phase 7 (~400 lines). Don't regenerate it.

---

## Edits, in order

### 1. Repoint primary eval scripts at LLaMA

#### `scripts/eval_headline_gemma.py`

Update the constants block at the top:

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CORPUS_PATH = "results/corpus_llama_d2.jsonl"
OUTPUT_PATH = "results/headline_llama_summary.json"
ZSCORES_PATH = "results/headline_llama_zscores.npz"
```

Also update any internal print messages that say "Gemma" to say "LLaMA" for clarity. Acceptance threshold (TPR > 0.90 at ≥150 tokens) is unchanged — LLaMA is what Kirchenbauer et al. originally tested on, so it should pass cleanly.

#### `scripts/eval_length_curves.py`

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CORPUS_PATH = "results/corpus_llama_d2.jsonl"
OUTPUT_PATH = "results/length_curves_llama.json"
```

#### `scripts/eval_robustness.py`

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CORPUS_PATH = "results/corpus_llama_d2.jsonl"
SUMMARY_PATH = "results/headline_llama_summary.json"
OUTPUT_PATH = "results/robustness_llama.json"
ZSCORES_PATH = "results/robustness_llama_zscores.npz"
```

#### `scripts/run_delta_sweep.py`

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HEADLINE_CORPUS = "results/corpus_llama_d2.jsonl"
OUTPUT_PATH = "results/delta_sweep_llama.json"
```

Also update the per-delta path inside `generate_for_delta`:
```python
corpus_path = f"results/corpus_llama_d{delta}.jsonl"
```

### 2. Repurpose the replication script for Mistral

#### `scripts/run_llama_replication.py` → rename to `scripts/run_mistral_replication.py`

This script already implements the right pattern (clean access check, generate corpus, save). Just swap the model:

```python
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
N_PROMPTS = 200  # 67 per dataset
DELTA = 2.0
OUTPUT_CORPUS = "results/corpus_mistral_d2.jsonl"
SUMMARY_PATH = "results/mistral_replication_summary.json"
```

**Important:** delete the access-check block that probes `meta-llama/Llama-3.1-8B-Instruct`. Mistral is ungated — accessing it via `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` will work without auth.

The script's existing graceful-skip behavior (if model is unavailable) should be kept — it's good defensive code in case the download is interrupted.

### 3. Add a Mistral evaluation script

Create new `scripts/eval_mistral.py` (parallel structure to `eval_headline_gemma.py`, but reduced scope):

```python
"""
Evaluate watermark detection on Mistral 7B corpus.
Reduced scope: TPR @ 1% FPR on full corpus + word-substitution robustness only.
"""

import json
import numpy as np
from transformers import AutoTokenizer
from watermark.detector import WatermarkDetector
from evaluation.metrics import compute_z_scores, compute_tpr_at_fpr
from evaluation.robustness import apply_word_substitution

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
CORPUS_PATH = "results/corpus_mistral_d2.jsonl"
OUTPUT_PATH = "results/mistral_replication_summary.json"
GAMMA = 0.5
SEED = 42
TARGET_FPR = 0.01
MIN_TOKENS = 150

# Load corpus
with open(CORPUS_PATH) as f:
    corpus = [json.loads(line) for line in f if line.strip()]

wm_items = [x for x in corpus if x["watermarked"]]
uwm_items = [x for x in corpus if not x["watermarked"]]
print(f"Loaded: {len(wm_items)} watermarked, {len(uwm_items)} unwatermarked")

# Load tokenizer & detector
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
vocab_size = len(tokenizer)
detector = WatermarkDetector(vocab_size=vocab_size, gamma=GAMMA, seed=SEED)

# Headline detectability
print("Computing headline z-scores...")
wm_z, uwm_z = compute_z_scores(corpus, detector, tokenizer)
calibrated_threshold = detector.calibrate_threshold(uwm_z, target_fpr=TARGET_FPR)
tpr_all = sum(z > calibrated_threshold for z in wm_z) / len(wm_z) if wm_z else 0.0

long_wm_z = [z for item, z in zip(wm_items, wm_z) if item["n_tokens"] >= MIN_TOKENS]
tpr_long = sum(z > calibrated_threshold for z in long_wm_z) / len(long_wm_z) if long_wm_z else 0.0
print(f"  TPR (all):          {tpr_all:.3f}")
print(f"  TPR (>={MIN_TOKENS}t): {tpr_long:.3f}")

# Word-substitution robustness at 10%
print("Word substitution robustness at 10%...")
sub_z = []
for item in wm_items:
    sub_text = apply_word_substitution(item["completion"], rate=0.10, seed=SEED)
    sub_ids = tokenizer.encode(sub_text, add_special_tokens=False)
    sub_z.append(detector.score_sequence(sub_ids).z_score)
tpr_sub = sum(z > calibrated_threshold for z in sub_z) / len(sub_z) if sub_z else 0.0
print(f"  TPR after 10% sub:  {tpr_sub:.3f}")

summary = {
    "model": MODEL_NAME,
    "delta": 2.0,
    "gamma": GAMMA,
    "n_wm": len(wm_items),
    "n_uwm": len(uwm_items),
    "calibrated_threshold": float(calibrated_threshold),
    "tpr_at_1pct_fpr_all": float(tpr_all),
    "tpr_at_1pct_fpr_long": float(tpr_long),
    "tpr_after_word_sub_10pct": float(tpr_sub),
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
```

### 4. Update `scripts/make_figures.py`

Replace all `*_gemma*` paths with `*_llama*`. Then add a new figure 6 (cross-family comparison):

```python
# Fig 6: Cross-family comparison — bar chart of LLaMA vs Mistral on key metrics
def fig6_cross_family(out_path):
    llama = json.load(open("results/headline_llama_summary.json"))
    mistral_path = "results/mistral_replication_summary.json"
    if not os.path.exists(mistral_path):
        print(f"SKIP fig6: missing {mistral_path}")
        return
    mistral = json.load(open(mistral_path))

    metrics = ["TPR (all)", "TPR (>=150t)", "TPR + 10% sub"]
    llama_values = [
        llama.get("tpr_at_1pct_fpr_all", 0),
        llama.get("tpr_at_1pct_fpr_long", 0),
        # Look up word-sub-10 TPR from robustness file
        json.load(open("results/robustness_llama.json")).get("word_sub_10pct", {}).get("tpr_at_1pct_fpr", 0),
    ]
    mistral_values = [
        mistral.get("tpr_at_1pct_fpr_all", 0),
        mistral.get("tpr_at_1pct_fpr_long", 0),
        mistral.get("tpr_after_word_sub_10pct", 0),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, llama_values, width, label="LLaMA 3.1 8B", color="#1f77b4")
    ax.bar(x + width/2, mistral_values, width, label="Mistral 7B", color="#ff7f0e")
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path}")

fig6_cross_family("figures/fig6_cross_family.pdf")
```

(Adjust the json key names in `robustness_llama.json` lookup to match whatever your existing structure is — I'm guessing at the schema.)

### 5. Verify

```bash
python -m py_compile \
  scripts/eval_headline_gemma.py \
  scripts/eval_length_curves.py \
  scripts/eval_robustness.py \
  scripts/run_delta_sweep.py \
  scripts/run_mistral_replication.py \
  scripts/eval_mistral.py \
  scripts/make_figures.py
```

All should compile cleanly.

### 6. Clean up stale Gemma artifacts before running

In Colab, run once:

```bash
rm -f results/corpus_gemma3_d2.jsonl
rm -f results/corpus_gemma3_d*.jsonl
rm -f results/*gemma*.json results/*gemma*.npz
```

(The LLaMA corpus at `results/corpus_llama_d2.jsonl` should remain untouched.)

Commit message suggestion: `Pivot to LLaMA-primary + Mistral-secondary; document Gemma compatibility failure`
