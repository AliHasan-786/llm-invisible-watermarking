# Colab Runbook — LLaMA primary + Mistral secondary

After Claude Code commits the patch, run these cells in order. **Don't skip the cleanup cell** — leftover Gemma files will confuse the resume logic.

```python
# === Cell 1: setup (mostly unchanged from before) ===
!nvidia-smi  # confirm GPU. A100 strongly preferred for Mistral 7B.
!pip install -q "transformers>=4.45" datasets accelerate sentencepiece protobuf scipy matplotlib

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/
!git clone https://github.com/AliHasan-786/llm-invisible-watermarking.git 2>/dev/null || echo "already cloned"
%cd llm-invisible-watermarking
!git pull
!git log --oneline -1

import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

# Pre-flight access check (Mistral is ungated; the others gated)
from huggingface_hub import HfApi
api = HfApi()
for repo in [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]:
    try:
        api.model_info(repo)
        print(f"  ✓ {repo}")
    except Exception as e:
        print(f"  ✗ {repo}: {e}")

!python -c "import os; print('HF_TOKEN is set:', bool(os.environ.get('HF_TOKEN')))"
```

```python
# === Cell 2: clean up stale Gemma artifacts ===
!rm -f results/corpus_gemma3_d*.jsonl
!rm -f results/*gemma*.json results/*gemma*.npz
!ls -la results/
# You should see: corpus_llama_d2.jsonl ONLY (everything else removed)
```

```python
# === Cell 3: verify LLaMA corpus is intact ===
!ls -la results/corpus_llama_d2.jsonl
!wc -l results/corpus_llama_d2.jsonl
# Expected: ~400 lines
```

```python
# === Cell 4: headline eval on LLaMA (fast, ~10 min) ===
!python scripts/eval_headline_gemma.py
# Expected: TPR (>=150t) > 0.90
```

```python
# === Cell 5: length curves on LLaMA (fast, ~5 min) ===
!python scripts/eval_length_curves.py
# Expected: TPR rises monotonically with token count
```

```python
# === Cell 6: robustness on LLaMA (medium, ~30-90 min, paraphrase is slow) ===
!python scripts/eval_robustness.py
# Expected: word-sub @ 10% still > 0.7; paraphrase drops noticeably
```

```python
# === Cell 7: delta sweep on LLaMA (slow, ~2-3 hours) ===
!python scripts/run_delta_sweep.py
# Expected: TPR + perplexity both rise with delta; knee around delta=2.0
```

```python
# === Cell 8: Mistral generation (medium, ~1 hour) ===
!python scripts/run_mistral_replication.py
# This will download Mistral weights (~14GB) on first run, then generate 200 prompts
# Resume logic is in place if interrupted
```

```python
# === Cell 9: Mistral eval (fast, <5 min) ===
!python scripts/eval_mistral.py
# Expected: TPR > 0.90 at >=150 tokens (LLaMA-comparable)
```

```python
# === Cell 10: generate all figures ===
!python scripts/make_figures.py
!ls -la results/ figures/
```

## Sequencing strategy

If you have one big Colab session: run sequentially (Cells 4 → 10). Total ~5–6 hours.

If you're disconnecting/reconnecting: prioritize like this:
- **Session 1 (90 min):** Cells 1, 2, 3, 4, 5, 6 — gets you headline + length + robustness on LLaMA. Even if you stop here, you have a strong report.
- **Session 2 (3 hours):** Cell 7 — δ-sweep. Long but unattended.
- **Session 3 (90 min):** Cells 8, 9 — Mistral. The "second model" story.
- **Session 4 (5 min):** Cell 10 — figures.

Cells 4–7 only depend on the existing LLaMA corpus, so if Mistral falls through for any reason, Cells 4–7 + 10 (skipping the cross-family fig6) gives you a complete LLaMA-only report.

## Watch points

- **Cell 7** runs longest. Verify Cell 4's TPR is > 0.90 before committing 3 hours to Cell 7.
- **Cell 8** downloads ~14GB. Make sure you have disk space and a stable connection.
- If any cell crashes, just re-run it. Resume logic skips completed work.
