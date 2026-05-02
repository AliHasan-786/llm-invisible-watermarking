# Gemma-3 Diagnostic — find the real error

The Cell 2 patch did not fix the bug. The errors keep happening at the same prompt indices, and Cell 2 is finishing in only ~4 minutes — far too fast for the model to be actually running. This means **every generation is failing instantly with the same exception, before any real GPU work happens**, and the script's broad `try/except` is masking the actual error message.

Run this in a fresh Colab cell (after Cell 1) to surface the real error:

```python
import os, sys, traceback
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # synchronous CUDA so the real error surfaces

sys.path.insert(0, "/content/drive/MyDrive/llm-invisible-watermarking")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermark.logits_processor import WatermarkLogitsProcessor

MODEL = "google/gemma-3-4b-it"

# ── Step 1: load tokenizer & model, print every relevant size ────────────────
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"  tokenizer.vocab_size = {tok.vocab_size}")
print(f"  len(tokenizer)        = {len(tok)}")

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

print(f"  model class           = {type(model).__name__}")
print(f"  model.config.vocab_size = {getattr(model.config, 'vocab_size', 'MISSING')}")
print(f"  text_config.vocab_size  = {getattr(getattr(model.config, 'text_config', None), 'vocab_size', 'no text_config')}")
out_emb = model.get_output_embeddings()
if out_emb is not None:
    print(f"  output embedding shape = {tuple(out_emb.weight.shape)}")

# ── Step 2: try to generate WITHOUT the watermark processor first ────────────
prompt = "The Eiffel Tower is located in"
inputs = tok(prompt, return_tensors="pt").to("cuda")

print("\n--- Test A: vanilla generate (no watermark) ---")
try:
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=True, temperature=1.0, top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    print("  PASS:", tok.decode(out[0], skip_special_tokens=True))
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

# ── Step 3: try with the watermark processor ─────────────────────────────────
print("\n--- Test B: with WatermarkLogitsProcessor ---")
processor = WatermarkLogitsProcessor(
    vocab_size=len(tok), delta=2.0, gamma=0.5, seed=42,
)
try:
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=True, temperature=1.0, top_p=0.95,
            logits_processor=[processor], pad_token_id=tok.pad_token_id,
        )
    print("  PASS:", tok.decode(out[0], skip_special_tokens=True))
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

# ── Step 4: inspect what the processor actually saw ──────────────────────────
print("\n--- Inspecting processor state ---")
print(f"  processor.vocab_size = {processor.vocab_size}")
sample_greenlist = processor._get_greenlist_ids(100)  # arbitrary prev token
print(f"  greenlist length     = {len(sample_greenlist)}")
print(f"  greenlist max idx    = {sample_greenlist.max().item()}")
print(f"  greenlist min idx    = {sample_greenlist.min().item()}")
```

What this tells us:

1. **If Test A fails** — the problem is loading/running Gemma-3 itself, unrelated to the watermark. Most likely the transformers version in Colab doesn't fully support Gemma-3, or the model loaded as a multimodal class instead of causal-LM. We'll need to either upgrade transformers or use `Gemma3ForCausalLM` explicitly.

2. **If Test A passes but Test B fails** — the watermark processor is the issue. Compare the printed sizes: if `output embedding shape[0] = 262208` and `len(tokenizer) = 262145`, the patch is dropping the wrong indices. We need to size the greenlist to the *output dim*, not the tokenizer size.

3. **If both pass** — something about the script's batching or prompt handling specifically breaks. Less likely but possible.

Paste the full output back, and we'll know exactly what to fix.
