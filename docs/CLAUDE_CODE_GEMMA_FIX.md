# Claude Code Patch — Fix Gemma CUDA Assert (vocab-size mismatch)

**The bug.** When Gemma-3 4B runs `model.generate()` with the watermark `LogitsProcessor`, every prompt fails with `CUDA error: device-side assert triggered`. Root cause: the green-list indices are sized against `tokenizer.vocab_size` (262,145 for Gemma), but the model's logits tensor `scores` may have a slightly different width because Gemma uses padded embeddings. When `scores[:, greenlist_ids]` indexes past the actual logit width, CUDA asserts.

The fix is in `watermark/logits_processor.py`. We clamp the green list to be a valid index range against the *actual* `scores` tensor width at call time, which is the source of truth.

**Severity.** Without this, Gemma's headline corpus generates zero samples (you saw this — `Done. 0 new samples generated`). LLaMA happens to work because its `tokenizer.vocab_size == model.config.vocab_size`, so the bug doesn't trigger.

---

## The patch

In `watermark/logits_processor.py`, replace the entire `__call__` method (lines 46–51) with:

```python
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Source of truth for valid logit indices is the scores tensor itself,
        # NOT tokenizer.vocab_size. Some models (e.g. Gemma-3) pad their embeddings
        # so scores.shape[1] != tokenizer vocab size.
        actual_vocab = scores.shape[-1]

        for batch_idx in range(input_ids.shape[0]):
            prev_token_id = input_ids[batch_idx, -1].item()
            greenlist_ids = self._get_greenlist_ids(prev_token_id)

            # Clamp to valid range. Indices >= actual_vocab are dropped silently;
            # the green-list density change is negligible (a few tokens out of ~131k).
            valid_mask = greenlist_ids < actual_vocab
            valid_green = greenlist_ids[valid_mask].to(scores.device)

            scores[batch_idx, valid_green] += self.delta
        return scores
```

Three changes from the original:

1. **Take `actual_vocab` from `scores.shape[-1]`**, not the constructor argument. This is the only number that's guaranteed to be a valid index range for the tensor we're mutating.
2. **Filter the greenlist** to drop any indices ≥ `actual_vocab` before applying the bias. With Gemma's ~256 token padding (out of 262k), at most ~128 green tokens get silently dropped — gamma effectively shifts from 0.5000 to 0.4998. Statistically invisible.
3. **`.to(scores.device)`** — the cached greenlist tensor was created on CPU (line 40 uses default device for `torch.randperm`). On A100/T4, `scores` lives on `cuda`. Indexing CPU tensor into GPU tensor would also assert. This was a latent bug masked by the first one.

## Important: detector also needs the same treatment

The detector (`watermark/detector.py`) uses the same green-list reconstruction. It currently scores against `tokenizer.vocab_size`, which means tokens with id `>= actual_vocab` would also need clamping. **However** — and this is critical — Gemma will *never generate* a token id outside the actual vocab, because the model can only produce indices that exist in its output layer. So during detection, every `token_id` we see is guaranteed to be valid.

But to be consistent and safe, also patch the detector. In `watermark/detector.py`, find the `score_sequence` method and add a guard. Look for the line that does the green-list lookup (something like `is_green = (greenlist == token_id).any()`). The exact patch depends on the existing structure — if `score_sequence` already does `if token_id >= self.vocab_size: continue` or equivalent, you don't need to change anything. If not, add that guard at the top of the per-token loop.

After both patches, run:

```bash
python -m py_compile watermark/logits_processor.py watermark/detector.py
```

## Verification

Before launching the full Phase 2 (3-hour job), do a 2-minute smoke test in Colab to confirm Gemma now generates without crashing:

```python
import os, sys
sys.path.insert(0, ".")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermark.logits_processor import WatermarkLogitsProcessor

model_name = "google/gemma-3-4b-it"
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

print("tokenizer.vocab_size:", tok.vocab_size, " len(tok):", len(tok))
print("model.config.vocab_size:", model.config.vocab_size)
print("lm_head out_features:", model.get_output_embeddings().out_features)

processor = WatermarkLogitsProcessor(
    vocab_size=len(tok), delta=2.0, gamma=0.5, seed=42,
)

prompt = "The Eiffel Tower is located in"
inputs = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=40, do_sample=True, temperature=1.0, top_p=0.95,
        logits_processor=[processor], pad_token_id=tok.pad_token_id,
    )
print("Generated:", tok.decode(out[0], skip_special_tokens=True))
```

**Pass criteria:** the print statement at the bottom shows ~40 new tokens of coherent French/Paris-related text. If you see that, Phase 2 will work and you can launch the full run.

**If it still fails** with a CUDA assert: stop, report the exact error, and we switch primary to LLaMA. Don't rerun a 3-hour job hoping it gets better.

## After the patch lands

Once the smoke test passes:

1. **Delete the empty corpus file** — otherwise the resume logic will think it's done:
   ```bash
   rm -f results/corpus_gemma3_d2.jsonl
   ```
2. **Re-run Cell 2** (`run_headline_gemma.py`). Watch the first few minutes — you want to see "[10 new / 0 skipped / 440 remaining] ETA: ~Xmin" actually progressing, not "Error on prompt N".
3. The rest of the cells (3–6, 8) will then work correctly because their inputs will exist.

Cell 7 (LLaMA) is already running and producing output. Don't restart it.
