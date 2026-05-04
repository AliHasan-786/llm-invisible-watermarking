# 3-minute speaker script: Invisible Watermarking for LLMs

Total budget 3:00. Hard cut-off. Timestamp markers at the end of each slide.

---

## Slide 1: title and motivation  (about 30 s, end at 0:30)

> Hi, I'm Ali, and with me on this is Ammar. Our project re-implements the Kirchenbauer watermark for large language models on Gemma 2 9B and LLaMA 3.1 8B.
>
> Why does this matter? AI-generated text is everywhere now: news, classrooms, customer support. The post-hoc detectors that classify by perplexity break down on short or paraphrased text. So instead of guessing afterward, we want to inject a statistical signal during generation that we can verify later with a single z-test. The challenge is keeping that signal strong enough to detect, while keeping the output fluent enough that no human notices.

---

## Slide 2: method  (about 40 s, end at 1:10)

> The watermark fits on one slide. On top, generation. At every decoding step we hash the previous token together with a secret seed using SHA-256, and that hash deterministically partitions the vocabulary into a green list, half the tokens, and a red list. Before softmax, we add a fixed bias delta of 2 to every green-list logit. The model itself is unchanged. We just wrap it with one extra LogitsProcessor.
>
> On the bottom, detection. Given any candidate text we replay the same hash, count how many tokens landed on their green list, and apply a one-sided z-test. We calibrate the threshold on a control corpus to give exactly 1 percent false positives.
>
> Hyperparameters are gamma 0.5, delta 2, seed 42, frozen across every experiment.

---

## Slide 3: headline and robustness  (about 50 s, end at 2:00)

> These are our headline numbers on Gemma 2 9B, with 201 watermarked plus 201 control completions.
>
> TPR at 1 percent FPR is 90 percent across all lengths, and 98.7 percent once you have at least 150 tokens. GPT-2 reference perplexity goes from 25.8 unwatermarked to 28.2 watermarked, a 9 percent overhead. So the watermark is statistically obvious to the detector and invisible to a reader. Same code on LLaMA 3.1 8B is even stronger: 98 percent TPR overall and a perplexity ratio of 1.01.
>
> Bottom row, robustness. We attack each watermarked sample. On Gemma, random word substitution at 5 percent drops TPR to 85. At 10 percent we are still at 84. Even at 20 percent, which is substantial editing, we are at 75. Token insertion and deletion at 10 percent leave it at 86. LLaMA holds up better on every one of those attacks.
>
> Bottom line: an adversary has to substantially rewrite the text, not just edit it, to evade detection.

---

## Slide 4: detectability versus quality  (about 40 s, end at 2:40)

> We also swept the bias delta. At delta 0.5 the signal is too weak, only 23 percent TPR. At delta 8 the watermark is overwhelming but the perplexity ratio blows up to 2.3 times, and the text is visibly degraded.
>
> Delta 2 is the knee. 90 percent TPR for only 14 percent extra perplexity. That matches the theoretical optimum from the original paper, and it's our recommended default.

---

## Slide 5: take-aways  (about 20 s, end at 3:00)

> Three contributions. One, we reproduced Kirchenbauer on a modern 9B instruction model that didn't exist when their paper was written, and every empirical claim still holds. Two, the same 30-line code path runs unchanged on LLaMA 3.1 8B and gets 98 percent TPR at a 1.01 perplexity ratio. Different family, different tokenizer, results actually stronger than Gemma. So the scheme is genuinely tokenizer-agnostic. Three, we documented a HuggingFace loader bug where Gemma 3 is treated as multimodal by default. That is a tooling issue rather than a watermark issue, and worth flagging for anyone trying to replicate this on next-generation models.
>
> Thank you.

---

## Pace cheat-sheet (printed in margin)

| Slide |  in  |  out  | content                      |
|------:|:----:|:-----:|------------------------------|
|   1   | 0:00 | 0:30  | motivation                   |
|   2   | 0:30 | 1:10  | method (TikZ schematic)      |
|   3   | 1:10 | 2:00  | headline + robustness        |
|   4   | 2:00 | 2:40  | delta knee                   |
|   5   | 2:40 | 3:00  | take-aways                   |

If you hit 2:00 still on slide 3, skip the bottom-row robustness narration and go straight to slide 4. The chart speaks for itself.
