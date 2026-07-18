# Explainer for Ali: Watermark Under Article 50

This file is your private study guide. It explains what the repository does, why the choices were made, and how to defend it in an interview without pretending the evaluation is further along than it is.

## The project in one minute

Language-model watermarking changes token sampling so generated text carries a statistical signal. A detector later looks for that signal without needing the original prompt or model weights.

The course project implemented the Kirchenbauer green-list scheme. It partitions the vocabulary into a pseudorandom "green" half at every generation step and adds a logit bias to those tokens. Detection replays the partition, counts green hits, and converts the excess into a z-score. The preserved baseline reached 90.0% TPR at a nominal 1% FPR on Gemma 2 9B and 98.0% on Llama 3.1 8B. The final report says an LLM paraphrase reduced Llama detection to 27.3%, but the raw paraphrase cache did not survive.

The extension asks whether SynthID-Text, a newer tournament-sampling watermark deployed in the Google ecosystem and available in open source, degrades more gracefully. The policy hook is Article 50(2) of the EU AI Act, which makes "detectable" and "robust" operationally important without supplying a universal numeric threshold.

## Architecture

```text
prompt corpus
    |
    +--> unwatermarked generation ----------------------+
    |                                                   |
    +--> Kirchenbauer generation --> fixed attacks -----+--> detectors
    |                                                   |       |
    +--> SynthID generation ------> fixed attacks ------+       v
                                                        held-out metrics
calibration controls ----------------------------------> thresholds
```

The important separation is calibration versus evaluation. The detector threshold is selected using only unwatermarked calibration outputs. TPR and realized FPR are then measured on held-out outputs. This avoids selecting an operating point on the same examples used to advertise performance.

## Core math

For the Kirchenbauer detector, each scored token is a Bernoulli trial under the null: it lands in a green list with probability `gamma`. If `X` of `T` tokens are green, then:

```text
z = (X - gamma*T) / sqrt(T*gamma*(1-gamma))
```

A high z-score means more green tokens than chance predicts. The threshold is empirical, because real text/token dependencies make the idealized binomial null imperfect.

TPR is the fraction of truly watermarked outputs detected. FPR is the fraction of unwatermarked outputs incorrectly detected. "TPR at 1% FPR" is not meaningful unless you explain how the threshold was calibrated and report the realized FPR.

SynthID does not use a fixed green list. Its tournament sampler uses pseudorandom g-values across multiple watermarking depths to influence sampled tokens while controlling distortion. Detection reconstructs those g-values from token n-grams and aggregates evidence. The preregistration uses the untrained weighted-mean detector as primary because a Bayesian detector would introduce a detector-training dataset and another source of overfitting.

## Why the P0 choices matter

- **Preregister before SynthID results:** prevents changing attacks or metrics after seeing which ones make a scheme look good.
- **Held-out controls:** makes the false-positive operating point more defensible.
- **Fixed two-model attack suite:** tests whether a result depends on one paraphraser's style.
- **One and two passes:** distinguishes light rewriting from determined laundering.
- **Length curves:** watermark evidence accumulates across tokens, so a single aggregate number hides the short-text failure mode.
- **No legal conclusion:** Article 50 motivates the measurement question but does not define 80% TPR as compliance.
- **Hashed caches:** every reported number should resolve to immutable generated text, detector scores, and a run configuration.

## What is complete versus planned

Complete in P0:

- baseline caches recovered and hashed;
- detector math tests;
- offline baseline verification;
- pinned Python dependencies;
- preregistered protocol;
- data/provenance card;
- current gate brief.

Not complete:

- SynthID integration;
- any new attack run;
- reproduction of the 27.3% paraphrase result;
- Article 50 report;
- interactive page;
- deployment.

Do not describe this as a completed SynthID evaluation until those artifacts exist.

## Ten hard interview questions and strong answers

### 1. Why is TPR at a fixed FPR the primary metric?

A detector can inflate recall simply by lowering its threshold and falsely flagging more human text. Fixing the false-positive operating point makes schemes comparable at a stated cost. I use held-out unwatermarked controls to calibrate the threshold and report both target and realized FPR because finite samples make exactly 1% impossible in some strata.

### 2. Why is 1% FPR still potentially too high?

At platform scale, 1% can mean millions of false flags. This study uses 1% because it matches the course baseline and is estimable with hundreds of controls. It also reports 0.1% and 5% when sample size supports them, and explicitly says robust low-FPR claims need much larger control sets.

### 3. Why not call the experiment an Article 50 compliance test?

The law uses qualitative language—effective, interoperable, robust, reliable, technically feasible—and does not define a universal TPR threshold or prescribe this dataset. The project supplies evidence relevant to that conversation. Compliance depends on legal interpretation, deployment context, and other technical and organizational measures.

### 4. What is the biggest validity threat?

The evaluation uses an open-source SynthID implementation, not Google's production Gemini deployment. Results can characterize the code and settings tested, not proprietary production behavior. The second major threat is author-run attacks, which may miss adaptive strategies.

### 5. Why use two paraphraser models and two passes?

One paraphraser can create a model-specific artifact. Two independent families test whether degradation generalizes. Two passes approximate a more determined laundering attack and directly test the preregistered hypothesis that both schemes fall below the study's 80% reference point.

### 6. Why use a weighted-mean SynthID detector rather than the Bayesian detector?

The Bayesian detector requires training, which creates extra choices around training data, class balance, and model selection. The weighted-mean detector requires no training and can be calibrated at the same held-out FPR protocol as the Kirchenbauer detector. Bayesian results can be exploratory unless separately preregistered.

### 7. What did repository cleanup uncover?

The clone ignored its entire results directory, so the public history had no reproducible results and contained stale empty stubs. The actual caches survived one directory above the clone, including byte-identical Gemma duplicates. I restored them under a manifest with SHA-256 hashes. The raw 27.3% paraphrase cache did not survive, so that claim is explicitly marked for replication.

### 8. How do you prevent cherry-picking?

The prompt sample, attacks, budgets, models, hypotheses, metrics, exclusions, and stopping rule are frozen before SynthID runs. Failed samples stay in a ledger. The primary table uses complete prompt pairs, an all-available sensitivity table is also shown, and any protocol change requires a dated deviation and new decision brief.

### 9. Why might watermark detection fall after paraphrasing?

The signal is encoded in token choices conditioned on local context. Paraphrasing changes both tokens and contexts, so the detector can no longer reconstruct the same sequence of biased choices. Surface edits preserve more of the sequence, while semantic rewriting can replace nearly all local evidence.

### 10. What would you build next if SynthID also collapses?

First, characterize where it fails: length, attack family, and quality tradeoff. Then test a preregistered defense such as a distortion-free or semantic-grouping variant. I would not quietly tune the current scheme after seeing failures; a defense experiment becomes a new phase with its own protocol.
