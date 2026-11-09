# Decision brief: historical replay release closeout

**Date:** 2026-11-09

**Assignment:** `watermark-release-closeout-1`

**Base:** merged `main` at `27f3f89ec2368f324ef24c3a2fcfd92c3214ca10`

**Gate:** source metadata and deployment-safety closeout; scientific execution remains stopped

## Recommended call

Accept the historical replay release as closed once this metadata revision is
on `main` and its bounded public-release workflow passes. Continue to describe
the live artifact as a historical Kirchenbauer coursework replay. Do not imply
that SynthID, the pilot, or the confirmatory Article 50 study ran.

## Actual release record

- Independently reviewed source:
  `e794de39b9100dc51bdbff398f0a35b986abcd4b`.
- Normal merge commit:
  `27f3f89ec2368f324ef24c3a2fcfd92c3214ca10`.
- Production project: `text-provenance-lab`.
- Stable URL: `https://text-provenance-lab.vercel.app`.
- Ready production deployment: `dpl_4DHxhKGFWBjyw3Q9cPM2nM8Wxga6`.
- Ready preview deployment: `dpl_Eg4jMgC9eKSNvetCKu6H3egZrjqZ`; browser
  access is Vercel-authentication protected and was not bypassed.
- Canceled first deployment: `dpl_7RrFSEz93CDNDp3go3jP888nbvFQ`.
- Portfolio-hub snapshot: `288ec81`; its deployment verification remains a
  separate coordinator task.

The live page was checked at the stable URL. The Llama paraphrase selection
shows the preserved 27.3% scores-only result and the provenance boundary. It
does not present a SynthID or confirmatory result.

## Deployment defect and permanent guard

The first Vercel CLI attempt unexpectedly targeted production and began
installing the repository's full research requirements, including Torch/CUDA
packages. The coordinator canceled it. The Vercel project was then configured
with `installCommand: "true"`, and the ready production deployment used that
setting.

This source revision makes the guard portable in `vercel.json` and tests it.
The static builder remains standard-library-only and publishes the same exact
12-file package: 11 allowlisted artifacts plus its build manifest. Raw corpus
JSONL, prompts/completions, NPZ caches, notebooks, credentials, Python research
code, and backend/functions are not in the output.

## Metadata boundary

`portfolio.json` now records the production URL and state. It distinguishes:

- `reviewedSourceCommit`: the exact independently reviewed release source;
- `sourceMergeCommit`: the merge that placed that source on `main`; and
- `metadataRevision: git:containing-commit`: an explicit instruction to use
  the Git revision containing the metadata file, avoiding an impossible
  self-referential commit hash.

The root portfolio handoff records the resulting containing revision after the
commit exists.

## Bounded CI

`.github/workflows/public-release.yml` is a no-cost, dependency-free public
release smoke workflow. It:

1. runs the standard-library static builder, verifying all 17 source hashes and
   the exact public-package inventory;
2. runs the 10 production browser-contract tests;
3. checks the shipped JavaScript syntax; and
4. verifies that Vercel dependency installation is disabled.

It does not install `requirements.txt`, Torch, CUDA, model weights, or provider
packages. It is accurately named and does not claim full scientific validation;
`make check` remains the complete local cache/detector/replay check.

## Remaining gates

- No new deployment is needed for this metadata-only closeout. The coordinator
  may connect Git after the safe source configuration is on `main`.
- The portfolio-hub deployment for snapshot `288ec81` still needs coordinator
  verification.
- Any pilot or confirmatory run still requires exact frozen revisions,
  personally supplied credentials, an appropriate no-cost GPU, and the existing
  scientific decision protocol.
- Paid compute, provider access, model inference, a scientific/legal-review
  waiver, and new public claims remain out of scope.

No old reviewer automation was invoked. No provider/model access, inference,
GPU run, dependency download, spend, or deployment occurred in this closeout.
