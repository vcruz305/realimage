# RealImage benchmark

This is RealImage's own source-disjoint holdout evaluation, not a
third-party or private-benchmark claim. It documents the methodology, the
leakage correction applied to the incumbent model, and the four-candidate
comparison the shipped model was chosen from.

Shipped model: `RealImage/broad-v1-modern-v1-1471e3ef` — FP32 ONNX, frozen
`OwensLab/commfor-model-384` ViT-S/16 backbone with a 384→32→1 GELU MLP
head. Raw decision threshold `0.646794855594635`, mapped to a displayed
**65%**. Q8 quantization was tested and rejected (3.40% / 3.40% / 6.73%
decision flips vs. FP32 on clean/web/hard, far over the 0.5% acceptance
bar), so FP32 ships. Full per-source training-data citations are in
`MODEL_CARD.md`.

## Corpus

21 sources, 4,650 images (2,455 AI / 2,195 real), 70/15/15
train/calibration/holdout split, seed 323. `flux-krea` and `nano-banana`
(locally-authored, non-redistributable) are eval-only — zero exposure in
train or calibration. `imagen4` and `recraft-v3` are held out entirely by
design as unseen-generator checks. See
`.bench-data/realimage-broad-v1/manifest.json` (not checked in — regenerate
with `scripts/split-realimage-broad-v1.mjs`) and `MODEL_CARD.md`'s training
table for exact per-source dataset IDs and licenses.

## Leakage correction (why this matters)

The prior shipped candidate (`RealImage/broad-v1-02195715`, since replaced)
was trained before this corpus was restructured: 55 of `flux-krea`'s 64
images were in its own train/calibration split, and later got relocated
into what is now the "holdout" directory. Reporting that candidate's
`flux-krea` recall as an "unseen generator" number would have been
contaminated by memorization (62/64 = 96.9% raw vs. 7/9 = 77.8% on the
literal 9 originally-held-out files). Every table below uses each
candidate's actual, verified-unseen generator set:

- **Incumbent (C0)** unseen set: `nano-banana`, `imagen4`, `recraft-v3`,
  `genimage-sd15`, `genimage-midjourney`, `gpt-image`, `midjourney-v6`,
  `sd35` (8 sources added after C0 was trained; `flux-krea` excluded).
- **C2/C3/C4** (trained on the corpus where `flux-krea` is eval-only) unseen
  set: `flux-krea`, `nano-banana`, `imagen4`, `recraft-v3`.

## Ship rule

A candidate is eligible only if, pooled per tier (not per transform):
real recall ≥92% clean / ≥90% web / ≥85% hard; web-tier balanced accuracy
≥90%; pooled unseen-generator web recall ≥80%. Among eligible candidates,
the winner maximizes `0.6·webBA + 0.4·hardBA`; a tie within 0.5pt favors
the public-only-corpus candidate over the incumbent.

## Four-candidate comparison

All four scored on the same 1,030-image holdout (3,090 for web/hard stress
derivatives), each with its own re-derived threshold.

| Candidate | Clean BA / AI-rec / real-rec | Web BA / AI-rec / real-rec | Hard BA / AI-rec / real-rec | Unseen-gen web recall |
| --- | ---: | ---: | ---: | ---: |
| C0 — incumbent, old corpus | 93.2 / 90.4 / 96.0 | 92.3 / 88.7 / 95.9 | 77.7 / 57.2 / 98.2 | 86.5% (n=1,494) |
| **C2 — expanded corpus + thumbnail aug (shipped)** | 95.4 / 93.3 / 97.6 | 94.4 / 91.1 / 97.7 | 83.9 / 73.1 / 94.7 | 88.0% (n=1,185) |
| C3 — expanded corpus, no augmentation | 96.2 / 93.9 / 98.5 | 94.9 / 91.4 / 98.4 | 77.3 / 55.6 / 99.0 | 87.7% (n=1,185) |
| C4 — expanded corpus, mild symmetric aug | 95.1 / 92.6 / 97.6 | 94.3 / 90.9 / 97.8 | 83.9 / 70.4 / 97.5 | 87.3% (n=1,185) |

All four candidates cleared every eligibility bar. Weighted score
(`0.6·webBA + 0.4·hardBA`): **C2 90.20** (shipped), C4 90.15, C3 87.84, C0
86.47. C2 and C4 are within the 0.5pt tie band but both are already
public-only-corpus candidates, so the tie-break doesn't apply; C2 wins on
raw score. Note for anyone re-deriving this: C4 has higher hard-tier real
recall than C2 (97.5% vs. 94.7%) — a defensible alternate pick if hard-tier
false-positive robustness is weighted more heavily than the rule's literal
score.

C3 (no augmentation) reproduces the incumbent's hard-tier collapse (55.6%
AI recall) despite the best clean/web numbers of the four — thumbnail
augmentation, not corpus expansion alone, is what buys hard-tier
robustness, at some cost to hard-tier real recall.

## Per-generator unseen-generator recall (web tier), shipped model (C2)

| Generator | n | Recall |
| --- | ---: | ---: |
| imagen4 | 660 | 92.9% |
| recraft-v3 | 141 | 88.7% |
| nano-banana | 192 | 85.9% |
| flux-krea | 192 | 72.9% |

`flux-krea` is the smallest and noisiest of the four (only 64 originals
total); its recall should be read with that in mind.

## Reproduce

```bash
npm ci
npm run build:fresh
npm run check
npm run benchmark -- /path/to/labeled-folder   # expects ai/ and real/ subfolders
```

No LLM, generator API, cloud inference service, or paid token is used
anywhere in this pipeline.

## Inherited upstream baseline (not a RealImage claim)

This project forked from Proofmark (see `README.md`'s Origins section and
`NOTICE.md`). Proofmark's own benchmark artifacts are preserved, unedited,
at `benchmark-results/upstream/` for provenance and regression-testing
purposes (`npm run test:fast-path`, `npm run audit:baseline-bias` still run
against them) — they report Proofmark's own 94.6% clean / 91.4% web
balanced accuracy on a differently-constructed, codec-confounded evaluation
set, using Node/native-ORT rather than installed Chrome. They are historical
context only, not evidence about `RealImage/broad-v1-modern-v1-1471e3ef`.

## What this does — and does not — prove

This is the project's own internal holdout, not an independent third-party
benchmark and not the bounty's private evaluation set. No detector
generalizes to every generator, edit, or adversarial input; treat scores as
a signal for user-controlled triage.
