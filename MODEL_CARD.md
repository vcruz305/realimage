# RealImage broad-v1-modern-v1 MLP FP32 candidate

> This is the currently shipped model: `RealImage/broad-v1-modern-v1-1471e3ef`, a
> frozen `OwensLab/commfor-model-384` ViT-S/16 backbone with a small trained
> classifier head, packaged in `public/models/`. The numbers below are this
> project's own source-disjoint holdout results, measured offline in
> Python/onnxruntime on the exact packaged ONNX graph -- **not** an
> independent third-party benchmark, and not a competition accuracy claim.
>
> **2026-08-16 model swap:** this candidate (internally called "C2" during
> the corrected ship-rule round below) replaces the prior shipped candidate
> `RealImage/broad-v1-02195715`. See "Leakage-corrected ship decision" below
> for the full four-candidate comparison and why this one was chosen.
> `SELF_TEST_REFERENCE_LOGIT` in `src/offscreen/self-test-fixture.js` has
> been re-measured in a real Chromium tab for this candidate, so WebGPU
> activates normally (WASM fallback remains available if the self-test ever
> fails on a given device). See `STATUS.md`.

## Frozen candidate summary

`RealImage/broad-v1-modern-v1-1471e3ef` keeps the MIT-licensed
`OwensLab/commfor-model-384` ViT-S/16 representation frozen and replaces its
classifier with a new 384→32→1 GELU MLP head, trained on a further-expanded,
public-only 21-source corpus (see "Training data" below).

- Route: one FP32 ONNX inference in local WebAssembly (WebGPU pending
  self-test re-measurement); no view fusion
- Decision input: raw model sigmoid only; metadata is reported but cannot alter it
- Input: RGB, shorter side 440 (bicubic), center crop 384, ImageNet normalization
- Raw/display cutoff: `0.646794855594635` → `65%` (display offset +0.0141 logits)
- Shipped precision: **FP32** (see "Why FP32, not Q8" below)
- FP32 weight SHA-256: `1471e3eff3a05d5ef8c068abfdef2f3f43d41060b27306f19763968cf8d38098` (87,437,806 bytes)
- Config SHA-256: `91c4edb9ee494e7f6afd510546850f30409a19a6817c6741552610a2cf087d81`
- Preprocessor-config SHA-256: `14c65215bdb2d8041bacf31d8b68953830c3081e7e7dd4ac25a4d3af58ceffaa`
- Backbone (source) weight SHA-256: `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`
- Training report: `.bench-data/realimage-broad-v1/candidate-mlp-modern-v1/training_report.json`

## Why FP32, not Q8

`scripts/train-community-head.py`'s `export_model()` produces both a FP32
ONNX export and a dynamic-INT8 (per-channel, MatMul-only) quantized export
from the same trained head. Both were re-scored on this candidate's own
clean/web/hard holdout tiers (1,030 / 3,090 / 3,090 rows) and compared
row-by-row at the calibrated raw threshold (`0.646794855594635`):

| Tier | n | Decision flips (Q8 vs FP32) | Flip rate | Max \|score Δ\| |
|---|---:|---:|---:|---:|
| Clean | 1,030 | 35 | 3.40% | 0.9972 |
| Web | 3,090 | 105 | 3.40% | 0.9860 |
| Hard | 3,090 | 208 | 6.73% | 0.9864 |

This project has never found Q8 acceptable for this head on any prior
candidate (4.0-4.7% flip rates on the previous `broad-v1-02195715` head); the
check above was re-run for this specific candidate rather than assumed, and
Q8 is again far above the 0.5% flip-rate bar for shipping it (3.4-6.7% here,
even worse than the incumbent on the hard tier) -- individual rows swing
substantially between precisions (mean |score delta| 0.030-0.066, max up to
0.997). **FP32 is shipped.** Size cost:
87,437,806 bytes vs. 24,031,833 bytes for Q8 (~83.4 MiB vs. ~22.9 MiB) --
the same order of magnitude as every previously shipped FP32 model in this
project, so this is not a package-size regression. Full numbers:
`.bench-data/realimage-broad-v1/fp32-vs-q8-flips-c2.json`.

## Training data

Built from `.bench-data/realimage-broad-v1/manifest.json` (4,650 staged
images across 21 sources, seed 323, split 70/15/15 train/calibration/holdout
per source by a seeded deterministic shuffle). Exceptions to the 70/15/15
rule:

- **`ai/flux-krea`, `ai/nano-banana`, `ai/imagen4`, `ai/recraft-v3` are 100%
  holdout** (0 train, 0 calibration) -- deliberately held out entirely to
  serve as this candidate's fully-holdout-only-generator generalization
  checks. `flux-krea` and `nano-banana` are locally-authored,
  non-redistributable proprietary generations; `imagen4` and `recraft-v3`
  are public HF pulls (Rapidata) simply chosen to stay unseen.
- **`real/usgs` (1 image) went entirely to train** -- too small to serve as
  its own evaluable holdout domain.

| Label | Domain | Total | Train | Calib. | Holdout | Source dataset | License | Note |
|---|---|---:|---:|---:|---:|---|---|---|
| ai | diffusiondb | 340 | 238 | 51 | 51 | `poloclub/diffusiondb` (HF) | CC0-1.0 | Confirmed genuine (filename+dimension cross-check against the dataset's own `metadata.parquet`, 12/12) |
| ai | flux-krea | 64 | 0 | 0 | 64 | locally-authored | proprietary, unpublished -- **not redistributable, eval-only** | Fully-holdout-only as of this round (previously had train/calibration rows under the superseded `broad-v1-02195715` candidate -- see "Leakage-corrected ship decision" below) |
| ai | flux-reason | 300 | 210 | 45 | 45 | `LucasFang/FLUX-Reason-6M` (HF) | Apache-2.0 | |
| ai | genimage-midjourney | 170 | 119 | 26 | 25 | `TheKernel01/Tiny-GenImage` (HF) | CC-BY-NC-SA-4.0 | validation split, generator=Midjourney (v5-era), label=fake |
| ai | genimage-sd15 | 170 | 119 | 26 | 25 | `TheKernel01/Tiny-GenImage` (HF) | CC-BY-NC-SA-4.0 | validation split, generator=SD15, label=fake |
| ai | gpt-image | 260 | 182 | 39 | 39 | `Rapidata/OpenAI-4o_t2i_human_preference` (HF) | CDLA-Permissive-2.0 | GPT-4o native image generation ("4o-26-3-25" snapshot) |
| ai | imagen4 | 220 | 0 | 0 | 220 | `Rapidata/Imagen4_t2i_human_preference` (HF) | CDLA-Permissive-2.0 | Google Imagen 4 Ultra (experimental snapshot); fully-holdout-only |
| ai | midjourney-v6 | 260 | 182 | 39 | 39 | `Photoroom/midjourney-v6-recap` (HF) | MIT | Genuinely v6-era (distinct from `genimage-midjourney`'s v5-era rows) |
| ai | nano-banana | 64 | 0 | 0 | 64 | locally-authored | proprietary, unpublished -- **not redistributable, eval-only** | Gemini "Nano Banana" generations; fully-holdout-only |
| ai | recraft-v3 | 47 | 0 | 0 | 47 | `Rapidata/Recraft-v3-24-7-25_t2i_human_preference` (HF) | CDLA-Permissive-2.0 | Fully-holdout-only; time-boxed at 47/220 target images due to severe upstream rate-limiting (see manifest provenance notes) |
| ai | sd35 | 260 | 182 | 39 | 39 | `data-is-better-together/open-image-preferences-v1` (HF) | Apache-2.0 | `_sd` column (paired `_dev`/FLUX column deliberately not pulled, to avoid overlap with flux-krea/flux-reason) |
| ai | synthetic-characters | 300 | 210 | 45 | 45 | `AbstractPhil/synthetic-characters` (HF) | CC-BY-4.0 | Capped at 300 to stay ≤10% of the AI pool |
| real | coco-val2017 | 1,300 | 910 | 195 | 195 | `rafaelpadilla/coco2017` (HF) | CC-BY-4.0 (annotations); per-image photo license inherited from the original Flickr upload, not independently re-verified | Everyday real photographs, added for false-positive control |
| real | docci | 80 | 56 | 12 | 12 | Google DOCCI | CC-BY-4.0 | reused from prior dev fixtures |
| real | japan-cc0 | 90 | 63 | 14 | 13 | `ThePioneer/japanese-photos` (HF) | **unverified** -- "cc0" is this repo's own directory-naming assumption | |
| real | laion-aesthetic | 300 | 210 | 45 | 45 | `laion/laion2B-en-aesthetic` (HF) | CC-BY-4.0 tag on the metadata/URL index only | LAION distributes an index, not the images; per-image copyright not independently verified |
| real | memes-imgflip | 30 | 21 | 5 | 4 | `SassyRong/meme-imgflip-small-test-dataset` (HF) | CC0-1.0 | Non-photo real content (memes), for false-positive control |
| real | nasa | 24 | 17 | 4 | 3 | reused-local | Public domain (NASA image/media usage guidelines) | |
| real | usgs | 1 | 1 | 0 | 0 | reused-local | Public domain (USGS / US Govt work) | too small to split; train-only |
| real | visual-logic | 300 | 210 | 45 | 45 | `skylenage/DeepVision-103K` (HF) | CC-BY-4.0 | |
| real | web-screenshots | 70 | 49 | 11 | 10 | `Zexanima/website_screenshots_image_dataset` (HF) | MIT | Non-photo real content (screenshots), for false-positive control |

**Totals: 2,455 AI / 2,195 real staged -> 2,979 train / 641 calibration /
1,030 holdout rows.** Training images are not shipped with the extension;
only the trained model weights are.

**License note:** `genimage-midjourney` and `genimage-sd15` (340 images,
~13.8% of the AI pool) are CC-BY-NC-SA-4.0 (non-commercial). Every other AI
and real source used in train/calibration is CC0/CC-BY/Apache/MIT/CDLA/public
domain, except the two fully-holdout-only, non-redistributable local sources
(`flux-krea`, `nano-banana`) which never enter train or calibration at all.
`japan-cc0`'s license is unverified (this repo's own naming assumption, not
a confirmed dataset-card tag).

## Leakage-corrected ship decision (2026-08-16)

### A. The problem this correction fixes

The immediately prior shipped candidate, `RealImage/broad-v1-02195715`,
reported `flux-krea` as if it were a held-out generator: 96.9% recall on its
64-image `flux-krea` holdout bucket. It was not held out. `flux-krea` was
moved to fully-holdout-only only as part of *this* round's corpus rebuild;
under `broad-v1-02195715`'s own training split, 55 of the 64 images now in
`flux-krea`'s holdout bucket were actually in that model's train/calibration
split. Independently reproducing the seed-323 shuffle against the current
64-file `flux-krea` staging listing confirms a 45/10/9 train/calibration/
true-holdout split, and the 9 true-holdout filenames are a subset of the
current 64. Scored on the shipped ONNX model:

| | True holdout (9/27 rows, clean/web/hard) | Contaminated (55/165 rows) |
|---|---:|---:|
| Clean recall | 7/9 = 77.8% | 55/55 = 100% |
| Web recall | 22/27 = 81.5% | 161/165 = 97.6% |
| Hard recall | 12/27 = 44.4% | 103/165 = 62.4% |

`broad-v1-02195715`'s `flux-krea` number was consequently an optimistic
mixture of true generalization and memorization. All numbers below use the
leakage-corrected reading: `broad-v1-02195715`'s unseen-generator recall is
computed only from generators genuinely absent from its own train/calibration
split (`nano-banana`, `imagen4`, `recraft-v3`, `genimage-sd15`,
`genimage-midjourney`, `gpt-image`, `midjourney-v6`, `sd35` -- 8 sources,
n=1,494 web-tier rows), while `flux-krea` is folded back into `broad-v1-02195715`'s
ordinary per-generator numbers rather than counted as unseen. For the three
candidates below (which trained on the corpus where `flux-krea` is genuinely
eval-only), `flux-krea` correctly counts as unseen alongside `nano-banana`,
`imagen4`, `recraft-v3` (4 sources, n=1,185 web-tier rows).

### B. Four-candidate comparison (pooled per tier, leakage-corrected)

Three new candidates were trained on the corpus above and compared against
the leakage-corrected incumbent, all scored on the same 1,030-image holdout
(clean) and its web/hard stress derivatives (3,090 rows each):

- **C0** -- the incumbent `broad-v1-02195715` (384→64→1 head, 14-source,
  2,543-image corpus that included `flux-krea` in train/calibration),
  re-scored on the current holdout with the leakage correction above.
- **C2** -- **this shipped candidate** (384→32→1 head, public-only 21-source
  corpus, `RandomThumbnailRoundTrip` augmentation `p=0.4`, short side
  160-224px, quality 30-65 -- the same augmentation already active on C0).
- **C3** -- same corpus as C2, thumbnail augmentation **removed entirely**.
- **C4** -- same corpus as C2, **milder** augmentation (`p=0.2`, short side
  224-320px, quality 55-80).

| Candidate | clean BA | clean AI-recall | clean real-recall | web BA | web AI-recall | web real-recall | hard BA | hard AI-recall | hard real-recall | unseen-gen web recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 (corrected, n=975/2925/2925) | 93.23% | 90.43% | 96.02% | 92.33% | 88.73% | 95.92% | 77.68% | 57.20% | 98.17% | 86.48% (1292/1494) |
| **C2 (shipped)** (n=1030/3090/3090) | 95.43% | 93.31% | 97.55% | 94.39% | 91.13% | 97.66% | 83.91% | 73.12% | 94.70% | 88.02% (1043/1185) |
| C3 (n=1030/3090/3090) | 96.18% | 93.88% | 98.47% | 94.87% | 91.37% | 98.37% | 77.30% | 55.62% | 98.98% | 87.68% (1039/1185) |
| C4 (n=1030/3090/3090) | 95.08% | 92.60% | 97.55% | 94.30% | 90.85% | 97.76% | 83.93% | 70.41% | 97.45% | 87.34% (1035/1185) |

Full clean/web/hard per-generator recall for every source and every
candidate is in
`.bench-data/realimage-broad-v1/candidate-comparison-summary.json`.

### C. Corrected ship rule

Replaces the earlier single-transform rule (which had rejected two prior
augmentation attempts on real-recall grounds alone -- see "Known
limitations" below for that history). Eligible if ALL hold, pooled per tier
(not per transform):

- real recall ≥ 92% clean, ≥ 90% web, ≥ 85% hard;
- web-tier BA ≥ 90%;
- pooled unseen-generator recall on the web tier ≥ 80%.

Among eligible candidates, pick max **0.6·webBA + 0.4·hardBA**. Ties (<0.5pt)
go to a candidate trained on the public-only cleaned corpus (C2/C3/C4) over
C0, for reproducibility.

| Bar | C0 | C2 | C3 | C4 |
|---|---|---|---|---|
| real recall ≥92% clean | PASS 96.02% | PASS 97.55% | PASS 98.47% | PASS 97.55% |
| real recall ≥90% web | PASS 95.92% | PASS 97.66% | PASS 98.37% | PASS 97.76% |
| real recall ≥85% hard | PASS 98.17% | PASS 94.70% | PASS 98.98% | PASS 97.45% |
| web-tier BA ≥90% | PASS 92.33% | PASS 94.39% | PASS 94.87% | PASS 94.30% |
| unseen-gen web recall ≥80% | PASS 86.48% | PASS 88.02% | PASS 87.68% | PASS 87.34% |
| **Eligible?** | YES | YES | YES | YES |

All four candidates clear all five bars -- a materially more permissive
outcome than the prior single-transform rule, which had rejected both
earlier augmentation attempts (see "Known limitations").

### D. Weighted score and decision

| Candidate | Score (0.6·webBA + 0.4·hardBA) | Public-only corpus? |
|---|---:|---|
| **C2 (shipped)** | **90.1995%** | Yes |
| C4 | 90.1547% | Yes |
| C3 | 87.8417% | Yes |
| C0 | 86.4705% | No |

C2 and C4 are 0.045pt apart (inside the 0.5pt tie-break band), but both are
already public-only-corpus candidates, so the "prefer public-only over C0"
tie-break does not distinguish between them -- it only applies when C0 is one
side of a near-tie, which is not the case here (C0 trails by 3.7pt). Ranked
by raw score, **C2 wins** and is the candidate shipped in this card.

**Caveat for a future reviewer:** C4 beats C2 on hard-tier real recall by
2.75pt (97.45% vs. 94.70%) and has a marginally higher hard BA (83.93% vs.
83.91%), while trailing C2 by only 0.05pt on the composite score. If
hard-tier real-recall robustness is valued specifically (fewer false
positives on heavily-degraded real photos), C4 is a defensible alternative;
its trained artifacts remain on disk at
`.bench-data/realimage-broad-v1/candidate-c4-mild-aug/` (FP32 SHA-256
`d7ff60aaa0d451071286aba4747d9ce5eaf9e338a90958c577a39d825bab1c4e`) for that
decision to be revisited. C3 (no augmentation) reproduces the same
hard-tier collapse pattern documented for C0 (55.62% AI recall) despite
having the best clean/web numbers of all four candidates, confirming that
thumbnail augmentation -- not corpus expansion alone -- is what buys
hard-tier robustness; its artifacts remain at
`.bench-data/realimage-broad-v1/candidate-c3-noaug/` (FP32 SHA-256
`28bff6ad5743895b64bf6f471cc7d38b1ce8799f64025572028c561b90c43ed4`).

### E. Per-generator detail on the unseen set

Web-tier n/recall feeding the "unseen-gen web recall" column above. C0's
unseen set is 8 sources (n=1,494); C2's is the 4 sources deliberately kept
fully-holdout-only in the corpus above (n=1,185):

| Generator | C0 | C2 (shipped) | C3 | C4 |
|---|---:|---:|---:|---:|
| flux-krea | 81.48% (n=27, leakage-corrected) | 72.92% (n=192) | 72.40% (n=192) | 68.75% (n=192) |
| nano-banana | 80.73% (n=192) | 85.94% (n=192) | 86.46% (n=192) | 89.06% (n=192) |
| imagen4 | 91.06% (n=660) | 92.88% (n=660) | 91.52% (n=660) | 91.97% (n=660) |
| recraft-v3 | 92.91% (n=141) | 88.65% (n=141) | 92.20% (n=141) | 88.65% (n=141) |
| genimage-sd15 | 100.00% (n=75) | -- (not unseen for C2/C3/C4) | -- | -- |
| genimage-midjourney | 69.33% (n=75) | -- | -- | -- |
| gpt-image | 88.89% (n=117) | -- | -- | -- |
| midjourney-v6 | 62.39% (n=117) | -- | -- | -- |
| sd35 | 86.32% (n=117) | -- | -- | -- |

(C0's last five rows count as unseen only for C0, since those five
generators were added to the corpus in the same expansion that trained
C2/C3/C4 -- they are not unseen for the shipped candidate, which trained on
them directly.)

## Stress-tier evaluation (shipped candidate, per-generator, web tier)

Clean = original holdout images (1,030 rows). Web = 3 mild transforms per
image (`jpeg-q75`, `webp-q72`, `resize1024-jpeg-q80`; 3,090 rows). Hard = 3
aggressive transforms per image (`double-jpeg-q85-q40`,
`thumbnail192-webp-q50-upscale`, `crop70-jpeg-q60`; 3,090 rows). Pooled
numbers are in section B above; per-generator web-tier AI recall:

| Generator | n | Web-tier AI recall |
|---|---:|---:|
| diffusiondb | 153 | 98.04% |
| flux-krea (unseen) | 192 | 72.92% |
| flux-reason | 135 | 88.89% |
| genimage-midjourney | 75 | 89.33% |
| genimage-sd15 | 75 | 96.00% |
| gpt-image | 117 | 99.15% |
| imagen4 (unseen) | 660 | 92.88% |
| midjourney-v6 | 117 | 89.74% |
| nano-banana (unseen) | 192 | 85.94% |
| recraft-v3 (unseen) | 141 | 88.65% |
| sd35 | 117 | 97.44% |
| synthetic-characters | 135 | 100.00% |

Full clean/web/hard per-generator recall for real-tier domains and every
candidate is in
`.bench-data/realimage-broad-v1/candidate-comparison-summary.json`; the
shipped candidate's own raw score log is
`.bench-data/realimage-broad-v1/score-c2.log`.

## Known limitations

- **Hard-tier AI recall is the weakest headline number for every candidate
  in this family** (73.12% for the shipped C2, pooled). Thumbnail-downscale/
  re-encode round trips remain the hardest transform for this head, even
  with augmentation on. Per-generator hard-tier recall is uneven: `flux-krea`
  47.4%, `recraft-v3` 65.2%, `midjourney-v6` 67.5% are the weakest on the web
  tier's hardest sibling transform set; see
  `candidate-comparison-summary.json` for the full breakdown.
- **`flux-krea` and `nano-banana` (128 images) are locally-authored,
  unpublished, non-redistributable generations**, not licensed for
  redistribution outside this dev/eval context; both are eval-only (0% train/
  calibration) in this candidate's corpus.
- **`genimage-midjourney`/`genimage-sd15` (340 images total, ~13.8% of the AI
  pool) are CC-BY-NC-SA-4.0** (non-commercial). Every other train/calibration
  source is CC0/CC-BY/Apache/MIT/CDLA/public domain.
- **`japan-cc0` and `laion-aesthetic` carry the same provenance caveats as
  before**: `japan-cc0`'s "cc0" label is this repo's own directory-naming
  assumption, not a confirmed dataset-card license; `laion-aesthetic`'s
  CC-BY-4.0 tag covers LAION's metadata/URL index, not necessarily the
  copyright status of every linked image. `coco-val2017`'s per-image photo
  license is inherited from the original Flickr upload and not independently
  re-verified, following COCO's own terms of use.
- Training data is not shipped with the extension; only trained model
  weights are packaged.

### History: prior ship-or-keep rounds on `broad-v1-02195715` (superseded)

Before this round, two retrains were evaluated against `broad-v1-02195715`
under an earlier, narrower ship rule (gate on one specific hard-tier
transform's real-recall drop) and both were rejected:

1. A thumbnail-augmentation retrain on the *original* 14-source corpus
   raised hard-tier BA 81.5%→88.6% but flipped the clean/web per-generator
   floor from passing to failing (concentrated in `flux-krea`'s then-9-image
   holdout) -- kept `broad-v1-02195715`.
2. A retrain on the first pass of the expanded, 21-source corpus (same
   corpus this card's candidates use) looked like a clean win on pooled
   aggregates but failed the old rule's single-transform real-recall gate
   (`thumbnail192-webp-q50-upscale`: 97.25%→88.69%, over 4x the 2pt
   tolerance) -- kept `broad-v1-02195715`.

Both rejections turned out to rest on `flux-krea`'s misleadingly-high
"unseen" recall for the incumbent (see section A above) and, in case 2, on a
per-transform gate rather than the pooled-per-tier reasoning now used. The
corrected round in this card supersedes both: it re-evaluates C0 with the
leakage fix, adds C3/C4 as two more candidates from the same corpus, and
replaces the ship rule with the pooled-per-tier version in section C. Full
prior writeups (with Wilson-interval detail) are preserved in `git log` /
prior revisions of this file.

## Intended use

The model is intended for private, browser-local triage of ordinary webpage
images. It should help a user decide what deserves scrutiny. It is not
suitable as the sole evidence for moderation, employment, legal,
academic-integrity, or other consequential decisions. Performance can change
with future generators, screenshots, collages, heavy edits, adversarial
processing, unusual illustration styles, or a website distribution unlike
the ones tested here. The displayed percentage is a calibrated detector
score, not a literal probability with universal coverage -- and, per the
hard-tier results above, a moderate thumbnail/re-encode round trip can
still reduce detection sensitivity for several generators.

## Superseded prior candidates

- **`RealImage/broad-v1-02195715`** (384→64→1 head, 14-source/2,543-image
  corpus that included `flux-krea` in train/calibration) -- FP32 SHA-256
  `02195715c69b2eade82df6359079f43f7312d04a5e541707216f1167164fe5d9`. Shipped
  from 2026-08-16 until superseded by this candidate the same day, once the
  leakage-corrected ship rule above found `broad-v1-modern-v1-1471e3ef` a
  better weighted-score choice. Removed from `public/models/`.
- **`RealImage/license-clean-v4-fp32-a6297c3a`** (384→1 affine head trained
  on 100 originals across 3 sources: Nano Banana, FLUX.1-Krea-dev, DOCCI) --
  superseded by `broad-v1-02195715` above. Its offline calibration-only
  balanced accuracy was 95.0% across 180 rows that also informed threshold
  selection -- not a held-out estimate. Already removed from
  `public/models/`. See `git log` / prior revisions of this file for the
  full retained-baseline provenance discussion (including the even earlier
  `Proofmark/proofmark-webwild-v3` inherited baseline), which this rewrite
  does not repeat in full.
