# Modern source-disjoint holdout, version 1

Status: preregistration scaffold. This document defines the cohort, transformations, evidence contract, and go/no-go rules before any candidate is scored.

The benchmark scripts in this repository do not download data. Originals and their source manifest live under `.bench-data/`, which is ignored by Git. Only the frozen specification, aggregate result JSON, and attribution/provenance records that are safe to publish belong in the repository.

## Purpose and non-claims

The existing WebWild v3 sealed set is useful for regression testing, but it is not a frontier-generator holdout. Modern-v1 must be source-family disjoint from every training, calibration, development, and earlier sealed source. Byte-hash separation alone is insufficient.

This is a screening benchmark, not proof that AI-image detection is universally reliable. Twenty-four originals per generator can expose large failures; it cannot establish a narrow confidence interval for each generator. Scores are detector decision scores, not calibrated probabilities that an image is AI-generated.

## Frozen cohort

Target 432 originals, balanced by label:

- 216 AI originals: 24 from each of nine generator families not used to fit or select the candidate.
- 216 real originals:
  - 96 author-originated camera photographs from Google DOCCI.
  - 120 individually licensed Wikimedia Commons files, 20 from each of six false-positive slices: recent phone/computational photography; HDR/night; human digital art; CGI/3D/procedural; screenshots/UI/diagrams; scans/paintings/film.

Preferred frontier diagnostic families are GPT Image 1.5, GPT Image 2, Nano Banana Pro, FLUX.2 Klein 9B, Z-Image Turbo, Recraft v2, Recraft v3, Midjourney 7, and Ideogram 2.0. They may enter the claim cohort only when the exact files have affirmative commercial-evaluation, local-copy, and derivative-work rights.

OpenFake is not automatically eligible. Its Hugging Face metadata and dataset-card licensing statements conflict, and some proprietary subsets are described as noncommercial. A cash-bounty submission may be commercial activity. Until written clarification is recorded, OpenFake rows may be used only in a separately marked private diagnostic and must not be used for training, redistributed, or represented as claim-safe. A row with unresolved rights cannot be placed in the source manifest; the build intentionally fails closed.

The strict fallback is self-owned output generated under terms that affirmatively allow the four required uses. If that fallback cannot cover proprietary frontier systems, disclose the missing coverage rather than relabeling an older model as a frontier test.

## Source manifest contract

Place normalized originals beneath `.bench-data/modern-v1/` and create `.bench-data/modern-v1/manifest.json`. Each file must be a single-frame PNG, JPEG, WebP, or AVIF, at least 384 by 384 pixels, with orientation normalized to 1. Do not strip an attribution record merely because encoded EXIF was removed.

The manifest schema is `proofmark-modern-holdout-source-v1`:

```json
{
  "schema": "proofmark-modern-holdout-source-v1",
  "benchmarkId": "modern-v1",
  "seed": 323,
  "frozenAt": "2026-08-15T08:00:00Z",
  "sourceDisjointness": {
    "confirmed": true,
    "policy": "source-group+source-row+sha256+dhash-v1",
    "checkedAgainst": [
      {
        "manifest": "benchmark-results/webwild-v1-locked-manifest.json",
        "sha256": "85f1c1a7c09b5b01454fa3daf90bb80a9dd86d95b9565d99dea0554779b1b538"
      },
      {
        "manifest": "benchmark-results/webwild-v2-locked-manifest.json",
        "sha256": "23e68a103d8ae14d9eb900d4f6b553cc09433c3944cec6800c57da7a48223e13"
      },
      {
        "manifest": "benchmark-results/webwild-v3-locked-manifest.json",
        "sha256": "20bdc3fc24c8529e8383b1b710aa2ebe8487909185eb5c94337606d65f153434"
      }
    ]
  },
  "rows": [
    {
      "familyId": "docci-test-000001",
      "label": "real",
      "domain": "camera",
      "slice": "docci-camera-general",
      "sourceDataset": "google/docci",
      "sourceRow": "test:1",
      "sourceGroup": "google/docci",
      "relativePath": "originals/real/docci-test-000001.jpg",
      "width": 2048,
      "height": 1536,
      "format": "jpeg",
      "sha256": "<64 lowercase hex characters>",
      "dHash": "<16 lowercase hex characters>",
      "license": {
        "status": "approved",
        "identifier": "CC-BY-4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "verifiedAt": "2026-08-15T08:00:00Z",
        "commercialUse": true,
        "evaluationUse": true,
        "derivatives": true,
        "localCopy": true,
        "attribution": "Exact creator and source attribution retained here"
      },
      "provenance": {
        "status": "verified",
        "sourceUrl": "https://huggingface.co/datasets/google/docci",
        "immutableRevision": "<repository commit and immutable row identity>",
        "creator": "Exact named creator or dataset author",
        "labelMethod": "author-originated camera photograph",
        "retrievedAt": "2026-08-15T08:00:00Z"
      }
    }
  ]
}
```

Angle-bracket values in the example are invalid placeholders and must be replaced. Booleans are evidence assertions, not defaults: every one must be explicitly `true`. `license.status` must be `approved`, `provenance.status` must be `verified`, and both URLs must use HTTPS. Ambiguous, missing, expired, or noncommercial rights must fail the build.

`familyId` identifies one semantic original and all of its derived views. It must not encode the expected label. `sourceRow` is the immutable upstream row/file identifier. `sourceGroup` is deliberately broad enough to prevent source-family leakage—for example, `google/docci`, not a separate group per DOCCI image. For a multi-generator diagnostic collection, use the narrowest upstream partition that genuinely prevents leakage and document the choice.

Before freezing, compute and record SHA-256 digests for every prior locked manifest. From PowerShell:

```powershell
Get-FileHash benchmark-results\webwild-v1-locked-manifest.json -Algorithm SHA256
Get-FileHash benchmark-results\webwild-v2-locked-manifest.json -Algorithm SHA256
Get-FileHash benchmark-results\webwild-v3-locked-manifest.json -Algorithm SHA256
```

`make-stress-matrix.mjs` verifies the declared digests and automatically requires every `benchmark-results/*-locked-manifest.json` present in the repository. It rejects overlap by source group, immutable dataset row, family identifier, exact SHA-256, or dHash distance at most 2 for a matching aspect ratio. It also rejects exact and near duplicates within modern-v1.

## Deterministic transform matrix

Every original receives all four transforms. Label, domain, and slice never influence transform choice or parameters.

1. `double-jpeg-q85-q45`: JPEG q85 4:2:0 followed by JPEG q45 4:2:0.
2. `thumbnail-256-webp-q60-upscale`: long edge to 256 pixels, WebP q60, then upscale to original dimensions and WebP q60 again.
3. `offcenter-crop-75-jpeg-q70`: 75% crop with deterministic off-center coordinates derived from the seed and encoded source SHA-256, then JPEG q70 4:2:0.
4. `screenshot-social-jpeg-q68`: sRGB/alpha flattening, resize into a light screenshot canvas, deterministic geometric UI overlay, PNG intermediate, then JPEG q68.

The output contains one clean view and four stress views per original—432 clean samples and 1,728 stress samples at the target cohort size. Sharp and libvips versions are recorded in the generated manifest.

Build from repository root:

```powershell
node scripts\make-stress-matrix.mjs .bench-data\modern-v1 .bench-data\modern-v1-matrix --seed=323
```

The output path must not already exist. This prevents an old partial matrix from being silently mixed with a new freeze.

## Score protocol

Freeze the source manifest, transforms, candidate artifact, threshold, and any fusion coefficient before looking at modern-v1 scores. Run the incumbent and one preregistered candidate once. Do not tune on modern-v1 and rerun.

```powershell
$env:PROOFMARK_MODEL_ID='Proofmark/proofmark-webwild-v3'
$env:PROOFMARK_RESULTS_PATH='benchmark-results\modern-v1-baseline-clean.json'
npm run benchmark -- .bench-data\modern-v1-matrix\clean --quiet
$env:PROOFMARK_RESULTS_PATH='benchmark-results\modern-v1-baseline-stress.json'
npm run benchmark -- .bench-data\modern-v1-matrix\stress --quiet

$env:PROOFMARK_MODEL_ID='Proofmark/proofmark-dual-head-v1'
$env:PROOFMARK_RESULTS_PATH='benchmark-results\modern-v1-candidate-clean.json'
npm run benchmark -- .bench-data\modern-v1-matrix\clean --quiet
$env:PROOFMARK_RESULTS_PATH='benchmark-results\modern-v1-candidate-stress.json'
npm run benchmark -- .bench-data\modern-v1-matrix\stress --quiet
```

The named candidate is illustrative until that immutable model exists. Do not publish results under that name for a different artifact. Preserve the model digest beside each result.

An aborted benchmark is an automatic no-go. For a completed result document, `summarize-slices.mjs` joins rows against the frozen matrix. Any missing result row, explicit error, wrong expected label, invalid prediction, or missing/out-of-range score remains in the denominator and is counted incorrect. Unexpected or duplicate paths invalidate the result document.

Summarize with a paired, label-stratified cluster bootstrap. All views of one original remain in the same bootstrap cluster:

```powershell
node scripts\summarize-slices.mjs `
  --manifest=.bench-data\modern-v1-matrix\manifest.json `
  --baseline-clean=benchmark-results\modern-v1-baseline-clean.json `
  --baseline-stress=benchmark-results\modern-v1-baseline-stress.json `
  --candidate-clean=benchmark-results\modern-v1-candidate-clean.json `
  --candidate-stress=benchmark-results\modern-v1-candidate-stress.json `
  --bootstrap=10000 `
  --seed=323 `
  --output=benchmark-results\modern-v1-comparison.json
```

Exit code 2 means the evidence is valid but one or more claim gates failed. Exit code 1 means the evidence itself is invalid. `--smoke` is only for synthetic wiring tests; it always marks output `SMOKE_NOT_CLAIM_ELIGIBLE` and can never produce `claimEligible: true`.

## Hard go/no-go gates

All gates are required:

- Cohort: at least 216 AI and 216 real original families; at least nine AI domains with at least 24 originals each; at least seven real slices with at least 20 originals each.
- Candidate clean balanced accuracy at least 82%; stress balanced accuracy at least 78%.
- Candidate macro AI recall at least 75% on clean and stress; no generator below 60% on either view.
- Candidate real recall at least 92% clean and 88% stress; no real slice below 85% on either view.
- Candidate improvement over baseline at least 3 percentage points of balanced accuracy and 5 points of macro AI recall on both clean and stress.
- Candidate real-recall delta no worse than -1 percentage point on clean or stress.
- Paired original-cluster bootstrap 95% confidence-interval lower bound for balanced-accuracy delta greater than zero on clean and stress.
- Baseline and candidate result coverage complete, with zero missing, errored, or invalid result rows.
- Separate existing-v3 regression: clean and WebWild v3 balanced-accuracy loss at most 1 point; ADM and Midjourney recall loss at most 2 points.
- Runtime/package regression: model growth below 0.1 MB, inference latency no more than 1.05 times baseline, unchanged peak-memory class, and maximum JS/Python/ORT score difference at most `1e-4`.

If any gate fails, keep the result as diagnostic evidence and do not make a superiority or bounty-winning accuracy claim.
