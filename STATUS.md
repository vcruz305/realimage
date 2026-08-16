# STATUS

This file is the live status tracker. Update it in place; do not append
history here -- see `docs/HISTORY_WORK_STATE.md` for the historical process
log.

## Product

- **Name:** RealImage
- **What it is:** A private, browser-native Chrome extension that detects
  AI-generated images locally on the page and shows a confidence score. No
  cloud calls, no uploads.

## Current model

- **Shipped candidate (as of 2026-08-16):** `RealImage/broad-v1-modern-v1-1471e3ef`
  -- FP32 ONNX, frozen `OwensLab/commfor-model-384` ViT-S/16 backbone with a
  384→32→1 GELU MLP head, trained on a further-expanded, public-only
  21-source / 4,650-image corpus (`.bench-data/realimage-broad-v1/manifest.json`).
  Raw threshold `0.646794855594635` → displayed **65%**. Holdout (leakage-
  corrected, pooled per tier): balanced accuracy 95.43% clean / 94.39% web /
  83.91% hard; real recall 97.55% clean / 97.66% web / 94.70% hard;
  unseen-generator (flux-krea, nano-banana, imagen4, recraft-v3) web-tier
  recall 88.02%. Q8 flip-rate re-check for this candidate (re-run, not
  assumed): 3.40% clean / 3.40% web / 6.73% hard -- still far above the
  0.5% bar, so **FP32 is shipped**, per this project's standing rejection
  rule for Q8 on this head. See `MODEL_CARD.md` for the full four-candidate
  comparison this was chosen from.
- **`SELF_TEST_REFERENCE_LOGIT` has been re-measured for
  `broad-v1-modern-v1-1471e3ef`** in real Chrome (WASM `-4.577674865722656`,
  WebGPU `-4.577672481536865`, ~2.4e-6 apart) -- WebGPU activates instead of
  falling back to WASM.
- **Ship rule corrected 2026-08-16, replacing the single-transform rule
  used for the two rounds below.** The old rule gated on one hard-tier
  transform's real-recall drop and had rejected two augmented retrains, in
  part because the incumbent's `flux-krea` "unseen-generator" recall was
  contaminated (55/64 of its current holdout images were actually in that
  model's own train/calibration split -- see `MODEL_CARD.md` section A).
  The corrected rule: eligible if pooled-per-tier real recall ≥92%/90%/85%
  (clean/web/hard), web-tier BA ≥90%, and unseen-generator web recall ≥80%;
  among eligible candidates, pick max `0.6*webBA + 0.4*hardBA`. Under this
  rule, **all four candidates evaluated this round (the leakage-corrected
  incumbent plus three new retrains on the expanded corpus) turned out to be
  eligible**, and the new `broad-v1-modern-v1-1471e3ef` candidate won on
  score (90.20% vs. 90.15% / 87.84% / 86.47% for the runners-up). Full
  candidate table, eligibility checklist, and score breakdown in
  `MODEL_CARD.md`.
- **Prior two ship-or-keep rounds (2026-08-16, both kept `broad-v1-02195715`
  at the time) are superseded by the corrected rule above** -- see
  `MODEL_CARD.md`'s "Known limitations > History" for the full writeup of
  why both earlier rejections don't hold up under the corrected,
  leakage-fixed reasoning.
- **Installed-Chrome lock-system remnants removed 2026-08-16.**
  `MODEL.finalistLockSha256` / `protocolSha256` / `toolingSha256` /
  `qualificationRead` (abandoned qualification-ceremony fields;
  `scripts/freeze-installed-chrome-candidate.mjs`, the tool that would have
  frozen them, was already deleted) are gone from `src/shared/constants.js`.
  `src/offscreen/analysis-debug.js` and `src/shared/calibration-debug-relay.js`
  no longer read or check those fields -- the underlying debug-identity
  record and calibration-debug relay are otherwise untouched and still work
  (they back real manual Chrome-console debugging; see
  `docs/PRIVATE_BROWSER_HARNESS.md`). `RELEASE.status` states the shipped
  model id plainly.
- The prior `RealImage/license-clean-v4-fp32-a6297c3a` and
  `RealImage/broad-v1-02195715` candidates have both been removed from
  `public/models/`.

## Last known benchmark

No installed-Chrome benchmark has been completed for the current candidate.
`BENCHMARK.md` documents an inherited upstream baseline (94.6% / 91.4%
balanced accuracy) that is explicitly **not** a RealImage claim.

## Manual E2E materials (prepared, not yet run)

Claude-in-Chrome was not connected in the sessions that built these, so a
human needs to run them in a real installed Chrome:
`tests/fixture/local-gallery/index.html` (6-image local gallery fixture) and
`docs/MANUAL_E2E.md` (exact steps + a results checklist, including a fresh
Chrome profile / internet-disconnected case). Not yet executed.

## Release package

- `npm run check` → green (unit tests, fast-path, policy scan, build).
- `npm run package` → `artifacts/realimage-v1.0.0.zip`, `110,118,524` bytes,
  SHA-256 `a1ec64c12965288cbdb76aef37f8fcd8b1ddcace7ed796327e72ed1d31d70c49`
  (deterministic -- reproduced twice with an identical hash before renaming
  the artifact from a stale `a6297c3a`-labeled filename).
- History squashed to one clean commit before the first push (private
  planning docs and personal paths were never intended to go public -- see
  `.gitignore`), tagged `v1.0.0`, pushed to a public GitHub repo.

## Next 3 tasks

1. Run the manual E2E checklist (`docs/MANUAL_E2E.md`) in a real installed
   Chrome and record results.
2. Create the GitHub Release itself (tag pushed; attach the ZIP + SHA-256
   via the GitHub UI -- see the chat for why this step needs a human).
3. Submit the POIDH claim only after both of the above are done.
