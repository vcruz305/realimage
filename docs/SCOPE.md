# Upstream Proofmark scope

> Historical baseline document. RealImage does not adopt these acceptance
> claims until they are reproduced through the installed Chrome path.

Proofmark targets POIDH Arbitrum bounty 323, **“local AI challenge: AI image detector for Chrome.”** The live brief was rechecked on 2026-08-13.

## Acceptance matrix

| Bounty requirement | Implementation | Verification |
| --- | --- | --- |
| MIT open source | Root `LICENSE`; source and Proofmark model changes released under MIT | Inspect repository and `MODEL_CARD.md` |
| Native Manifest V3 | Service worker, content script, popup, options page, and offscreen inference document | Inspect `dist/manifest.json`; load `dist/` unpacked |
| Browser-local inference | Frozen FP32 ONNX ViT runs through Transformers.js / ONNX Runtime WebAssembly inside the offscreen document | `npm run test:e2e`; readiness check in Chrome |
| No cloud inference or external API | Remote model loading is disabled; production has no inference endpoint, telemetry, API key, or localhost dependency | `npm run test:policy`; disconnect after build |
| Offline after setup | Model and ONNX WebAssembly runtime are bundled in `dist/`; no later inference assets download | `npm run model:verify`; inspect network/policy |
| Automatic ordinary-page analysis and treatment | Content script discovers visible and lazy-loaded `<img>` elements; likely-AI images blur by default, with hide and label-only modes | Chrome E2E verifies automatic blur plus reveal/reapply |
| Confidence on every analyzed image | Overlay badge shows a percentage and verdict; click opens evidence | Chrome E2E and manual fixture |
| Watermark/metadata support | Local PNG/JPEG/WebP/AVIF scans surface generation parameters, C2PA labels, SynthID labels, and known embedded watermark markers | `src/analysis/forensics.test.js` |
| Complete reproducible build/install | Pinned source revision, hashes, scripts, `npm ci`, local build, and unpacked-install instructions | `README.md`; `npm run package` |
| ≥75% balanced accuracy at 65% | **Pending.** The inherited 94.6%/91.4% rows are Node/native-ORT baseline evidence with codec/aspect confounding, not RealImage or installed-Chrome results. | Frozen codec/aspect-balanced benchmark and installed-Chrome harness required before submission |

## Bounty threat model

The maintainers will build from source in a clean Chrome profile, disable internet after initial model setup, block native localhost APIs, and evaluate at a 65% confidence threshold. Proofmark does not need a setup server: Node and Python are development/build tools only. The installed extension contains JavaScript, ONNX weights, and WebAssembly assets and keeps pixels in the browser.

## Deliberate boundaries

- Confidence is probabilistic evidence, not proof of authorship.
- Generic invisible watermarks cannot all be decoded without each vendor's private detector. Proofmark reports accessible metadata and marker labels without claiming cryptographic verification.
- Very small images, Chrome-internal pages, CSS backgrounds, video frames, and canvases are outside this release. Ordinary webpage `<img>` elements are in scope.
- The maintainer benchmark is private. The checked-in harness and sealed public benchmark provide reproducible evidence, not a claim that our sample is identical to theirs.
