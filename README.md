# RealImage

RealImage is a Manifest V3 Chrome extension that detects AI-generated images
locally, in your browser, as you browse. It runs a bundled ONNX vision model
on-device, shows a confidence score on top of each image, and can blur, hide,
or label images that cross the AI line. There is no inference server, no API
key, no telemetry, and no upload path — the model and runtime ship inside the
extension, and no network request is made for analysis, ever.

## What it does

A frozen, MIT-licensed ViT-S/16 backbone (`OwensLab/commfor-model-384`) with
a small replacement classifier head scores every sufficiently large image on
a page. The score is calibrated so **65% is the decision line**: at or above
it, an image is treated as likely AI-generated (default behavior: blur it).
Metadata (C2PA, IPTC digital-source-type declarations, generator strings in
PNG/JPEG chunks) is surfaced as extra context in the detail view but never
changes the model's decision.

## Features

- **Fully local inference.** The ~87 MB model and the ONNX runtime ship
  inside the extension; nothing is uploaded, and the build itself fails
  loudly if it would need to reach the network for a model file.
- **WebGPU with a verified WASM fallback.** Each offscreen document tries
  WebGPU first, checked against a stored reference logit; on a missing
  adapter, load failure, or self-test mismatch it falls back to
  multi-threaded WebAssembly automatically — including on machines with no
  usable GPU, which previously could fail outright (see Origins below).
- **Works on `http(s)`, `file://`, and local-network pages**, not just
  public websites — useful for checking images before you ever post them.
- **Blur, hide, or label** images that cross the line, adjustable per-page
  from the popup with no reload required.
- **Declared-AI detection.** High-confidence embedded declarations (C2PA,
  IPTC digital-source-type, Google's AI metadata) surface as a distinct
  "Declared AI-generated" badge, shown alongside the model's own score.
- **Page-caption evidence.** When a page's own caption or alt text
  explicitly declares an image AI-generated (e.g. "Generated using Stable
  Diffusion") but the model alone would score it low, that declaration is
  factored in — narrow phrase matching only, never a bare mention of a
  generator's name, to avoid false positives on pages that are merely
  discussing AI images.
- **Top-down scanning.** Images are analyzed in the order they appear on
  the page, starting from the top, rather than in arbitrary arrival order.
- **Live popup.** The popup keeps updating while a page is still being
  scanned, instead of freezing at whatever it saw the instant it opened.
- **Backend readiness check.** The Options page reports which backend is
  active (WebGPU or WASM) and the thread count once the model is warm.

## Build

Requirements: Node.js 20.19+ (or 22.12+) and npm 10+.

```bash
npm ci
npm run build:fresh
```

This verifies the bundled model weights and runtime, then builds `dist/`.
**No network request is made for the model or runtime** — both are committed
directly in this repository (`public/models/`, `node_modules/@huggingface/transformers`)
and only ever read from disk; `build:fresh` fails loudly rather than
substituting a download if a bundled file is missing.

Load it: open `chrome://extensions`, enable **Developer mode**, click **Load
unpacked**, and select the `dist/` directory. To test on local `file://`
pages, open the extension's card and enable **Allow access to file URLs**.

## Inference backend

Each offscreen document tries WebGPU first, verified with a one-time
self-test against a stored reference logit; on any mismatch, load failure,
or timeout it falls back to WebAssembly (12 threads when the page is
cross-origin isolated). Whichever backend is active is shown on the
**Options page** (popup → Settings → **Run local readiness check**), which
reports the backend name and thread count once the model is warm.

## Verification

```bash
npm test                    # unit tests
npm run test:fast-path      # inherited-baseline decision-parity regression
npm run test:policy         # no remote-inference references in source
npm run build                # model hash verification + MV3 build
```

`npm run check` runs the release-critical subset together; `npm run package`
runs `check` and produces a deterministic ZIP.

Run your own labeled folder:

```bash
npm run benchmark -- /path/to/folder   # expects ai/ and real/ subfolders
```

## Results

Source-disjoint holdout, this project's own measurement (not a third-party
benchmark), model `RealImage/broad-v1-modern-v1-1471e3ef`:

| Tier | Balanced accuracy | AI recall | Real recall |
| --- | ---: | ---: | ---: |
| Clean | 95.4% | 93.3% | 97.6% |
| Web (moderate degradation) | 94.4% | 91.1% | 97.7% |
| Hard (aggressive degradation) | 83.9% | 73.1% | 94.7% |

**Unseen-generator recall** (web tier, four generators with zero exposure in
train/calibration — `flux-krea`, `nano-banana`, `imagen4`, `recraft-v3`,
n=1,185 pooled): **88.0%**. Per-generator: `imagen4` 92.9% (n=660),
`recraft-v3` 88.7% (n=141), `nano-banana` 85.9% (n=192), `flux-krea` 72.9%
(n=192, the smallest and noisiest source).

Full methodology, the four-candidate comparison this model was selected
from, and per-source dataset citations are in `BENCHMARK.md` and
`MODEL_CARD.md`. INT8 quantization was tested and rejected (3.4–6.7% decision
flips vs. FP32, well over the 0.5% bar) — the extension ships the 87 MB FP32
model.

## Origins

RealImage began as a fork of the Proofmark extension, MIT-licensed, at
commit `ef986acb51c9ed6768d512bfc76174070940458b`. Since then it has:

- retrained the classifier head from scratch on a public-only, 21-source,
  4,650-image, multi-generator corpus with a leakage-corrected,
  source-disjoint holdout (see `MODEL_CARD.md`);
- added a WebGPU inference path with a self-test-verified WASM fallback,
  then fixed a real bug in that fallback: the bundled ONNX runtime caches
  its first session-creation attempt even on failure, so a machine with no
  usable GPU adapter could poison the WASM fallback with the same stale
  WebGPU error and fail outright. A cheap adapter preflight now skips the
  doomed WebGPU attempt entirely on such machines instead of triggering it;
- added `file://` and local-network page support, then fixed a real bug
  there too: Chrome's tainted-canvas rule blocks reading pixels back out of
  a `file://` image via canvas, independent of extension permissions, so
  local images always failed. `file://` sources now go through the same
  direct background-fetch path as `http(s)` instead of a canvas capture
  (with a same-origin private-network fetch policy that still blocks a
  public page from reaching loopback/private-network image targets);
- added narrow, page-caption-based declarative evidence (distinct from the
  file-metadata evidence system, which remains display-only and never
  changes the decision score) for the case where a page's own caption
  explicitly declares an image AI-generated but the model alone misses it;
- made image analysis order top-down instead of arrival order, and fixed a
  scroll-performance bug where badge placement's occlusion-avoidance check
  (up to 9 hit-tests per badge) re-ran on every scrolled frame, causing
  badges to visibly lag behind their image and snap into place once
  scrolling stopped;
- made the popup update live while a page keeps scanning, instead of
  freezing at whatever it saw the instant it was opened;
- removed a Facebook-linked-original fetch path that spoofed a User-Agent
  via `declarativeNetRequest`;
- fixed a ghost-badge bug on images that go hidden before analysis starts;
- hardened message validation, admission control, and caching against
  untrusted page-controlled input;
- removed an unaudited leftover model directory and abandoned
  hash-lock/qualification-ceremony tooling.

Fork-specific changes are not attributable to the upstream Proofmark
authors. See `NOTICE.md` for the full attribution chain.

## Limitations

- Hard-tier (aggressively cropped/recompressed/thumbnailed) AI recall is
  the weakest number in this family of models — 73.1% pooled, lower for
  some individual generators. A detector score is a lead, not a verdict.
- Two locally-authored AI sources (`flux-krea`, `nano-banana`) are
  non-redistributable and used for evaluation only, never for training.
- Two AI sources (`genimage-midjourney`, `genimage-sd15`, from
  `TheKernel01/Tiny-GenImage`) are CC-BY-NC-SA-4.0 — used for training,
  disclosed here since the license terms differ from the rest of the corpus.
- No detector is universally reliable, especially against novel generators,
  heavy edits, or adversarial inputs. Treat scores as signals for
  user-controlled triage, not a sole basis for consequential decisions.

## License

MIT — see `LICENSE`. RealImage is a fork of Proofmark; both copyright
notices are retained per the MIT license terms. See `NOTICE.md`.
