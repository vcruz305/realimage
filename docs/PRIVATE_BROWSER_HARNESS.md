# Private installed-Chrome benchmark harness

This harness exercises the built RealImage extension through the real installed-Chrome path while keeping all source images under the ignored `.bench-data/` tree. It is diagnostic infrastructure, not a substitute for the frozen benchmark protocol in `MODERN_HOLDOUT_SPEC.md`.

## Safety boundary

`serve-private-browser-benchmark.mjs` has no downloader and no remote URL input. At startup it resolves the configured root, manifest, and every image through the filesystem and refuses anything outside this repository's real `.bench-data` directory. This includes `..` traversal and symlinks that resolve out of the tree.

The HTTP server:

- binds only to `127.0.0.1`;
- rejects non-loopback `Host` headers, non-`GET`/`HEAD` methods, and every route except the harness, opaque image assets, and same-origin redirects;
- never places local paths or original filenames in the page;
- verifies every supplied row SHA-256 (and byte count, when present) against the
  contained file before serving any page;
- applies a CSP that permits images and fetches only from the same origin, plus local `blob:`/`data:` image fixtures;
- sends no-cache, COOP, COEP, CORP, referrer, permissions, and MIME-sniffing protections.

The page does not copy, rewrite, upload, or embed the private images into tracked assets. It serves bytes from their existing ignored location on demand. Keep `.bench-data/` ignored, and do not publish browser screenshots that contain images whose rights do not permit publication.

## Start it

Build a deterministic matrix first if needed:

```powershell
node scripts\make-stress-matrix.mjs .bench-data\modern-v1 .bench-data\modern-v1-matrix --seed=323
```

Then start the fixture from the repository root:

```powershell
node scripts\serve-private-browser-benchmark.mjs `
  --root=.bench-data\modern-v1-matrix `
  --manifest=.bench-data\modern-v1-matrix\manifest.json `
  --port=4174
```

The defaults are those same paths and port, so this is equivalent:

```powershell
node scripts\serve-private-browser-benchmark.mjs
```

To exercise the repository's existing five-image private diagnostic without building a matrix, point the root at `.bench-data` and its local diagnostic manifest:

```powershell
node scripts\serve-private-browser-benchmark.mjs `
  --root=.bench-data `
  --manifest=.bench-data\private-diagnostic-manifest.json `
  --port=4174
```

Frozen matrix manifests use `rows[].relativePath`. Small private diagnostic manifests may instead use `samples[].file`; the harness derives opaque deterministic sample identities from the recorded SHA-256 (or, if absent, a hash of the contained relative path). A supplied SHA-256 is authoritative and a mismatch aborts startup. Both forms receive identical containment checks.

The printed URL is always loopback. Stop the server with Ctrl+C.

## Run it in the real Chrome profile

1. Build the extension and load the repository's `dist` directory as an unpacked extension in Chrome.
2. Open `http://127.0.0.1:4174/` in that same Chrome profile.
3. Leave the tab in the foreground. The harness scrolls one image into view at a time, waits for the extension's actual badge, opens its detail panel, and then advances.
4. Wait until the root element has `data-realimage-harness-terminal="true"`.

The default page runs 12 clean samples through ordinary same-origin image URLs. Use query parameters to make small, auditable batches:

```text
http://127.0.0.1:4174/?view=clean&label=all&offset=0&limit=12&delivery=direct
http://127.0.0.1:4174/?view=stress&label=ai&offset=0&limit=12&delivery=direct
http://127.0.0.1:4174/?view=clean&label=all&offset=0&limit=4&delivery=web-stress
http://127.0.0.1:4174/?view=clean&label=all&offset=0&limit=20&delivery=direct&execution=burst
http://127.0.0.1:4174/?view=clean&label=all&offset=0&limit=60&delivery=direct&execution=gallery
```

`execution` controls how the selected cases enter the real extension path:

- `sequential` is the unchanged default. It scrolls to, loads, and waits for one
  case at a time.
- `burst` uses a compact grid, attaches every selected source together, and
  waits for all cases collectively. The 20-case example deliberately exceeds
  the offscreen document's one-active/eight-waiting capacity and verifies that
  page-side backpressure eventually produces a terminal result for every image.
- `gallery` attaches every selected source up front, then moves through a
  compact static gallery. The 60-case example crosses the default 40-live-record
  window without removing DOM nodes, exercising terminal off-viewport eviction,
  treatment cleanup, and cached re-entry behavior.

Supported `delivery` values are:

- `direct`: ordinary same-origin `img.src`;
- `query`: same bytes with a cache-key query string;
- `srcset`: responsive `srcset`/`sizes` selection;
- `delayed`: the source is attached after a deterministic delay;
- `blob`: a same-origin fetch is converted to a browser `blob:` URL;
- `shadow`: the image lives in an open shadow root;
- `redirect`: a same-origin HTTP redirect leads to the opaque asset route;
- `source-swap`: a one-pixel data fixture is replaced with the real same-origin source;
- `web-stress`: runs all eight delivery forms for each selected source row.

`web-stress` accepts at most four source rows per page, producing at most 32
cases. Sequential and non-direct runs accept at most 32 source rows. Direct
`burst` and `gallery` runs accept at most 60 source rows; the intended release
gates are 20 and 60 respectively. `offset` paginates the already
sample-ID-sorted manifest rows. `view` may be `all` or a view present in the
manifest, and `label` may be `all`, `ai`, or `real`.

Optional `caseTimeoutMs` and `totalTimeoutMs` query parameters are bounded to 5–120 seconds and 10–600 seconds respectively. A missing, errored, or timed-out result remains explicit in terminal state; it is never silently dropped.

## Inspect terminal DOM state

The page exposes the same state in three places:

- `window.__REALIMAGE_HARNESS_STATE__` for direct Chrome evaluation;
- JSON text in `#realimage-harness-state`;
- `data-realimage-harness-status` and `data-realimage-harness-terminal` on `<html>` for polling.

Useful DevTools expressions:

```js
document.documentElement.dataset.realimageHarnessTerminal
window.__REALIMAGE_HARNESS_STATE__
JSON.parse(document.querySelector('#realimage-harness-state').textContent)
```

Terminal state uses schema `realimage-private-browser-harness-state-v1`. Each
page also contains its pinned manifest SHA-256, loopback origin, and random
32-hex `harnessRunId`; every image URL carries that run ID to defeat stale
worker/HTTP cache reuse. Every case contains deterministic `caseId`, `sampleId`,
`domId`, expected label, view, transform, delivery, and status. Completed cases
also contain the extension's predicted class, displayed score, exact decision
threshold taken from the badge, correctness, badge text, detail-panel evidence,
total time, inference time, and runtime summary. The `model` object records the
numeric displayed model score and model-only verdict independently of badge
color or copy. `declaration` is either `null` or separately records the exact
allowlisted metadata-declaration type, summary, detail copy, and sources. Totals
include incorrect predictions, extension errors, image errors, and timeouts.

Burst and gallery completion is fail-closed: when the extension is missing,
each selected case is counted as an extension error; when a per-case or global
deadline expires, every unresolved case is explicitly marked timed out before
the harness becomes terminal. A terminal page therefore never hides queued or
unfinished burst/gallery cases outside the totals.

The DOM/model score is the badge's rounded display percentage, not the detector's full-precision score. The harness fails closed instead of inferring a model verdict from badge text or treating an amber metadata declaration as a model-positive result. For exact raw scores and the stage-by-stage timing object, inspect Chrome's extension/offscreen debug entries named `proofmark:analysis`; their opaque asset ID, page-run ID, origin, and input digest can be joined to the harness terminal case. Never describe the displayed score as a calibrated probability.

The separately frozen candidate gate on `127.0.0.1:4176` also mirrors the
canonical runtime-ready v2 and analysis v3 objects into the isolated
content-script console under `realimage:calibration-runtime-ready` and
`realimage:calibration-analysis`. This narrow relay exists only for the exact
six sequential 30-row URLs in the pre-Chrome capture plan, emits JSON strings
for tab-debug collection, and is stripped for ordinary pages. It does not
publish raw debug through this harness's DOM state or any storage channel.

For evidence intended to support a competition claim, use this harness to verify the installed browser path and failure behavior, then preserve the frozen manifest, extension/model digests, Chrome and platform versions, raw debug output, and the aggregate result produced by the preregistered benchmark scorer. The harness's expected-label captions are for operator auditing and are not model inputs.

For full-precision candidate parity, runtime-identity checks, and preregistered
latency gates, follow [`INSTALLED_CHROME_PARITY_GATE.md`](INSTALLED_CHROME_PARITY_GATE.md).
The raw internal debug channel is deliberately separate from this page's
rounded DOM state.

## Focused tests

Run only this harness's tests:

```powershell
npx vitest run tests\private-browser-benchmark.test.js
```

The tests cover containment, digest rejection, URL rejection, deterministic case IDs, random cache-busting run IDs, all delivery variants, path redaction, CSP, loopback-host enforcement, opaque asset serving, same-origin redirects, traversal rejection, method rejection, separate model/declaration collection, and malformed result-contract rejection.
