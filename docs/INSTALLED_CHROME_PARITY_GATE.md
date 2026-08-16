# Frozen installed-Chrome parity and latency gate

This gate answers one narrow question: does an already-frozen candidate produce
the same threshold decisions in the packaged Chrome route that it produced in
the offline calibration route, with acceptable numeric drift and latency? It
does **not** select a model, tune a threshold, open qualification data, or
support a public accuracy claim.

## Safety and scope

The auditor reads only paths explicitly supplied on its command line. Every
input and output must resolve inside the repository's ignored `.bench-data/`
tree. It rejects both lexical and symlink escapes, any `..` row path, remote row
URLs, and visible or dot-prefixed path components beginning with `qualification`,
`private-diagnostic`, or `private_diagnostic`. It never resolves or opens an
image path from a manifest; the existing loopback server remains the only
component that serves calibration image bytes.

The gate requires exactly 180 distinct observations joined to complete terminal
harness pages. Missing/error/timed-out rows, extra rows, duplicates, sequence
gaps, a changed extension session, an unknown opaque asset ID, a mismatched
input SHA-256, malformed numerics, an unjoinable page run, or any
candidate/config/runtime identity mismatch fails closed.

## What the extension records

The offscreen document emits two internal-only console events. Raw values are
not added to the web page DOM, content-script result history, popup, or public
message surface.

- `proofmark:runtime-ready` carries schema
  `realimage-runtime-ready-debug-v2`. It records a random offscreen-session ID;
  the candidate/finalist/protocol/tooling/policy pins; the extension version; model,
  config, preprocessor-config, WASM, and loader identities; the preprocessing
  values actually instantiated by Transformers.js; backend/isolation/thread
  readiness; and browser/platform identity.
- `proofmark:analysis` carries schema `realimage-analysis-debug-v3`. It adds a
  strictly increasing success sequence, opaque loopback `assetId` and random
  harness-page run ID, the acquired input byte count and SHA-256, the exact
  model logit and full-precision sigmoid,
  fused and unrounded display scores, both thresholds and decisions, decoded
  format/dimensions, and stage-by-stage timings.

The source URL is never logged. Only the loopback origin, a 20-hex harness asset
ID, and a 32-hex page-run ID are retained for loopback `/asset/<id>/...` and
`/redirect/<id>` routes; ordinary web URLs are reported only as `remote-http`.
The harness verifies every manifest-provided image SHA-256 before serving it,
and a new page-run query key prevents the extension worker or HTTP cache from
reusing a prior result for the same sample ID.

For the frozen FP32 Chrome gate only, those same canonical records pass through
an internal relay to the isolated content-script console so Chrome tab debug
logs can collect them without attaching to the offscreen target. The relay is
fail-closed and exists only when all of these facts agree: exact candidate and
model-only identities, raw threshold `0.001812748527039188`, display threshold
`0.65`, fused score equal to the raw model sigmoid within `1e-12`, a fresh and
non-coalesced request, the exact `http://127.0.0.1:4176` origin, one of the six
pinned 30-row sequential page shapes, and an opaque asset URL carrying that
page's 32-hex run ID. Background and content layers independently strip the
relay for every other caller. It never uses DOM events, DOM attributes, page
globals, or browser/page storage.

The isolated-world log labels are `realimage:calibration-runtime-ready` and
`realimage:calibration-analysis`; the second console argument is an exact JSON
string. Each new harness page logs one identical runtime-ready v2 record and 30
analysis v3 records. Collect a page before navigating, verify the six readiness
records are canonically identical, and retain one in the normalized capture.
The original `proofmark:*` offscreen records remain the canonical source and
are unchanged by this capture-only relay.

`npm run model:verify` checksum-verifies the model weight, both model JSON
files, and both local runtime files before every build. Changing any candidate
or dependency therefore requires updating its frozen identities; copying new
bytes under an old ID makes the build fail.

## Frozen reference contract

Before installed-Chrome inference, create one JSON document with schema
`realimage-installed-chrome-parity-reference-v1`. The candidate lane owns this
artifact and must freeze/hash it with its finalist lock. Required fields are:

```json
{
  "schema": "realimage-installed-chrome-parity-reference-v1",
  "frozenBeforeChrome": true,
  "qualificationRead": false,
  "candidateId": "must equal identity.candidate.id",
  "expectedObservations": 180,
  "manifests": [
    { "sha256": "<64 lowercase hex>", "rows": 60 }
  ],
  "identity": {
    "extensionVersion": "0.2.0",
    "candidate": {
      "id": "immutable candidate identifier",
      "finalistLockSha256": "<sha256>",
      "protocolSha256": "<sha256>",
      "toolingSha256": "<sha256>",
      "decisionPolicy": "model-only",
      "qualificationRead": false
    },
    "model": {
      "id": "...",
      "upstreamId": "...",
      "revision": "...",
      "dtype": "...",
      "inputSize": 384,
      "resizeSize": 440,
      "outputActivation": "sigmoid",
      "weightFile": "...",
      "sourceWeightSha256": "<sha256>",
      "sourceModelSha256": "<sha256>",
      "weightSha256": "<sha256>",
      "configSha256": "<sha256>",
      "preprocessorConfigSha256": "<sha256>"
    },
    "runtimeArtifacts": {
      "transformersJsVersion": "...",
      "onnxRuntimeWebVersion": "...",
      "wasmModule": "...",
      "wasmSha256": "<sha256>",
      "wasmLoader": "...",
      "wasmLoaderSha256": "<sha256>"
    }
  },
  "preprocessing": {
    "processorType": "ViTImageProcessor",
    "doResize": true,
    "shortestEdge": 440,
    "resample": 3,
    "doCenterCrop": true,
    "cropSize": { "width": 384, "height": 384 },
    "doRescale": true,
    "rescaleFactor": 0.00392156862745098,
    "doNormalize": true,
    "imageMean": [0.485, 0.456, 0.406],
    "imageStd": [0.229, 0.224, 0.225],
    "doConvertRgb": true,
    "doFlipChannelOrder": false
  },
  "rawThreshold": 0.001812748527039188,
  "displayThreshold": 0.65,
  "runtimeRequirements": {
    "backend": "WebAssembly",
    "crossOriginIsolated": true,
    "minimumActiveThreads": 1,
    "minimumChromeMajor": 148
  },
  "gates": {
    "maxSigmoidError": 0.02,
    "zeroThresholdFlips": true,
    "requireModelOnlyDecision": true,
    "latency": {
      "excludeFirst": 1,
      "maximumMedianInferenceMs": 120,
      "maximumP95InferenceMs": 150,
      "maximumSingleInferenceMs": 250,
      "maximumMedianTotalMs": 180
    },
    "decisions": [
      {
        "id": "jpeg75-docci",
        "match": { "transformId": "square-jpeg75-768", "domain": "DOCCI", "label": "real" },
        "expectedRows": 30,
        "minimumCorrect": 27
      }
    ]
  },
  "rows": [
    { "sampleId": "...", "rawSigmoidScore": 0.123, "predicted": "real" }
  ]
}
```

Every manifest row must contain the exact acquired-file `sha256`. The
`manifests` array may contain one 180-row manifest or several pinned
manifests whose counts total 180. `rows` must be the exact union of their
distinct sample IDs. `predicted` must agree with `rawSigmoidScore >=
rawThreshold`.

Decision gates must match exactly `label`, `transformId`, and `domain`; they
must contain exactly one gate for every such manifest group and partition all
180 observations. The auditor refuses weaker minima than 90% real recall or
70% AI recall. For the clean-head calibration layout, preregister three gates per transform:
DOCCI `27/30`, Nano Banana `11/15`, and FLUX Krea `11/15`.

All four latency maxima shown above are mandatory, may only be stricter, and
exactly the first event is excluded. The report uses nearest-rank percentiles
over the contiguous extension sequence. Choose stricter limits before Chrome
scores are inspected; changing them after a run retires the candidate under
this protocol.

## Installed Chrome run

1. Finish and hash the reference, candidate model, configs, runtime assets, and
   unpacked extension tree. Run `npm run check`.
2. In Chrome, manually reload the unpacked extension from the freshly built
   `dist` directory. Chrome's internal extension-management page must remain a
   manual operator step.
3. From the first pinned harness tab, preserve the exact JSON string following
   `realimage:calibration-runtime-ready`. Each of the six pages emits one copy;
   require all six to be canonically identical and place exactly one parsed
   object in the normalized capture.
4. Serve only the pinned development/calibration manifests with
   `serve-private-browser-benchmark.mjs`. Use `delivery=direct`. Sequential
   batches of at most 30 are required for batch-one latency; cover every
   manifest row once with non-overlapping `offset` values. Do not reuse a URL
   for different bytes or candidates.
5. Wait for every page to reach `terminal=true`, `status=complete`, with every
   case `complete` and zero image/extension errors or timeouts. Preserve each
   untouched `window.__REALIMAGE_HARNESS_STATE__` value.
6. After each page becomes terminal, collect exactly 30 JSON strings following
   `realimage:calibration-analysis` from that tab's isolated content-script
   console before navigating. Close unrelated image tabs: the combined 180
   analysis sequence numbers must be contiguous. Do not transcribe the rounded
   badge score. The frozen capture plan contains the six exact URLs and server
   commands.
7. Normalize the untouched payload objects into one ignored capture file:

```json
{
  "schema": "realimage-installed-chrome-capture-v1",
  "qualificationRead": false,
  "browser": {
    "name": "Chrome",
    "version": "151.0.7922.138",
    "platform": "Windows"
  },
  "runtimeReady": { "schema": "realimage-runtime-ready-debug-v2" },
  "harnessPages": [
    { "schema": "realimage-private-browser-harness-state-v1", "terminal": true }
  ],
  "analyses": [
    { "schema": "realimage-analysis-debug-v3" }
  ]
}
```

`browser.version` must match the major version in the runtime-ready user agent
(minor/build components may be reduced to zero by Chrome);
`browser.platform` must exactly match its runtime platform. The readiness and
analysis placeholders above stand for the complete logged objects, not partial
records. Only the two exact `realimage:calibration-*` isolated-world relay
labels are capture inputs; arbitrary page-console messages are not. The session ID, contiguous
sequence, page-run ID, terminal state, and input digest make accidental mixing
or page spoofing fail, but they do not cryptographically authenticate a capture
fabricated by an operator with filesystem access.

## Audit

Run the focused tests:

```powershell
npm run test:chrome-parity
```

Run the gate, repeating `--manifest` for every pinned calibration manifest:

```powershell
npm run gate:chrome-parity -- `
  --manifest=.bench-data\candidate\chrome-gate\jpeg75.manifest.json `
  --manifest=.bench-data\candidate\chrome-gate\webp72.manifest.json `
  --manifest=.bench-data\candidate\chrome-gate\thumbnail.manifest.json `
  --reference=.bench-data\candidate\chrome-gate\frozen-reference.json `
  --capture=.bench-data\candidate\chrome-gate\chrome-capture.json `
  --output=.bench-data\candidate\chrome-gate\parity-report.json
```

The output path's parent must already exist and the file must not exist. A gate
failure still writes a report and exits nonzero. Structurally unsafe input exits
nonzero without scoring. A passing report has
`realimage-installed-chrome-parity-report-v1`, `eligible: true`, zero failures,
zero threshold flips, the exact maximum/mean sigmoid errors, preregistered
decision-gate counts, and p50/p90/p95/max/mean inference and total latency.
It also pins the exact reference and capture file SHA-256 values so the later
joint execution lock can reject a substituted report input.

A pass authorizes only the next already-preregistered stage. It is not
qualification authorization by itself.
