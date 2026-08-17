import { MESSAGE, MODEL, RELEASE } from '../shared/constants.js';
import { createCalibrationDebugRelay } from '../shared/calibration-debug-relay.js';
import { calibrateDecisionScore, collectPageContextEvidence, fuseEvidence, inspectEncodedImage } from '../analysis/forensics.js';
import { acquireImageInput } from './image-input.js';
import { SerialInferenceQueue } from './inference-queue.js';
import { extractModelLogit } from './model-output.js';
import { validateDecodedDimensions } from './source-policy.js';
import { createUnavailableResponse } from './unavailable-response.js';
import { createAnalysisDebugRecord, createRuntimeReadyDebugRecord, opaqueSourceIdentity } from './analysis-debug.js';
import { isTrustedOffscreenInternalSender } from './message-source.js';
import { selectBackendPlan, selectWasmThreadCount, WEBGPU_SELF_TEST_TIMEOUT_MS } from './runtime-config.js';
import { createSelfTestPixelData, SELF_TEST_IMAGE_SIZE, SELF_TEST_REFERENCE_LOGIT } from './self-test-fixture.js';

const requestedThreads = selectWasmThreadCount(navigator.hardwareConcurrency);
const activeThreads = globalThis.crossOriginIsolated ? requestedThreads : 1;
// Offscreen documents expose only a restricted subset of chrome.runtime.
// In particular, Chrome does not provide runtime.getManifest() here. Keep the
// version in the same frozen release constants used to generate the manifest.
const extensionVersion = RELEASE.version;
const extensionId = chrome.runtime.id;
const debugSessionId = randomHex(16);

let detectorPromise;
let transformersRuntimePromise;
let modelState = runtimeState('cold');
let readyDebugRecord;
let debugAnalysisSequence = 0;
const inferenceQueue = new SerialInferenceQueue({ maxQueued: 8 });

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender?.id !== chrome.runtime.id || !message || typeof message !== 'object') return false;
  if (message.type === MESSAGE.OFFSCREEN_PING) {
    if (!isTrustedOffscreenInternalSender(sender, chrome.runtime.id)) return false;
    sendResponse({ ok: true, receiver: 'offscreen', nonce: message.nonce });
    return false;
  }
  if (message.type === MESSAGE.OFFSCREEN_ANALYZE) {
    if (!isTrustedOffscreenInternalSender(sender, chrome.runtime.id)) return false;
    const requestId = typeof message.payload?.requestId === 'string' ? message.payload.requestId : undefined;
    inferenceQueue.run(() => analyzeImage(message.payload), message.payload?.priority).then(sendResponse).catch((error) => {
      sendResponse(createUnavailableResponse(error, requestId, humanizeError));
    });
    return true;
  }
  if (message.type === MESSAGE.WARM_MODEL) {
    getDetector().then(() => {
      console.debug('realimage:runtime-ready', readyDebugRecord);
      sendResponse({ ok: true, ...modelState, debug: readyDebugRecord });
    }).catch((error) => {
      sendResponse({ ok: false, ...modelState, error: humanizeError(error) });
    });
    return true;
  }
  if (message.type === MESSAGE.MODEL_STATUS) {
    sendResponse({ ok: true, ...modelState, debug: readyDebugRecord });
  }
});

async function getDetector() {
  if (!detectorPromise) {
    modelState = runtimeState('loading');
    detectorPromise = loadDetector().catch((error) => {
      detectorPromise = undefined;
      modelState = runtimeState('error', humanizeError(error));
      readyDebugRecord = undefined;
      throw error;
    });
  }
  return detectorPromise;
}

/**
 * Try WebGPU first, verified by one self-test inference against a stored
 * reference logit (self-test-fixture.js); fall back to the WASM path on any
 * load failure, self-test mismatch, or timeout. This runs once per offscreen
 * document — no retry loop, and no result is cached across sessions.
 */
async function loadDetector() {
  const { AutoImageProcessor, AutoModelForImageClassification, RawImage } = await getTransformersRuntime();
  const processor = await AutoImageProcessor.from_pretrained(MODEL.id);
  const webgpuProbe = await probeWebgpuBackend({ AutoModelForImageClassification, RawImage, processor });
  const plan = selectBackendPlan({
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
    referenceLogit: SELF_TEST_REFERENCE_LOGIT,
    webgpuSelfTest: webgpuProbe.outcome
  });
  const model = plan.backend === 'webgpu'
    ? webgpuProbe.model
    : await AutoModelForImageClassification.from_pretrained(MODEL.id, {
        dtype: MODEL.dtype,
        device: 'wasm'
      });
  modelState = runtimeState('ready', null, plan);
  readyDebugRecord = createRuntimeReadyDebugRecord({
    extensionVersion,
    extensionId,
    sessionId: debugSessionId,
    processor,
    modelState
  });
  console.debug('realimage:runtime-ready', readyDebugRecord);
  return { processor, model, RawImage, debug: readyDebugRecord };
}

async function probeWebgpuBackend({ AutoModelForImageClassification, RawImage, processor }) {
  // Bundled @huggingface/transformers (createInferenceSession) caches the
  // FIRST InferenceSession.create() call's promise in a module-level
  // wasmInitPromise to avoid double-initializing the shared ORT-web
  // runtime -- but it caches that promise even when it rejects. If the
  // WebGPU attempt below were allowed to fail inside a real session
  // creation call, every later session creation (including the WASM
  // fallback in loadDetector()) would immediately re-throw that same
  // stale WebGPU error instead of attempting WASM at all. A cheap, direct
  // adapter check here — with no ONNX Runtime session creation involved —
  // lets a missing/broken GPU adapter (the overwhelmingly common WebGPU
  // failure mode) skip the doomed attempt entirely, so it can never poison
  // that shared promise. This does not fully eliminate the underlying
  // library bug (a WebGPU session that fails for some other reason after
  // the adapter check passes would still poison the WASM fallback), but it
  // closes the failure mode this project has actually observed.
  if (!(await hasUsableWebgpuAdapter())) {
    return { outcome: { ok: false }, model: undefined };
  }
  try {
    const model = await withTimeout(
      AutoModelForImageClassification.from_pretrained(MODEL.id, { dtype: MODEL.dtype, device: 'webgpu' }),
      WEBGPU_SELF_TEST_TIMEOUT_MS
    );
    const selfTestImage = new RawImage(createSelfTestPixelData(SELF_TEST_IMAGE_SIZE), SELF_TEST_IMAGE_SIZE, SELF_TEST_IMAGE_SIZE, 3);
    const logit = await withTimeout(runSelfTestInference(model, processor, selfTestImage), WEBGPU_SELF_TEST_TIMEOUT_MS);
    return { outcome: { ok: true, logit }, model };
  } catch {
    // Expected on machines without usable WebGPU. Any failure here — an
    // unsupported device, a load error, a self-test logit outside tolerance
    // (checked by selectBackendPlan), or a timeout — falls back to the WASM
    // path below and never surfaces as an analysis error.
    return { outcome: { ok: false }, model: undefined };
  }
}

async function hasUsableWebgpuAdapter() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

async function runSelfTestInference(model, processor, image) {
  const inputs = await processor(image);
  const output = await model(inputs);
  return extractModelLogit(output);
}

function withTimeout(promise, timeoutMs) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('The WebGPU self-test timed out.')), timeoutMs);
    Promise.resolve(promise).then(
      (value) => { clearTimeout(timer); resolve(value); },
      (error) => { clearTimeout(timer); reject(error); }
    );
  });
}

async function analyzeImage({ source, fallbackDataUrl, requestId, pageUrl, pageContextText }) {
  const startedAt = performance.now();
  const transformersRuntimeTask = getTransformersRuntime();
  const detectorTask = getDetector();
  // Acquisition can fail before we await the parallel cold-start task. Attach
  // a handler now so a simultaneous model-load failure never becomes an
  // unhandled rejection; awaiting the original promise below still propagates it.
  void detectorTask.catch(() => {});
  const encodedInput = await acquireImageInput({ source, fallbackDataUrl, pageUrl });
  const fetchedAt = performance.now();
  const { RawImage } = await transformersRuntimeTask;
  const rawImage = await RawImage.fromBlob(encodedInput.blob);
  validateDecodedDimensions(rawImage.width, rawImage.height);
  const buffer = encodedInput.buffer;
  const decodedAt = performance.now();
  const detector = await detectorTask;
  const modelReadyAt = performance.now();
  const inputs = await detector.processor(rawImage);
  const preprocessedAt = performance.now();
  const output = await detector.model(inputs);
  const inferredAt = performance.now();
  const modelLogit = extractModelLogit(output);
  const modelAiScore = sigmoid(modelLogit);
  const encoded = inspectEncodedImage(buffer, encodedInput.mimeType);
  // Candidate qualification is deliberately model-only. Embedded file-metadata
  // evidence (encoded.evidence) is still surfaced for explanation, but it does
  // not alter the decision score -- this file's own dormant fuseEvidence()
  // export exists and is exercised in tests/benchmarking, but is intentionally
  // not called against encoded.evidence here.
  //
  // Page-adjacent caption text is different: it is a new, narrow, explicit
  // declarative signal (see collectPageContextEvidence -- a bare generator
  // name is not enough, an explicit generation verb is required) that the
  // model demonstrably misses on real search-result captions. It reuses the
  // same fuseEvidence() floor (score >= 0.985 once matched) rather than a
  // second scoring mechanism, but is applied narrowly to only this new
  // evidence source so the existing model-only policy for file metadata is
  // left exactly as it was.
  const pageContextEvidence = collectPageContextEvidence(pageContextText);
  const uncalibratedScore = pageContextEvidence.length > 0
    ? fuseEvidence(modelAiScore, { evidence: pageContextEvidence })
    : modelAiScore;
  const score = calibrateDecisionScore(
    uncalibratedScore,
    MODEL.calibration.rawThreshold,
    MODEL.calibration.displayThreshold
  );
  const finishedAt = performance.now();
  const timings = {
    fetchMs: Math.round(fetchedAt - startedAt),
    decodeMs: Math.round(decodedAt - fetchedAt),
    modelWaitMs: Math.round(modelReadyAt - decodedAt),
    preprocessMs: Math.round(preprocessedAt - modelReadyAt),
    inferenceMs: Math.round(inferredAt - preprocessedAt),
    evidenceMs: Math.round(finishedAt - inferredAt),
    totalMs: Math.round(finishedAt - startedAt)
  };
  const sourceIdentity = opaqueSourceIdentity(source);
  const inputSha256 = sourceIdentity.kind === 'loopback-harness-asset'
    ? await sha256Hex(buffer)
    : undefined;

  const analysisDebugRecord = createAnalysisDebugRecord({
    extensionVersion,
    extensionId,
    sessionId: debugSessionId,
    sequence: ++debugAnalysisSequence,
    preprocessing: detector.debug.preprocessing,
    modelState,
    requestId,
    source,
    modelLogit,
    modelSigmoid: modelAiScore,
    fusedScore: uncalibratedScore,
    decisionScore: score,
    rawThreshold: MODEL.calibration.rawThreshold,
    displayThreshold: MODEL.calibration.displayThreshold,
    format: encoded.format,
    width: rawImage.width,
    height: rawImage.height,
    inputSha256,
    inputByteLength: inputSha256 === undefined ? undefined : buffer.byteLength,
    timings
  });
  console.debug('realimage:analysis', analysisDebugRecord);
  const calibrationDebugRelay = createCalibrationDebugRelay({
    pageUrl,
    source,
    runtimeReady: detector.debug,
    analysis: analysisDebugRecord
  });

  return {
    ok: true,
    requestId,
    score,
    modelScore: modelAiScore,
    model: MODEL.id,
    backend: modelState.backend,
    runtime: modelState.runtime,
    format: encoded.format,
    dimensions: { width: rawImage.width, height: rawImage.height },
    evidence: [...encoded.evidence, ...encoded.watermarks, ...encoded.provenance, ...pageContextEvidence],
    watermarks: encoded.watermarks,
    provenance: encoded.provenance,
    timings,
    elapsedMs: timings.totalMs,
    ...(calibrationDebugRelay ? { calibrationDebugRelay } : {})
  };
}

function getTransformersRuntime() {
  if (!transformersRuntimePromise) {
    transformersRuntimePromise = import('@huggingface/transformers').then((runtime) => {
      const { env } = runtime;
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      // The model ships inside the extension. Chromium's Cache API rejects
      // chrome-extension:// request keys, so Transformers.js' default browser
      // cache only produces a warning after each local asset fetch.
      env.useBrowserCache = false;
      env.localModelPath = chrome.runtime.getURL('models/');
      env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('wasm/');
      env.backends.onnx.wasm.numThreads = activeThreads;
      return runtime;
    }).catch((error) => {
      transformersRuntimePromise = undefined;
      throw error;
    });
  }
  return transformersRuntimePromise;
}

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function humanizeError(error) {
  const message = error instanceof Error ? error.message : String(error);
  if (/model.*not found|no available backend/i.test(message)) {
    return `The local detector model could not be loaded. ${message}`;
  }
  return message;
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function randomHex(byteLength) {
  const bytes = crypto.getRandomValues(new Uint8Array(byteLength));
  return [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function runtimeState(state, error = null, plan) {
  return {
    state,
    backend: plan?.backend === 'webgpu' ? 'WebGPU' : 'WebAssembly',
    error,
    runtime: {
      crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
      requestedThreads: plan?.requestedThreads ?? requestedThreads,
      activeThreads: plan?.activeThreads ?? activeThreads
    }
  };
}
