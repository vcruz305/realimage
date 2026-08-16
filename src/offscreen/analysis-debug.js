import { LOCAL_RUNTIME, MODEL } from '../shared/constants.js';

export const ANALYSIS_DEBUG_SCHEMA = 'realimage-analysis-debug-v3';
export const RUNTIME_READY_DEBUG_SCHEMA = 'realimage-runtime-ready-debug-v2';

const SHA256 = /^[a-f0-9]{64}$/;
const DEBUG_SESSION_ID = /^[a-f0-9]{32}$/;
const HARNESS_RUN_ID = /^[a-f0-9]{32}$/;

/**
 * Build the immutable identity used by the installed-Chrome parity gate. The
 * values are release constants whose corresponding files are checksum-verified
 * before a build. Dynamic browser/profile details deliberately live outside
 * this identity so an unpacked extension ID cannot change the candidate ID.
 */
export function detectorIdentity(extensionVersion) {
  return {
    extensionVersion: requiredString(extensionVersion, 'extension version'),
    candidate: {
      id: requiredString(MODEL.candidateId, 'candidate ID'),
      decisionPolicy: exactString(MODEL.decisionPolicy, 'model-only', 'decision policy')
    },
    model: {
      id: MODEL.id,
      upstreamId: MODEL.upstreamId,
      revision: MODEL.revision,
      dtype: MODEL.dtype,
      inputSize: MODEL.inputSize,
      resizeSize: MODEL.resizeSize,
      outputActivation: MODEL.outputActivation,
      weightFile: MODEL.weightFile,
      sourceWeightSha256: MODEL.sourceWeightSha256,
      sourceModelSha256: MODEL.sourceModelSha256,
      weightSha256: MODEL.weightSha256,
      configSha256: MODEL.configSha256,
      preprocessorConfigSha256: MODEL.preprocessorConfigSha256
    },
    runtimeArtifacts: { ...LOCAL_RUNTIME }
  };
}

/**
 * Record the processor settings actually instantiated by Transformers.js. A
 * config digest alone proves which bytes shipped; this snapshot additionally
 * proves that the browser runtime interpreted the important resize/crop/color
 * and normalization fields as preregistered.
 */
export function snapshotPreprocessor(processor) {
  // Transformers.js image processors are callable instances: their runtime
  // type is "function", with the interpreted preprocessing configuration
  // stored on that function object. Accept both that real runtime shape and a
  // plain object used by focused validators, while still rejecting primitives.
  if (!processor || !['object', 'function'].includes(typeof processor)) {
    throw new TypeError('A loaded image processor is required.');
  }
  return {
    processorType: requiredString(processor.config?.image_processor_type, 'processor type'),
    doResize: requiredBoolean(processor.do_resize, 'do_resize'),
    shortestEdge: positiveInteger(processor.size?.shortest_edge, 'size.shortest_edge'),
    resample: nonNegativeInteger(processor.resample, 'resample'),
    doCenterCrop: requiredBoolean(processor.do_center_crop, 'do_center_crop'),
    cropSize: {
      width: positiveInteger(processor.crop_size?.width, 'crop_size.width'),
      height: positiveInteger(processor.crop_size?.height, 'crop_size.height')
    },
    doRescale: requiredBoolean(processor.do_rescale, 'do_rescale'),
    rescaleFactor: finiteNumber(processor.rescale_factor, 'rescale_factor'),
    doNormalize: requiredBoolean(processor.do_normalize, 'do_normalize'),
    imageMean: finiteVector(processor.image_mean, 3, 'image_mean'),
    imageStd: finiteVector(processor.image_std, 3, 'image_std'),
    doConvertRgb: requiredBoolean(processor.do_convert_rgb, 'do_convert_rgb'),
    doFlipChannelOrder: requiredBoolean(processor.do_flip_channel_order, 'do_flip_channel_order')
  };
}

export function createRuntimeReadyDebugRecord({
  extensionVersion,
  extensionId,
  sessionId,
  processor,
  modelState,
  navigatorLike = globalThis.navigator
}) {
  const runtime = snapshotReadyRuntime(modelState);
  return {
    schema: RUNTIME_READY_DEBUG_SCHEMA,
    sessionId: debugSessionId(sessionId, 'debug session ID'),
    identity: detectorIdentity(extensionVersion),
    preprocessing: snapshotPreprocessor(processor),
    runtime,
    environment: snapshotEnvironment(navigatorLike, extensionId)
  };
}

export function createAnalysisDebugRecord({
  extensionVersion,
  extensionId,
  sessionId,
  sequence,
  preprocessing,
  modelState,
  requestId,
  source,
  modelLogit,
  modelSigmoid,
  fusedScore,
  decisionScore,
  rawThreshold,
  displayThreshold,
  format,
  width,
  height,
  inputSha256,
  inputByteLength,
  timings,
  navigatorLike = globalThis.navigator
}) {
  const logit = finiteNumber(modelLogit, 'model logit');
  const sigmoid = probability(modelSigmoid, 'model sigmoid score');
  const stable = stableSigmoid(logit);
  if (Math.abs(stable - sigmoid) > 1e-12) {
    throw new RangeError('The logged sigmoid score does not match the logged model logit.');
  }
  const raw = openProbability(rawThreshold, 'raw threshold');
  const display = openProbability(displayThreshold, 'display threshold');
  const fused = probability(fusedScore, 'fused score');
  const decision = probability(decisionScore, 'decision score');
  const expectedDecision = calibrateForDisplay(fused, raw, display);
  if (Math.abs(expectedDecision - decision) > 1e-12) {
    throw new RangeError('The logged decision score does not match the fused score calibration.');
  }
  const sourceIdentity = opaqueSourceIdentity(source);
  const input = {
    format: requiredString(format, 'image format').toLowerCase(),
    width: positiveInteger(width, 'image width'),
    height: positiveInteger(height, 'image height')
  };
  if (inputSha256 !== undefined || inputByteLength !== undefined) {
    input.sha256 = sha256String(inputSha256, 'input SHA-256');
    input.byteLength = positiveInteger(inputByteLength, 'input byte length');
  }
  return {
    schema: ANALYSIS_DEBUG_SCHEMA,
    sessionId: debugSessionId(sessionId, 'debug session ID'),
    sequence: positiveInteger(sequence, 'analysis sequence'),
    requestId: requiredString(requestId, 'request ID'),
    source: sourceIdentity,
    identity: detectorIdentity(extensionVersion),
    preprocessing: clonePreprocessing(preprocessing),
    runtime: snapshotReadyRuntime(modelState),
    environment: snapshotEnvironment(navigatorLike, extensionId),
    input,
    scores: {
      modelLogit: logit,
      modelSigmoid: sigmoid,
      fused,
      decision,
      rawThreshold: raw,
      displayThreshold: display,
      rawPredicted: sigmoid >= raw ? 'ai' : 'real',
      displayedPredicted: decision >= display ? 'ai' : 'real'
    },
    timings: snapshotTimings(timings)
  };
}

export function opaqueSourceIdentity(source) {
  let url;
  try {
    url = new URL(requiredString(source, 'image source'));
  } catch {
    return { kind: 'unsupported' };
  }
  if (url.protocol === 'blob:') return { kind: 'blob' };
  if (url.protocol === 'data:') return { kind: 'data' };
  if (!['http:', 'https:'].includes(url.protocol)) return { kind: 'unsupported' };

  const loopback = ['127.0.0.1', 'localhost'].includes(url.hostname.toLowerCase());
  const match = loopback && /\/(?:asset|redirect)\/([a-f0-9]{20})(?:\/|$)/i.exec(url.pathname);
  const harnessRunId = url.searchParams.get('realimageRun')?.toLowerCase();
  return match
    ? {
        kind: 'loopback-harness-asset',
        assetId: match[1].toLowerCase(),
        harnessRunId: HARNESS_RUN_ID.test(harnessRunId || '') ? harnessRunId : undefined,
        origin: url.origin
      }
    : { kind: 'remote-http' };
}

export function stableSigmoid(value) {
  const number = finiteNumber(value, 'sigmoid input');
  if (number >= 0) return 1 / (1 + Math.exp(-number));
  const exponential = Math.exp(number);
  return exponential / (1 + exponential);
}

function calibrateForDisplay(score, rawThreshold, displayThreshold) {
  if (score === 0 || score === 1) return score;
  const logit = (value) => Math.log(value / (1 - value));
  return stableSigmoid(logit(score) - logit(rawThreshold) + logit(displayThreshold));
}

function snapshotReadyRuntime(modelState) {
  if (!modelState || modelState.state !== 'ready' || modelState.error != null) {
    throw new Error('The model runtime is not ready.');
  }
  const runtime = modelState.runtime;
  if (!runtime || typeof runtime !== 'object') throw new TypeError('Runtime details are unavailable.');
  return {
    state: 'ready',
    backend: requiredString(modelState.backend, 'runtime backend'),
    crossOriginIsolated: requiredBoolean(runtime.crossOriginIsolated, 'crossOriginIsolated'),
    requestedThreads: positiveInteger(runtime.requestedThreads, 'requested threads'),
    activeThreads: positiveInteger(runtime.activeThreads, 'active threads')
  };
}

function snapshotEnvironment(navigatorLike, extensionId) {
  return {
    extensionId: requiredString(extensionId, 'extension ID'),
    userAgent: requiredString(navigatorLike?.userAgent, 'browser user agent'),
    platform: requiredString(navigatorLike?.userAgentData?.platform || navigatorLike?.platform, 'browser platform'),
    hardwareConcurrency: positiveInteger(navigatorLike?.hardwareConcurrency, 'hardware concurrency')
  };
}

function snapshotTimings(timings) {
  if (!timings || typeof timings !== 'object') throw new TypeError('Timing details are required.');
  const output = {};
  for (const field of ['fetchMs', 'decodeMs', 'modelWaitMs', 'preprocessMs', 'inferenceMs', 'evidenceMs', 'totalMs']) {
    const value = finiteNumber(timings[field], `timings.${field}`);
    if (value < 0) throw new RangeError(`timings.${field} must be non-negative.`);
    output[field] = value;
  }
  return output;
}

function clonePreprocessing(value) {
  if (!value || typeof value !== 'object') throw new TypeError('Observed preprocessing details are required.');
  return JSON.parse(JSON.stringify(value));
}

function probability(value, label) {
  const number = finiteNumber(value, label);
  if (number < 0 || number > 1) throw new RangeError(`${label} must be in [0, 1].`);
  return number;
}

function openProbability(value, label) {
  const number = probability(value, label);
  if (number === 0 || number === 1) throw new RangeError(`${label} must be between zero and one.`);
  return number;
}

function finiteVector(value, length, label) {
  if (!Array.isArray(value) || value.length !== length) throw new TypeError(`${label} must contain ${length} values.`);
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`));
}

function finiteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new TypeError(`${label} must be finite.`);
  return value;
}

function requiredString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') throw new TypeError(`${label} must be a non-empty string.`);
  return value.trim();
}

function requiredBoolean(value, label) {
  if (typeof value !== 'boolean') throw new TypeError(`${label} must be boolean.`);
  return value;
}

function exactString(value, expected, label) {
  const observed = requiredString(value, label);
  if (observed !== expected) throw new RangeError(`${label} must be ${expected}.`);
  return observed;
}

function sha256String(value, label) {
  const observed = requiredString(value, label).toLowerCase();
  if (!SHA256.test(observed)) throw new TypeError(`${label} must be a lowercase SHA-256.`);
  return observed;
}

function debugSessionId(value, label) {
  const observed = requiredString(value, label).toLowerCase();
  if (!DEBUG_SESSION_ID.test(observed)) throw new TypeError(`${label} must be 32 lowercase hexadecimal characters.`);
  return observed;
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${label} must be a positive integer.`);
  return value;
}

function nonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`${label} must be a non-negative integer.`);
  return value;
}
