import { MODEL, RELEASE } from './constants.js';

export const CALIBRATION_DEBUG_RELAY_SCHEMA = 'realimage-calibration-debug-relay-v1';

const ANALYSIS_SCHEMA = 'realimage-analysis-debug-v3';
const READY_SCHEMA = 'realimage-runtime-ready-debug-v2';
const HARNESS_ORIGIN = 'http://127.0.0.1:4176';
const SESSION_ID = /^[a-f0-9]{32}$/;
const ASSET_ID = /^[a-f0-9]{20}$/;
const SHA256 = /^[a-f0-9]{64}$/;

/**
 * Creates an internal-only relay for the exact frozen calibration harness
 * shape. The trusted page URL came from Chrome's MessageSender in background
 * validation; the untrusted payload cannot choose it.
 */
export function createCalibrationDebugRelay({ pageUrl, source, runtimeReady, analysis }) {
  const context = calibrationHarnessContext(pageUrl, source);
  if (!context || !validDebugPair(runtimeReady, analysis, context)) return undefined;
  return {
    schema: CALIBRATION_DEBUG_RELAY_SCHEMA,
    runtimeReady,
    analysis
  };
}

/**
 * Removes relay material before a result enters content-script records, popup
 * state, or the DOM. Only a fresh, non-coalesced result from the same page,
 * source, and request is returned for isolated-world console logging.
 */
export function takeCalibrationDebugRelay(result, { pageUrl, source }) {
  if (!result || typeof result !== 'object') return { publicResult: result, relay: undefined };
  const { calibrationDebugRelay: candidate, ...publicResult } = result;
  const relay = validRelayForCaller(result, candidate, pageUrl, source)
    ? candidate
    : undefined;
  return { publicResult, relay };
}

/** Strip raw debug before the background responds to any non-harness caller. */
export function restrictCalibrationDebugRelay(result, { pageUrl, source }) {
  if (!result || typeof result !== 'object' || !Object.hasOwn(result, 'calibrationDebugRelay')) return result;
  if (validRelayForCaller(result, result.calibrationDebugRelay, pageUrl, source)) return result;
  const { calibrationDebugRelay: _privateRelay, ...publicResult } = result;
  return publicResult;
}

export function calibrationHarnessContext(pageUrl, source) {
  let page;
  let image;
  try {
    page = new URL(String(pageUrl));
    image = new URL(String(source));
  } catch {
    return undefined;
  }
  if (page.origin !== HARNESS_ORIGIN || image.origin !== HARNESS_ORIGIN) return undefined;
  if (page.username || page.password || page.hash || image.username || image.password || image.hash) return undefined;
  if (!['/', '/index.html'].includes(page.pathname)) return undefined;
  const expectedPageParameters = {
    view: 'all',
    label: 'all',
    delivery: 'direct',
    execution: 'sequential',
    limit: '30'
  };
  if (page.searchParams.size !== 6) return undefined;
  for (const [name, value] of Object.entries(expectedPageParameters)) {
    if (page.searchParams.get(name) !== value) return undefined;
  }
  const offset = page.searchParams.get('offset');
  if (!['0', '30'].includes(offset)) return undefined;

  if (image.searchParams.size !== 1) return undefined;
  const harnessRunId = image.searchParams.get('realimageRun')?.toLowerCase();
  if (!SESSION_ID.test(harnessRunId || '')) return undefined;
  const match = /^\/asset\/([a-f0-9]{20})\/image\.(?:png|jpe?g|webp|avif)$/i.exec(image.pathname);
  if (!match) return undefined;
  return {
    origin: HARNESS_ORIGIN,
    assetId: match[1].toLowerCase(),
    harnessRunId,
    offset: Number(offset)
  };
}

function validDebugPair(runtimeReady, analysis, context) {
  if (!context || runtimeReady?.schema !== READY_SCHEMA || analysis?.schema !== ANALYSIS_SCHEMA) return false;
  if (!SESSION_ID.test(runtimeReady.sessionId || '') || analysis.sessionId !== runtimeReady.sessionId) return false;
  if (analysis.identity?.extensionVersion !== RELEASE.version) return false;
  if (runtimeReady.identity?.extensionVersion !== RELEASE.version) return false;
  if (analysis.identity?.candidate?.id !== MODEL.candidateId) return false;
  if (analysis.identity?.candidate?.decisionPolicy !== 'model-only') return false;
  if (canonicalJson(analysis.identity) !== canonicalJson(runtimeReady.identity)) return false;
  if (canonicalJson(analysis.preprocessing) !== canonicalJson(runtimeReady.preprocessing)) return false;
  if (
    analysis.source?.kind !== 'loopback-harness-asset'
    || analysis.source.origin !== context.origin
    || analysis.source.assetId !== context.assetId
    || analysis.source.harnessRunId !== context.harnessRunId
  ) return false;
  if (!SHA256.test(analysis.input?.sha256 || '') || !Number.isSafeInteger(analysis.input?.byteLength) || analysis.input.byteLength <= 0) return false;
  if (analysis.scores?.rawThreshold !== MODEL.calibration.rawThreshold) return false;
  if (analysis.scores?.displayThreshold !== MODEL.calibration.displayThreshold) return false;
  if (
    !Number.isFinite(analysis.scores?.modelSigmoid)
    || !Number.isFinite(analysis.scores?.fused)
    || Math.abs(analysis.scores.fused - analysis.scores.modelSigmoid) > 1e-12
  ) return false;
  return typeof analysis.requestId === 'string' && analysis.requestId.length > 0;
}

function validRelayForCaller(result, candidate, pageUrl, source) {
  return candidate?.schema === CALIBRATION_DEBUG_RELAY_SCHEMA
    && result.cached === false
    && result.coalesced === false
    && result.requestId === candidate.analysis?.requestId
    && validDebugPair(candidate.runtimeReady, candidate.analysis, calibrationHarnessContext(pageUrl, source));
}

function canonicalJson(value) {
  return JSON.stringify(sortJson(value));
}

function sortJson(value) {
  if (Array.isArray(value)) return value.map(sortJson);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortJson(value[key])]));
  }
  return value;
}
