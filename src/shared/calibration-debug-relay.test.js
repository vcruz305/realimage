import { describe, expect, it } from 'vitest';
import { MODEL, RELEASE } from './constants.js';
import {
  createAnalysisDebugRecord,
  createRuntimeReadyDebugRecord,
  snapshotPreprocessor
} from '../offscreen/analysis-debug.js';
import {
  CALIBRATION_DEBUG_RELAY_SCHEMA,
  calibrationHarnessContext,
  createCalibrationDebugRelay,
  restrictCalibrationDebugRelay,
  takeCalibrationDebugRelay
} from './calibration-debug-relay.js';
import { sanitizeAnalyzePayload } from '../background/validation.js';

const pageUrl = 'http://127.0.0.1:4176/?view=all&label=all&offset=0&limit=30&delivery=direct&execution=sequential';
const harnessRunId = '2'.repeat(32);
const assetId = '0123456789abcdefabcd';
const source = `http://127.0.0.1:4176/asset/${assetId}/image.jpg?realimageRun=${harnessRunId}`;
const processor = {
  config: { image_processor_type: 'ViTImageProcessor' },
  do_resize: true,
  size: { shortest_edge: 440 },
  resample: 3,
  do_center_crop: true,
  crop_size: { width: 384, height: 384 },
  do_rescale: true,
  rescale_factor: 1 / 255,
  do_normalize: true,
  image_mean: [0.485, 0.456, 0.406],
  image_std: [0.229, 0.224, 0.225],
  do_convert_rgb: true,
  do_flip_channel_order: false
};
const modelState = {
  state: 'ready',
  backend: 'WebAssembly',
  error: null,
  runtime: { crossOriginIsolated: true, requestedThreads: 4, activeThreads: 4 }
};
const navigatorLike = {
  userAgent: 'Mozilla/5.0 Chrome/151.0.7922.138 Safari/537.36',
  userAgentData: { platform: 'Windows' },
  hardwareConcurrency: 16
};

describe('loopback-only calibration debug relay', () => {
  it('relays and then strips a fresh exact candidate record', () => {
    const { runtimeReady, analysis } = records();
    const calibrationDebugRelay = createCalibrationDebugRelay({ pageUrl, source, runtimeReady, analysis });
    expect(calibrationDebugRelay).toMatchObject({ schema: CALIBRATION_DEBUG_RELAY_SCHEMA, analysis: { requestId: 'pm-calibration-1' } });

    const { publicResult, relay } = takeCalibrationDebugRelay({
      ok: true,
      requestId: 'pm-calibration-1',
      cached: false,
      coalesced: false,
      score: 0.7,
      calibrationDebugRelay
    }, { pageUrl, source });
    expect(relay).toBe(calibrationDebugRelay);
    expect(publicResult).not.toHaveProperty('calibrationDebugRelay');
    expect(JSON.stringify(publicResult)).not.toContain('modelSigmoid');
  });

  it.each([
    ['ordinary remote page', 'https://example.test/?view=all&label=all&offset=0&limit=30&delivery=direct&execution=sequential', source],
    ['localhost alias', pageUrl.replace('127.0.0.1', 'localhost'), source.replace('127.0.0.1', 'localhost')],
    ['wrong port', pageUrl.replace('4176', '4175'), source.replace('4176', '4175')],
    ['missing run', pageUrl, source.split('?')[0]],
    ['wrong execution', pageUrl.replace('execution=sequential', 'execution=burst'), source],
    ['unexpected query', `${pageUrl}&extra=1`, source],
    ['page fragment', `${pageUrl}#spoof`, source],
    ['asset fragment', pageUrl, `${source}#spoof`]
  ])('does not relay on %s', (_label, candidatePage, candidateSource) => {
    const { runtimeReady, analysis } = records();
    expect(createCalibrationDebugRelay({ pageUrl: candidatePage, source: candidateSource, runtimeReady, analysis })).toBeUndefined();
  });

  it('strips but refuses cached, coalesced, mismatched, or non-model-only records', () => {
    const { runtimeReady, analysis } = records();
    const candidate = createCalibrationDebugRelay({ pageUrl, source, runtimeReady, analysis });
    for (const patch of [
      { cached: true, coalesced: false },
      { cached: false, coalesced: true },
      { cached: false, coalesced: false, requestId: 'different' }
    ]) {
      const output = takeCalibrationDebugRelay({ ok: true, requestId: 'pm-calibration-1', ...patch, calibrationDebugRelay: candidate }, { pageUrl, source });
      expect(output.relay).toBeUndefined();
      expect(output.publicResult).not.toHaveProperty('calibrationDebugRelay');
    }
    const changed = structuredClone(candidate);
    changed.analysis.scores.fused += 0.1;
    expect(takeCalibrationDebugRelay({
      ok: true,
      requestId: 'pm-calibration-1',
      cached: false,
      coalesced: false,
      calibrationDebugRelay: changed
    }, { pageUrl, source }).relay).toBeUndefined();
  });

  it('strips raw debug in background before any ordinary page receives it', () => {
    const { runtimeReady, analysis } = records();
    const calibrationDebugRelay = createCalibrationDebugRelay({ pageUrl, source, runtimeReady, analysis });
    const result = {
      ok: true,
      requestId: 'pm-calibration-1',
      cached: false,
      coalesced: false,
      score: 0.7,
      calibrationDebugRelay
    };
    expect(restrictCalibrationDebugRelay(result, { pageUrl, source })).toBe(result);
    const ordinary = restrictCalibrationDebugRelay(result, {
      pageUrl: 'http://127.0.0.1:4176/ordinary-page',
      source
    });
    expect(ordinary).not.toHaveProperty('calibrationDebugRelay');
    expect(JSON.stringify(ordinary)).not.toContain('modelSigmoid');
  });

  it('cannot be enabled by a pageUrl field spoofed in a page message', () => {
    const extensionId = 'abcdefghijklmnopabcdefghijklmnop';
    const sanitized = sanitizeAnalyzePayload({
      requestId: 'pm-spoof',
      source: 'https://example.test/image.jpg',
      naturalWidth: 768,
      naturalHeight: 768,
      pageUrl
    }, {
      id: extensionId,
      url: 'https://example.test/gallery',
      tab: { id: 7, url: 'https://example.test/gallery' }
    }, extensionId);
    expect(sanitized.pageUrl).toBe('https://example.test/gallery');

    const { runtimeReady, analysis } = records();
    const calibrationDebugRelay = createCalibrationDebugRelay({ pageUrl, source, runtimeReady, analysis });
    const exposed = restrictCalibrationDebugRelay({
      ok: true,
      requestId: 'pm-calibration-1',
      cached: false,
      coalesced: false,
      calibrationDebugRelay
    }, { pageUrl: sanitized.pageUrl, source });
    expect(exposed).not.toHaveProperty('calibrationDebugRelay');
  });

  it.each([
    ['candidate ID', (candidate) => { candidate.analysis.identity.candidate.id = 'forged'; }],
    ['decision policy', (candidate) => { candidate.analysis.identity.candidate.decisionPolicy = 'fusion'; }],
    ['raw threshold', (candidate) => { candidate.analysis.scores.rawThreshold = 0.5; }],
    ['input digest', (candidate) => { candidate.analysis.input.sha256 = 'not-a-digest'; }],
    ['asset binding', (candidate) => { candidate.analysis.source.assetId = 'f'.repeat(20); }],
    ['page-run binding', (candidate) => { candidate.analysis.source.harnessRunId = 'f'.repeat(32); }]
  ])('strips a relay with a forged %s', (_field, mutate) => {
    const { runtimeReady, analysis } = records();
    const candidate = createCalibrationDebugRelay({ pageUrl, source, runtimeReady, analysis });
    mutate(candidate);
    const output = takeCalibrationDebugRelay({
      ok: true,
      requestId: 'pm-calibration-1',
      cached: false,
      coalesced: false,
      calibrationDebugRelay: candidate
    }, { pageUrl, source });
    expect(output.relay).toBeUndefined();
    expect(output.publicResult).not.toHaveProperty('calibrationDebugRelay');
  });

  it('recognizes only the two frozen page offsets', () => {
    expect(calibrationHarnessContext(pageUrl, source)).toMatchObject({ assetId, harnessRunId, offset: 0 });
    expect(calibrationHarnessContext(pageUrl.replace('offset=0', 'offset=30'), source)).toMatchObject({ offset: 30 });
    expect(calibrationHarnessContext(pageUrl.replace('offset=0', 'offset=1'), source)).toBeUndefined();
  });
});

function records() {
  const sessionId = '1'.repeat(32);
  const extensionId = 'abcdefghijklmnopabcdefghijklmnop';
  const preprocessing = snapshotPreprocessor(processor);
  const runtimeReady = createRuntimeReadyDebugRecord({
    extensionVersion: RELEASE.version,
    extensionId,
    sessionId,
    processor,
    modelState,
    navigatorLike
  });
  const modelSigmoid = 0.8;
  const modelLogit = Math.log(modelSigmoid / (1 - modelSigmoid));
  const decisionScore = calibrate(modelSigmoid, MODEL.calibration.rawThreshold, MODEL.calibration.displayThreshold);
  const analysis = createAnalysisDebugRecord({
    extensionVersion: RELEASE.version,
    extensionId,
    sessionId,
    sequence: 1,
    preprocessing,
    modelState,
    requestId: 'pm-calibration-1',
    source,
    modelLogit,
    modelSigmoid,
    fusedScore: modelSigmoid,
    decisionScore,
    rawThreshold: MODEL.calibration.rawThreshold,
    displayThreshold: MODEL.calibration.displayThreshold,
    format: 'jpeg',
    width: 768,
    height: 768,
    inputSha256: 'a'.repeat(64),
    inputByteLength: 12345,
    timings: { fetchMs: 2, decodeMs: 3, modelWaitMs: 0, preprocessMs: 5, inferenceMs: 80, evidenceMs: 1, totalMs: 91 },
    navigatorLike
  });
  return { runtimeReady, analysis };
}

function calibrate(score, rawThreshold, displayThreshold) {
  const logit = (value) => Math.log(value / (1 - value));
  return 1 / (1 + Math.exp(-(logit(score) - logit(rawThreshold) + logit(displayThreshold))));
}
