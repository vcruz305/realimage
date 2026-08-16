import { describe, expect, it } from 'vitest';
import {
  ANALYSIS_DEBUG_SCHEMA,
  RUNTIME_READY_DEBUG_SCHEMA,
  createAnalysisDebugRecord,
  createRuntimeReadyDebugRecord,
  opaqueSourceIdentity,
  snapshotPreprocessor
} from './analysis-debug.js';

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
const extensionId = 'abcdefghijklmnopabcdefghijklmnop';
const sessionId = '1'.repeat(32);
const harnessRunId = '2'.repeat(32);

describe('internal installed-Chrome debug records', () => {
  it('requires the optional input digest and byte length as an atomic pair', () => {
    const common = {
      extensionVersion: '0.1.0', extensionId, sessionId, sequence: 1,
      preprocessing: snapshotPreprocessor(processor), modelState,
      requestId: 'pm-ordinary-remote', source: 'https://example.com/image.jpg',
      modelLogit: 0, modelSigmoid: 0.5, fusedScore: 0.5, decisionScore: 0.5,
      rawThreshold: 0.5, displayThreshold: 0.5,
      format: 'jpeg', width: 256, height: 256,
      timings: { fetchMs: 1, decodeMs: 1, modelWaitMs: 1, preprocessMs: 1, inferenceMs: 1, evidenceMs: 1, totalMs: 6 },
      navigatorLike
    };
    expect(createAnalysisDebugRecord(common).input).not.toHaveProperty('sha256');
    expect(() => createAnalysisDebugRecord({ ...common, inputByteLength: 10 })).toThrow('input SHA-256');
    expect(() => createAnalysisDebugRecord({ ...common, inputSha256: 'a'.repeat(64) })).toThrow('input byte length');
  });

  it('accepts the callable processor shape returned by Transformers.js', () => {
    const callableProcessor = Object.assign(function loadedImageProcessor() {}, processor);

    expect(typeof callableProcessor).toBe('function');
    expect(snapshotPreprocessor(callableProcessor)).toEqual(snapshotPreprocessor(processor));
    expect(() => snapshotPreprocessor('not-a-processor')).toThrow('loaded image processor');
  });

  it('records checksum identity and runtime-observed preprocessing only after readiness', () => {
    const record = createRuntimeReadyDebugRecord({
      extensionVersion: '0.1.0',
      extensionId,
      sessionId,
      processor,
      modelState,
      navigatorLike
    });

    expect(record).toMatchObject({
      schema: RUNTIME_READY_DEBUG_SCHEMA,
      sessionId,
      identity: {
        extensionVersion: '0.1.0',
        candidate: {
          decisionPolicy: 'model-only'
        },
        model: {
          weightSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
          configSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
          preprocessorConfigSha256: expect.stringMatching(/^[a-f0-9]{64}$/)
        },
        runtimeArtifacts: {
          wasmSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
          wasmLoaderSha256: expect.stringMatching(/^[a-f0-9]{64}$/)
        }
      },
      preprocessing: {
        processorType: 'ViTImageProcessor',
        shortestEdge: 440,
        cropSize: { width: 384, height: 384 },
        doConvertRgb: true
      },
      runtime: { state: 'ready', backend: 'WebAssembly', activeThreads: 4 },
      environment: { platform: 'Windows', hardwareConcurrency: 16 }
    });
    expect(() => createRuntimeReadyDebugRecord({
      extensionVersion: '0.1.0',
      extensionId,
      sessionId,
      processor,
      modelState: { ...modelState, state: 'loading' },
      navigatorLike
    })).toThrow('not ready');
  });

  it('emits exact raw scores and a joinable opaque asset without leaking the URL', () => {
    const preprocessing = snapshotPreprocessor(processor);
    const logit = -1.25;
    const modelSigmoid = 1 / (1 + Math.exp(-logit));
    const rawThreshold = 0.3;
    const displayThreshold = 0.65;
    const decisionScore = 1 / (1 + Math.exp(-(
      Math.log(modelSigmoid / (1 - modelSigmoid))
      - Math.log(rawThreshold / (1 - rawThreshold))
      + Math.log(displayThreshold / (1 - displayThreshold))
    )));
    const event = createAnalysisDebugRecord({
      extensionVersion: '0.1.0',
      extensionId,
      sessionId,
      sequence: 7,
      preprocessing,
      modelState,
      requestId: 'pm-safe-1',
      source: `http://127.0.0.1:4176/asset/0123456789abcdefabcd/image.jpg?realimageRun=${harnessRunId}&cache=secret`,
      modelLogit: logit,
      modelSigmoid,
      fusedScore: modelSigmoid,
      decisionScore,
      rawThreshold,
      displayThreshold,
      format: 'jpeg',
      width: 768,
      height: 768,
      inputSha256: 'a'.repeat(64),
      inputByteLength: 12345,
      timings: { fetchMs: 2, decodeMs: 3, modelWaitMs: 0, preprocessMs: 5, inferenceMs: 80, evidenceMs: 1, totalMs: 91 },
      navigatorLike
    });

    expect(event).toMatchObject({
      schema: ANALYSIS_DEBUG_SCHEMA,
      sessionId,
      sequence: 7,
      source: { kind: 'loopback-harness-asset', assetId: '0123456789abcdefabcd', harnessRunId, origin: 'http://127.0.0.1:4176' },
      input: { sha256: 'a'.repeat(64), byteLength: 12345 },
      scores: { modelLogit: logit, modelSigmoid, rawPredicted: 'real' },
      timings: { inferenceMs: 80, totalMs: 91 }
    });
    expect(JSON.stringify(event)).not.toContain('cache=secret');
    expect(JSON.stringify(event)).not.toContain('/asset/');
  });

  it('redacts non-harness origins and rejects inconsistent logit/sigmoid pairs', () => {
    expect(opaqueSourceIdentity('https://images.example/private-name.jpg')).toEqual({ kind: 'remote-http' });
    expect(opaqueSourceIdentity('blob:https://example.test/secret')).toEqual({ kind: 'blob' });
    expect(() => createAnalysisDebugRecord({
      extensionVersion: '0.1.0',
      extensionId,
      sessionId,
      sequence: 1,
      preprocessing: snapshotPreprocessor(processor),
      modelState,
      requestId: 'pm-bad',
      source: 'http://127.0.0.1:4176/asset/0123456789abcdefabcd/image.jpg',
      modelLogit: 0,
      modelSigmoid: 0.7,
      fusedScore: 0.7,
      decisionScore: 0.7,
      rawThreshold: 0.5,
      displayThreshold: 0.65,
      format: 'jpeg',
      width: 768,
      height: 768,
      timings: { fetchMs: 1, decodeMs: 1, modelWaitMs: 0, preprocessMs: 1, inferenceMs: 1, evidenceMs: 1, totalMs: 5 },
      navigatorLike
    })).toThrow('does not match');
  });
});
