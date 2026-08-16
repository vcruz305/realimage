import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it } from 'vitest';
import {
  CHROME_CAPTURE_SCHEMA,
  PARITY_REFERENCE_SCHEMA,
  REQUIRED_OBSERVATIONS,
  auditInstalledChromeParity,
  loadGuardedParityInputs
} from '../scripts/audit-installed-chrome-parity.mjs';
import {
  createAnalysisDebugRecord,
  createRuntimeReadyDebugRecord,
  detectorIdentity,
  snapshotPreprocessor
} from '../src/offscreen/analysis-debug.js';
import { formatScorePercent } from '../src/shared/score-display.js';

const transforms = ['square-jpeg75-768', 'square-webp72-768', 'thumbnail-jpeg75-512'];
const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const temporaryRoots = [];
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
const harnessOrigin = 'http://127.0.0.1:4176';

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe('frozen installed-Chrome parity auditor', () => {
  it('passes exactly 180 complete, identity-matched, flip-free calibration observations', () => {
    const fixture = makeFixture();
    const report = auditInstalledChromeParity(fixture);

    expect(report).toMatchObject({
      eligible: true,
      qualificationRead: false,
      observations: { required: 180, captured: 180, matched: 180 },
      parity: { thresholdFlips: 0, maxAllowedSigmoidError: 0.02 },
      latency: { measuredObservations: 179, medianInferenceMs: 80 },
      failures: []
    });
    expect(report.decisionGates).toHaveLength(9);
    expect(report.decisionGates.every((gate) => gate.pass)).toBe(true);
    expect(report.rows).toHaveLength(REQUIRED_OBSERVATIONS);
  });

  it('fails closed on candidate identity or runtime-preprocessing drift', () => {
    const fixture = makeFixture();
    fixture.capture.runtimeReady.identity.model.weightSha256 = 'f'.repeat(64);
    fixture.capture.runtimeReady.preprocessing.shortestEdge = 999;
    const report = auditInstalledChromeParity(fixture);

    expect(report.eligible).toBe(false);
    expect(report.failures.map((item) => item.code)).toEqual(expect.arrayContaining([
      'IDENTITY_MISMATCH',
      'PREPROCESSING_MISMATCH'
    ]));
  });

  it('fails on any excessive sigmoid error and any threshold decision flip', () => {
    const fixture = makeFixture();
    const sampleId = fixture.reference.rows[0].sampleId;
    fixture.reference.rows[0].rawSigmoidScore = 0.49;
    fixture.reference.rows[0].predicted = 'real';
    fixture.capture.analyses[0] = makeEvent(fixture.manifests[0].document.rows[0], 0, 0.51);
    const flipReport = auditInstalledChromeParity(fixture);
    expect(flipReport.eligible).toBe(false);
    expect(flipReport.parity.thresholdFlips).toBe(1);
    expect(flipReport.failures.map((item) => item.code)).toContain('THRESHOLD_FLIPS');
    expect(flipReport.rows.find((row) => row.sampleId === sampleId).thresholdFlip).toBe(true);

    const errorFixture = makeFixture();
    errorFixture.capture.analyses[0] = makeEvent(errorFixture.manifests[0].document.rows[0], 0, 0.85);
    const errorReport = auditInstalledChromeParity(errorFixture);
    expect(errorReport.eligible).toBe(false);
    expect(errorReport.failures.map((item) => item.code)).toContain('SIGMOID_ERROR_LIMIT');
  });

  it('rejects missing, duplicate, or non-calibration observations structurally', () => {
    const missing = makeFixture();
    missing.capture.analyses.pop();
    expect(() => auditInstalledChromeParity(missing)).toThrow('exactly 180');

    const duplicate = makeFixture();
    duplicate.capture.analyses[1] = structuredClone(duplicate.capture.analyses[0]);
    duplicate.capture.analyses[1].requestId = 'different-request';
    expect(() => auditInstalledChromeParity(duplicate)).toThrow('Duplicate Chrome observation');

    const forbidden = makeFixture();
    forbidden.manifests[0].document.rows[0].relativePath = '.bench-data/qualification/image.jpg';
    expect(() => auditInstalledChromeParity(forbidden)).toThrow('forbidden qualification/private-diagnostic');

    const hiddenForbidden = makeFixture();
    hiddenForbidden.manifests[0].document.rows[0].relativePath = '.bench-data/.qualification/image.jpg';
    expect(() => auditInstalledChromeParity(hiddenForbidden)).toThrow('forbidden qualification/private-diagnostic');
  });

  it('fails a preregistered latency or independent source/transform decision gate', () => {
    const fixture = makeFixture();
    fixture.reference.gates.latency.maximumP95InferenceMs = 79;
    fixture.reference.gates.decisions[0].minimumCorrect = 30;
    fixture.capture.analyses[0] = makeEvent(fixture.manifests[0].document.rows[0], 0, 0.8);
    const report = auditInstalledChromeParity(fixture);

    expect(report.eligible).toBe(false);
    expect(report.failures.map((item) => item.code)).toEqual(expect.arrayContaining(['LATENCY_LIMIT', 'DECISION_GATE']));
  });

  it('fails captures stitched across sessions, sequences, page runs, or substituted bytes', () => {
    const fixture = makeFixture();
    fixture.capture.analyses[40].sequence += 1;
    fixture.capture.analyses[80].sessionId = 'f'.repeat(32);
    fixture.capture.analyses[120].source.harnessRunId = 'e'.repeat(32);
    fixture.capture.analyses[160].input.sha256 = 'd'.repeat(64);
    const report = auditInstalledChromeParity(fixture);

    expect(report.eligible).toBe(false);
    expect(report.failures.map((item) => item.code)).toEqual(expect.arrayContaining([
      'NONCONTIGUOUS_ANALYSIS_SEQUENCE',
      'DEBUG_SESSION_MISMATCH',
      'HARNESS_RUN_MISMATCH',
      'INPUT_DIGEST_MISMATCH'
    ]));
  });

  it('rejects a partial/error page, UI threshold drift, or weakened frozen gates', () => {
    const partial = makeFixture();
    partial.capture.harnessPages[0].cases[0].status = 'extension-error';
    partial.capture.harnessPages[0].totals.complete = 29;
    partial.capture.harnessPages[0].totals.extensionErrors = 1;
    expect(() => auditInstalledChromeParity(partial)).toThrow('terminal success');

    const threshold = makeFixture();
    threshold.capture.harnessPages[0].cases[0].decisionThreshold = 0.7;
    expect(() => auditInstalledChromeParity(threshold)).toThrow('decision threshold');

    const weakDecision = makeFixture();
    weakDecision.reference.gates.decisions[0].minimumCorrect = 1;
    expect(() => auditInstalledChromeParity(weakDecision)).toThrow('minimumCorrect must be at least');

    const weakLatency = makeFixture();
    weakLatency.reference.gates.latency.maximumMedianInferenceMs = 121;
    expect(() => auditInstalledChromeParity(weakLatency)).toThrow('must not exceed 120');
  });

  it('loads only explicitly named non-qualification artifacts contained in .bench-data', async () => {
    const benchRoot = resolve(projectRoot, '.bench-data');
    await mkdir(benchRoot, { recursive: true });
    const root = await mkdtemp(join(benchRoot, 'chrome-parity-io-'));
    temporaryRoots.push(root);
    const fixture = makeFixture();
    const manifestPaths = [];
    for (const [index, manifest] of fixture.manifests.entries()) {
      const bytes = `${JSON.stringify(manifest.document, null, 2)}\n`;
      const path = join(root, `manifest-${index}.json`);
      await writeFile(path, bytes);
      manifest.sha256 = sha256(bytes);
      manifestPaths.push(path);
    }
    fixture.reference.manifests = fixture.manifests.map((item) => ({ sha256: item.sha256, rows: item.document.rows.length }));
    fixture.capture.harnessPages = makeHarnessPages(fixture.manifests);
    const referencePath = join(root, 'reference.json');
    const capturePath = join(root, 'capture.json');
    await writeFile(referencePath, `${JSON.stringify(fixture.reference, null, 2)}\n`);
    await writeFile(capturePath, `${JSON.stringify(fixture.capture, null, 2)}\n`);

    const loaded = await loadGuardedParityInputs({ projectRoot, manifestPaths, referencePath, capturePath });
    expect(auditInstalledChromeParity(loaded).eligible).toBe(true);

    await expect(loadGuardedParityInputs({
      projectRoot,
      manifestPaths,
      referencePath: join(benchRoot, 'qualification', 'never-open.json'),
      capturePath
    })).rejects.toThrow('forbidden qualification/private-diagnostic');
    await expect(loadGuardedParityInputs({
      projectRoot,
      manifestPaths,
      referencePath: resolve(benchRoot, '..', 'package.json'),
      capturePath
    })).rejects.toThrow('must remain inside .bench-data');
  });
});

function makeFixture() {
  const manifests = transforms.map((transformId, transformIndex) => {
    const rows = [];
    for (let index = 0; index < 60; index += 1) {
      const domain = index < 30 ? 'DOCCI' : index < 45 ? 'Nano Banana' : 'FLUX Krea';
      const label = domain === 'DOCCI' ? 'real' : 'ai';
      const sampleId = `cal-${String(transformIndex).padStart(2, '0')}-${String(index).padStart(3, '0')}:${transformId}`;
      rows.push({
        sampleId,
        familyId: `family-${String(index).padStart(3, '0')}`,
        label,
        domain,
        slice: 'synthetic-calibration-wiring',
        transformId,
        view: 'single-official',
        relativePath: `.bench-data/synthetic-calibration/${transformId}/${sampleId}.jpg`,
        sha256: sha256(`fixture-bytes:${sampleId}`)
      });
    }
    const document = {
      schema: 'synthetic-calibration-manifest-v1',
      benchmarkId: `synthetic-${transformId}`,
      rows
    };
    return { document, sha256: sha256(JSON.stringify(document)) };
  });
  const allRows = manifests.flatMap((item) => item.document.rows);
  const preprocessing = snapshotPreprocessor(processor);
  const identity = detectorIdentity('0.1.0');
  const runtimeReady = createRuntimeReadyDebugRecord({
    extensionVersion: '0.1.0',
    extensionId,
    sessionId,
    processor,
    modelState,
    navigatorLike
  });
  const rawThreshold = 0.5;
  const reference = {
    schema: PARITY_REFERENCE_SCHEMA,
    frozenBeforeChrome: true,
    qualificationRead: false,
    candidateId: identity.candidate.id,
    expectedObservations: REQUIRED_OBSERVATIONS,
    manifests: manifests.map((item) => ({ sha256: item.sha256, rows: item.document.rows.length })),
    identity,
    preprocessing,
    rawThreshold,
    displayThreshold: 0.65,
    runtimeRequirements: {
      backend: 'WebAssembly',
      crossOriginIsolated: true,
      minimumActiveThreads: 1,
      minimumChromeMajor: 148
    },
    gates: {
      maxSigmoidError: 0.02,
      zeroThresholdFlips: true,
      requireModelOnlyDecision: true,
      latency: {
        excludeFirst: 1,
        maximumMedianInferenceMs: 100,
        maximumP95InferenceMs: 120,
        maximumSingleInferenceMs: 150,
        maximumMedianTotalMs: 140
      },
      decisions: transforms.flatMap((transformId) => [
        { id: `${transformId}:docci`, match: { transformId, domain: 'DOCCI', label: 'real' }, expectedRows: 30, minimumCorrect: 27 },
        { id: `${transformId}:nano`, match: { transformId, domain: 'Nano Banana', label: 'ai' }, expectedRows: 15, minimumCorrect: 11 },
        { id: `${transformId}:flux`, match: { transformId, domain: 'FLUX Krea', label: 'ai' }, expectedRows: 15, minimumCorrect: 11 }
      ])
    },
    rows: allRows.map((row) => {
      const rawSigmoidScore = row.label === 'ai' ? 0.8 : 0.2;
      return { sampleId: row.sampleId, rawSigmoidScore, predicted: row.label };
    })
  };
  const capture = {
    schema: CHROME_CAPTURE_SCHEMA,
    qualificationRead: false,
    browser: { name: 'Chrome', version: '151.0.7922.138', platform: 'Windows' },
    runtimeReady,
    harnessPages: makeHarnessPages(manifests),
    analyses: allRows.map((row, index) => makeEvent(row, index, row.label === 'ai' ? 0.8 : 0.2))
  };
  return {
    manifests,
    reference,
    capture,
    inputArtifacts: { referenceSha256: 'a'.repeat(64), captureSha256: 'b'.repeat(64) }
  };
}

function makeEvent(row, index, modelSigmoid) {
  const modelLogit = Math.log(modelSigmoid / (1 - modelSigmoid));
  const rawThreshold = 0.5;
  const displayThreshold = 0.65;
  const decisionScore = calibrate(modelSigmoid, rawThreshold, displayThreshold);
  return createAnalysisDebugRecord({
    extensionVersion: '0.1.0',
    extensionId,
    sessionId,
    sequence: index + 1,
    preprocessing: snapshotPreprocessor(processor),
    modelState,
    requestId: `pm-synthetic-${index}`,
    source: `${harnessOrigin}/asset/${stableAssetId(row.sampleId)}/image.jpg?realimageRun=${harnessRunIdFor(index)}`,
    modelLogit,
    modelSigmoid,
    fusedScore: modelSigmoid,
    decisionScore,
    rawThreshold,
    displayThreshold,
    format: 'jpeg',
    width: 768,
    height: 768,
    inputSha256: row.sha256,
    inputByteLength: 12345,
    timings: { fetchMs: 2, decodeMs: 4, modelWaitMs: 0, preprocessMs: 6, inferenceMs: 80, evidenceMs: 1, totalMs: 93 },
    navigatorLike
  });
}

function makeHarnessPages(manifests) {
  return manifests.flatMap((manifest, manifestIndex) => [0, 30].map((offset) => {
    const globalOffset = manifestIndex * 60 + offset;
    const harnessRunId = harnessRunIdFor(globalOffset);
    const cases = manifest.document.rows.slice(offset, offset + 30).map((row) => {
      const modelSigmoid = row.label === 'ai' ? 0.8 : 0.2;
      const decision = calibrate(modelSigmoid, 0.5, 0.65);
      const predicted = decision >= 0.65 ? 'ai' : 'real';
      return {
        caseId: `${row.sampleId}::direct`,
        assetId: stableAssetId(row.sampleId),
        sampleId: row.sampleId,
        familyId: row.familyId,
        expected: row.label,
        view: row.view,
        transformId: row.transformId,
        domain: row.domain,
        slice: row.slice,
        delivery: 'direct',
        status: 'complete',
        predicted,
        decisionThreshold: 0.65,
        score: Number(formatScorePercent(decision, 0.65)) / 100
      };
    });
    const correct = cases.filter((row) => row.predicted === row.expected).length;
    return {
      schema: 'realimage-private-browser-harness-state-v1',
      harnessRunId,
      origin: harnessOrigin,
      benchmarkId: manifest.document.benchmarkId,
      manifestSchema: manifest.document.schema,
      manifestSha256: manifest.sha256,
      status: 'complete',
      terminal: true,
      selection: { delivery: 'direct', execution: 'sequential', offset, limit: cases.length, availableRows: 60 },
      totals: {
        expected: cases.length,
        complete: cases.length,
        correct,
        incorrect: cases.length - correct,
        extensionErrors: 0,
        imageErrors: 0,
        timedOut: 0
      },
      cases
    };
  }));
}

function harnessRunIdFor(globalIndex) {
  const pageIndex = Math.floor(globalIndex / 30) + 1;
  return pageIndex.toString(16).padStart(32, '0');
}

function calibrate(score, rawThreshold, displayThreshold) {
  const logit = (value) => Math.log(value / (1 - value));
  return 1 / (1 + Math.exp(-(logit(score) - logit(rawThreshold) + logit(displayThreshold))));
}

function stableAssetId(sampleId) {
  return createHash('sha256').update(`asset\0${sampleId}`).digest('hex').slice(0, 20);
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}
