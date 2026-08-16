import { createHash } from 'node:crypto';
import { readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  ANALYSIS_DEBUG_SCHEMA,
  RUNTIME_READY_DEBUG_SCHEMA,
  stableSigmoid
} from '../src/offscreen/analysis-debug.js';
import { formatScorePercent } from '../src/shared/score-display.js';

export const PARITY_REFERENCE_SCHEMA = 'realimage-installed-chrome-parity-reference-v1';
export const CHROME_CAPTURE_SCHEMA = 'realimage-installed-chrome-capture-v1';
export const PARITY_REPORT_SCHEMA = 'realimage-installed-chrome-parity-report-v1';
export const REQUIRED_OBSERVATIONS = 180;

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const PROJECT_ROOT = resolve(dirname(SCRIPT_PATH), '..');
const SHA256 = /^[a-f0-9]{64}$/;
const ASSET_ID = /^[a-f0-9]{20}$/;
const DEBUG_SESSION_ID = /^[a-f0-9]{32}$/;
const HARNESS_RUN_ID = /^[a-f0-9]{32}$/;
const EXTENSION_ID = /^[a-p]{32}$/;
const LABELS = new Set(['ai', 'real']);
const MATCH_FIELDS = new Set(['label', 'transformId', 'domain', 'view', 'slice']);
const LATENCY_LIMITS = Object.freeze({
  maximumMedianInferenceMs: 'medianInferenceMs',
  maximumP95InferenceMs: 'p95InferenceMs',
  maximumSingleInferenceMs: 'maximumInferenceMs',
  maximumMedianTotalMs: 'medianTotalMs'
});
const MAXIMUM_LATENCY_LIMITS = Object.freeze({
  maximumMedianInferenceMs: 120,
  maximumP95InferenceMs: 150,
  maximumSingleInferenceMs: 250,
  maximumMedianTotalMs: 180
});
const MAXIMUM_SIGMOID_ERROR = 0.02;
const MINIMUM_RECALL = Object.freeze({ ai: 0.7, real: 0.9 });

export function auditInstalledChromeParity({ manifests, reference, capture, inputArtifacts = undefined }) {
  const manifestRows = validateManifests(manifests, reference);
  const frozen = validateReference(reference, manifestRows);
  validateCaptureEnvelope(capture);
  const harness = validateHarnessPages(capture.harnessPages, manifestRows, frozen);

  const failures = [];
  const ready = capture.runtimeReady;
  if (canonicalJson(ready.identity) !== canonicalJson(frozen.identity)) {
    failures.push(failure('IDENTITY_MISMATCH', 'Installed model/config/runtime identity differs from the frozen reference.'));
  }
  if (canonicalJson(ready.preprocessing) !== canonicalJson(frozen.preprocessing)) {
    failures.push(failure('PREPROCESSING_MISMATCH', 'Runtime-observed preprocessing differs from the frozen reference.'));
  }
  applyRuntimeGates(ready, capture.browser, frozen.runtimeRequirements, failures);

  const referenceBySample = new Map(frozen.rows.map((row) => [row.sampleId, row]));
  const manifestByAsset = new Map(manifestRows.map((row) => [row.assetId, row]));
  const seenAssets = new Set();
  const seenRequests = new Set();
  const observations = [];
  let previousSequence;

  for (const [captureIndex, event] of capture.analyses.entries()) {
    validateAnalysisEvent(event, captureIndex);
    if (event.sessionId !== ready.sessionId) {
      failures.push(failure('DEBUG_SESSION_MISMATCH', `Chrome debug session changed at capture index ${captureIndex}.`, { captureIndex }));
    }
    if (previousSequence != null && event.sequence !== previousSequence + 1) {
      failures.push(failure('NONCONTIGUOUS_ANALYSIS_SEQUENCE', `Chrome analysis sequence jumped from ${previousSequence} to ${event.sequence}.`, { captureIndex }));
    }
    previousSequence = event.sequence;
    if (seenRequests.has(event.requestId)) throw new Error(`Duplicate Chrome requestId: ${event.requestId}`);
    seenRequests.add(event.requestId);
    const assetId = event.source.assetId;
    if (seenAssets.has(assetId)) throw new Error(`Duplicate Chrome observation for asset ${assetId}.`);
    seenAssets.add(assetId);
    const manifestRow = manifestByAsset.get(assetId);
    if (!manifestRow) throw new Error(`Chrome returned an unregistered calibration asset: ${assetId}`);
    const expected = referenceBySample.get(manifestRow.sampleId);
    if (!expected) throw new Error(`Frozen Python reference is missing ${manifestRow.sampleId}.`);
    const harnessCase = harness.byAsset.get(assetId);
    if (!harnessCase) throw new Error(`No terminal harness case exists for ${manifestRow.sampleId}.`);
    if (event.source.harnessRunId !== harnessCase.harnessRunId || event.source.origin !== harnessCase.origin) {
      failures.push(failure('HARNESS_RUN_MISMATCH', `Chrome source did not come from the terminal harness page for ${manifestRow.sampleId}.`, { sampleId: manifestRow.sampleId }));
    }
    if (event.input.sha256 !== manifestRow.sha256) {
      failures.push(failure('INPUT_DIGEST_MISMATCH', `Chrome analyzed bytes other than the pinned bytes for ${manifestRow.sampleId}.`, { sampleId: manifestRow.sampleId }));
    }

    const identityMismatch = canonicalJson(event.identity) !== canonicalJson(frozen.identity);
    const preprocessingMismatch = canonicalJson(event.preprocessing) !== canonicalJson(frozen.preprocessing);
    const runtimeMismatch = canonicalJson(event.runtime) !== canonicalJson(ready.runtime);
    const environmentMismatch = canonicalJson(event.environment) !== canonicalJson(ready.environment);
    if (identityMismatch || preprocessingMismatch || runtimeMismatch || environmentMismatch) {
      failures.push(failure(
        'PER_OBSERVATION_CONTEXT_MISMATCH',
        `Chrome context changed for ${manifestRow.sampleId}.`,
        { sampleId: manifestRow.sampleId }
      ));
    }

    const scores = event.scores;
    const reconstructed = stableSigmoid(scores.modelLogit);
    if (Math.abs(reconstructed - scores.modelSigmoid) > 1e-12) {
      failures.push(failure('LOGIT_SIGMOID_MISMATCH', `Logged logit/sigmoid mismatch for ${manifestRow.sampleId}.`, { sampleId: manifestRow.sampleId }));
    }
    if (scores.rawThreshold !== frozen.rawThreshold || scores.displayThreshold !== frozen.displayThreshold) {
      failures.push(failure('THRESHOLD_IDENTITY_MISMATCH', `Chrome thresholds differ for ${manifestRow.sampleId}.`, { sampleId: manifestRow.sampleId }));
    }
    if (frozen.requireModelOnlyDecision && Math.abs(scores.fused - scores.modelSigmoid) > 1e-12) {
      failures.push(failure('NON_MODEL_SCORE_FUSION', `Calibration row ${manifestRow.sampleId} did not use its raw model score.`, { sampleId: manifestRow.sampleId }));
    }

    const chromePredicted = scores.modelSigmoid >= frozen.rawThreshold ? 'ai' : 'real';
    const displayedPredicted = scores.decision >= frozen.displayThreshold ? 'ai' : 'real';
    const expectedDisplay = calibrateForDisplay(scores.modelSigmoid, frozen.rawThreshold, frozen.displayThreshold);
    if (Math.abs(expectedDisplay - scores.decision) > 1e-12 || chromePredicted !== displayedPredicted) {
      failures.push(failure('DISPLAY_MAPPING_MISMATCH', `Visible decision mapping changed for ${manifestRow.sampleId}.`, { sampleId: manifestRow.sampleId }));
    }
    if (scores.rawPredicted !== chromePredicted || scores.displayedPredicted !== displayedPredicted) {
      failures.push(failure('LOGGED_DECISION_MISMATCH', `Logged decision fields are inconsistent for ${manifestRow.sampleId}.`, { sampleId: manifestRow.sampleId }));
    }
    const expectedBadgeScore = Number(formatScorePercent(scores.decision, frozen.displayThreshold)) / 100;
    if (
      harnessCase.decisionThreshold !== frozen.displayThreshold
      || harnessCase.predicted !== displayedPredicted
      || harnessCase.score !== expectedBadgeScore
    ) {
      failures.push(failure('VISIBLE_UI_MISMATCH', `Visible badge output differs from the frozen unrounded decision for ${manifestRow.sampleId}.`, { sampleId: manifestRow.sampleId }));
    }

    const absoluteSigmoidError = Math.abs(scores.modelSigmoid - expected.rawSigmoidScore);
    const thresholdFlip = chromePredicted !== expected.predicted;
    observations.push({
      sampleId: manifestRow.sampleId,
      label: manifestRow.label,
      transformId: manifestRow.transformId,
      domain: manifestRow.domain,
      view: manifestRow.view,
      slice: manifestRow.slice,
      captureIndex,
      expectedRawSigmoidScore: expected.rawSigmoidScore,
      chromeRawSigmoidScore: scores.modelSigmoid,
      absoluteSigmoidError,
      expectedPredicted: expected.predicted,
      chromePredicted,
      thresholdFlip,
      correct: chromePredicted === manifestRow.label,
      timings: { ...event.timings }
    });
  }

  if (seenAssets.size !== manifestRows.length) {
    const missing = manifestRows.filter((row) => !seenAssets.has(row.assetId)).map((row) => row.sampleId);
    throw new Error(`Chrome capture is missing ${missing.length} frozen observations: ${missing.slice(0, 5).join(', ')}`);
  }

  const maximumAbsoluteSigmoidError = Math.max(...observations.map((row) => row.absoluteSigmoidError));
  const meanAbsoluteSigmoidError = mean(observations.map((row) => row.absoluteSigmoidError));
  const thresholdFlips = observations.filter((row) => row.thresholdFlip || row.chromePredicted !== row.expectedPredicted).length;
  if (maximumAbsoluteSigmoidError > frozen.maxSigmoidError) {
    failures.push(failure('SIGMOID_ERROR_LIMIT', `Maximum sigmoid error ${maximumAbsoluteSigmoidError} exceeds ${frozen.maxSigmoidError}.`));
  }
  if (thresholdFlips !== 0) {
    failures.push(failure('THRESHOLD_FLIPS', `${thresholdFlips} Python-to-Chrome threshold decisions changed.`));
  }

  const decisionGates = frozen.decisionGates.map((gate) => evaluateDecisionGate(gate, observations, failures));
  const partitionViolations = observations.filter((row) => (
    frozen.decisionGates.filter((gate) => Object.entries(gate.match).every(([field, expected]) => row[field] === expected)).length !== 1
  ));
  if (partitionViolations.length) {
    failures.push(failure(
      'DECISION_GATE_PARTITION',
      `${partitionViolations.length} observations are not covered by exactly one decision gate.`
    ));
  }
  const latency = evaluateLatency(frozen.latencyGate, observations, failures);
  const reportRows = observations
    .slice()
    .sort((left, right) => left.sampleId.localeCompare(right.sampleId));

  return {
    schema: PARITY_REPORT_SCHEMA,
    eligible: failures.length === 0,
    qualificationRead: false,
    candidateId: frozen.candidateId,
    identitySha256: sha256(canonicalJson(frozen.identity)),
    inputArtifacts: normalizeInputArtifactDigests(inputArtifacts),
    manifestPins: manifests.map((item) => ({ sha256: item.sha256, rows: item.document.rows.length })).sort(comparePins),
    observations: {
      required: REQUIRED_OBSERVATIONS,
      captured: capture.analyses.length,
      matched: observations.length
    },
    browser: { ...capture.browser },
    captureProvenance: {
      debugSessionId: ready.sessionId,
      firstAnalysisSequence: capture.analyses[0].sequence,
      lastAnalysisSequence: capture.analyses.at(-1).sequence,
      terminalHarnessPages: harness.pages.length
    },
    runtime: { ...ready.runtime },
    preprocessing: ready.preprocessing,
    parity: {
      maxAllowedSigmoidError: frozen.maxSigmoidError,
      maximumAbsoluteSigmoidError,
      meanAbsoluteSigmoidError,
      thresholdFlips
    },
    latency,
    decisionGates,
    failures,
    rows: reportRows
  };
}

export async function loadGuardedParityInputs({ projectRoot = PROJECT_ROOT, manifestPaths, referencePath, capturePath }) {
  if (!Array.isArray(manifestPaths) || manifestPaths.length === 0) throw new Error('At least one calibration --manifest is required.');
  const manifests = [];
  for (const path of manifestPaths) {
    const artifact = await readGuardedJson(projectRoot, path, 'calibration manifest');
    manifests.push({ sha256: artifact.sha256, document: artifact.document });
  }
  const referenceArtifact = await readGuardedJson(projectRoot, referencePath, 'parity reference');
  const captureArtifact = await readGuardedJson(projectRoot, capturePath, 'Chrome capture');
  return {
    manifests,
    reference: referenceArtifact.document,
    capture: captureArtifact.document,
    inputArtifacts: {
      referenceSha256: referenceArtifact.sha256,
      captureSha256: captureArtifact.sha256
    }
  };
}

function validateManifests(manifests, reference) {
  if (!Array.isArray(manifests) || manifests.length === 0) throw new Error('Calibration manifests are required.');
  const pins = manifests.map((item, index) => {
    if (!item || !SHA256.test(item.sha256 || '')) throw new Error(`Manifest ${index} has no valid SHA-256.`);
    if (!Array.isArray(item.document?.rows) || item.document.rows.length === 0) throw new Error(`Manifest ${index} has no rows.`);
    return { sha256: item.sha256, rows: item.document.rows.length };
  }).sort(comparePins);
  const expectedPins = reference?.manifests;
  if (!Array.isArray(expectedPins) || expectedPins.length !== pins.length) throw new Error('Frozen manifest pin count changed.');
  const normalizedExpectedPins = expectedPins.map((pin, index) => {
    if (!SHA256.test(pin?.sha256 || '') || !Number.isSafeInteger(pin?.rows) || pin.rows <= 0) {
      throw new Error(`Frozen manifest pin ${index} is invalid.`);
    }
    return { sha256: pin.sha256, rows: pin.rows };
  }).sort(comparePins);
  if (canonicalJson(pins) !== canonicalJson(normalizedExpectedPins)) throw new Error('Calibration manifest SHA-256 or row-count pins changed.');

  const rows = [];
  const sampleIds = new Set();
  const assetIds = new Set();
  for (const [manifestIndex, item] of manifests.entries()) {
    for (const [rowIndex, source] of item.document.rows.entries()) {
      const field = `manifests[${manifestIndex}].rows[${rowIndex}]`;
      const sampleId = requiredString(source?.sampleId, `${field}.sampleId`);
      if (sampleIds.has(sampleId)) throw new Error(`Duplicate calibration sampleId: ${sampleId}`);
      sampleIds.add(sampleId);
      const label = requiredString(source?.label, `${field}.label`).toLowerCase();
      if (!LABELS.has(label)) throw new Error(`${field}.label must be ai or real.`);
      const relativePath = requiredString(source?.relativePath, `${field}.relativePath`);
      assertSafeRelativeCalibrationPath(relativePath, `${field}.relativePath`);
      if (!SHA256.test(source?.sha256 || '')) throw new Error(`${field}.sha256 must be a lowercase SHA-256.`);
      const assetId = stableId(`asset\0${sampleId}`, 20);
      if (assetIds.has(assetId)) throw new Error(`Opaque asset ID collision for ${sampleId}.`);
      assetIds.add(assetId);
      rows.push({
        sampleId,
        assetId,
        manifestSha256: item.sha256,
        sha256: source.sha256,
        label,
        transformId: requiredString(source?.transformId, `${field}.transformId`),
        domain: requiredString(source?.domain, `${field}.domain`),
        view: optionalString(source?.view),
        slice: optionalString(source?.slice)
      });
    }
  }
  if (rows.length !== REQUIRED_OBSERVATIONS) {
    throw new Error(`Installed-Chrome parity requires exactly ${REQUIRED_OBSERVATIONS} calibration observations; received ${rows.length}.`);
  }
  return rows;
}

function validateReference(reference, manifestRows) {
  if (!reference || reference.schema !== PARITY_REFERENCE_SCHEMA) throw new Error(`Reference schema must be ${PARITY_REFERENCE_SCHEMA}.`);
  if (reference.frozenBeforeChrome !== true || reference.qualificationRead !== false) {
    throw new Error('Reference must be frozen before Chrome and attest qualificationRead=false.');
  }
  if (reference.expectedObservations !== REQUIRED_OBSERVATIONS) throw new Error(`Reference must freeze ${REQUIRED_OBSERVATIONS} observations.`);
  const candidateId = requiredString(reference.candidateId, 'reference.candidateId');
  validateIdentity(reference.identity, 'reference.identity');
  if (candidateId !== reference.identity.candidate.id) throw new Error('reference.candidateId must match reference.identity.candidate.id.');
  validatePreprocessing(reference.preprocessing, 'reference.preprocessing');
  const rawThreshold = openProbability(reference.rawThreshold, 'reference.rawThreshold');
  const displayThreshold = openProbability(reference.displayThreshold, 'reference.displayThreshold');
  const maxSigmoidError = finiteNumber(reference.gates?.maxSigmoidError, 'reference.gates.maxSigmoidError');
  if (maxSigmoidError <= 0 || maxSigmoidError > MAXIMUM_SIGMOID_ERROR) {
    throw new Error(`maxSigmoidError must be in (0, ${MAXIMUM_SIGMOID_ERROR}].`);
  }
  if (reference.gates?.zeroThresholdFlips !== true) throw new Error('zeroThresholdFlips must be preregistered as true.');
  if (reference.gates?.requireModelOnlyDecision !== true) throw new Error('requireModelOnlyDecision must be preregistered as true.');
  const latencyGate = validateLatencyGate(reference.gates?.latency);
  const runtimeRequirements = validateRuntimeRequirements(reference.runtimeRequirements);
  const decisionGates = validateDecisionGates(reference.gates?.decisions, manifestRows);

  if (!Array.isArray(reference.rows) || reference.rows.length !== REQUIRED_OBSERVATIONS) {
    throw new Error(`Reference rows must contain exactly ${REQUIRED_OBSERVATIONS} observations.`);
  }
  const manifestIds = new Set(manifestRows.map((row) => row.sampleId));
  const seen = new Set();
  const rows = reference.rows.map((row, index) => {
    const sampleId = requiredString(row?.sampleId, `reference.rows[${index}].sampleId`);
    if (seen.has(sampleId)) throw new Error(`Duplicate reference sampleId: ${sampleId}`);
    seen.add(sampleId);
    if (!manifestIds.has(sampleId)) throw new Error(`Reference contains an unregistered sample: ${sampleId}`);
    const rawSigmoidScore = probability(row?.rawSigmoidScore, `reference.rows[${index}].rawSigmoidScore`);
    const predicted = requiredString(row?.predicted, `reference.rows[${index}].predicted`).toLowerCase();
    if (!LABELS.has(predicted) || predicted !== (rawSigmoidScore >= rawThreshold ? 'ai' : 'real')) {
      throw new Error(`Reference prediction is inconsistent for ${sampleId}.`);
    }
    return { sampleId, rawSigmoidScore, predicted };
  });
  if (seen.size !== manifestIds.size) throw new Error('Reference and calibration manifest row sets differ.');
  return {
    candidateId,
    identity: reference.identity,
    preprocessing: reference.preprocessing,
    rawThreshold,
    displayThreshold,
    maxSigmoidError,
    requireModelOnlyDecision: true,
    latencyGate,
    runtimeRequirements,
    decisionGates,
    rows
  };
}

function validateCaptureEnvelope(capture) {
  if (!capture || capture.schema !== CHROME_CAPTURE_SCHEMA) throw new Error(`Capture schema must be ${CHROME_CAPTURE_SCHEMA}.`);
  if (capture.qualificationRead !== false) throw new Error('Chrome capture must attest qualificationRead=false.');
  const browserName = requiredString(capture.browser?.name, 'capture.browser.name');
  if (browserName !== 'Chrome') throw new Error('Installed-Chrome capture must identify browser.name exactly as Chrome.');
  requiredString(capture.browser?.version, 'capture.browser.version');
  requiredString(capture.browser?.platform, 'capture.browser.platform');
  if (capture.runtimeReady?.schema !== RUNTIME_READY_DEBUG_SCHEMA) throw new Error(`Missing ${RUNTIME_READY_DEBUG_SCHEMA} readiness record.`);
  if (!DEBUG_SESSION_ID.test(capture.runtimeReady?.sessionId || '')) throw new Error('runtimeReady.sessionId must be 32 lowercase hexadecimal characters.');
  validateIdentity(capture.runtimeReady.identity, 'capture.runtimeReady.identity');
  validatePreprocessing(capture.runtimeReady.preprocessing, 'capture.runtimeReady.preprocessing');
  validateRuntime(capture.runtimeReady.runtime, 'capture.runtimeReady.runtime');
  validateEnvironment(capture.runtimeReady.environment, 'capture.runtimeReady.environment');
  if (!Array.isArray(capture.analyses) || capture.analyses.length !== REQUIRED_OBSERVATIONS) {
    throw new Error(`Chrome capture must contain exactly ${REQUIRED_OBSERVATIONS} analysis records.`);
  }
  if (!Array.isArray(capture.harnessPages) || capture.harnessPages.length === 0) {
    throw new Error('Chrome capture must contain terminal harnessPages records.');
  }
}

function validateHarnessPages(pages, manifestRows, frozen) {
  const manifestByAsset = new Map(manifestRows.map((row) => [row.assetId, row]));
  const seenRuns = new Set();
  const byAsset = new Map();
  const normalizedPages = pages.map((page, pageIndex) => {
    const field = `capture.harnessPages[${pageIndex}]`;
    if (!page || page.schema !== 'realimage-private-browser-harness-state-v1') {
      throw new Error(`${field}.schema must be realimage-private-browser-harness-state-v1.`);
    }
    if (!HARNESS_RUN_ID.test(page.harnessRunId || '')) throw new Error(`${field}.harnessRunId must be 32 lowercase hexadecimal characters.`);
    if (seenRuns.has(page.harnessRunId)) throw new Error(`Duplicate terminal harness run: ${page.harnessRunId}`);
    seenRuns.add(page.harnessRunId);
    const origin = validateLoopbackOrigin(page.origin, `${field}.origin`);
    if (!SHA256.test(page.manifestSha256 || '')) throw new Error(`${field}.manifestSha256 must be a lowercase SHA-256.`);
    if (page.terminal !== true || page.status !== 'complete') throw new Error(`${field} is not a successfully completed terminal page.`);
    if (page.selection?.delivery !== 'direct' || page.selection?.execution !== 'sequential') {
      throw new Error(`${field} must be a direct, sequential batch-one page.`);
    }
    if (!Array.isArray(page.cases) || page.cases.length < 1 || page.cases.length > 30) {
      throw new Error(`${field}.cases must contain from 1 through 30 rows.`);
    }
    if (page.selection.limit !== page.cases.length) throw new Error(`${field}.selection.limit must equal its captured case count.`);
    nonNegativeInteger(page.selection.offset, `${field}.selection.offset`);
    const totals = page.totals;
    for (const name of ['expected', 'complete', 'correct', 'incorrect', 'extensionErrors', 'imageErrors', 'timedOut']) {
      nonNegativeInteger(totals?.[name], `${field}.totals.${name}`);
    }
    if (
      totals.expected !== page.cases.length
      || totals.complete !== page.cases.length
      || totals.correct + totals.incorrect !== page.cases.length
      || totals.extensionErrors !== 0
      || totals.imageErrors !== 0
      || totals.timedOut !== 0
    ) {
      throw new Error(`${field}.totals does not attest one terminal success for every case.`);
    }

    for (const [caseIndex, item] of page.cases.entries()) {
      const caseField = `${field}.cases[${caseIndex}]`;
      if (!ASSET_ID.test(item?.assetId || '')) throw new Error(`${caseField}.assetId is invalid.`);
      if (byAsset.has(item.assetId)) throw new Error(`Duplicate terminal harness case for asset ${item.assetId}.`);
      const manifestRow = manifestByAsset.get(item.assetId);
      if (!manifestRow) throw new Error(`${caseField} refers to an unregistered calibration asset.`);
      if (page.manifestSha256 !== manifestRow.manifestSha256) throw new Error(`${caseField} is attached to the wrong pinned manifest.`);
      if (
        item.status !== 'complete'
        || item.delivery !== 'direct'
        || item.sampleId !== manifestRow.sampleId
        || item.expected !== manifestRow.label
        || item.transformId !== manifestRow.transformId
        || item.domain !== manifestRow.domain
        || optionalString(item.view) !== manifestRow.view
        || optionalString(item.slice) !== manifestRow.slice
      ) {
        throw new Error(`${caseField} does not match its frozen manifest row or terminal-success contract.`);
      }
      if (!LABELS.has(item.predicted)) throw new Error(`${caseField}.predicted must be ai or real.`);
      const score = probability(item.score, `${caseField}.score`);
      const decisionThreshold = openProbability(item.decisionThreshold, `${caseField}.decisionThreshold`);
      byAsset.set(item.assetId, {
        harnessRunId: page.harnessRunId,
        origin,
        predicted: item.predicted,
        score,
        decisionThreshold
      });
    }
    return { harnessRunId: page.harnessRunId, origin, manifestSha256: page.manifestSha256, rows: page.cases.length };
  });
  if (byAsset.size !== manifestRows.length) {
    const missing = manifestRows.filter((row) => !byAsset.has(row.assetId)).map((row) => row.sampleId);
    throw new Error(`Terminal harness pages are missing ${missing.length} frozen observations: ${missing.slice(0, 5).join(', ')}`);
  }
  for (const item of byAsset.values()) {
    if (item.decisionThreshold !== frozen.displayThreshold) {
      throw new Error('A terminal harness badge used a decision threshold other than the frozen display threshold.');
    }
  }
  return { byAsset, pages: normalizedPages };
}

function validateAnalysisEvent(event, index) {
  const field = `capture.analyses[${index}]`;
  if (!event || event.schema !== ANALYSIS_DEBUG_SCHEMA) throw new Error(`${field}.schema must be ${ANALYSIS_DEBUG_SCHEMA}.`);
  if (!DEBUG_SESSION_ID.test(event.sessionId || '')) throw new Error(`${field}.sessionId must be 32 lowercase hexadecimal characters.`);
  positiveInteger(event.sequence, `${field}.sequence`);
  requiredString(event.requestId, `${field}.requestId`);
  if (
    event.source?.kind !== 'loopback-harness-asset'
    || !ASSET_ID.test(event.source?.assetId || '')
    || !HARNESS_RUN_ID.test(event.source?.harnessRunId || '')
  ) {
    throw new Error(`${field} is not joinable to an opaque loopback calibration asset.`);
  }
  validateLoopbackOrigin(event.source?.origin, `${field}.source.origin`);
  validateIdentity(event.identity, `${field}.identity`);
  validatePreprocessing(event.preprocessing, `${field}.preprocessing`);
  validateRuntime(event.runtime, `${field}.runtime`);
  validateEnvironment(event.environment, `${field}.environment`);
  requiredString(event.input?.format, `${field}.input.format`);
  positiveInteger(event.input?.width, `${field}.input.width`);
  positiveInteger(event.input?.height, `${field}.input.height`);
  if (!SHA256.test(event.input?.sha256 || '')) throw new Error(`${field}.input.sha256 must be a lowercase SHA-256.`);
  positiveInteger(event.input?.byteLength, `${field}.input.byteLength`);
  finiteNumber(event.scores?.modelLogit, `${field}.scores.modelLogit`);
  for (const name of ['modelSigmoid', 'fused', 'decision', 'rawThreshold', 'displayThreshold']) {
    probability(event.scores?.[name], `${field}.scores.${name}`);
  }
  for (const name of ['rawPredicted', 'displayedPredicted']) {
    if (!LABELS.has(event.scores?.[name])) throw new Error(`${field}.scores.${name} must be ai or real.`);
  }
  validateTimings(event.timings, `${field}.timings`);
}

function validateIdentity(identity, field) {
  if (!identity || typeof identity !== 'object') throw new Error(`${field} is required.`);
  requiredString(identity.extensionVersion, `${field}.extensionVersion`);
  const candidate = identity.candidate;
  requiredString(candidate?.id, `${field}.candidate.id`);
  if (candidate?.decisionPolicy !== 'model-only') throw new Error(`${field}.candidate.decisionPolicy must be model-only.`);
  const model = identity.model;
  for (const name of ['id', 'upstreamId', 'revision', 'dtype', 'outputActivation', 'weightFile']) {
    requiredString(model?.[name], `${field}.model.${name}`);
  }
  positiveInteger(model?.inputSize, `${field}.model.inputSize`);
  positiveInteger(model?.resizeSize, `${field}.model.resizeSize`);
  for (const name of ['sourceWeightSha256', 'sourceModelSha256', 'weightSha256', 'configSha256', 'preprocessorConfigSha256']) {
    if (!SHA256.test(model?.[name] || '')) throw new Error(`${field}.model.${name} must be a lowercase SHA-256.`);
  }
  const runtime = identity.runtimeArtifacts;
  requiredString(runtime?.transformersJsVersion, `${field}.runtimeArtifacts.transformersJsVersion`);
  requiredString(runtime?.onnxRuntimeWebVersion, `${field}.runtimeArtifacts.onnxRuntimeWebVersion`);
  requiredString(runtime?.wasmModule, `${field}.runtimeArtifacts.wasmModule`);
  requiredString(runtime?.wasmLoader, `${field}.runtimeArtifacts.wasmLoader`);
  for (const name of ['wasmSha256', 'wasmLoaderSha256']) {
    if (!SHA256.test(runtime?.[name] || '')) throw new Error(`${field}.runtimeArtifacts.${name} must be a lowercase SHA-256.`);
  }
}

function validatePreprocessing(value, field) {
  if (!value || typeof value !== 'object') throw new Error(`${field} is required.`);
  requiredString(value.processorType, `${field}.processorType`);
  requiredBoolean(value.doResize, `${field}.doResize`);
  positiveInteger(value.shortestEdge, `${field}.shortestEdge`);
  nonNegativeInteger(value.resample, `${field}.resample`);
  requiredBoolean(value.doCenterCrop, `${field}.doCenterCrop`);
  positiveInteger(value.cropSize?.width, `${field}.cropSize.width`);
  positiveInteger(value.cropSize?.height, `${field}.cropSize.height`);
  requiredBoolean(value.doRescale, `${field}.doRescale`);
  finiteNumber(value.rescaleFactor, `${field}.rescaleFactor`);
  requiredBoolean(value.doNormalize, `${field}.doNormalize`);
  finiteVector(value.imageMean, 3, `${field}.imageMean`);
  finiteVector(value.imageStd, 3, `${field}.imageStd`);
  requiredBoolean(value.doConvertRgb, `${field}.doConvertRgb`);
  requiredBoolean(value.doFlipChannelOrder, `${field}.doFlipChannelOrder`);
}

function validateRuntimeRequirements(value) {
  if (!value || typeof value !== 'object') throw new Error('reference.runtimeRequirements is required.');
  return {
    backend: requiredString(value.backend, 'reference.runtimeRequirements.backend'),
    crossOriginIsolated: requiredBoolean(value.crossOriginIsolated, 'reference.runtimeRequirements.crossOriginIsolated'),
    minimumActiveThreads: positiveInteger(value.minimumActiveThreads, 'reference.runtimeRequirements.minimumActiveThreads'),
    minimumChromeMajor: positiveInteger(value.minimumChromeMajor, 'reference.runtimeRequirements.minimumChromeMajor')
  };
}

function applyRuntimeGates(ready, browser, requirements, failures) {
  if (ready.runtime.backend !== requirements.backend) failures.push(failure('BACKEND_MISMATCH', `Expected ${requirements.backend}; received ${ready.runtime.backend}.`));
  if (ready.runtime.crossOriginIsolated !== requirements.crossOriginIsolated) failures.push(failure('ISOLATION_MISMATCH', 'crossOriginIsolated differs from the frozen requirement.'));
  if (ready.runtime.activeThreads < requirements.minimumActiveThreads) failures.push(failure('THREAD_COUNT', `Active thread count ${ready.runtime.activeThreads} is below ${requirements.minimumActiveThreads}.`));
  const chromeMajor = parseChromeMajor(browser.version);
  const userAgentMajor = parseUserAgentChromeMajor(ready.environment.userAgent);
  if (chromeMajor !== userAgentMajor) failures.push(failure('CHROME_IDENTITY_MISMATCH', 'Capture browser version and runtime user agent major disagree.'));
  if (chromeMajor < requirements.minimumChromeMajor) failures.push(failure('CHROME_VERSION', `Chrome ${chromeMajor} is below required major ${requirements.minimumChromeMajor}.`));
  if (browser.platform !== ready.environment.platform) failures.push(failure('PLATFORM_MISMATCH', 'Capture and runtime platform identities disagree.'));
  if (ready.runtime.activeThreads > ready.runtime.requestedThreads) failures.push(failure('THREAD_COUNT', 'Active thread count exceeds the requested thread count.'));
  if (ready.runtime.requestedThreads > ready.environment.hardwareConcurrency) failures.push(failure('THREAD_COUNT', 'Requested thread count exceeds navigator.hardwareConcurrency.'));
}

function validateRuntime(value, field) {
  if (!value || value.state !== 'ready') throw new Error(`${field}.state must be ready.`);
  requiredString(value.backend, `${field}.backend`);
  requiredBoolean(value.crossOriginIsolated, `${field}.crossOriginIsolated`);
  positiveInteger(value.requestedThreads, `${field}.requestedThreads`);
  positiveInteger(value.activeThreads, `${field}.activeThreads`);
}

function validateEnvironment(value, field) {
  if (!EXTENSION_ID.test(value?.extensionId || '')) throw new Error(`${field}.extensionId must be a 32-character Chrome extension ID.`);
  requiredString(value?.userAgent, `${field}.userAgent`);
  requiredString(value?.platform, `${field}.platform`);
  positiveInteger(value?.hardwareConcurrency, `${field}.hardwareConcurrency`);
}

function validateLatencyGate(value) {
  if (!value || typeof value !== 'object') throw new Error('A preregistered latency gate is required.');
  const excludeFirst = value.excludeFirst;
  if (excludeFirst !== 1) throw new Error('latency.excludeFirst must be exactly 1.');
  const result = { excludeFirst };
  for (const name of Object.keys(LATENCY_LIMITS)) {
    const limit = finiteNumber(value[name], `reference.gates.latency.${name}`);
    if (limit <= 0) throw new Error(`reference.gates.latency.${name} must be positive.`);
    if (limit > MAXIMUM_LATENCY_LIMITS[name]) {
      throw new Error(`reference.gates.latency.${name} must not exceed ${MAXIMUM_LATENCY_LIMITS[name]} ms.`);
    }
    result[name] = limit;
  }
  return result;
}

function evaluateLatency(gate, observations, failures) {
  const measured = observations.slice(gate.excludeFirst);
  if (measured.length === 0) throw new Error('Latency warm-up exclusion removed every observation.');
  const inference = measured.map((row) => row.timings.inferenceMs);
  const total = measured.map((row) => row.timings.totalMs);
  const metrics = {
    excludedWarmupObservations: gate.excludeFirst,
    measuredObservations: measured.length,
    medianInferenceMs: percentile(inference, 0.5),
    p90InferenceMs: percentile(inference, 0.9),
    p95InferenceMs: percentile(inference, 0.95),
    maximumInferenceMs: Math.max(...inference),
    meanInferenceMs: mean(inference),
    medianTotalMs: percentile(total, 0.5),
    p95TotalMs: percentile(total, 0.95),
    maximumTotalMs: Math.max(...total),
    meanTotalMs: mean(total),
    limits: { ...gate }
  };
  for (const [limitName, metricName] of Object.entries(LATENCY_LIMITS)) {
    if (gate[limitName] == null) continue;
    if (metrics[metricName] > gate[limitName]) failures.push(failure('LATENCY_LIMIT', `${metricName} ${metrics[metricName]} ms exceeds ${gate[limitName]} ms.`, { metric: metricName }));
  }
  return metrics;
}

function validateDecisionGates(value, manifestRows) {
  if (!Array.isArray(value) || value.length === 0) throw new Error('At least one preregistered decision gate is required.');
  const groups = new Map();
  for (const row of manifestRows) {
    const key = canonicalJson({ label: row.label, transformId: row.transformId, domain: row.domain });
    const group = groups.get(key) || { label: row.label, transformId: row.transformId, domain: row.domain, rows: 0 };
    group.rows += 1;
    groups.set(key, group);
  }
  if (value.length !== groups.size) throw new Error('Decision gates must contain exactly one label/domain/transform gate for every manifest group.');
  const ids = new Set();
  const groupKeys = new Set();
  const gates = value.map((gate, index) => {
    const id = requiredString(gate?.id, `reference.gates.decisions[${index}].id`);
    if (ids.has(id)) throw new Error(`Duplicate decision gate ID: ${id}`);
    ids.add(id);
    if (!gate.match || typeof gate.match !== 'object') throw new Error(`Decision gate ${id} must declare a match object.`);
    const matchFields = Object.keys(gate.match).sort();
    if (canonicalJson(matchFields) !== canonicalJson(['domain', 'label', 'transformId'])) {
      throw new Error(`Decision gate ${id} must match exactly label, domain, and transformId.`);
    }
    const match = {};
    for (const [field, raw] of Object.entries(gate.match)) {
      if (!MATCH_FIELDS.has(field)) throw new Error(`Decision gate ${id} uses unsupported match field ${field}.`);
      match[field] = field === 'label'
        ? requiredString(raw, `decision gate ${id}.match.${field}`).toLowerCase()
        : requiredString(raw, `decision gate ${id}.match.${field}`);
    }
    if (!LABELS.has(match.label)) throw new Error(`Decision gate ${id} has an invalid label.`);
    const groupKey = canonicalJson({ label: match.label, transformId: match.transformId, domain: match.domain });
    if (groupKeys.has(groupKey) || !groups.has(groupKey)) throw new Error(`Decision gate ${id} does not identify one unique manifest group.`);
    groupKeys.add(groupKey);
    const expectedRows = positiveInteger(gate.expectedRows, `decision gate ${id}.expectedRows`);
    if (expectedRows !== groups.get(groupKey).rows) throw new Error(`Decision gate ${id} expectedRows does not match its frozen manifest group.`);
    const minimumCorrect = positiveInteger(gate.minimumCorrect, `decision gate ${id}.minimumCorrect`);
    if (minimumCorrect > expectedRows) throw new Error(`Decision gate ${id} minimumCorrect exceeds expectedRows.`);
    const minimumAllowed = Math.ceil(expectedRows * MINIMUM_RECALL[match.label]);
    if (minimumCorrect < minimumAllowed) {
      throw new Error(`Decision gate ${id} minimumCorrect must be at least ${minimumAllowed}/${expectedRows}.`);
    }
    return { id, match, expectedRows, minimumCorrect };
  });
  if (groupKeys.size !== groups.size) throw new Error('Decision gates do not cover every frozen manifest group.');
  return gates;
}

function evaluateDecisionGate(gate, observations, failures) {
  const selected = observations.filter((row) => Object.entries(gate.match).every(([field, expected]) => row[field] === expected));
  const correct = selected.filter((row) => row.correct).length;
  const pass = selected.length === gate.expectedRows && correct >= gate.minimumCorrect;
  if (!pass) failures.push(failure('DECISION_GATE', `${gate.id} observed ${correct}/${selected.length}; requires ${gate.minimumCorrect}/${gate.expectedRows}.`, { gateId: gate.id }));
  return { ...gate, observedRows: selected.length, correct, pass };
}

function validateTimings(value, field) {
  if (!value || typeof value !== 'object') throw new Error(`${field} is required.`);
  for (const name of ['fetchMs', 'decodeMs', 'modelWaitMs', 'preprocessMs', 'inferenceMs', 'evidenceMs', 'totalMs']) {
    const timing = finiteNumber(value[name], `${field}.${name}`);
    if (timing < 0) throw new Error(`${field}.${name} must be non-negative.`);
  }
  const componentTotal = ['fetchMs', 'decodeMs', 'modelWaitMs', 'preprocessMs', 'inferenceMs', 'evidenceMs']
    .reduce((sum, name) => sum + value[name], 0);
  if (Math.abs(componentTotal - value.totalMs) > 4) {
    throw new Error(`${field}.totalMs is inconsistent with its stage timings.`);
  }
}

async function readGuardedJson(projectRoot, configuredPath, label) {
  const benchRoot = await realpath(resolve(projectRoot, '.bench-data')).catch(() => {
    throw new Error('The ignored .bench-data root does not exist.');
  });
  const lexicalPath = isAbsolute(String(configuredPath)) ? resolve(String(configuredPath)) : resolve(projectRoot, String(configuredPath));
  assertContained(benchRoot, lexicalPath, `${label} must remain inside .bench-data.`);
  assertNoForbiddenComponents(relative(benchRoot, lexicalPath), `${label} path`);
  const path = await realpath(lexicalPath).catch(() => { throw new Error(`${label} does not exist.`); });
  assertContained(benchRoot, path, `${label} resolves outside .bench-data.`);
  assertNoForbiddenComponents(relative(benchRoot, path), `${label} resolved path`);
  if (!(await stat(path)).isFile()) throw new Error(`${label} must be a regular file.`);
  const bytes = await readFile(path);
  let document;
  try {
    document = JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
  return { document, sha256: sha256(bytes) };
}

function assertSafeRelativeCalibrationPath(value, field) {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value) || value.startsWith('\\\\') || isAbsolute(value) || value.includes('\0')) {
    throw new Error(`${field} must be a local relative calibration path.`);
  }
  if (value.split(/[\\/]+/).includes('..')) throw new Error(`${field} must not contain path traversal.`);
  assertNoForbiddenComponents(value, field);
}

function assertNoForbiddenComponents(value, field) {
  const components = String(value).split(/[\\/]+/).filter(Boolean);
  const forbidden = components.find((component) => /^\.*(?:qualification|private[-_]?diagnostic)(?:[-_.]|$)/i.test(component));
  if (forbidden) throw new Error(`${field} refers to forbidden qualification/private-diagnostic data.`);
}

function assertContained(parent, candidate, message) {
  const pathFromParent = relative(parent, candidate);
  if (pathFromParent === '..' || pathFromParent.startsWith(`..${sep}`) || isAbsolute(pathFromParent)) throw new Error(message);
}

function validateLoopbackOrigin(value, field) {
  const input = requiredString(value, field);
  let url;
  try {
    url = new URL(input);
  } catch {
    throw new Error(`${field} must be a loopback HTTP origin.`);
  }
  if (
    url.protocol !== 'http:'
    || !['127.0.0.1', 'localhost'].includes(url.hostname.toLowerCase())
    || url.origin !== input
    || url.username
    || url.password
  ) {
    throw new Error(`${field} must be a loopback HTTP origin.`);
  }
  return url.origin;
}

function parseChromeMajor(value) {
  const match = /^(\d+)\./.exec(requiredString(value, 'Chrome version'));
  if (!match) throw new Error('Chrome version must begin with a numeric major and dot.');
  return Number(match[1]);
}

function parseUserAgentChromeMajor(value) {
  const match = /Chrome\/(\d+)\./.exec(requiredString(value, 'Chrome user agent'));
  if (!match) throw new Error('Runtime user agent does not identify Chrome with a versioned user agent token.');
  return Number(match[1]);
}

function calibrateForDisplay(score, rawThreshold, displayThreshold) {
  if (score === 0 || score === 1) return score;
  const logit = (value) => Math.log(value / (1 - value));
  return stableSigmoid(logit(score) - logit(rawThreshold) + logit(displayThreshold));
}

function percentile(values, fraction) {
  const sorted = values.slice().sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(fraction * sorted.length) - 1)];
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function comparePins(left, right) {
  return left.sha256.localeCompare(right.sha256) || left.rows - right.rows;
}

function failure(code, message, context = undefined) {
  return context ? { code, message, ...context } : { code, message };
}

function stableId(value, length = 16) {
  return createHash('sha256').update(String(value)).digest('hex').slice(0, length);
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJson(value) {
  return JSON.stringify(sortJson(value));
}

function normalizeInputArtifactDigests(value) {
  if (value == null) throw new Error('Guarded reference and capture artifact SHA-256 metadata is required.');
  if (!SHA256.test(value.referenceSha256 || '') || !SHA256.test(value.captureSha256 || '')) {
    throw new Error('Input artifact SHA-256 metadata is invalid.');
  }
  return {
    referenceSha256: value.referenceSha256,
    captureSha256: value.captureSha256
  };
}

function sortJson(value) {
  if (Array.isArray(value)) return value.map(sortJson);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortJson(value[key])]));
  }
  return value;
}

function requiredString(value, field) {
  if (typeof value !== 'string' || value.trim() === '') throw new Error(`${field} must be a non-empty string.`);
  return value.trim();
}

function optionalString(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function requiredBoolean(value, field) {
  if (typeof value !== 'boolean') throw new Error(`${field} must be boolean.`);
  return value;
}

function finiteNumber(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${field} must be finite.`);
  return value;
}

function probability(value, field) {
  const number = finiteNumber(value, field);
  if (number < 0 || number > 1) throw new Error(`${field} must be in [0, 1].`);
  return number;
}

function openProbability(value, field) {
  const number = probability(value, field);
  if (number === 0 || number === 1) throw new Error(`${field} must be between zero and one.`);
  return number;
}

function positiveInteger(value, field) {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${field} must be a positive integer.`);
  return value;
}

function nonNegativeInteger(value, field) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${field} must be a non-negative integer.`);
  return value;
}

function finiteVector(value, length, field) {
  if (!Array.isArray(value) || value.length !== length) throw new Error(`${field} must contain ${length} values.`);
  value.forEach((item, index) => finiteNumber(item, `${field}[${index}]`));
}

function parseArgs(argv) {
  const options = { manifestPaths: [] };
  for (const argument of argv) {
    if (argument === '--help' || argument === '-h') options.help = true;
    else if (argument.startsWith('--manifest=')) options.manifestPaths.push(argument.slice('--manifest='.length));
    else if (argument.startsWith('--reference=')) options.referencePath = argument.slice('--reference='.length);
    else if (argument.startsWith('--capture=')) options.capturePath = argument.slice('--capture='.length);
    else if (argument.startsWith('--output=')) options.outputPath = argument.slice('--output='.length);
    else throw new Error(`Unknown option: ${argument}`);
  }
  return options;
}

async function writeGuardedReport(projectRoot, configuredPath, report) {
  const benchRoot = await realpath(resolve(projectRoot, '.bench-data'));
  const path = isAbsolute(String(configuredPath)) ? resolve(String(configuredPath)) : resolve(projectRoot, String(configuredPath));
  assertContained(benchRoot, path, 'Parity report must remain inside .bench-data.');
  assertNoForbiddenComponents(relative(benchRoot, path), 'parity report path');
  const parent = await realpath(dirname(path)).catch(() => { throw new Error('Parity report parent directory must already exist.'); });
  assertContained(benchRoot, parent, 'Parity report parent resolves outside .bench-data.');
  await writeFile(path, `${JSON.stringify(report, null, 2)}\n`, { flag: 'wx' });
  return { sha256: sha256(await readFile(path)), path };
}

async function runCli() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log('Usage: node scripts/audit-installed-chrome-parity.mjs --manifest=PATH [--manifest=PATH ...] --reference=PATH --capture=PATH --output=PATH');
    console.log(`All inputs and output must stay in non-qualification .bench-data paths; exactly ${REQUIRED_OBSERVATIONS} direct calibration observations are required.`);
    return;
  }
  if (!options.referencePath || !options.capturePath || !options.outputPath || options.manifestPaths.length === 0) {
    throw new Error('Required: one or more --manifest, plus --reference, --capture, and --output.');
  }
  const inputs = await loadGuardedParityInputs({ projectRoot: PROJECT_ROOT, ...options });
  const report = auditInstalledChromeParity(inputs);
  const artifact = await writeGuardedReport(PROJECT_ROOT, options.outputPath, report);
  console.log(JSON.stringify({ eligible: report.eligible, failures: report.failures, report: artifact }, null, 2));
  if (!report.eligible) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await runCli().catch((error) => {
    console.error(`Installed-Chrome parity gate failed closed: ${error.message}`);
    process.exitCode = 1;
  });
}
