import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';

const MANIFEST_SCHEMA = 'proofmark-modern-holdout-matrix-v1';
const DISJOINT_POLICY = 'source-group+source-row+sha256+dhash-v1';
const REQUIRED_TRANSFORMS = new Set([
  'double-jpeg-q85-q45',
  'thumbnail-256-webp-q60-upscale',
  'offcenter-crop-75-jpeg-q70',
  'screenshot-social-jpeg-q68'
]);
const CLAIM_GATES = Object.freeze({
  minimumAiDomains: 9,
  minimumAiFamiliesPerDomain: 24,
  minimumAiFamilies: 216,
  minimumRealSlices: 7,
  minimumRealFamiliesPerSlice: 20,
  minimumRealFamilies: 216,
  cleanBalancedAccuracy: 0.82,
  stressBalancedAccuracy: 0.78,
  cleanAiMacroRecall: 0.75,
  stressAiMacroRecall: 0.75,
  minimumGeneratorRecall: 0.60,
  cleanRealRecall: 0.92,
  stressRealRecall: 0.88,
  minimumRealSliceRecall: 0.85,
  balancedAccuracyDelta: 0.03,
  aiMacroRecallDelta: 0.05,
  maximumRealRecallLoss: 0.01,
  bootstrapLowerBound: 0
});

await main().catch((error) => {
  console.error(`Slice summary failed: ${error.message}`);
  process.exitCode = 1;
});

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }
  for (const name of ['manifest', 'baseline-clean', 'baseline-stress', 'candidate-clean', 'candidate-stress']) {
    if (!options[name]) throw new Error(`Missing required --${name}=PATH.`);
  }

  const bootstrapIterations = options.bootstrap === undefined ? 10_000 : parseInteger(options.bootstrap, '--bootstrap');
  if (bootstrapIterations < 200) throw new Error('--bootstrap must be at least 200 iterations.');
  const seed = options.seed === undefined ? 323 : parseInteger(options.seed, '--seed');
  const manifest = await readJson(resolve(options.manifest));
  await validateManifest(manifest);

  const expected = {
    clean: manifest.rows.filter((row) => row.view === 'clean'),
    stress: manifest.rows.filter((row) => row.view === 'stress')
  };
  const resultDocuments = {
    baselineClean: await readJson(resolve(options['baseline-clean'])),
    baselineStress: await readJson(resolve(options['baseline-stress'])),
    candidateClean: await readJson(resolve(options['candidate-clean'])),
    candidateStress: await readJson(resolve(options['candidate-stress']))
  };
  const runs = {
    baseline: {
      clean: evaluateRun('baseline-clean', resultDocuments.baselineClean, expected.clean),
      stress: evaluateRun('baseline-stress', resultDocuments.baselineStress, expected.stress)
    },
    candidate: {
      clean: evaluateRun('candidate-clean', resultDocuments.candidateClean, expected.clean),
      stress: evaluateRun('candidate-stress', resultDocuments.candidateStress, expected.stress)
    }
  };

  const comparisons = {};
  for (const [index, view] of ['clean', 'stress'].entries()) {
    comparisons[view] = compareRuns(
      runs.baseline[view],
      runs.candidate[view],
      expected[view],
      bootstrapIterations,
      seed + index * 10_007
    );
  }

  const cohort = summarizeCohort(expected.clean);
  const gateResults = evaluateGates(cohort, runs, comparisons);
  const smoke = options.smoke === true;
  const summary = {
    schema: 'proofmark-modern-holdout-summary-v1',
    benchmarkId: manifest.benchmarkId,
    sourceManifestSha256: manifest.sourceManifest?.sha256,
    mode: smoke ? 'SMOKE_NOT_CLAIM_ELIGIBLE' : 'CLAIM_GATE',
    claimEligible: !smoke && gateResults.every((gate) => gate.pass),
    policy: {
      missingResultsCountAsIncorrect: true,
      erroredOrInvalidResultsCountAsIncorrect: true,
      bootstrapUnit: 'original familyId; all transforms remain in the sampled cluster',
      bootstrapPairing: 'candidate and baseline use the identical sampled original clusters',
      bootstrapIterations,
      seed
    },
    cohort,
    runs: {
      baseline: { clean: publicRun(runs.baseline.clean), stress: publicRun(runs.baseline.stress) },
      candidate: { clean: publicRun(runs.candidate.clean), stress: publicRun(runs.candidate.stress) }
    },
    comparisons,
    gates: {
      thresholds: CLAIM_GATES,
      allPassed: gateResults.every((gate) => gate.pass),
      results: gateResults
    }
  };

  const serialized = `${JSON.stringify(summary, null, 2)}\n`;
  if (options.output) {
    await writeFile(resolve(options.output), serialized);
    console.log(`Detailed summary: ${resolve(options.output)}`);
  } else {
    process.stdout.write(serialized);
  }
  console.log(`Candidate clean BA: ${(runs.candidate.clean.metrics.balancedAccuracy * 100).toFixed(1)}%`);
  console.log(`Candidate stress BA: ${(runs.candidate.stress.metrics.balancedAccuracy * 100).toFixed(1)}%`);
  console.log(`Claim gates: ${gateResults.filter((gate) => gate.pass).length}/${gateResults.length} passed${smoke ? ' (smoke mode cannot qualify a claim)' : ''}`);
  if (!smoke && !summary.claimEligible) process.exitCode = 2;
}

function parseArgs(argv) {
  const options = {};
  for (const value of argv) {
    if (value === '--help' || value === '-h') options.help = true;
    else if (value === '--smoke') options.smoke = true;
    else if (/^--[a-z-]+=/.test(value)) {
      const separator = value.indexOf('=');
      options[value.slice(2, separator)] = value.slice(separator + 1);
    } else throw new Error(`Unknown argument: ${value}`);
  }
  return options;
}

function printUsage() {
  console.log('Usage: node scripts/summarize-slices.mjs --manifest=PATH --baseline-clean=PATH --baseline-stress=PATH --candidate-clean=PATH --candidate-stress=PATH [--output=PATH] [--bootstrap=10000] [--seed=323]');
  console.log('Use --smoke only for synthetic wiring tests; smoke output is always marked ineligible for a claim.');
}

async function validateManifest(manifest) {
  if (!manifest || manifest.schema !== MANIFEST_SCHEMA) throw new Error(`Manifest schema must be ${MANIFEST_SCHEMA}.`);
  if (typeof manifest.benchmarkId !== 'string' || !manifest.benchmarkId) throw new Error('Manifest benchmarkId is required.');
  if (manifest.sourceDisjointness?.confirmed !== true || manifest.sourceDisjointness?.policy !== DISJOINT_POLICY) {
    throw new Error('Manifest does not carry a confirmed source-disjointness audit.');
  }
  if (!manifest.transformationPolicy?.labelBlind || !manifest.transformationPolicy?.everyTransformAppliedToEveryOriginal) {
    throw new Error('Manifest does not guarantee an identical label-blind transform matrix.');
  }
  const declaredTransforms = new Set((manifest.transformationPolicy.transforms || []).map((item) => item.id));
  if (!sameSet(declaredTransforms, REQUIRED_TRANSFORMS)) throw new Error('Manifest transform declaration is incomplete or unexpected.');
  if (!Array.isArray(manifest.rows) || manifest.rows.length === 0) throw new Error('Manifest rows must be non-empty.');

  const sourceReference = manifest.sourceManifest;
  if (!sourceReference || sourceReference.schema !== 'proofmark-modern-holdout-source-v1') {
    throw new Error('Manifest must reference its frozen source manifest.');
  }
  if (typeof sourceReference.path !== 'string' || !sourceReference.path || !/^[a-f0-9]{64}$/.test(sourceReference.sha256 || '')) {
    throw new Error('Manifest source reference needs a local path and lowercase SHA-256.');
  }
  const sourcePath = resolve(sourceReference.path);
  const sourceBytes = await readFile(sourcePath).catch((error) => {
    throw new Error(`Cannot verify frozen source manifest ${sourcePath}: ${error.message}`);
  });
  if (sha256(sourceBytes) !== sourceReference.sha256) throw new Error('Frozen source-manifest SHA-256 mismatch.');
  let sourceManifest;
  try {
    sourceManifest = JSON.parse(sourceBytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Invalid frozen source manifest JSON: ${error.message}`);
  }
  if (sourceManifest.schema !== sourceReference.schema || sourceManifest.benchmarkId !== manifest.benchmarkId) {
    throw new Error('Frozen source manifest identity does not match the matrix.');
  }
  if (sourceManifest.sourceDisjointness?.confirmed !== true || sourceManifest.sourceDisjointness?.policy !== DISJOINT_POLICY) {
    throw new Error('Frozen source manifest lacks a confirmed disjointness audit.');
  }
  for (const [index, checked] of (sourceManifest.sourceDisjointness.checkedAgainst || []).entries()) {
    if (typeof checked.manifest !== 'string' || !/^[a-f0-9]{64}$/.test(checked.sha256 || '')) {
      throw new Error(`Frozen source manifest exclusion ${index} is incomplete.`);
    }
    const exclusionBytes = await readFile(resolve(checked.manifest)).catch((error) => {
      throw new Error(`Cannot reverify exclusion manifest ${checked.manifest}: ${error.message}`);
    });
    if (sha256(exclusionBytes) !== checked.sha256) throw new Error(`Exclusion-manifest SHA-256 mismatch: ${checked.manifest}`);
  }

  const sampleIds = new Set();
  const paths = new Set();
  const families = new Map();
  for (const [index, row] of manifest.rows.entries()) {
    const prefix = `manifest.rows[${index}]`;
    for (const field of ['sampleId', 'familyId', 'label', 'domain', 'slice', 'transformId', 'benchmarkPath']) {
      if (typeof row[field] !== 'string' || !row[field]) throw new Error(`${prefix}.${field} is required.`);
    }
    if (row.label !== 'ai' && row.label !== 'real') throw new Error(`${prefix}.label must be ai or real.`);
    if (row.view !== 'clean' && row.view !== 'stress') throw new Error(`${prefix}.view must be clean or stress.`);
    if ((row.view === 'clean') !== (row.transformId === 'original')) throw new Error(`${prefix} has an inconsistent view/transformId.`);
    validateBenchmarkPath(row.benchmarkPath, `${prefix}.benchmarkPath`);
    if (sampleIds.has(row.sampleId)) throw new Error(`Duplicate sampleId: ${row.sampleId}`);
    if (paths.has(canonicalPath(row.benchmarkPath))) throw new Error(`Duplicate benchmarkPath: ${row.benchmarkPath}`);
    sampleIds.add(row.sampleId);
    paths.add(canonicalPath(row.benchmarkPath));
    validateRights(row, prefix);
    const existingLabel = families.get(row.familyId);
    if (existingLabel && existingLabel !== row.label) throw new Error(`familyId ${row.familyId} crosses labels.`);
    families.set(row.familyId, row.label);
  }

  const rowsByFamily = groupBy(manifest.rows, (row) => row.familyId);
  for (const [familyId, rows] of rowsByFamily) {
    const clean = rows.filter((row) => row.view === 'clean');
    const stress = rows.filter((row) => row.view === 'stress');
    if (clean.length !== 1 || clean[0].transformId !== 'original') throw new Error(`${familyId} must have exactly one original.`);
    if (!sameSet(new Set(stress.map((row) => row.transformId)), REQUIRED_TRANSFORMS) || stress.length !== REQUIRED_TRANSFORMS.size) {
      throw new Error(`${familyId} does not have the complete stress matrix.`);
    }
  }

  const cleanByFamily = new Map(manifest.rows.filter((row) => row.view === 'clean').map((row) => [row.familyId, row]));
  if (!Array.isArray(sourceManifest.rows) || sourceManifest.rows.length !== cleanByFamily.size) {
    throw new Error('Frozen source rows do not match the matrix family count.');
  }
  for (const source of sourceManifest.rows) {
    const clean = cleanByFamily.get(source.familyId);
    if (!clean || clean.sha256 !== source.sha256 || clean.label !== source.label || clean.domain !== source.domain || clean.slice !== source.slice || clean.sourceDataset !== source.sourceDataset || String(clean.sourceRow) !== String(source.sourceRow) || clean.sourceGroup !== source.sourceGroup) {
      throw new Error(`Matrix original does not match frozen source family ${source.familyId}.`);
    }
  }
}

function validateRights(row, prefix) {
  const license = row.license;
  if (!license || license.status !== 'approved') throw new Error(`${prefix} has unresolved license evidence.`);
  for (const field of ['identifier', 'url', 'verifiedAt', 'attribution']) {
    if (typeof license[field] !== 'string' || !license[field]) throw new Error(`${prefix}.license.${field} is required.`);
  }
  requireHttps(license.url, `${prefix}.license.url`);
  requireIsoDate(license.verifiedAt, `${prefix}.license.verifiedAt`);
  for (const field of ['commercialUse', 'evaluationUse', 'derivatives', 'localCopy']) {
    if (license[field] !== true) throw new Error(`${prefix}.license.${field} is not approved.`);
  }
  const provenance = row.provenance;
  if (!provenance || provenance.status !== 'verified') throw new Error(`${prefix} has unresolved provenance evidence.`);
  for (const field of ['sourceUrl', 'immutableRevision', 'creator', 'labelMethod', 'retrievedAt']) {
    if (typeof provenance[field] !== 'string' || !provenance[field]) throw new Error(`${prefix}.provenance.${field} is required.`);
  }
  requireHttps(provenance.sourceUrl, `${prefix}.provenance.sourceUrl`);
  requireIsoDate(provenance.retrievedAt, `${prefix}.provenance.retrievedAt`);
}

function evaluateRun(name, document, expectedRows) {
  if (!document || !Array.isArray(document.rows)) throw new Error(`${name} result document must contain rows.`);
  if (typeof document.model !== 'string' || !document.model) throw new Error(`${name} result document must identify its model.`);
  const expectedByPath = new Map(expectedRows.map((row) => [canonicalPath(row.benchmarkPath), row]));
  const actualByPath = new Map();
  for (const [index, row] of document.rows.entries()) {
    if (!row || typeof row.file !== 'string') throw new Error(`${name}.rows[${index}].file is required.`);
    const path = canonicalPath(row.file);
    if (!expectedByPath.has(path)) throw new Error(`${name} contains an unexpected result path: ${row.file}`);
    if (actualByPath.has(path)) throw new Error(`${name} contains a duplicate result path: ${row.file}`);
    actualByPath.set(path, row);
  }

  const outcomes = expectedRows.map((expected) => {
    const actual = actualByPath.get(canonicalPath(expected.benchmarkPath));
    let failure = null;
    if (!actual) failure = 'missing-result';
    else if (actual.error) failure = 'reported-error';
    else if (actual.expected !== expected.label) failure = 'expected-label-mismatch';
    else if (actual.predicted !== 'ai' && actual.predicted !== 'real') failure = 'invalid-prediction';
    else if (!Number.isFinite(actual.score) || actual.score < 0 || actual.score > 1) failure = 'invalid-score';
    const correct = failure === null && actual.predicted === expected.label;
    return {
      sampleId: expected.sampleId,
      familyId: expected.familyId,
      label: expected.label,
      domain: expected.domain,
      slice: expected.slice,
      transformId: expected.transformId,
      correct,
      failure
    };
  });
  return {
    name,
    model: document.model,
    outcomes,
    coverage: summarizeCoverage(outcomes),
    metrics: calculateMetrics(outcomes)
  };
}

function calculateMetrics(outcomes) {
  const ai = rate(outcomes.filter((row) => row.label === 'ai'));
  const real = rate(outcomes.filter((row) => row.label === 'real'));
  if (!ai.total || !real.total) throw new Error('Both labels must have at least one expected sample.');
  const aiDomains = groupStats(outcomes.filter((row) => row.label === 'ai'), (row) => row.domain, 'domain');
  const realSlices = groupStats(outcomes.filter((row) => row.label === 'real'), (row) => row.slice, 'slice');
  const transforms = groupStats(outcomes, (row) => row.transformId, 'transformId');
  return {
    aiRecall: ai.recall,
    realRecall: real.recall,
    balancedAccuracy: (ai.recall + real.recall) / 2,
    aiMacroRecall: mean(aiDomains.map((row) => row.recall)),
    minimumGeneratorRecall: Math.min(...aiDomains.map((row) => row.recall)),
    minimumRealSliceRecall: Math.min(...realSlices.map((row) => row.recall)),
    totals: { ai, real },
    aiDomains,
    realSlices,
    transforms
  };
}

function compareRuns(baseline, candidate, expectedRows, iterations, seed) {
  const delta = {
    balancedAccuracy: candidate.metrics.balancedAccuracy - baseline.metrics.balancedAccuracy,
    aiRecall: candidate.metrics.aiRecall - baseline.metrics.aiRecall,
    realRecall: candidate.metrics.realRecall - baseline.metrics.realRecall,
    aiMacroRecall: candidate.metrics.aiMacroRecall - baseline.metrics.aiMacroRecall
  };
  return {
    delta,
    balancedAccuracyBootstrap: pairedClusterBootstrap(baseline, candidate, expectedRows, iterations, seed)
  };
}

function pairedClusterBootstrap(baseline, candidate, expectedRows, iterations, seed) {
  const baselineBySample = new Map(baseline.outcomes.map((row) => [row.sampleId, row]));
  const candidateBySample = new Map(candidate.outcomes.map((row) => [row.sampleId, row]));
  const rowsByFamily = groupBy(expectedRows, (row) => row.familyId);
  const familyLabels = new Map();
  for (const [familyId, rows] of rowsByFamily) {
    const labels = new Set(rows.map((row) => row.label));
    if (labels.size !== 1) throw new Error(`Bootstrap family ${familyId} crosses labels.`);
    familyLabels.set(familyId, rows[0].label);
  }
  const aiFamilies = [...familyLabels].filter(([, label]) => label === 'ai').map(([familyId]) => familyId);
  const realFamilies = [...familyLabels].filter(([, label]) => label === 'real').map(([familyId]) => familyId);
  if (!aiFamilies.length || !realFamilies.length) throw new Error('Bootstrap requires both AI and real families.');

  const random = mulberry32(seed >>> 0);
  const deltas = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const selected = [
      ...resample(aiFamilies, random),
      ...resample(realFamilies, random)
    ];
    const totals = {
      baseline: { ai: 0, aiCorrect: 0, real: 0, realCorrect: 0 },
      candidate: { ai: 0, aiCorrect: 0, real: 0, realCorrect: 0 }
    };
    for (const familyId of selected) {
      for (const expected of rowsByFamily.get(familyId)) {
        for (const [key, outcomes] of [['baseline', baselineBySample], ['candidate', candidateBySample]]) {
          const outcome = outcomes.get(expected.sampleId);
          totals[key][expected.label] += 1;
          totals[key][`${expected.label}Correct`] += Number(outcome.correct);
        }
      }
    }
    const baselineBa = balancedAccuracy(totals.baseline);
    const candidateBa = balancedAccuracy(totals.candidate);
    deltas.push(candidateBa - baselineBa);
  }
  deltas.sort((left, right) => left - right);
  return {
    unit: 'familyId',
    stratifiedByLabel: true,
    iterations,
    seed,
    observedDelta: candidate.metrics.balancedAccuracy - baseline.metrics.balancedAccuracy,
    meanDelta: mean(deltas),
    confidenceLevel: 0.95,
    lower: quantile(deltas, 0.025),
    upper: quantile(deltas, 0.975)
  };
}

function summarizeCohort(cleanRows) {
  const families = new Map(cleanRows.map((row) => [row.familyId, row]));
  const originals = [...families.values()];
  const aiDomains = groupStats(originals.filter((row) => row.label === 'ai'), (row) => row.domain, 'domain');
  const realSlices = groupStats(originals.filter((row) => row.label === 'real'), (row) => row.slice, 'slice');
  return {
    aiFamilies: originals.filter((row) => row.label === 'ai').length,
    realFamilies: originals.filter((row) => row.label === 'real').length,
    aiDomains: aiDomains.map(({ domain, total }) => ({ domain, families: total })),
    realSlices: realSlices.map(({ slice, total }) => ({ slice, families: total }))
  };
}

function evaluateGates(cohort, runs, comparisons) {
  const candidate = runs.candidate;
  const baseline = runs.baseline;
  const gates = [];
  const add = (id, value, operator, threshold) => gates.push({ id, value, operator, threshold, pass: compare(value, operator, threshold) });

  add('cohort-ai-families', cohort.aiFamilies, '>=', CLAIM_GATES.minimumAiFamilies);
  add('cohort-ai-domains', cohort.aiDomains.length, '>=', CLAIM_GATES.minimumAiDomains);
  add('cohort-ai-minimum-per-domain', Math.min(...cohort.aiDomains.map((row) => row.families)), '>=', CLAIM_GATES.minimumAiFamiliesPerDomain);
  add('cohort-real-families', cohort.realFamilies, '>=', CLAIM_GATES.minimumRealFamilies);
  add('cohort-real-slices', cohort.realSlices.length, '>=', CLAIM_GATES.minimumRealSlices);
  add('cohort-real-minimum-per-slice', Math.min(...cohort.realSlices.map((row) => row.families)), '>=', CLAIM_GATES.minimumRealFamiliesPerSlice);
  add('candidate-clean-balanced-accuracy', candidate.clean.metrics.balancedAccuracy, '>=', CLAIM_GATES.cleanBalancedAccuracy);
  add('candidate-stress-balanced-accuracy', candidate.stress.metrics.balancedAccuracy, '>=', CLAIM_GATES.stressBalancedAccuracy);
  add('candidate-clean-ai-macro-recall', candidate.clean.metrics.aiMacroRecall, '>=', CLAIM_GATES.cleanAiMacroRecall);
  add('candidate-stress-ai-macro-recall', candidate.stress.metrics.aiMacroRecall, '>=', CLAIM_GATES.stressAiMacroRecall);
  add('candidate-clean-minimum-generator-recall', candidate.clean.metrics.minimumGeneratorRecall, '>=', CLAIM_GATES.minimumGeneratorRecall);
  add('candidate-stress-minimum-generator-recall', candidate.stress.metrics.minimumGeneratorRecall, '>=', CLAIM_GATES.minimumGeneratorRecall);
  add('candidate-clean-real-recall', candidate.clean.metrics.realRecall, '>=', CLAIM_GATES.cleanRealRecall);
  add('candidate-stress-real-recall', candidate.stress.metrics.realRecall, '>=', CLAIM_GATES.stressRealRecall);
  add('candidate-clean-minimum-real-slice-recall', candidate.clean.metrics.minimumRealSliceRecall, '>=', CLAIM_GATES.minimumRealSliceRecall);
  add('candidate-stress-minimum-real-slice-recall', candidate.stress.metrics.minimumRealSliceRecall, '>=', CLAIM_GATES.minimumRealSliceRecall);
  add('clean-balanced-accuracy-delta', comparisons.clean.delta.balancedAccuracy, '>=', CLAIM_GATES.balancedAccuracyDelta);
  add('stress-balanced-accuracy-delta', comparisons.stress.delta.balancedAccuracy, '>=', CLAIM_GATES.balancedAccuracyDelta);
  add('clean-ai-macro-recall-delta', comparisons.clean.delta.aiMacroRecall, '>=', CLAIM_GATES.aiMacroRecallDelta);
  add('stress-ai-macro-recall-delta', comparisons.stress.delta.aiMacroRecall, '>=', CLAIM_GATES.aiMacroRecallDelta);
  add('clean-real-recall-loss', comparisons.clean.delta.realRecall, '>=', -CLAIM_GATES.maximumRealRecallLoss);
  add('stress-real-recall-loss', comparisons.stress.delta.realRecall, '>=', -CLAIM_GATES.maximumRealRecallLoss);
  add('clean-bootstrap-lower-bound', comparisons.clean.balancedAccuracyBootstrap.lower, '>', CLAIM_GATES.bootstrapLowerBound);
  add('stress-bootstrap-lower-bound', comparisons.stress.balancedAccuracyBootstrap.lower, '>', CLAIM_GATES.bootstrapLowerBound);
  add('baseline-clean-complete', baseline.clean.coverage.failures, '===', 0);
  add('baseline-stress-complete', baseline.stress.coverage.failures, '===', 0);
  add('candidate-clean-complete', candidate.clean.coverage.failures, '===', 0);
  add('candidate-stress-complete', candidate.stress.coverage.failures, '===', 0);
  return gates;
}

function publicRun(run) {
  return {
    model: run.model,
    coverage: run.coverage,
    metrics: run.metrics,
    failureExamples: run.outcomes.filter((row) => row.failure).slice(0, 25)
  };
}

function summarizeCoverage(outcomes) {
  const byFailure = {};
  for (const row of outcomes) {
    if (row.failure) byFailure[row.failure] = (byFailure[row.failure] || 0) + 1;
  }
  const correct = outcomes.filter((row) => row.correct).length;
  return {
    expected: outcomes.length,
    correct,
    incorrect: outcomes.length - correct,
    failures: Object.values(byFailure).reduce((sum, value) => sum + value, 0),
    byFailure
  };
}

function groupStats(rows, key, field) {
  return [...groupBy(rows, key)]
    .map(([value, items]) => ({ [field]: value, ...rate(items) }))
    .sort((left, right) => String(left[field]).localeCompare(String(right[field])));
}

function rate(rows) {
  const correct = rows.filter((row) => row.correct).length;
  return { total: rows.length, correct, recall: rows.length ? correct / rows.length : 0 };
}

function balancedAccuracy(totals) {
  return ((totals.aiCorrect / totals.ai) + (totals.realCorrect / totals.real)) / 2;
}

function resample(values, random) {
  return Array.from({ length: values.length }, () => values[Math.floor(random() * values.length)]);
}

function mulberry32(seed) {
  let value = seed;
  return () => {
    value |= 0;
    value = value + 0x6d2b79f5 | 0;
    let result = Math.imul(value ^ value >>> 15, 1 | value);
    result = result + Math.imul(result ^ result >>> 7, 61 | result) ^ result;
    return ((result ^ result >>> 14) >>> 0) / 4294967296;
  };
}

function quantile(sorted, probability) {
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function compare(value, operator, threshold) {
  if (operator === '>=') return value >= threshold;
  if (operator === '>') return value > threshold;
  if (operator === '===') return value === threshold;
  throw new Error(`Unsupported gate operator: ${operator}`);
}

function validateBenchmarkPath(path, field) {
  if (isAbsolute(path) || path.includes('\\') || path.split('/').includes('..') || path.startsWith('/')) {
    throw new Error(`${field} must be a normalized relative path.`);
  }
}

function canonicalPath(value) {
  return String(value).replaceAll('\\', '/').replace(/^\.\//, '').toLowerCase();
}

function sameSet(left, right) {
  return left.size === right.size && [...left].every((value) => right.has(value));
}

function groupBy(values, key) {
  const groups = new Map();
  for (const value of values) {
    const id = key(value);
    const group = groups.get(id) || [];
    group.push(value);
    groups.set(id, group);
  }
  return groups;
}

function parseInteger(value, field) {
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new Error(`${field} must be an integer.`);
  return result;
}

function requireHttps(value, field) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${field} must be a valid URL.`);
  }
  if (parsed.protocol !== 'https:') throw new Error(`${field} must use HTTPS.`);
}

function requireIsoDate(value, field) {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value) || Number.isNaN(Date.parse(value))) {
    throw new Error(`${field} must be an ISO-8601 UTC timestamp.`);
  }
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

async function readJson(path) {
  let bytes;
  try {
    bytes = await readFile(path);
  } catch (error) {
    throw new Error(`Cannot read ${path}: ${error.message}`);
  }
  try {
    return JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Invalid JSON in ${path}: ${error.message}`);
  }
}
