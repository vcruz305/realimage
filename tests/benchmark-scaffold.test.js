import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import sharp from 'sharp';
import { afterEach, describe, expect, it } from 'vitest';

const execute = promisify(execFile);
const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const matrixScript = resolve(projectRoot, 'scripts/make-stress-matrix.mjs');
const summaryScript = resolve(projectRoot, 'scripts/summarize-slices.mjs');
const temporaryRoots = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe('modern holdout benchmark scaffold', () => {
  it('builds the same deterministic transform matrix for both labels and fails closed on rights', async () => {
    const fixture = await createFixture();
    const outputRoot = join(fixture.root, '.bench-data', 'matrix');
    await execute(process.execPath, [matrixScript, fixture.inputRoot, outputRoot, '--seed=323'], { cwd: fixture.root });

    const matrix = JSON.parse(await readFile(join(outputRoot, 'manifest.json'), 'utf8'));
    expect(matrix.schema).toBe('proofmark-modern-holdout-matrix-v1');
    expect(matrix.counts).toEqual({
      originalFamilies: 4,
      aiFamilies: 2,
      realFamilies: 2,
      cleanSamples: 4,
      stressSamples: 16,
      totalSamples: 20
    });
    const transformSets = new Map();
    for (const label of ['ai', 'real']) {
      transformSets.set(label, new Set(matrix.rows.filter((row) => row.label === label).map((row) => row.transformId)));
    }
    expect([...transformSets.get('ai')].sort()).toEqual([...transformSets.get('real')].sort());
    expect(new Set(matrix.rows.filter((row) => row.view === 'stress').map((row) => row.transformId)).size).toBe(4);

    const badManifest = structuredClone(fixture.manifest);
    badManifest.rows[0].license.commercialUse = false;
    const badPath = join(fixture.inputRoot, 'bad-manifest.json');
    await writeFile(badPath, `${JSON.stringify(badManifest, null, 2)}\n`);
    await expect(execute(process.execPath, [
      matrixScript,
      fixture.inputRoot,
      join(fixture.root, '.bench-data', 'bad-matrix'),
      `--manifest=${badPath}`,
      '--seed=323'
    ], { cwd: fixture.root })).rejects.toMatchObject({
      stderr: expect.stringContaining('license.commercialUse must be explicitly true')
    });
  }, 30_000);

  it('keeps missing outputs in the denominator and reports a paired family-cluster bootstrap', async () => {
    const fixture = await createFixture();
    const outputRoot = join(fixture.root, '.bench-data', 'matrix');
    await execute(process.execPath, [matrixScript, fixture.inputRoot, outputRoot, '--seed=323'], { cwd: fixture.root });
    const matrix = JSON.parse(await readFile(join(outputRoot, 'manifest.json'), 'utf8'));
    const clean = matrix.rows.filter((row) => row.view === 'clean');
    const stress = matrix.rows.filter((row) => row.view === 'stress');
    const resultsRoot = join(fixture.root, 'results');
    await mkdir(resultsRoot, { recursive: true });

    const paths = {
      baselineClean: join(resultsRoot, 'baseline-clean.json'),
      baselineStress: join(resultsRoot, 'baseline-stress.json'),
      candidateClean: join(resultsRoot, 'candidate-clean.json'),
      candidateStress: join(resultsRoot, 'candidate-stress.json'),
      summary: join(resultsRoot, 'summary.json')
    };
    await writeResult(paths.baselineClean, 'baseline', clean, false);
    await writeResult(paths.baselineStress, 'baseline', stress, false);
    await writeResult(paths.candidateClean, 'candidate', clean, true);
    await writeResult(paths.candidateStress, 'candidate', stress, true, stress[0].sampleId);

    await execute(process.execPath, [
      summaryScript,
      `--manifest=${join(outputRoot, 'manifest.json')}`,
      `--baseline-clean=${paths.baselineClean}`,
      `--baseline-stress=${paths.baselineStress}`,
      `--candidate-clean=${paths.candidateClean}`,
      `--candidate-stress=${paths.candidateStress}`,
      '--bootstrap=200',
      '--seed=323',
      '--smoke',
      `--output=${paths.summary}`
    ], { cwd: fixture.root });

    const summary = JSON.parse(await readFile(paths.summary, 'utf8'));
    expect(summary.mode).toBe('SMOKE_NOT_CLAIM_ELIGIBLE');
    expect(summary.claimEligible).toBe(false);
    expect(summary.runs.candidate.stress.coverage).toMatchObject({
      expected: 16,
      correct: 15,
      incorrect: 1,
      failures: 1,
      byFailure: { 'missing-result': 1 }
    });
    expect(summary.runs.candidate.stress.metrics.totals.ai.total).toBe(8);
    expect(summary.comparisons.stress.balancedAccuracyBootstrap).toMatchObject({
      unit: 'familyId',
      stratifiedByLabel: true,
      iterations: 200,
      seed: 10330
    });
    expect(summary.comparisons.stress.balancedAccuracyBootstrap.lower).toBeGreaterThan(0);
  }, 30_000);
});

async function createFixture() {
  const root = await mkdtemp(join(tmpdir(), 'proofmark-modern-holdout-'));
  temporaryRoots.push(root);
  const inputRoot = join(root, '.bench-data', 'modern-v1');
  await mkdir(join(inputRoot, 'originals'), { recursive: true });

  const priorManifestPath = join(root, 'prior-locked-manifest.json');
  const priorManifest = {
    rows: [{
      familyId: 'prior-family',
      sourceDataset: 'prior/synthetic',
      sourceRow: 'old:1',
      sourceGroup: 'prior/synthetic',
      sha256: 'f'.repeat(64)
    }]
  };
  const priorBytes = Buffer.from(`${JSON.stringify(priorManifest, null, 2)}\n`);
  await writeFile(priorManifestPath, priorBytes);

  const definitions = [
    { familyId: 'ai-generator-a-001', label: 'ai', domain: 'generator-a', slice: 'generator-a', width: 400, height: 400, pattern: 1 },
    { familyId: 'ai-generator-b-001', label: 'ai', domain: 'generator-b', slice: 'generator-b', width: 424, height: 400, pattern: 2 },
    { familyId: 'real-camera-001', label: 'real', domain: 'camera', slice: 'camera-general', width: 400, height: 432, pattern: 3 },
    { familyId: 'real-art-001', label: 'real', domain: 'human-art', slice: 'human-digital-art', width: 460, height: 400, pattern: 4 }
  ];
  const rows = [];
  for (const [index, definition] of definitions.entries()) {
    const relativePath = `originals/${definition.familyId}.png`;
    const bytes = await patternedPng(definition.width, definition.height, definition.pattern);
    await writeFile(join(inputRoot, relativePath), bytes);
    rows.push({
      familyId: definition.familyId,
      label: definition.label,
      domain: definition.domain,
      slice: definition.slice,
      sourceDataset: `synthetic/${definition.domain}`,
      sourceRow: `row:${index + 1}`,
      sourceGroup: `synthetic/${definition.domain}`,
      relativePath,
      width: definition.width,
      height: definition.height,
      format: 'png',
      sha256: hash(bytes),
      dHash: await differenceHash(bytes),
      license: {
        status: 'approved',
        identifier: 'CC0-1.0',
        url: 'https://creativecommons.org/publicdomain/zero/1.0/',
        verifiedAt: '2026-08-15T08:00:00Z',
        commercialUse: true,
        evaluationUse: true,
        derivatives: true,
        localCopy: true,
        attribution: 'Synthetic test fixture generated by this test.'
      },
      provenance: {
        status: 'verified',
        sourceUrl: 'https://example.test/proofmark-synthetic-fixture',
        immutableRevision: `fixture-${index + 1}`,
        creator: 'Proofmark test suite',
        labelMethod: 'deterministic synthetic test fixture',
        retrievedAt: '2026-08-15T08:00:00Z'
      }
    });
  }

  const manifest = {
    schema: 'proofmark-modern-holdout-source-v1',
    benchmarkId: 'synthetic-modern-v1',
    seed: 323,
    frozenAt: '2026-08-15T08:00:00Z',
    sourceDisjointness: {
      confirmed: true,
      policy: 'source-group+source-row+sha256+dhash-v1',
      checkedAgainst: [{ manifest: priorManifestPath, sha256: hash(priorBytes) }]
    },
    rows
  };
  await writeFile(join(inputRoot, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return { root, inputRoot, manifest };
}

async function patternedPng(width, height, pattern) {
  const pixels = Buffer.alloc(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 3;
      pixels[offset] = (x * (11 + pattern) + y * pattern * 3) % 256;
      pixels[offset + 1] = (y * (17 + pattern) + x * pattern * 5) % 256;
      pixels[offset + 2] = ((x ^ y) * (23 + pattern) + pattern * 29) % 256;
    }
  }
  return sharp(pixels, { raw: { width, height, channels: 3 } }).png({ compressionLevel: 9 }).toBuffer();
}

async function differenceHash(bytes) {
  const pixels = await sharp(bytes).resize(9, 8, { fit: 'fill' }).greyscale().raw().toBuffer();
  let value = 0n;
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      value = (value << 1n) | BigInt(pixels[y * 9 + x] > pixels[y * 9 + x + 1]);
    }
  }
  return value.toString(16).padStart(16, '0');
}

async function writeResult(path, model, expectedRows, correct, omittedSampleId) {
  const rows = expectedRows
    .filter((row) => row.sampleId !== omittedSampleId)
    .map((row) => ({
      file: row.benchmarkPath,
      expected: row.label,
      predicted: correct ? row.label : row.label === 'ai' ? 'real' : 'ai',
      score: correct ? row.label === 'ai' ? 0.9 : 0.1 : row.label === 'ai' ? 0.1 : 0.9
    }));
  await writeFile(path, `${JSON.stringify({ model, rows }, null, 2)}\n`);
}

function hash(value) {
  return createHash('sha256').update(value).digest('hex');
}
