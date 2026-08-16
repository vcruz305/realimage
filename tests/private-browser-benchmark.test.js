import { request } from 'node:http';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it } from 'vitest';
import { formatScorePercent } from '../src/shared/score-display.js';
import {
  HARNESS_STATE_SCHEMA,
  createHarnessServer,
  loadPrivateBenchmark,
  parseHarnessBadgeContract,
  parseHarnessDeclarationDetails,
  renderHarnessPage,
  stableId
} from '../scripts/serve-private-browser-benchmark.mjs';

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const benchRoot = resolve(projectRoot, '.bench-data');
const temporaryRoots = [];
const onePixelPng = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64'
);

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe('private installed-extension browser harness', () => {
  it('records a declared-below-threshold badge as a real model verdict plus a separate declaration', () => {
    const contract = parseHarnessBadgeContract({
      badgeKind: 'declared',
      modelVerdict: 'real',
      modelScorePercent: '0.8',
      decisionThreshold: '0.65',
      declarationType: 'edited',
      declarationSummary: 'Declared AI-edited'
    });

    expect(contract).toEqual({
      ok: true,
      model: {
        verdict: 'real',
        score: 0.008,
        displayedScorePercent: 0.8,
        decisionThreshold: 0.65
      },
      declaration: {
        type: 'edited',
        summary: 'Declared AI-edited'
      }
    });

    expect(parseHarnessDeclarationDetails({
      count: 1,
      titleCount: 1,
      bodyCount: 1,
      sourcesCount: 1,
      titleInBlock: true,
      bodyInBlock: true,
      sourcesInBlock: true,
      title: ' Metadata declares AI editing ',
      body: ' Recognized metadata says AI was used.  Metadata is not proof. ',
      sources: ' IPTC digital-source declaration · Google AI edit declaration '
    }, contract.declaration)).toEqual({
      ok: true,
      declaration: {
        type: 'edited',
        summary: 'Declared AI-edited',
        detailTitle: 'Metadata declares AI editing',
        detailBody: 'Recognized metadata says AI was used. Metadata is not proof.',
        sources: 'IPTC digital-source declaration · Google AI edit declaration'
      }
    });
  });

  it.each([
    {
      name: 'AI',
      snapshot: {
        badgeKind: 'ai',
        modelVerdict: 'ai',
        modelScorePercent: '93.6',
        decisionThreshold: '0.65'
      },
      expected: { verdict: 'ai', score: 0.936, displayedScorePercent: 93.6, decisionThreshold: 0.65 }
    },
    {
      name: 'real',
      snapshot: {
        badgeKind: 'real',
        modelVerdict: 'real',
        modelScorePercent: '12.4',
        decisionThreshold: '0.65'
      },
      expected: { verdict: 'real', score: 0.124, displayedScorePercent: 12.4, decisionThreshold: 0.65 }
    }
  ])('preserves ordinary $name badge behavior through explicit model fields', ({ snapshot, expected }) => {
    expect(parseHarnessBadgeContract(snapshot)).toEqual({ ok: true, model: expected, declaration: null });
    expect(parseHarnessDeclarationDetails({ count: 0, titleCount: 0, bodyCount: 0, sourcesCount: 0 }, null)).toEqual({ ok: true, declaration: null });
  });

  it('accepts formatter-valid real results immediately below every supported threshold', () => {
    for (let thresholdPercent = 50; thresholdPercent <= 95; thresholdPercent += 1) {
      const threshold = thresholdPercent / 100;
      const modelScorePercent = formatScorePercent(threshold - 0.000499, threshold);
      expect(parseHarnessBadgeContract({
        badgeKind: 'real',
        modelVerdict: 'real',
        modelScorePercent,
        decisionThreshold: String(threshold)
      }), `threshold=${threshold} displayed=${modelScorePercent}`).toMatchObject({ ok: true });
    }
  });

  it.each([
    ['missing score', { badgeKind: 'real', modelVerdict: 'real', decisionThreshold: '0.65' }],
    ['nonnumeric score', { badgeKind: 'real', modelVerdict: 'real', modelScorePercent: '12.4% AI', decisionThreshold: '0.65' }],
    ['noncanonical score', { badgeKind: 'ai', modelVerdict: 'ai', modelScorePercent: '6.5e1', decisionThreshold: '0.65' }],
    ['verdict mismatch', { badgeKind: 'declared', modelVerdict: 'ai', modelScorePercent: '0.8', decisionThreshold: '0.65', declarationType: 'edited', declarationSummary: 'Declared AI-edited' }],
    ['missing declaration summary', { badgeKind: 'declared', modelVerdict: 'real', modelScorePercent: '0.8', decisionThreshold: '0.65', declarationType: 'edited' }],
    ['unknown declaration type', { badgeKind: 'declared', modelVerdict: 'real', modelScorePercent: '0.8', decisionThreshold: '0.65', declarationType: 'synthetic', declarationSummary: 'Declared AI-edited' }],
    ['altered declaration summary', { badgeKind: 'declared', modelVerdict: 'real', modelScorePercent: '0.8', decisionThreshold: '0.65', declarationType: 'edited', declarationSummary: 'Declared AI-edited <img src=x onerror=alert(1)>' }],
    ['ambiguous badge kind', { badgeKind: '', modelVerdict: 'real', modelScorePercent: '0.8', decisionThreshold: '0.65' }]
  ])('fails closed for a malformed badge contract: %s', (_name, snapshot) => {
    expect(parseHarnessBadgeContract(snapshot)).toMatchObject({ ok: false });
  });

  it('fails closed on missing or contradictory declaration details and retains HTML-like text as inert data', () => {
    const declaration = { type: 'edited', summary: 'Declared AI-edited' };
    expect(parseHarnessDeclarationDetails({ count: 0, titleCount: 0, bodyCount: 0, sourcesCount: 0 }, declaration)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 2, titleCount: 1, bodyCount: 1, sourcesCount: 1, title: 'x', body: 'y', sources: 'z' }, declaration)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 1, titleCount: 2, bodyCount: 1, sourcesCount: 1, title: 'title', body: 'body', sources: 'source' }, declaration)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 1, titleCount: 1, bodyCount: 0, sourcesCount: 1, title: 'title', body: 'body', sources: 'source' }, declaration)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 1, titleCount: 1, bodyCount: 1, sourcesCount: 2, title: 'title', body: 'body', sources: 'source' }, declaration)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 1, titleCount: 1, bodyCount: 1, sourcesCount: 1, titleInBlock: false, bodyInBlock: true, sourcesInBlock: true, title: 'title', body: 'body', sources: 'source' }, declaration)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 1, titleCount: 1, bodyCount: 1, sourcesCount: 1, titleInBlock: true, bodyInBlock: true, sourcesInBlock: true, title: '', body: 'body', sources: 'source' }, declaration)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 1, titleCount: 1, bodyCount: 1, sourcesCount: 1, titleInBlock: true, bodyInBlock: true, sourcesInBlock: true, title: 'title', body: 'body', sources: 'source' }, null)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails({ count: 0, titleCount: 2, bodyCount: 3, sourcesCount: 4 }, null)).toMatchObject({ ok: false });
    expect(parseHarnessDeclarationDetails(undefined, null)).toMatchObject({ ok: false });

    const inert = '<img src=x onerror=alert(1)> & <script>alert(2)</script>';
    const parsed = parseHarnessDeclarationDetails({
      count: 1,
      titleCount: 1,
      bodyCount: 1,
      sourcesCount: 1,
      titleInBlock: true,
      bodyInBlock: true,
      sourcesInBlock: true,
      title: inert,
      body: `Caveat: ${inert}`,
      sources: `Source: ${inert}`
    }, declaration);
    expect(parsed.ok).toBe(true);
    expect(parsed.declaration.detailTitle).toBe(inert);
    expect(parsed.declaration.detailBody).toBe(`Caveat: ${inert}`);
    expect(parsed.declaration.sources).toBe(`Source: ${inert}`);
  });

  it('loads only contained local manifest files and derives stable opaque IDs', async () => {
    const fixture = await createFixture();
    const benchmark = await loadPrivateBenchmark({
      projectRoot,
      benchRoot,
      root: fixture.imageRoot,
      manifestPath: fixture.manifestPath
    });

    expect(benchmark.rows.map((row) => row.sampleId)).toEqual(['ai-family:original', 'real-family:original']);
    expect(benchmark.rows[0]).toMatchObject({
      assetId: stableId('asset\0ai-family:original', 20),
      label: 'ai',
      view: 'clean',
      mimeType: 'image/png'
    });

    const projectRelativeManifest = structuredClone(fixture.manifest);
    projectRelativeManifest.rows[0].relativePath = relativePathFromProject(
      join(fixture.imageRoot, 'clean', 'real', 'private-real.png')
    );
    projectRelativeManifest.rows[1].relativePath = relativePathFromProject(
      join(fixture.imageRoot, 'clean', 'ai', 'private-ai.png')
    );
    await writeFile(fixture.manifestPath, `${JSON.stringify(projectRelativeManifest)}\n`);
    const projectRelative = await loadPrivateBenchmark({
      projectRoot,
      benchRoot,
      root: fixture.imageRoot,
      manifestPath: fixture.manifestPath
    });
    expect(projectRelative.rows).toHaveLength(2);

    const remoteManifest = structuredClone(fixture.manifest);
    remoteManifest.rows[0].relativePath = 'https://example.test/private.png';
    await writeFile(fixture.manifestPath, `${JSON.stringify(remoteManifest)}\n`);
    await expect(loadPrivateBenchmark({ projectRoot, benchRoot, root: fixture.imageRoot, manifestPath: fixture.manifestPath }))
      .rejects.toThrow('must be a local relative path');

    const escapingManifest = structuredClone(fixture.manifest);
    escapingManifest.rows[0].relativePath = '../outside.png';
    await writeFile(join(fixture.root, 'outside.png'), onePixelPng);
    await writeFile(fixture.manifestPath, `${JSON.stringify(escapingManifest)}\n`);
    await expect(loadPrivateBenchmark({ projectRoot, benchRoot, root: fixture.imageRoot, manifestPath: fixture.manifestPath }))
      .rejects.toThrow('escapes the configured image root');

    const mismatchedDigest = structuredClone(fixture.manifest);
    mismatchedDigest.rows[0].sha256 = 'a'.repeat(64);
    await writeFile(fixture.manifestPath, `${JSON.stringify(mismatchedDigest)}\n`);
    await expect(loadPrivateBenchmark({ projectRoot, benchRoot, root: fixture.imageRoot, manifestPath: fixture.manifestPath }))
      .rejects.toThrow('SHA-256 mismatch');

    const privateDiagnostic = {
      schema: 'realimage-private-diagnostic-v1',
      samples: [{
        file: 'clean/ai/private-ai.png',
        label: 'ai',
        slice: 'private-smoke',
        sha256: createHash('sha256').update(onePixelPng).digest('hex')
      }]
    };
    await writeFile(fixture.manifestPath, `${JSON.stringify(privateDiagnostic)}\n`);
    const diagnostic = await loadPrivateBenchmark({ projectRoot, benchRoot, root: fixture.imageRoot, manifestPath: fixture.manifestPath });
    const identity = createHash('sha256').update(onePixelPng).digest('hex').slice(0, 16);
    expect(diagnostic.rows[0]).toMatchObject({
      sampleId: `local-${identity}:original`,
      familyId: `local-${identity}`,
      label: 'ai',
      slice: 'private-smoke'
    });
  });

  it('renders deterministic labeled cases plus all same-origin delivery stresses', async () => {
    const fixture = await createFixture();
    const benchmark = await loadPrivateBenchmark({ projectRoot, benchRoot, root: fixture.imageRoot, manifestPath: fixture.manifestPath });
    const page = renderHarnessPage(benchmark, new URLSearchParams('view=clean&label=ai&delivery=web-stress&limit=1'));
    const config = extractClientConfig(page.html);

    expect(page.html).toContain(HARNESS_STATE_SCHEMA);
    expect(page.html).toContain('id="realimage-harness-state"');
    expect(page.html).toContain('window.__REALIMAGE_HARNESS_STATE__');
    expect(page.html).toContain('data-realimage-harness-terminal="false"');
    expect(config.harnessRunId).toMatch(/^[a-f0-9]{32}$/);
    expect(config.manifestSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(config.cases[0].assetUrl).toContain(`realimageRun=${config.harnessRunId}`);
    expect(page.html).toContain('modelScorePercent: badge.dataset.proofmarkModelScorePercent');
    expect(page.html).toContain('declarationType: badge.dataset.proofmarkDeclarationType');
    expect(page.html).toContain("row.predicted = contract.model.verdict");
    expect(page.html).toContain("row.status = 'extension-error'");
    expect(page.html).toContain('const linkedBadgeId = image.dataset.proofmarkBadgeId');
    expect(page.html).toContain('panel.dataset.anchor !== badge.dataset.proofmarkId');
    expect(page.html).toContain("panel.querySelectorAll('[data-proofmark-declaration-details]')");
    expect(page.html).toContain('titleCount: declarationTitleNodes.length');
    expect(page.html).toContain('stateNode.textContent = JSON.stringify(state, null, 2)');
    expect(page.html).not.toContain('scoreMatch');
    expect(page.html).not.toContain("if (config.selection.execution !== 'sequential')");
    expect(page.html).toContain(`"domId":"tg-case-${stableId('ai-family:original::direct')}"`);
    for (const delivery of ['direct', 'query', 'srcset', 'delayed', 'blob', 'shadow', 'redirect', 'source-swap']) {
      expect(page.html).toContain(`"caseId":"ai-family:original::${delivery}"`);
    }
    expect(page.html).not.toContain(fixture.root);
    expect(page.html).not.toContain('private-ai.png');
    expect(page.html).not.toMatch(/https?:\/\//i);
    expect(page.headers['content-security-policy']).toContain("connect-src 'self'");
    expect(page.headers['content-security-policy']).toContain("img-src 'self' blob: data:");
    expect(page.headers['content-security-policy']).not.toMatch(/https?:/i);
  });

  it('renders a deterministic 20-case collective burst without relaxing sequential limits', () => {
    const benchmark = createSyntheticBenchmark(60);
    const page = renderHarnessPage(
      benchmark,
      new URLSearchParams('view=clean&delivery=direct&execution=burst&offset=0&limit=20')
    );
    const config = extractClientConfig(page.html);

    expect(config.selection).toMatchObject({
      execution: 'burst',
      delivery: 'direct',
      offset: 0,
      limit: 20,
      availableRows: 60
    });
    expect(config.cases).toHaveLength(20);
    expect(config.cases.map((item) => item.caseId)).toEqual(
      Array.from({ length: 20 }, (_, index) => `sample-${String(index).padStart(3, '0')}::direct`)
    );
    expect(new Set(config.cases.map((item) => item.domId)).size).toBe(20);
    expect(page.html).toContain('data-realimage-harness-execution="burst"');
    expect(page.html).toContain("config.selection.execution === 'burst'");
    expect(page.html).toContain("row.status = 'extension-error'");
    expect(() => new Function(extractInlineScript(page.html))).not.toThrow();
    expect(() => renderHarnessPage(
      benchmark,
      new URLSearchParams('view=clean&delivery=direct&execution=sequential&limit=60')
    )).toThrow('limit must be an integer from 1 through 32');
  });

  it('renders a deterministic 60-case static gallery with terminal fail-closed accounting', () => {
    const benchmark = createSyntheticBenchmark(60);
    const page = renderHarnessPage(
      benchmark,
      new URLSearchParams('view=clean&delivery=direct&execution=gallery&offset=0&limit=60')
    );
    const config = extractClientConfig(page.html);

    expect(config.selection).toMatchObject({
      execution: 'gallery',
      delivery: 'direct',
      offset: 0,
      limit: 60,
      availableRows: 60
    });
    expect(config.cases).toHaveLength(60);
    expect(config.cases.at(-1)).toMatchObject({
      caseId: 'sample-059::direct',
      domId: `tg-case-${stableId('sample-059::direct')}`
    });
    expect(page.html).toContain('data-realimage-harness-execution="gallery"');
    expect(page.html).toContain('markUnsettledTimedOut');
    expect(page.html).toContain("['complete', 'extension-error', 'image-error', 'timed-out']");
    expect(() => renderHarnessPage(
      benchmark,
      new URLSearchParams('view=clean&delivery=web-stress&execution=gallery&limit=60')
    )).toThrow('limit must be an integer from 1 through 4');
    expect(extractClientConfig(renderHarnessPage(
      benchmark,
      new URLSearchParams('view=clean&delivery=query&execution=gallery')
    ).html).selection.limit).toBe(12);
  });

  it('serves only opaque asset routes on loopback and refuses hostile hosts and methods', async () => {
    const fixture = await createFixture();
    const benchmark = await loadPrivateBenchmark({ projectRoot, benchRoot, root: fixture.imageRoot, manifestPath: fixture.manifestPath });
    const server = createHarnessServer(benchmark);
    await new Promise((resolveListen, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolveListen);
    });
    const address = server.address();
    const origin = `http://127.0.0.1:${address.port}`;

    try {
      const page = await fetch(`${origin}/?view=clean&delivery=direct&limit=1`);
      expect(page.status).toBe(200);
      expect(page.headers.get('cross-origin-opener-policy')).toBe('same-origin');
      expect(page.headers.get('cross-origin-embedder-policy')).toBe('require-corp');

      const ai = benchmark.rows.find((row) => row.label === 'ai');
      const assetPath = `/asset/${ai.assetId}/image.png`;
      const asset = await fetch(`${origin}${assetPath}`);
      expect(asset.status).toBe(200);
      expect(asset.headers.get('cross-origin-resource-policy')).toBe('cross-origin');
      expect(Buffer.from(await asset.arrayBuffer())).toEqual(onePixelPng);

      const redirected = await fetch(`${origin}/redirect/${ai.assetId}`, { redirect: 'manual' });
      expect(redirected.status).toBe(302);
      expect(redirected.headers.get('location')).toBe(assetPath);

      expect((await fetch(`${origin}/asset/../../package.json`)).status).toBe(404);
      expect((await fetch(`${origin}/`, { method: 'POST' })).status).toBe(405);
      const hostile = await rawRequest({ port: address.port, path: '/', hostHeader: 'attacker.example' });
      expect(hostile.status).toBe(403);
      expect(hostile.body).toContain('Loopback Host header required');
    } finally {
      await new Promise((resolveClose) => server.close(resolveClose));
    }
  });
});

async function createFixture() {
  await mkdir(benchRoot, { recursive: true });
  const root = await mkdtemp(join(benchRoot, 'private-browser-test-'));
  temporaryRoots.push(root);
  const imageRoot = join(root, 'matrix');
  await mkdir(join(imageRoot, 'clean', 'ai'), { recursive: true });
  await mkdir(join(imageRoot, 'clean', 'real'), { recursive: true });
  await writeFile(join(imageRoot, 'clean', 'ai', 'private-ai.png'), onePixelPng);
  await writeFile(join(imageRoot, 'clean', 'real', 'private-real.png'), onePixelPng);
  const manifest = {
    schema: 'proofmark-modern-holdout-matrix-v1',
    benchmarkId: 'private-test-v1',
    rows: [
      {
        sampleId: 'real-family:original',
        familyId: 'real-family',
        label: 'real',
        view: 'clean',
        transformId: 'original',
        relativePath: 'clean/real/private-real.png'
      },
      {
        sampleId: 'ai-family:original',
        familyId: 'ai-family',
        label: 'ai',
        view: 'clean',
        transformId: 'original',
        relativePath: 'clean/ai/private-ai.png'
      }
    ]
  };
  const manifestPath = join(imageRoot, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return { root, imageRoot, manifestPath, manifest };
}

function rawRequest({ port, path, hostHeader }) {
  return new Promise((resolveRequest, reject) => {
    const outgoing = request({ hostname: '127.0.0.1', port, path, headers: { host: hostHeader } }, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => resolveRequest({ status: response.statusCode, body: Buffer.concat(chunks).toString('utf8') }));
    });
    outgoing.on('error', reject);
    outgoing.end();
  });
}

function createSyntheticBenchmark(count) {
  return {
    benchmarkId: 'synthetic-browser-gates-v1',
    manifestSchema: 'synthetic-v1',
    rows: Array.from({ length: count }, (_, index) => {
      const sampleId = `sample-${String(index).padStart(3, '0')}`;
      return {
        assetId: stableId(`asset\0${sampleId}`, 20),
        sampleId,
        familyId: `family-${String(index).padStart(3, '0')}`,
        label: index % 2 === 0 ? 'ai' : 'real',
        view: 'clean',
        transformId: 'original',
        extension: '.png'
      };
    })
  };
}

function extractClientConfig(html) {
  const match = html.match(/const config = (\{.*\});\r?\n/);
  if (!match) throw new Error('Rendered harness did not contain its client configuration.');
  return JSON.parse(match[1]);
}

function extractInlineScript(html) {
  const match = html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/);
  if (!match) throw new Error('Rendered harness did not contain its inline runner.');
  return match[1];
}

function relativePathFromProject(path) {
  return path.slice(projectRoot.length + 1).replaceAll('\\', '/');
}
