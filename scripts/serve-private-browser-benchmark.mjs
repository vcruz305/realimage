import { createHash, randomBytes } from 'node:crypto';
import { readFile, realpath, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { basename, dirname, extname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const PROJECT_ROOT = resolve(dirname(SCRIPT_PATH), '..');
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif']);
const LABELS = new Set(['ai', 'real']);
const EXECUTIONS = new Set(['sequential', 'burst', 'gallery']);
const DELIVERIES = Object.freeze([
  'direct',
  'query',
  'srcset',
  'delayed',
  'blob',
  'shadow',
  'redirect',
  'source-swap'
]);

export const HARNESS_STATE_SCHEMA = 'realimage-private-browser-harness-state-v1';

/**
 * Parse the public, rounded model-result contract emitted on a terminal badge.
 * This deliberately does not infer a model result from badge copy or color: an
 * amber metadata declaration is separate from the detector's numeric verdict.
 */
export function parseHarnessBadgeContract(snapshot) {
  const fail = (error) => Object.freeze({ ok: false, error });
  if (!snapshot || typeof snapshot !== 'object') return fail('Badge result fields are missing.');

  const verdict = snapshot.modelVerdict;
  if (verdict !== 'ai' && verdict !== 'real') return fail('Model verdict must be exactly ai or real.');

  const parseDecimal = (value, label, minimum, maximum) => {
    if (typeof value !== 'string' || !/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(value)) {
      return { error: `${label} must be a canonical non-negative decimal string.` };
    }
    const number = Number(value);
    if (!Number.isFinite(number) || number < minimum || number > maximum) {
      return { error: `${label} must be within ${minimum} through ${maximum}.` };
    }
    return { number };
  };

  const displayedScore = parseDecimal(snapshot.modelScorePercent, 'Model score percent', 0, 100);
  if (displayedScore.error) return fail(displayedScore.error);
  if (!/^(?:0|[1-9]\d*)\.\d$/.test(snapshot.modelScorePercent)) {
    return fail('Model score percent must use the one-decimal public display format.');
  }
  const threshold = parseDecimal(snapshot.decisionThreshold, 'Decision threshold', 0, 1);
  if (threshold.error) return fail(threshold.error);

  const score = Number((displayedScore.number / 100).toFixed(12));
  // A one-decimal percentage can round an AI score down by at most 0.05
  // percentage points. Real-side formatting is separately guaranteed never to
  // round up across the threshold by formatScorePercent().
  if (verdict === 'real' && score >= threshold.number) {
    return fail('Real model verdict conflicts with the displayed score and decision threshold.');
  }
  if (verdict === 'ai' && score + 0.000500000001 < threshold.number) {
    return fail('AI model verdict conflicts with the displayed score and decision threshold.');
  }

  const declarationType = snapshot.declarationType;
  const declarationSummary = snapshot.declarationSummary;
  const hasDeclarationType = typeof declarationType === 'string' && declarationType.length > 0;
  const hasDeclarationSummary = typeof declarationSummary === 'string' && declarationSummary.length > 0;
  if (hasDeclarationType !== hasDeclarationSummary) {
    return fail('Declaration type and summary must either both be present or both be absent.');
  }

  let declaration = null;
  if (hasDeclarationType) {
    const expectedSummaries = {
      generated: 'Declared AI-generated',
      edited: 'Declared AI-edited',
      composite: 'Declared AI composite',
      mixed: 'Declared AI involvement'
    };
    if (!Object.hasOwn(expectedSummaries, declarationType)) {
      return fail('Declaration type is not allowlisted.');
    }
    const compactSummary = declarationSummary.replace(/\s+/g, ' ').trim();
    if (compactSummary !== expectedSummaries[declarationType]) {
      return fail('Declaration summary does not match its allowlisted type.');
    }
    declaration = Object.freeze({ type: declarationType, summary: compactSummary });
  } else if (declarationType != null || declarationSummary != null) {
    return fail('Empty declaration fields are invalid.');
  }

  if (!['ai', 'real', 'declared'].includes(snapshot.badgeKind)) return fail('Badge kind is missing or ambiguous.');
  const expectedBadgeKind = verdict === 'ai' ? 'ai' : declaration ? 'declared' : 'real';
  if (snapshot.badgeKind !== expectedBadgeKind) {
    return fail('Badge kind conflicts with the model verdict or declaration fields.');
  }

  return Object.freeze({
    ok: true,
    model: Object.freeze({
      verdict,
      score,
      displayedScorePercent: displayedScore.number,
      decisionThreshold: threshold.number
    }),
    declaration
  });
}

/**
 * Merge declaration detail text captured via explicit inert DOM hooks. Text is
 * retained as data only; the harness publishes it through textContent/JSON.
 */
export function parseHarnessDeclarationDetails(snapshot, declaration) {
  const fail = (error) => Object.freeze({ ok: false, error });
  if (!snapshot || typeof snapshot !== 'object' || !Number.isSafeInteger(snapshot.count) || snapshot.count < 0) {
    return fail('Declaration detail count is missing or invalid.');
  }
  const count = snapshot.count;
  for (const field of ['titleCount', 'bodyCount', 'sourcesCount']) {
    if (!Number.isSafeInteger(snapshot[field]) || snapshot[field] < 0) {
      return fail('Declaration detail hook counts are missing or invalid.');
    }
  }
  if (!declaration) {
    return count === 0 && snapshot.titleCount === 0 && snapshot.bodyCount === 0 && snapshot.sourcesCount === 0
      ? Object.freeze({ ok: true, declaration: null })
      : fail('Declaration details or hooks were present without declaration badge fields.');
  }
  if (!['generated', 'edited', 'composite', 'mixed'].includes(declaration.type) || typeof declaration.summary !== 'string') {
    return fail('Declaration badge fields are malformed.');
  }
  if (count !== 1) {
    return fail('Exactly one declaration detail block is required.');
  }
  for (const field of ['titleCount', 'bodyCount', 'sourcesCount']) {
    if (snapshot[field] !== 1) return fail('Every declaration detail hook must occur exactly once.');
  }
  if (!snapshot.titleInBlock || !snapshot.bodyInBlock || !snapshot.sourcesInBlock) {
    return fail('Every declaration detail hook must belong to the sole declaration block.');
  }

  const readText = (value, label, maximum) => {
    if (typeof value !== 'string') return { error: `${label} is missing.` };
    const compact = value.replace(/\s+/g, ' ').trim();
    if (!compact || compact.length > maximum) return { error: `${label} must be non-empty and at most ${maximum} characters.` };
    return { text: compact };
  };
  const title = readText(snapshot.title, 'Declaration title', 240);
  if (title.error) return fail(title.error);
  const body = readText(snapshot.body, 'Declaration body', 1_200);
  if (body.error) return fail(body.error);
  const sources = readText(snapshot.sources, 'Declaration sources', 600);
  if (sources.error) return fail(sources.error);

  return Object.freeze({
    ok: true,
    declaration: Object.freeze({
      type: declaration.type,
      summary: declaration.summary,
      detailTitle: title.text,
      detailBody: body.text,
      sources: sources.text
    })
  });
}

export async function loadPrivateBenchmark({
  projectRoot = PROJECT_ROOT,
  benchRoot = resolve(projectRoot, '.bench-data'),
  root = resolve(benchRoot, 'modern-v1-matrix'),
  manifestPath = resolve(root, 'manifest.json')
} = {}) {
  const realBenchRoot = await realpath(resolve(benchRoot)).catch(() => {
    throw new Error(`Private benchmark root does not exist: ${resolve(benchRoot)}`);
  });
  const configuredRoot = resolveConfiguredPath(projectRoot, root, '--root');
  const configuredManifest = resolveConfiguredPath(projectRoot, manifestPath, '--manifest');
  const realRoot = await realpath(configuredRoot).catch(() => {
    throw new Error(`Benchmark image root does not exist: ${configuredRoot}`);
  });
  assertContained(realBenchRoot, realRoot, 'Benchmark image root must remain inside .bench-data.');
  if (!(await stat(realRoot)).isDirectory()) throw new Error('Benchmark image root must be a directory.');

  const realManifest = await realpath(configuredManifest).catch(() => {
    throw new Error(`Benchmark manifest does not exist: ${configuredManifest}`);
  });
  assertContained(realBenchRoot, realManifest, 'Benchmark manifest must remain inside .bench-data.');
  if (!(await stat(realManifest)).isFile()) throw new Error('Benchmark manifest must be a regular file.');

  let manifest;
  let manifestBytes;
  try {
    manifestBytes = await readFile(realManifest);
    manifest = JSON.parse(manifestBytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Could not parse benchmark manifest: ${error.message}`);
  }
  const sourceRows = Array.isArray(manifest.rows) ? manifest.rows : manifest.samples;
  const collectionName = Array.isArray(manifest.rows) ? 'rows' : 'samples';
  if (!Array.isArray(sourceRows) || sourceRows.length === 0) {
    throw new Error('Benchmark manifest must contain a non-empty rows or samples array.');
  }

  const rows = [];
  const sampleIds = new Set();
  const assetIds = new Set();
  for (const [index, source] of sourceRows.entries()) {
    const field = `${collectionName}[${index}]`;
    if (!source || typeof source !== 'object') throw new Error(`${field} must be an object.`);
    const label = requiredString(source.label, `${field}.label`).toLowerCase();
    if (!LABELS.has(label)) throw new Error(`${field}.label must be ai or real.`);
    const sourcePath = source.relativePath ?? source.file;
    const relativePath = validateRelativePath(sourcePath, `${field}.${source.relativePath == null ? 'file' : 'relativePath'}`);
    const expectedSha256 = source.sha256 == null
      ? undefined
      : validateSha256(source.sha256, `${field}.sha256`);
    const contentIdentity = expectedSha256 ? expectedSha256.slice(0, 16) : stableId(relativePath);
    const familyId = optionalString(source.familyId) || `local-${contentIdentity}`;
    const transformId = optionalString(source.transformId) || 'original';
    const sampleId = optionalString(source.sampleId) || `${familyId}:${transformId}`;
    if (sampleIds.has(sampleId)) throw new Error(`Duplicate sampleId: ${sampleId}`);
    sampleIds.add(sampleId);

    // Frozen benchmark builders record paths relative to the repository so the
    // manifest remains unambiguous when its image root moves. Accept that
    // `.bench-data/...` form as well as paths relative to the configured root;
    // the containment checks below remain authoritative in both cases.
    const lexicalPath = relativePath.replaceAll('\\', '/').startsWith('.bench-data/')
      ? resolve(projectRoot, relativePath)
      : resolve(realRoot, relativePath);
    assertContained(realRoot, lexicalPath, `${field} image path escapes the configured image root.`);
    const absolutePath = await realpath(lexicalPath).catch(() => {
      throw new Error(`Benchmark image does not exist for ${sampleId}.`);
    });
    assertContained(realRoot, absolutePath, `Benchmark image resolves outside the configured image root: ${sampleId}`);
    if (!(await stat(absolutePath)).isFile()) throw new Error(`Benchmark image must be a regular file: ${sampleId}`);
    const imageBytes = await readFile(absolutePath);
    const observedSha256 = sha256(imageBytes);
    if (expectedSha256 && observedSha256 !== expectedSha256) {
      throw new Error(`Benchmark image SHA-256 mismatch for ${sampleId}.`);
    }
    if (source.bytes != null && (!Number.isSafeInteger(source.bytes) || source.bytes <= 0 || source.bytes !== imageBytes.byteLength)) {
      throw new Error(`Benchmark image byte count mismatch for ${sampleId}.`);
    }
    const extension = extname(absolutePath).toLowerCase();
    if (!IMAGE_EXTENSIONS.has(extension)) throw new Error(`Unsupported benchmark image extension for ${sampleId}: ${extension}`);

    const assetId = stableId(`asset\0${sampleId}`, 20);
    if (assetIds.has(assetId)) throw new Error(`Asset identifier collision for ${sampleId}.`);
    assetIds.add(assetId);
    rows.push(Object.freeze({
      assetId,
      sampleId,
      familyId,
      label,
      view: optionalString(source.view) || 'all',
      transformId,
      domain: optionalString(source.domain),
      slice: optionalString(source.slice),
      sha256: observedSha256,
      byteLength: imageBytes.byteLength,
      absolutePath,
      extension,
      mimeType: imageMimeType(extension)
    }));
  }

  rows.sort((left, right) => left.sampleId.localeCompare(right.sampleId));
  return Object.freeze({
    benchmarkId: optionalString(manifest.benchmarkId) || basename(realRoot),
    manifestSchema: optionalString(manifest.schema) || 'unspecified',
    manifestSha256: sha256(manifestBytes),
    root: realRoot,
    manifestPath: realManifest,
    rows: Object.freeze(rows)
  });
}

export function createHarnessServer(benchmark) {
  if (!benchmark?.rows?.length) throw new Error('A loaded benchmark with at least one row is required.');
  const assets = new Map(benchmark.rows.map((row) => [row.assetId, row]));

  return createServer(async (request, response) => {
    try {
      setBaseHeaders(response);
      if (!isLoopbackHost(request.headers.host)) {
        send(response, request.method, 403, 'Loopback Host header required.', 'text/plain; charset=utf-8');
        return;
      }
      if (request.method !== 'GET' && request.method !== 'HEAD') {
        response.setHeader('allow', 'GET, HEAD');
        send(response, request.method, 405, 'Method not allowed.', 'text/plain; charset=utf-8');
        return;
      }

      const url = new URL(request.url || '/', 'http://127.0.0.1');
      if (url.pathname === '/' || url.pathname === '/index.html') {
        const page = renderHarnessPage(benchmark, url.searchParams);
        for (const [name, value] of Object.entries(page.headers)) response.setHeader(name, value);
        send(response, request.method, 200, page.html, 'text/html; charset=utf-8');
        return;
      }

      const redirectMatch = /^\/redirect\/([a-f0-9]{20})$/.exec(url.pathname);
      if (redirectMatch) {
        const row = assets.get(redirectMatch[1]);
        if (!row) {
          send(response, request.method, 404, 'Unknown asset.', 'text/plain; charset=utf-8');
          return;
        }
        response.statusCode = 302;
        response.setHeader('location', withHarnessRun(assetUrl(row), validatedHarnessRun(url.searchParams.get('realimageRun'))));
        response.end();
        return;
      }

      const assetMatch = /^\/asset\/([a-f0-9]{20})\/image\.(?:png|jpe?g|webp|avif|gif)$/.exec(url.pathname);
      if (assetMatch) {
        const row = assets.get(assetMatch[1]);
        if (!row || url.pathname !== assetUrl(row)) {
          send(response, request.method, 404, 'Unknown asset.', 'text/plain; charset=utf-8');
          return;
        }
        const bytes = await readFile(row.absolutePath);
        response.setHeader('cross-origin-resource-policy', 'cross-origin');
        response.setHeader('content-length', String(bytes.length));
        send(response, request.method, 200, bytes, row.mimeType, { contentLengthAlreadySet: true });
        return;
      }

      send(response, request.method, 404, 'Not found.', 'text/plain; charset=utf-8');
    } catch (error) {
      send(response, request.method, 500, `Fixture error: ${error.message}`, 'text/plain; charset=utf-8');
    }
  });
}

export function renderHarnessPage(benchmark, searchParams = new URLSearchParams()) {
  const selection = selectCases(benchmark, searchParams);
  const nonce = randomBytes(18).toString('base64');
  const harnessRunId = randomBytes(16).toString('hex');
  const clientConfig = {
    schema: HARNESS_STATE_SCHEMA,
    harnessRunId,
    benchmarkId: benchmark.benchmarkId,
    manifestSchema: benchmark.manifestSchema,
    manifestSha256: benchmark.manifestSha256,
    selection: selection.options,
    cases: selection.cases.map((item) => ({
      caseId: item.caseId,
      assetId: item.row.assetId,
      sampleId: item.row.sampleId,
      familyId: item.row.familyId,
      label: item.row.label,
      view: item.row.view,
      transformId: item.row.transformId,
      domain: item.row.domain,
      slice: item.row.slice,
      delivery: item.delivery,
      domId: item.domId,
      assetUrl: withHarnessRun(assetUrl(item.row), harnessRunId),
      redirectUrl: withHarnessRun(`/redirect/${item.row.assetId}`, harnessRunId)
    }))
  };
  const serializedConfig = JSON.stringify(clientConfig).replaceAll('<', '\\u003c');
  const html = `<!doctype html>
<html lang="en" data-realimage-harness-status="booting" data-realimage-harness-terminal="false" data-realimage-harness-execution="${selection.options.execution}">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>RealImage private browser benchmark</title>
    <style nonce="${nonce}">
      :root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; background: #eef2f6; color: #14233a; }
      * { box-sizing: border-box; }
      body { margin: 0; }
      header { position: sticky; z-index: 3; top: 0; padding: 14px 20px; background: #101b2dcc; color: white; backdrop-filter: blur(12px); }
      header h1 { margin: 0 0 4px; font: 700 18px/1.2 ui-serif, Georgia, serif; }
      header p { margin: 0; font-size: 12px; opacity: .86; }
      main { width: min(620px, 100% - 24px); margin: 24px auto 160px; }
      #realimage-harness-summary { padding: 12px 14px; border: 1px solid #ccd7e4; border-radius: 12px; background: white; font: 600 13px/1.4 ui-monospace, monospace; }
      #realimage-harness-gallery { display: grid; gap: 80px; margin-top: 28px; }
      figure { min-height: 380px; margin: 0; padding: 14px; border-radius: 16px; background: white; box-shadow: 0 10px 30px #162b4817; }
      .image-slot, .shadow-host { display: grid; min-height: 330px; place-items: center; overflow: hidden; border-radius: 10px; background: repeating-conic-gradient(#e8edf3 0 25%, #f6f8fa 0 50%) 50% / 28px 28px; }
      img { display: block; width: 100%; max-height: 520px; object-fit: contain; }
      figcaption { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; padding: 10px 2px 0; font: 12px/1.4 ui-monospace, monospace; }
      figcaption strong { color: #72411b; text-transform: uppercase; }
      figure[data-status="complete"] { outline: 2px solid #2f855a; }
      figure[data-status$="error"], figure[data-status="timed-out"] { outline: 2px solid #c53030; }
      details { margin-top: 40px; padding: 14px; border-radius: 12px; background: #101b2d; color: #dce7f5; }
      pre { max-height: 460px; overflow: auto; margin: 10px 0 0; white-space: pre-wrap; word-break: break-word; font: 11px/1.45 ui-monospace, monospace; }
      html[data-realimage-harness-execution="burst"] main { width: min(1180px, 100% - 16px); margin: 8px auto 24px; }
      html[data-realimage-harness-execution="burst"] #realimage-harness-gallery { grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; margin-top: 8px; }
      html[data-realimage-harness-execution="burst"] figure { min-height: 106px; padding: 4px; border-radius: 7px; }
      html[data-realimage-harness-execution="burst"] .image-slot,
      html[data-realimage-harness-execution="burst"] .shadow-host { min-height: 78px; border-radius: 5px; }
      html[data-realimage-harness-execution="burst"] img { width: 100%; height: 78px; object-fit: cover; }
      html[data-realimage-harness-execution="burst"] figcaption { gap: 2px; padding: 3px 0 0; font-size: 7px; }
      html[data-realimage-harness-execution="gallery"] main { width: min(980px, 100% - 20px); }
      html[data-realimage-harness-execution="gallery"] #realimage-harness-gallery { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 26px 18px; }
      html[data-realimage-harness-execution="gallery"] figure { min-height: 226px; padding: 8px; }
      html[data-realimage-harness-execution="gallery"] .image-slot,
      html[data-realimage-harness-execution="gallery"] .shadow-host { min-height: 178px; }
      html[data-realimage-harness-execution="gallery"] img { max-height: 190px; }
      html[data-realimage-harness-execution="gallery"] figcaption { padding-top: 5px; font-size: 9px; }
      @media (max-width: 720px) {
        html[data-realimage-harness-execution="burst"] #realimage-harness-gallery { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        html[data-realimage-harness-execution="gallery"] #realimage-harness-gallery { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>RealImage private browser benchmark</h1>
      <p id="realimage-harness-status">Booting local-only harness…</p>
    </header>
    <main>
      <div id="realimage-harness-summary" role="status" aria-live="polite"></div>
      <section id="realimage-harness-gallery" aria-label="Private benchmark cases"></section>
      <details>
        <summary>Machine-readable terminal state</summary>
        <pre id="realimage-harness-state"></pre>
      </details>
    </main>
    <script nonce="${nonce}">
      (() => {
        'use strict';
        const config = ${serializedConfig};
        const parseHarnessBadgeContract = ${parseHarnessBadgeContract.toString()};
        const parseHarnessDeclarationDetails = ${parseHarnessDeclarationDetails.toString()};
        const gallery = document.querySelector('#realimage-harness-gallery');
        const summary = document.querySelector('#realimage-harness-summary');
        const statusLine = document.querySelector('#realimage-harness-status');
        const stateNode = document.querySelector('#realimage-harness-state');
        const imageByCase = new Map();
        const assignedBadges = new Set();
        const badgeByCase = new Map();
        const objectUrls = [];
        const started = performance.now();
        const state = {
          schema: config.schema,
          harnessRunId: config.harnessRunId,
          origin: location.origin,
          benchmarkId: config.benchmarkId,
          manifestSchema: config.manifestSchema,
          manifestSha256: config.manifestSha256,
          status: 'waiting-extension',
          terminal: false,
          selection: config.selection,
          totals: { expected: config.cases.length, complete: 0, correct: 0, incorrect: 0, extensionErrors: 0, imageErrors: 0, timedOut: 0 },
          cases: config.cases.map((item) => ({
            caseId: item.caseId,
            assetId: item.assetId,
            sampleId: item.sampleId,
            familyId: item.familyId,
            expected: item.label,
            view: item.view,
            transformId: item.transformId,
            domain: item.domain,
            slice: item.slice,
            delivery: item.delivery,
            domId: item.domId,
            status: 'queued'
          }))
        };
        window.__REALIMAGE_HARNESS_STATE__ = state;

        for (const [index, item] of config.cases.entries()) {
          const row = state.cases[index];
          const figure = document.createElement('figure');
          figure.id = item.domId;
          figure.dataset.caseId = item.caseId;
          figure.dataset.sampleId = item.sampleId;
          figure.dataset.label = item.label;
          figure.dataset.delivery = item.delivery;
          figure.dataset.status = row.status;
          const slot = document.createElement('div');
          slot.className = item.delivery === 'shadow' ? 'shadow-host' : 'image-slot';
          const image = document.createElement('img');
          image.alt = '';
          image.decoding = 'async';
          image.loading = 'eager';
          image.dataset.realimageCaseId = item.caseId;
          if (item.delivery === 'shadow') {
            const shadow = slot.attachShadow({ mode: 'open' });
            const style = document.createElement('style');
            style.textContent = ':host{display:grid;width:100%;min-height:330px;place-items:center}img{display:block;width:100%;max-height:520px;object-fit:contain}';
            shadow.append(style, image);
          } else {
            slot.append(image);
          }
          const caption = document.createElement('figcaption');
          const identity = document.createElement('span');
          identity.textContent = item.sampleId + ' · ' + item.delivery;
          const label = document.createElement('strong');
          label.textContent = 'expected ' + item.label;
          caption.append(identity, label);
          figure.append(slot, caption);
          gallery.append(figure);
          imageByCase.set(item.caseId, image);
          image.addEventListener('load', () => { row.image = { loaded: true, naturalWidth: image.naturalWidth, naturalHeight: image.naturalHeight, sourceKind: sourceKind(image.currentSrc) }; publish(); });
          image.addEventListener('error', () => { row.image = { loaded: false, error: 'Browser could not decode the local fixture.' }; publish(); });
        }

        publish();
        void run();

        async function run() {
          const overlay = await waitFor(() => document.querySelector('#proofmark-overlay-layer'), 10_000);
          if (!overlay) {
            for (const row of state.cases) {
              row.status = 'extension-error';
              row.error = 'RealImage did not attach its page overlay.';
              row.finishedAfterMs = rounded(performance.now() - started);
            }
            finish('extension-missing');
            return;
          }
          state.status = 'running';
          publish();
          const deadline = performance.now() + config.selection.totalTimeoutMs;
          if (config.selection.execution === 'burst') await runBurst(deadline);
          else if (config.selection.execution === 'gallery') await runGallery(deadline);
          else await runSequential(deadline);
          window.scrollTo({ top: 0, behavior: 'instant' });
          finish('complete');
        }

        async function runSequential(deadline) {
          for (let index = 0; index < config.cases.length; index += 1) {
            const item = config.cases[index];
            const row = state.cases[index];
            const figure = document.getElementById(item.domId);
            const image = imageByCase.get(item.caseId);
            if (performance.now() >= deadline) {
              row.status = 'timed-out';
              row.error = 'Global harness deadline elapsed before this case started.';
              publish();
              continue;
            }
            row.status = 'loading-image';
            row.startedAfterMs = rounded(performance.now() - started);
            figure.dataset.status = row.status;
            figure.scrollIntoView({ block: 'center', behavior: 'instant' });
            await nextPaint();
            const badgesBefore = new Set(document.querySelectorAll('#proofmark-overlay-layer .proofmark-badge'));
            try {
              await activateImage(item, image);
              const loaded = await waitFor(() => image.complete && image.naturalWidth > 0, 15_000);
              if (!loaded) throw new Error('Local image did not load within 15 seconds.');
            } catch (error) {
              row.status = 'image-error';
              row.error = String(error && error.message || error);
              row.finishedAfterMs = rounded(performance.now() - started);
              figure.dataset.status = row.status;
              publish();
              continue;
            }

            row.status = 'waiting-result';
            figure.dataset.status = row.status;
            publish();
            const remaining = Math.max(1, Math.min(config.selection.caseTimeoutMs, deadline - performance.now()));
            const badge = await waitFor(() => findNewBadge(badgesBefore), remaining);
            if (!badge) {
              row.status = 'timed-out';
              row.error = 'RealImage did not create a badge before the case deadline.';
              row.finishedAfterMs = rounded(performance.now() - started);
              figure.dataset.status = row.status;
              publish();
              continue;
            }
            assignedBadges.add(badge);
            const terminalBadge = await waitFor(() => !badge.classList.contains('proofmark-badge--loading') && badge, Math.max(1, Math.min(config.selection.caseTimeoutMs, deadline - performance.now())));
            if (!terminalBadge) {
              row.status = 'timed-out';
              row.error = 'RealImage badge remained in its loading state.';
            } else if (badge.classList.contains('proofmark-badge--error')) {
              row.status = 'extension-error';
              row.error = badge.title || badge.getAttribute('aria-label') || badge.textContent.trim();
              row.badgeText = compact(badge.textContent);
            } else {
              collectResult(row, badge);
            }
            row.finishedAfterMs = rounded(performance.now() - started);
            figure.dataset.status = row.status;
            publish();
          }
        }

        async function runBurst(deadline) {
          const caseDeadlines = new Map();
          await activateAllCases(deadline, (item) => {
            caseDeadlines.set(item.caseId, performance.now() + config.selection.caseTimeoutMs);
          });
          await nextPaint();

          while (state.cases.some((row) => row.status === 'waiting-result')) {
            const now = performance.now();
            for (let index = 0; index < config.cases.length; index += 1) {
              const item = config.cases[index];
              const row = state.cases[index];
              if (row.status !== 'waiting-result') continue;
              const badge = claimBadgeForCase(item);
              if (badge && settleRowFromBadge(row, badge)) continue;
              const caseDeadline = Math.min(deadline, caseDeadlines.get(item.caseId) || deadline);
              if (now >= caseDeadline) markTimedOut(row, badge
                ? 'RealImage badge remained in its loading state.'
                : 'RealImage did not create a badge before the case deadline.');
            }
            publish();
            if (performance.now() >= deadline) break;
            if (state.cases.some((row) => row.status === 'waiting-result')) await delay(50);
          }
          markUnsettledTimedOut('The collective burst deadline elapsed before this case completed.');
        }

        async function runGallery(deadline) {
          await activateAllCases(deadline);
          for (let index = 0; index < config.cases.length; index += 1) {
            const item = config.cases[index];
            const row = state.cases[index];
            if (isTerminalRow(row)) continue;
            if (performance.now() >= deadline) {
              markTimedOut(row, 'Global gallery deadline elapsed before this case entered view.');
              publish();
              continue;
            }

            const figure = document.getElementById(item.domId);
            row.status = 'waiting-result';
            row.analysisStartedAfterMs = rounded(performance.now() - started);
            figure.dataset.status = row.status;
            figure.scrollIntoView({ block: 'center', behavior: 'instant' });
            await nextPaint();
            const remaining = Math.max(1, Math.min(config.selection.caseTimeoutMs, deadline - performance.now()));
            const badge = await waitFor(() => claimBadgeForCase(item), remaining);
            if (!badge) {
              markTimedOut(row, 'RealImage did not create a badge before the gallery case deadline.');
              publish();
              continue;
            }
            const terminalBadge = await waitFor(
              () => !badge.classList.contains('proofmark-badge--loading') && badge,
              Math.max(1, Math.min(config.selection.caseTimeoutMs, deadline - performance.now()))
            );
            if (!terminalBadge) markTimedOut(row, 'RealImage badge remained in its loading state.');
            else settleRowFromBadge(row, badge);
            publish();
          }
          markUnsettledTimedOut('The gallery deadline elapsed before this case completed.');
        }

        async function activateAllCases(deadline, beforeActivation = () => {}) {
          await Promise.all(config.cases.map(async (item, index) => {
            const row = state.cases[index];
            if (performance.now() >= deadline) {
              markTimedOut(row, 'Global harness deadline elapsed before this image loaded.');
              return;
            }
            const image = imageByCase.get(item.caseId);
            row.status = 'loading-image';
            row.startedAfterMs = rounded(performance.now() - started);
            beforeActivation(item, row);
            publish();
            try {
              await activateImage(item, image);
              const loadBudget = Math.max(1, Math.min(15_000, deadline - performance.now()));
              const loaded = await waitFor(() => image.complete && image.naturalWidth > 0, loadBudget);
              if (!loaded) throw new Error('Local image did not load within its bounded deadline.');
              row.status = 'waiting-result';
            } catch (error) {
              row.status = 'image-error';
              row.error = String(error && error.message || error);
              row.finishedAfterMs = rounded(performance.now() - started);
            }
            publish();
          }));
        }

        function claimBadgeForCase(item) {
          const existing = badgeByCase.get(item.caseId);
          if (existing?.isConnected) return existing;
          const image = imageByCase.get(item.caseId);
          const badge = findBadgeForImage(image);
          if (!badge) return undefined;
          assignedBadges.add(badge);
          badgeByCase.set(item.caseId, badge);
          return badge;
        }

        function findBadgeForImage(image) {
          if (!image?.isConnected) return undefined;
          const linkedBadgeId = image.dataset.proofmarkBadgeId;
          if (linkedBadgeId) {
            return [...document.querySelectorAll('#proofmark-overlay-layer .proofmark-badge')]
              .find((badge) => badge.dataset.proofmarkId === linkedBadgeId && !assignedBadges.has(badge));
          }
          const imageRect = image.getBoundingClientRect();
          let best;
          let bestDistance = Number.POSITIVE_INFINITY;
          for (const badge of document.querySelectorAll('#proofmark-overlay-layer .proofmark-badge')) {
            if (assignedBadges.has(badge)) continue;
            const width = badge.offsetWidth;
            const expectedTop = Math.max(6, imageRect.top + 8);
            const expectedLeft = Math.min(innerWidth - width - 6, Math.max(6, imageRect.right - width - 8));
            const actualTop = Number.parseFloat(badge.style.top);
            const actualLeft = Number.parseFloat(badge.style.left);
            if (!Number.isFinite(actualTop) || !Number.isFinite(actualLeft)) continue;
            const distance = Math.abs(actualTop - expectedTop) + Math.abs(actualLeft - expectedLeft);
            if (distance <= 6 && distance < bestDistance) {
              best = badge;
              bestDistance = distance;
            }
          }
          return best;
        }

        function settleRowFromBadge(row, badge) {
          if (badge.classList.contains('proofmark-badge--loading')) return false;
          if (badge.classList.contains('proofmark-badge--error')) {
            row.status = 'extension-error';
            row.error = badge.title || badge.getAttribute('aria-label') || badge.textContent.trim();
            row.badgeText = compact(badge.textContent);
          } else {
            collectResult(row, badge);
          }
          row.finishedAfterMs = rounded(performance.now() - started);
          return true;
        }

        function markTimedOut(row, message) {
          row.status = 'timed-out';
          row.error = message;
          row.finishedAfterMs = rounded(performance.now() - started);
        }

        function markUnsettledTimedOut(message) {
          for (const row of state.cases) {
            if (!isTerminalRow(row)) markTimedOut(row, message);
          }
          publish();
        }

        function isTerminalRow(row) {
          return ['complete', 'extension-error', 'image-error', 'timed-out'].includes(row.status);
        }

        function findNewBadge(before) {
          return [...document.querySelectorAll('#proofmark-overlay-layer .proofmark-badge')]
            .find((badge) => !before.has(badge) && !assignedBadges.has(badge));
        }

        function collectResult(row, badge) {
          row.badgeText = compact(badge.textContent);
          row.badgeAriaLabel = badge.getAttribute('aria-label') || '';
          const badgeKinds = ['ai', 'real', 'declared'].filter((kind) => badge.classList.contains('proofmark-badge--' + kind));
          const contract = parseHarnessBadgeContract({
            badgeKind: badgeKinds.length === 1 ? badgeKinds[0] : '',
            modelVerdict: badge.dataset.proofmarkModelVerdict,
            modelScorePercent: badge.dataset.proofmarkModelScorePercent,
            decisionThreshold: badge.dataset.proofmarkDecisionThreshold,
            declarationType: badge.dataset.proofmarkDeclarationType,
            declarationSummary: badge.dataset.proofmarkDeclarationSummary
          });
          if (!contract.ok) {
            row.status = 'extension-error';
            row.error = 'RealImage badge result contract invalid: ' + contract.error;
            row.predicted = 'unknown';
            row.score = null;
            row.decisionThreshold = null;
            row.declaration = null;
            row.correct = false;
            return;
          }

          row.model = contract.model;
          row.predicted = contract.model.verdict;
          row.score = contract.model.score;
          row.decisionThreshold = contract.model.decisionThreshold;
          row.declaration = contract.declaration;
          row.correct = row.predicted === row.expected;
          badge.click();
          const panel = document.querySelector('#proofmark-overlay-layer .proofmark-detail');
          if (!panel || panel.hidden || !badge.dataset.proofmarkId || panel.dataset.anchor !== badge.dataset.proofmarkId) {
            row.status = 'extension-error';
            row.error = 'RealImage detail panel did not open for the matching badge.';
            row.correct = false;
            return;
          }
          row.detail = {
            verdict: compact(panel.querySelector('.proofmark-detail__verdict')?.textContent || ''),
            scoreText: compact(panel.querySelector('.proofmark-detail__score')?.textContent || ''),
            meta: compact(panel.querySelector('.proofmark-detail__meta')?.textContent || ''),
            evidence: [...panel.querySelectorAll('.proofmark-detail__evidence li')].map((node) => compact(node.textContent))
          };
          const total = row.detail.meta.match(/total\\s+(\\d+(?:\\.\\d+)?)\\s*ms/i);
          const inference = row.detail.meta.match(/inference\\s+(\\d+(?:\\.\\d+)?)\\s*ms/i);
          const threads = row.detail.meta.match(/(?:·|^)\\s*(\\d+)\\s+threads?$/i);
          row.timings = { totalMs: total ? Number(total[1]) : null, inferenceMs: inference ? Number(inference[1]) : null };
          row.runtime = { threads: threads ? Number(threads[1]) : null, summary: row.detail.meta };

          const declarationNodes = [...panel.querySelectorAll('[data-proofmark-declaration-details]')];
          const declarationNode = declarationNodes[0];
          const declarationTitleNodes = [...panel.querySelectorAll('[data-proofmark-declaration-title]')];
          const declarationBodyNodes = [...panel.querySelectorAll('[data-proofmark-declaration-body]')];
          const declarationSourceNodes = [...panel.querySelectorAll('[data-proofmark-declaration-sources]')];
          const declarationDetails = parseHarnessDeclarationDetails({
            count: declarationNodes.length,
            titleCount: declarationTitleNodes.length,
            bodyCount: declarationBodyNodes.length,
            sourcesCount: declarationSourceNodes.length,
            titleInBlock: declarationTitleNodes[0]?.closest('[data-proofmark-declaration-details]') === declarationNode,
            bodyInBlock: declarationBodyNodes[0]?.closest('[data-proofmark-declaration-details]') === declarationNode,
            sourcesInBlock: declarationSourceNodes[0]?.closest('[data-proofmark-declaration-details]') === declarationNode,
            title: declarationTitleNodes[0]?.textContent,
            body: declarationBodyNodes[0]?.textContent,
            sources: declarationSourceNodes[0]?.textContent
          }, contract.declaration);
          if (!declarationDetails.ok) {
            row.status = 'extension-error';
            row.error = 'RealImage declaration detail contract invalid: ' + declarationDetails.error;
            row.declaration = contract.declaration;
            row.correct = false;
            return;
          }
          row.declaration = declarationDetails.declaration;
          row.status = 'complete';
        }

        async function activateImage(item, image) {
          if (item.delivery === 'delayed') await delay(250);
          if (item.delivery === 'blob') {
            const response = await fetch(item.assetUrl, { cache: 'no-store', credentials: 'omit', redirect: 'error' });
            if (!response.ok) throw new Error('Same-origin blob fixture returned HTTP ' + response.status + '.');
            const objectUrl = URL.createObjectURL(await response.blob());
            objectUrls.push(objectUrl);
            image.src = objectUrl;
            return;
          }
          if (item.delivery === 'srcset') {
            image.sizes = '(max-width: 700px) 92vw, 580px';
            image.srcset = addQuery(item.assetUrl, 'candidate', 'small') + ' 480w, ' + addQuery(item.assetUrl, 'candidate', 'large') + ' 960w';
            image.src = addQuery(item.assetUrl, 'fallback', '1');
            return;
          }
          if (item.delivery === 'source-swap') {
            image.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==';
            await delay(250);
            image.src = addQuery(item.assetUrl, 'swapped', '1');
            return;
          }
          if (item.delivery === 'query') {
            image.src = addQuery(item.assetUrl, 'cache-key', item.caseId);
            return;
          }
          image.src = item.delivery === 'redirect' ? item.redirectUrl : item.assetUrl;
        }

        function finish(status) {
          state.status = status;
          state.terminal = true;
          state.durationMs = rounded(performance.now() - started);
          publish();
          for (const url of objectUrls) URL.revokeObjectURL(url);
        }

        function publish() {
          state.totals.complete = state.cases.filter((row) => row.status === 'complete').length;
          state.totals.correct = state.cases.filter((row) => row.status === 'complete' && row.correct).length;
          state.totals.incorrect = state.cases.filter((row) => row.status === 'complete' && !row.correct).length;
          state.totals.extensionErrors = state.cases.filter((row) => row.status === 'extension-error').length;
          state.totals.imageErrors = state.cases.filter((row) => row.status === 'image-error').length;
          state.totals.timedOut = state.cases.filter((row) => row.status === 'timed-out').length;
          document.documentElement.dataset.realimageHarnessStatus = state.status;
          document.documentElement.dataset.realimageHarnessTerminal = String(state.terminal);
          statusLine.textContent = state.status + ' · ' + state.totals.complete + '/' + state.totals.expected + ' complete';
          summary.textContent = 'status=' + state.status + ' expected=' + state.totals.expected + ' complete=' + state.totals.complete + ' correct=' + state.totals.correct + ' errors=' + (state.totals.extensionErrors + state.totals.imageErrors) + ' timeouts=' + state.totals.timedOut;
          stateNode.textContent = JSON.stringify(state, null, 2);
          for (const row of state.cases) {
            const figure = document.getElementById(row.domId);
            if (figure) figure.dataset.status = row.status;
          }
        }

        function sourceKind(value) {
          if (value.startsWith('blob:')) return 'blob';
          if (value.startsWith('data:')) return 'data';
          return 'same-origin-http';
        }
        function compact(value) { return String(value).replace(/\\s+/g, ' ').trim(); }
        function addQuery(value, name, content) { return value + (value.includes('?') ? '&' : '?') + encodeURIComponent(name) + '=' + encodeURIComponent(content); }
        function rounded(value) { return Math.round(value * 100) / 100; }
        function delay(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
        function nextPaint() { return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))); }
        async function waitFor(probe, timeoutMs) {
          const end = performance.now() + timeoutMs;
          while (performance.now() < end) {
            const result = probe();
            if (result) return result;
            await delay(50);
          }
          return undefined;
        }
      })();
    </script>
  </body>
</html>`;
  return {
    html,
    headers: {
      'content-security-policy': `default-src 'none'; base-uri 'none'; connect-src 'self'; img-src 'self' blob: data:; object-src 'none'; frame-src 'none'; script-src 'nonce-${nonce}'; style-src 'nonce-${nonce}'; form-action 'none'`,
      'cross-origin-opener-policy': 'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
      'cross-origin-resource-policy': 'same-origin',
      'permissions-policy': 'camera=(), microphone=(), geolocation=(), payment=(), usb=()'
    }
  };
}

export function stableId(value, length = 16) {
  return createHash('sha256').update(String(value)).digest('hex').slice(0, length);
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function selectCases(benchmark, searchParams) {
  const availableViews = new Set(benchmark.rows.map((row) => row.view));
  const defaultView = availableViews.has('clean') ? 'clean' : 'all';
  const view = searchParams.get('view') || defaultView;
  const label = searchParams.get('label') || 'all';
  const delivery = searchParams.get('delivery') || 'direct';
  const execution = searchParams.get('execution') || 'sequential';
  if (view !== 'all' && !availableViews.has(view)) throw new Error(`Unknown view: ${view}`);
  if (label !== 'all' && !LABELS.has(label)) throw new Error(`label must be all, ai, or real.`);
  if (delivery !== 'web-stress' && !DELIVERIES.includes(delivery)) throw new Error(`Unknown delivery: ${delivery}`);
  if (!EXECUTIONS.has(execution)) throw new Error('execution must be sequential, burst, or gallery.');
  const expandedDirectRun = delivery === 'direct' && execution !== 'sequential';
  const maximumLimit = delivery === 'web-stress' ? 4 : expandedDirectRun ? 60 : 32;
  const defaultLimit = delivery === 'web-stress'
    ? 4
    : expandedDirectRun
      ? execution === 'burst' ? 20 : 60
      : 12;
  const offset = boundedInteger(searchParams.get('offset'), 0, Number.MAX_SAFE_INTEGER, 0, 'offset');
  const limit = boundedInteger(searchParams.get('limit'), 1, maximumLimit, defaultLimit, 'limit');
  const caseTimeoutMs = boundedInteger(searchParams.get('caseTimeoutMs'), 5_000, 120_000, 45_000, 'caseTimeoutMs');
  const totalTimeoutMs = boundedInteger(searchParams.get('totalTimeoutMs'), 10_000, 600_000, 180_000, 'totalTimeoutMs');
  const filtered = benchmark.rows.filter((row) => (view === 'all' || row.view === view) && (label === 'all' || row.label === label));
  const selected = filtered.slice(offset, offset + limit);
  const deliveries = delivery === 'web-stress' ? DELIVERIES : [delivery];
  const cases = selected.flatMap((row) => deliveries.map((caseDelivery) => {
    const caseId = `${row.sampleId}::${caseDelivery}`;
    return { row, delivery: caseDelivery, caseId, domId: `tg-case-${stableId(caseId)}` };
  }));
  return {
    options: { view, label, delivery, execution, offset, limit, caseTimeoutMs, totalTimeoutMs, availableRows: filtered.length },
    cases
  };
}

function resolveConfiguredPath(projectRoot, value, optionName) {
  const input = String(value);
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(input) || input.startsWith('\\\\')) {
    throw new Error(`${optionName} must be a local filesystem path.`);
  }
  return isAbsolute(input) ? resolve(input) : resolve(projectRoot, input);
}

function validateRelativePath(value, field) {
  const input = requiredString(value, field);
  if (input.includes('\0') || /^[a-z][a-z0-9+.-]*:\/\//i.test(input) || input.startsWith('//') || input.startsWith('\\\\') || isAbsolute(input)) {
    throw new Error(`${field} must be a local relative path.`);
  }
  return input;
}

function assertContained(parent, candidate, message) {
  const pathFromParent = relative(parent, candidate);
  if (pathFromParent === '..' || pathFromParent.startsWith(`..${sep}`) || isAbsolute(pathFromParent)) throw new Error(message);
}

function requiredString(value, field) {
  if (typeof value !== 'string' || value.trim() === '') throw new Error(`${field} must be a non-empty string.`);
  return value.trim();
}

function optionalString(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function boundedInteger(value, minimum, maximum, fallback, name) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} through ${maximum}.`);
  }
  return parsed;
}

function assetUrl(row) {
  const extension = row.extension === '.jpeg' ? '.jpg' : row.extension;
  return `/asset/${row.assetId}/image${extension}`;
}

function withHarnessRun(value, harnessRunId) {
  return harnessRunId ? `${value}?realimageRun=${encodeURIComponent(harnessRunId)}` : value;
}

function validatedHarnessRun(value) {
  if (value == null || value === '') return undefined;
  if (!/^[a-f0-9]{32}$/.test(value)) throw new Error('Invalid harness run identifier.');
  return value;
}

function validateSha256(value, field) {
  const input = requiredString(value, field).toLowerCase();
  if (!/^[a-f0-9]{64}$/.test(input)) throw new Error(`${field} must be a SHA-256.`);
  return input;
}

function imageMimeType(extension) {
  if (extension === '.jpg' || extension === '.jpeg') return 'image/jpeg';
  if (extension === '.png') return 'image/png';
  if (extension === '.webp') return 'image/webp';
  if (extension === '.avif') return 'image/avif';
  if (extension === '.gif') return 'image/gif';
  return 'application/octet-stream';
}

function setBaseHeaders(response) {
  response.setHeader('cache-control', 'no-store, max-age=0');
  response.setHeader('x-content-type-options', 'nosniff');
  response.setHeader('referrer-policy', 'no-referrer');
}

function send(response, method, status, body, contentType, { contentLengthAlreadySet = false } = {}) {
  const bytes = Buffer.isBuffer(body) ? body : Buffer.from(String(body));
  response.statusCode = status;
  response.setHeader('content-type', contentType);
  if (!contentLengthAlreadySet) response.setHeader('content-length', String(bytes.length));
  response.end(method === 'HEAD' ? undefined : bytes);
}

function isLoopbackHost(value) {
  if (typeof value !== 'string') return false;
  try {
    const hostname = new URL(`http://${value}`).hostname.toLowerCase();
    return hostname === '127.0.0.1' || hostname === 'localhost';
  } catch {
    return false;
  }
}

function parseArgs(argv) {
  const options = {};
  for (const argument of argv) {
    if (argument === '--help' || argument === '-h') options.help = true;
    else if (argument.startsWith('--root=')) options.root = argument.slice('--root='.length);
    else if (argument.startsWith('--manifest=')) options.manifestPath = argument.slice('--manifest='.length);
    else if (argument.startsWith('--port=')) options.port = boundedInteger(argument.slice('--port='.length), 0, 65_535, 4174, 'port');
    else throw new Error(`Unknown option: ${argument}`);
  }
  return options;
}

async function runCli() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log('Usage: node scripts/serve-private-browser-benchmark.mjs [--root=.bench-data/modern-v1-matrix] [--manifest=PATH] [--port=4174]');
    console.log('The root, manifest, and every served image must resolve inside this repository\'s ignored .bench-data directory.');
    return;
  }
  const root = options.root || resolve(PROJECT_ROOT, '.bench-data/modern-v1-matrix');
  const manifestPath = options.manifestPath || resolve(root, 'manifest.json');
  const benchmark = await loadPrivateBenchmark({ root, manifestPath });
  const server = createHarnessServer(benchmark);
  server.requestTimeout = 30_000;
  server.headersTimeout = 15_000;
  const port = options.port ?? 4174;
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  console.log(`Private RealImage browser harness: http://127.0.0.1:${address.port}/`);
  console.log(`Loaded ${benchmark.rows.length} manifest rows from ${benchmark.benchmarkId}. No remote fetches are implemented.`);
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.once(signal, () => server.close(() => process.exit(0)));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await runCli().catch((error) => {
    console.error(`Private browser harness failed: ${error.message}`);
    process.exitCode = 1;
  });
}
