// Pulls a set of source_dataset partitions out of the HF-hosted aggregator
// "Zitacron/real-vs-ai-corpus" (which itself re-tags many underlying AI-generator
// and real-photo datasets, each carrying its own per-row source_license) via the
// public datasets-server REST API, and stages the accepted files into
// .bench-data/realimage-broad-v1/_staging/<ai|real>/<name>/ for the later
// split-and-manifest step. Adapted from scripts/prepare-webwild-sources.mjs.
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import sharp from 'sharp';

const DATASET = 'Zitacron/real-vs-ai-corpus';
const output = resolve(process.argv[2] || '.bench-data/realimage-broad-v1/_staging');
const concurrency = Number(process.env.PROOFMARK_FETCH_CONCURRENCY || 4);

// Offset ranges were located by sampling scripts/map-hf-corpus-sources.mjs across
// the dataset's 12,336,207-row train split (see also scripts/_scratch-map-sources.mjs
// probes). Each source_dataset value is a distinct upstream AI generator or real-photo
// collection with its own license, reported per-row in source_license.
const specifications = [
  { key: 'synthetic-characters', label: 'ai', source: 'AbstractPhil/synthetic-characters', license: 'cc-by-4.0', start: 0, end: 140_000, count: 300 },
  { key: 'flux-reason', label: 'ai', source: 'LucasFang/FLUX-Reason-6M', license: 'apache-2.0', start: 180_000, end: 6_000_000, count: 300 },
  { key: 'visual-logic', label: 'real', source: 'skylenage/DeepVision-103K', license: 'cc-by-4.0', start: 150_000, end: 174_000, count: 300 },
  { key: 'laion-aesthetic', label: 'real', source: 'laion/laion2B-en-aesthetic', license: 'cc-by-4.0', start: 6_050_000, end: 12_330_000, count: 300 }
];

await mkdir(output, { recursive: true });
const manifest = [];

for (const specification of specifications) {
  const directory = `${output}/${specification.label}/${specification.key}`;
  await mkdir(directory, { recursive: true });
  const exactHashes = new Set();
  const candidates = await fetchCandidates(specification);
  const accepted = [];
  for (let cursor = 0; cursor < candidates.length && accepted.length < specification.count; cursor += concurrency) {
    const batch = candidates.slice(cursor, cursor + concurrency);
    const downloaded = await Promise.all(batch.map((candidate) => downloadCandidate(candidate, specification).catch(() => null)));
    for (const item of downloaded) {
      if (!item || accepted.length >= specification.count || exactHashes.has(item.sha256)) continue;
      exactHashes.add(item.sha256);
      const fileName = `${item.sha256.slice(0, 16)}.${item.extension}`;
      await writeFile(`${directory}/${fileName}`, item.bytes);
      const record = {
        label: specification.label,
        domain: specification.key,
        fileName,
        sourceDataset: specification.source,
        sourceLicense: item.sourceLicense || specification.license,
        sourceRow: item.rowIndex,
        width: item.width,
        height: item.height,
        format: item.format,
        sha256: item.sha256
      };
      accepted.push(record);
      manifest.push(record);
    }
    process.stderr.write(`\r${specification.key}: ${accepted.length}/${specification.count}`);
  }
  process.stderr.write('\n');
  console.log(`${specification.key}: staged ${accepted.length}/${specification.count} (source=${specification.source}, license=${specification.license})`);
}

await mkdir(output, { recursive: true });
await writeFile(`${output}/zitacron-pull-manifest.json`, `${JSON.stringify({ dataset: DATASET, generatedAt: new Date().toISOString(), rows: manifest }, null, 2)}\n`);
console.log(`Prepared ${manifest.length} licensed images from ${DATASET} in ${output}`);

async function fetchCandidates(specification) {
  const batchCount = 8;
  const batchLength = 100;
  const span = specification.end - specification.start - batchLength;
  const starts = Array.from({ length: batchCount }, (_, index) =>
    Math.floor(specification.start + (span * index) / (batchCount - 1))
  );
  const batches = [];
  for (const start of starts) {
    const parameters = new URLSearchParams({
      dataset: DATASET,
      config: 'default',
      split: 'train',
      offset: String(start),
      length: String(batchLength)
    });
    const response = await fetchWithRetry(`https://datasets-server.huggingface.co/rows?${parameters}`);
    const payload = await response.json();
    batches.push(...(payload.rows || []));
    await delay(150);
  }
  return batches.filter((item) =>
    item.row?.source_dataset === specification.source &&
    Number(item.row?.label) === (specification.label === 'ai' ? 1 : 0) &&
    item.row?.image?.src
  );
}

async function downloadCandidate(candidate, specification) {
  const response = await fetchWithRetry(candidate.row.image.src);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < 1_000 || bytes.length > 25_000_000) return null;
  const metadata = await sharp(bytes).metadata();
  if (!metadata.width || !metadata.height || Math.min(metadata.width, metadata.height) < 96) return null;
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return {
    bytes,
    rowIndex: candidate.row_idx,
    sourceLicense: candidate.row.source_license,
    width: metadata.width,
    height: metadata.height,
    format: metadata.format,
    extension: extensionFor(metadata.format),
    sha256
  };
}

function extensionFor(format) {
  if (format === 'jpeg') return 'jpg';
  if (['png', 'webp', 'avif', 'gif'].includes(format)) return format;
  return 'img';
}

async function fetchWithRetry(url) {
  let lastError;
  for (let attempt = 1; attempt <= 10; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      lastError = new Error(`HTTP ${response.status}`);
      const retryAfter = Number(response.headers.get('retry-after') || 0);
      await delay(Math.max(retryAfter * 1_000, attempt * 1_500));
      continue;
    } catch (error) {
      lastError = error;
      await delay(attempt * 1_000);
    }
  }
  throw lastError || new Error(`Unable to fetch ${url}`);
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}
