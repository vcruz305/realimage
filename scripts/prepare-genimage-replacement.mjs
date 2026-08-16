// Pulls the ai/genimage-sd15 and ai/genimage-midjourney replacement sources
// (added 2026-08-16 to replace the retired, unverified-provenance
// ai/sd15-stock folder -- see manifest provenanceNotes and MODEL_CARD.md)
// from the licensed HF dataset TheKernel01/Tiny-GenImage (CC-BY-NC-SA-4.0),
// via the public datasets-server REST API. Same fetch/dedup/retry pattern as
// scripts/prepare-realimage-broad-v1.mjs, adapted for this dataset's
// {label, generator} schema instead of {label, source_dataset}.
import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import sharp from 'sharp';

const DATASET = 'TheKernel01/Tiny-GenImage';
const SPLIT = 'validation';
const TOTAL_ROWS = 7000;
const output = resolve(process.argv[2] || '.bench-data/realimage-broad-v1/_staging');
const concurrency = Number(process.env.PROOFMARK_FETCH_CONCURRENCY || 6);

const GENERATOR_NAMES = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD14', 'SD15', 'VQDM', 'Wukong'];

const specifications = [
  { key: 'genimage-sd15', generatorId: GENERATOR_NAMES.indexOf('SD15'), count: 170 },
  { key: 'genimage-midjourney', generatorId: GENERATOR_NAMES.indexOf('Midjourney'), count: 170 }
];

const manifest = [];
let allRowsCache = null;

for (const specification of specifications) {
  const directory = `${output}/ai/${specification.key}`;
  await mkdir(directory, { recursive: true });
  const exactHashes = new Set();
  const candidates = await fetchCandidates(specification);
  console.error(`${specification.key}: found ${candidates.length} candidate rows`);
  const accepted = [];
  for (let cursor = 0; cursor < candidates.length && accepted.length < specification.count; cursor += concurrency) {
    const batch = candidates.slice(cursor, cursor + concurrency);
    const downloaded = await Promise.all(batch.map((candidate) => downloadCandidate(candidate).catch(() => null)));
    for (const item of downloaded) {
      if (!item || accepted.length >= specification.count || exactHashes.has(item.sha256)) continue;
      exactHashes.add(item.sha256);
      const fileName = `${item.sha256.slice(0, 16)}.${item.extension}`;
      await writeFile(`${directory}/${fileName}`, item.bytes);
      const record = {
        label: 'ai',
        domain: specification.key,
        fileName,
        sourceDataset: DATASET,
        sourceLicense: 'cc-by-nc-sa-4.0',
        sourceSplit: SPLIT,
        sourceRow: item.rowIndex,
        generator: GENERATOR_NAMES[specification.generatorId],
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
  if (accepted.length !== specification.count) {
    throw new Error(`Only accepted ${accepted.length}/${specification.count} images for ${specification.key}`);
  }
}

await writeFile(`${output}/tiny-genimage-pull-manifest.json`, `${JSON.stringify({ dataset: DATASET, license: 'cc-by-nc-sa-4.0', split: SPLIT, generatedAt: new Date().toISOString(), rows: manifest }, null, 2)}\n`);
console.log(`Prepared ${manifest.length} licensed replacement images from ${DATASET} in ${output}`);

async function fetchAllRows() {
  if (allRowsCache) return allRowsCache;
  const batchLength = 100;
  const starts = [];
  for (let start = 0; start < TOTAL_ROWS; start += batchLength) starts.push(start);
  const batches = [];
  const rowConcurrency = 3;
  for (let cursor = 0; cursor < starts.length; cursor += rowConcurrency) {
    const group = starts.slice(cursor, cursor + rowConcurrency);
    const results = await Promise.all(group.map(async (start) => {
      const parameters = new URLSearchParams({ dataset: DATASET, config: 'default', split: SPLIT, offset: String(start), length: String(batchLength) });
      const response = await fetchWithRetry(`https://datasets-server.huggingface.co/rows?${parameters}`);
      const payload = await response.json();
      return payload.rows || [];
    }));
    for (const rows of results) batches.push(...rows);
    process.stderr.write(`\rscanned ${Math.min(cursor + group.length, starts.length) * batchLength}/${TOTAL_ROWS} rows`);
    await delay(400);
  }
  process.stderr.write('\n');
  allRowsCache = batches;
  return batches;
}

async function fetchCandidates(specification) {
  const rows = await fetchAllRows();
  return rows.filter((item) => Number(item.row?.generator) === specification.generatorId && Number(item.row?.label) === 1 && item.row?.image?.src);
}

async function downloadCandidate(candidate) {
  const response = await fetchWithRetry(candidate.row.image.src);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < 1_000 || bytes.length > 25_000_000) return null;
  const metadata = await sharp(bytes).metadata();
  if (!metadata.width || !metadata.height || Math.min(metadata.width, metadata.height) < 96) return null;
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return {
    bytes,
    rowIndex: candidate.row_idx,
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
  for (let attempt = 1; attempt <= 16; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      lastError = new Error(`HTTP ${response.status}`);
      const retryAfter = Number(response.headers.get('retry-after') || 0);
      await delay(Math.max(retryAfter * 1_000, attempt * 1_000, 1_500));
      continue;
    } catch (error) {
      lastError = error;
    }
    await delay(attempt * 1_000);
  }
  throw lastError || new Error(`Unable to fetch ${url}`);
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}
