// Pulls ai/sd35 (added 2026-08-16 for the "modern generator" corpus
// expansion -- see manifest.json provenanceNotes) from
// data-is-better-together/open-image-preferences-v1 on HF (Apache-2.0
// license). That dataset pits two generators against each other per prompt:
// a column pair named "*_dev" (FLUX.1-dev) and a column pair named "*_sd"
// (a Stable-Diffusion-3.x-class checkpoint -- the exact checkpoint id is not
// machine-readable from the parquet metadata or dataset card, only inferable
// from the column-naming convention and the dataset's own "flux" /
// "stable-diffusion" tags; recorded here as SD3.x-class rather than an exact
// pinned version, consistent with this repo's practice of flagging
// uncertain provenance honestly rather than guessing -- see
// provenanceNote in the manifest entry this script produces).
// We only pull the "_sd" columns (never "_dev", to avoid overlapping with
// the existing ai/flux-krea / ai/flux-reason FLUX sources). No category
// filter is applied (the datasets-server rows endpoint is too slow on this
// 11GB/8667-row/4-image-columns dataset to scan for a specific category
// value within the time-box; an unfiltered sample still covers a wide style
// mix, including photographic prompts, which is fine for generator-identity
// diversity even though it is not exclusively photoreal).
import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import sharp from 'sharp';

const DATASET = 'data-is-better-together/open-image-preferences-v1';
const LICENSE = 'apache-2.0';
const SPLIT = 'cleaned';
const TOTAL_ROWS = 8_667;
const COUNT = 260;
const IMAGE_COLUMNS = ['image_quality_sd', 'image_simplified_sd'];
const output = resolve(process.argv[2] || '.bench-data/realimage-broad-v1/_staging');
const concurrency = Number(process.env.PROOFMARK_FETCH_CONCURRENCY || 3);

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

async function main() {
  const directory = `${output}/ai/sd35`;
  await mkdir(directory, { recursive: true });
  const existingFileNames = new Set(await readdir(directory));
  if (existingFileNames.size >= COUNT) {
    console.log(`sd35: already has ${existingFileNames.size}/${COUNT} staged, nothing to do (resumed run)`);
    return;
  }
  const remaining = COUNT - existingFileNames.size;
  console.log(`sd35: resuming, ${existingFileNames.size} already staged, need ${remaining} more`);
  await mkdir(`${output}/_checkpoints`, { recursive: true });
  const checkpointPath = `${output}/_checkpoints/sd35.json`;
  const exactHashes = new Set();
  let newlyAdded = 0;
  const manifest = [];

  const random = mulberry32(0x53443335); // 'SD35' -- fixed seed for reproducibility
  const rowIndices = [];
  const seen = new Set();
  while (rowIndices.length < TOTAL_ROWS) {
    const index = Math.floor(random() * TOTAL_ROWS);
    if (seen.has(index)) continue;
    seen.add(index);
    rowIndices.push(index);
  }

  const pageLength = concurrency; // one in-flight row-fetch batch at a time, sized to concurrency to bound simultaneous requests to datasets-server
  const startCursor = await readCheckpoint(checkpointPath);
  if (startCursor > 0) console.log(`sd35: resuming row scan from cursor=${startCursor} (checkpoint)`);
  for (let cursor = startCursor; cursor < rowIndices.length && newlyAdded < remaining; cursor += pageLength) {
    const batchIndices = rowIndices.slice(cursor, cursor + pageLength);
    // datasets-server rows endpoint only supports contiguous offset/length,
    // not arbitrary index lists, so we fetch each shuffled index as its own
    // length=1 request (kept small and serialized in pageLength-sized
    // groups below to avoid bursting the datasets-server rate limit).
    const rows = await Promise.all(batchIndices.map((index) => fetchRow(index).catch(() => null)));
    await delay(200);
    await writeFile(checkpointPath, JSON.stringify({ cursor: cursor + pageLength })).catch(() => {});
    const candidates = [];
    for (const row of rows) {
      if (!row) continue;
      for (const column of IMAGE_COLUMNS) {
        if (row[column]) candidates.push(row[column]);
      }
    }
    for (let i = 0; i < candidates.length && newlyAdded < remaining; i += concurrency) {
      const batch = candidates.slice(i, i + concurrency);
      const downloaded = await Promise.all(batch.map((candidate) => downloadCandidate(candidate).catch(() => null)));
      for (const item of downloaded) {
        if (!item || newlyAdded >= remaining || exactHashes.has(item.sha256)) continue;
        exactHashes.add(item.sha256);
        const fileName = `${item.sha256.slice(0, 16)}.${item.extension}`;
        const record = {
          label: 'ai',
          domain: 'sd35',
          fileName,
          sourceDataset: DATASET,
          sourceLicense: LICENSE,
          sourceSplit: SPLIT,
          width: item.width,
          height: item.height,
          format: item.format,
          sha256: item.sha256,
        };
        manifest.push(record);
        if (existingFileNames.has(fileName)) continue; // identical content already staged from a prior run -- nothing to do
        await writeFile(`${directory}/${fileName}`, item.bytes);
        existingFileNames.add(fileName);
        newlyAdded += 1;
      }
      await delay(150);
    }
    process.stderr.write(`\rsd35: ${existingFileNames.size}/${COUNT} (row cursor=${cursor})`);
  }
  process.stderr.write('\n');
  const finalCount = (await readdir(directory)).length;
  if (finalCount < COUNT) {
    throw new Error(`Only staged ${finalCount}/${COUNT} images for sd35`);
  }

  await writeFile(`${output}/open-image-preferences-sd35-pull-manifest.json`, `${JSON.stringify({ dataset: DATASET, license: LICENSE, split: SPLIT, generatedAt: new Date().toISOString(), rows: manifest }, null, 2)}\n`);
  console.log(`Prepared ${manifest.length} newly-staged SD3.x images from ${DATASET} in ${output} (${finalCount} total staged)`);
}

async function fetchRow(index) {
  const parameters = new URLSearchParams({ dataset: DATASET, config: 'default', split: SPLIT, offset: String(index), length: '1' });
  const response = await fetchWithRetry(`https://datasets-server.huggingface.co/rows?${parameters}`);
  const payload = await response.json();
  return payload.rows?.[0]?.row || null;
}

async function downloadCandidate(imageField) {
  const response = await fetchWithRetry(imageField.src);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < 1_000 || bytes.length > 25_000_000) return null;
  const metadata = await sharp(bytes).metadata();
  if (!metadata.width || !metadata.height || Math.min(metadata.width, metadata.height) < 256) return null;
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return {
    bytes,
    width: metadata.width,
    height: metadata.height,
    format: metadata.format,
    extension: extensionFor(metadata.format),
    sha256,
  };
}

function extensionFor(format) {
  if (format === 'jpeg') return 'jpg';
  if (['png', 'webp', 'avif', 'gif'].includes(format)) return format;
  return 'img';
}

async function fetchWithRetry(url) {
  let lastError;
  for (let attempt = 1; attempt <= 24; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      lastError = new Error(`HTTP ${response.status}`);
      const retryAfter = Number(response.headers.get('retry-after') || 0);
      const isRateLimited = response.status === 429;
      const backoff = isRateLimited
        ? Math.min(8_000 * attempt, 45_000) + Math.random() * 2_000
        : Math.max(attempt * 1_000, 1_500);
      await delay(Math.max(retryAfter * 1_000, backoff));
      continue;
    } catch (error) {
      lastError = error;
      await delay(Math.min(2_000 * attempt, 20_000));
    }
  }
  throw lastError || new Error(`Unable to fetch ${url}`);
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function readCheckpoint(path) {
  try {
    const raw = await readFile(path, 'utf8');
    const parsed = JSON.parse(raw);
    return Number(parsed.cursor) || 0;
  } catch {
    return 0;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
