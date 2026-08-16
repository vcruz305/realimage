// Pulls ai/midjourney-v6 (added 2026-08-16 for the "modern generator" corpus
// expansion -- see manifest.json provenanceNotes) from Photoroom's
// Midjourney v6 recaptioning dataset on HF (MIT license). This is genuinely
// v6-era Midjourney (NOT the v5-era images already in ai/genimage-midjourney),
// satisfying the requirement for a newer-generation Midjourney sample.
// Images are sampled from a deterministic pseudo-random walk over row
// offsets (the dataset has 1.2M rows; we only need a few hundred) using the
// same fetch/dedup/retry pattern as scripts/prepare-genimage-replacement.mjs.
//
// Resumable by design: a checkpoint file (outside the domain directory, so
// it never gets mistaken for a corpus image by split-realimage-broad-v1.mjs)
// records how many pseudo-random offsets have already been consumed, so a
// re-run after a crash/rate-limit fast-forwards past already-fetched offsets
// instead of re-downloading them from scratch.
import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import sharp from 'sharp';

const DATASET = 'Photoroom/midjourney-v6-recap';
const LICENSE = 'mit';
const SPLIT = 'train';
const TOTAL_ROWS = 1_235_432;
const COUNT = 260;
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
  const directory = `${output}/ai/midjourney-v6`;
  await mkdir(directory, { recursive: true });
  const existingFileNames = new Set(await readdir(directory));
  if (existingFileNames.size >= COUNT) {
    console.log(`midjourney-v6: already has ${existingFileNames.size}/${COUNT} staged, nothing to do (resumed run)`);
    return;
  }
  const remaining = COUNT - existingFileNames.size;
  console.log(`midjourney-v6: resuming, ${existingFileNames.size} already staged, need ${remaining} more`);

  await mkdir(`${output}/_checkpoints`, { recursive: true });
  const checkpointPath = `${output}/_checkpoints/midjourney-v6.json`;
  let processedOffsetCount = await readCheckpoint(checkpointPath);

  const exactHashes = new Set();
  const manifest = [];
  let newlyAdded = 0;

  const random = mulberry32(0x4d4a3652); // 'MJ6R' -- fixed seed for reproducibility
  const pageLength = 50;
  const maxStart = TOTAL_ROWS - pageLength;
  for (let i = 0; i < processedOffsetCount; i += 1) random(); // fast-forward past already-consumed offsets, no network involved

  while (newlyAdded < remaining) {
    const offset = Math.floor(random() * maxStart);
    processedOffsetCount += 1;
    const rows = await fetchRows(offset, pageLength);
    await writeFile(checkpointPath, JSON.stringify({ processedOffsetCount })).catch(() => {});
    const candidates = rows.map((item) => item.row.image).filter(Boolean);
    for (let cursor = 0; cursor < candidates.length && newlyAdded < remaining; cursor += concurrency) {
      const batch = candidates.slice(cursor, cursor + concurrency);
      const downloaded = await Promise.all(batch.map((candidate) => downloadCandidate(candidate).catch(() => null)));
      for (const item of downloaded) {
        if (!item || newlyAdded >= remaining || exactHashes.has(item.sha256)) continue;
        exactHashes.add(item.sha256);
        const fileName = `${item.sha256.slice(0, 16)}.${item.extension}`;
        const record = {
          label: 'ai',
          domain: 'midjourney-v6',
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
        if (existingFileNames.has(fileName)) continue; // already on disk from a prior run of this script -- identical content, nothing to do
        await writeFile(`${directory}/${fileName}`, item.bytes);
        existingFileNames.add(fileName);
        newlyAdded += 1;
      }
      await delay(150);
    }
    process.stderr.write(`\rmidjourney-v6: ${existingFileNames.size}/${COUNT} (offsets consumed=${processedOffsetCount})`);
  }
  process.stderr.write('\n');
  const finalCount = (await readdir(directory)).length;
  if (finalCount < COUNT) {
    throw new Error(`Only staged ${finalCount}/${COUNT} images for midjourney-v6`);
  }

  await writeFile(`${output}/midjourney-v6-pull-manifest.json`, `${JSON.stringify({ dataset: DATASET, license: LICENSE, split: SPLIT, generatedAt: new Date().toISOString(), rows: manifest }, null, 2)}\n`);
  console.log(`Prepared ${manifest.length} newly-staged Midjourney v6 images from ${DATASET} in ${output} (${finalCount} total staged)`);
}

async function fetchRows(offset, length) {
  const parameters = new URLSearchParams({ dataset: DATASET, config: 'default', split: SPLIT, offset: String(offset), length: String(length) });
  const response = await fetchWithRetry(`https://datasets-server.huggingface.co/rows?${parameters}`);
  const payload = await response.json();
  return payload.rows || [];
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
    return Number(parsed.processedOffsetCount) || 0;
  } catch {
    return 0;
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
