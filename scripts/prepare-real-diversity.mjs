// Pulls real-photo and real-non-photo diversity sources (added 2026-08-16
// -- see manifest.json provenanceNotes) to control false positives on
// ordinary web content: everyday real photographs the model has never seen,
// plus non-photo real graphics (screenshots / memes / UI chrome) since a
// detector trained only on "photo vs AI-photo" can otherwise mis-fire on
// ordinary web pages that aren't photos at all.
//
// real/coco-val2017   <- rafaelpadilla/coco2017 (val split). COCO's own
//   annotations are CC-BY-4.0; the photographs themselves were sourced from
//   Flickr under a mix of per-photographer licenses (COCO's official terms
//   of use point users to the original Flickr license for each image,
//   rather than asserting a single blanket license) -- same "verified
//   dataset-level tag, unverified per-image copyright" situation already
//   documented for real/laion-aesthetic in this manifest, recorded here with
//   the same honesty rather than glossing over it.
// real/web-screenshots <- Zexanima/website_screenshots_image_dataset (MIT)
// real/memes-imgflip   <- SassyRong/meme-imgflip-small-test-dataset (CC0-1.0)
//
// Same fetch/dedup/retry pattern as scripts/prepare-genimage-replacement.mjs.
import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import sharp from 'sharp';

const output = resolve(process.argv[2] || '.bench-data/realimage-broad-v1/_staging');
const concurrency = Number(process.env.PROOFMARK_FETCH_CONCURRENCY || 3);

const specifications = [
  {
    key: 'coco-val2017',
    dataset: 'rafaelpadilla/coco2017',
    split: 'val',
    imageColumn: 'image',
    license: 'cc-by-4.0 (annotations); per-image photo license inherited from original Flickr upload, not independently re-verified per COCO terms of use',
    count: 1300,
    minDimension: 128,
  },
  {
    key: 'web-screenshots',
    dataset: 'Zexanima/website_screenshots_image_dataset',
    split: 'train',
    imageColumn: 'image',
    license: 'mit',
    count: 70,
    minDimension: 128,
  },
  {
    key: 'memes-imgflip',
    dataset: 'SassyRong/meme-imgflip-small-test-dataset',
    split: 'train',
    imageColumn: 'image',
    license: 'cc0-1.0',
    count: 30,
    minDimension: 96,
  },
];

const manifest = [];

for (const specification of specifications) {
  const directory = `${output}/real/${specification.key}`;
  await mkdir(directory, { recursive: true });
  const existingFileNames = new Set(await readdir(directory));
  if (existingFileNames.size >= specification.count) {
    console.log(`${specification.key}: already has ${existingFileNames.size}/${specification.count} staged, skipping (resumed run)`);
    continue;
  }
  const remaining = specification.count - existingFileNames.size;
  console.log(`${specification.key}: resuming, ${existingFileNames.size} already staged, need ${remaining} more`);
  // Checkpoint lives OUTSIDE the domain directory -- split-realimage-broad-v1.mjs
  // treats every file under _staging/<label>/<domain>/ as an image to copy.
  await mkdir(`${output}/_checkpoints`, { recursive: true });
  const checkpointPath = `${output}/_checkpoints/${specification.key}.json`;
  const exactHashes = new Set();
  let newlyAdded = 0;
  let offset = await readCheckpoint(checkpointPath);
  if (offset > 0) console.log(`${specification.key}: resuming row scan from offset=${offset} (checkpoint)`);
  const pageLength = 100;
  let consecutiveEmpty = 0;

  while (newlyAdded < remaining && consecutiveEmpty < 5) {
    const rows = await fetchRows(specification, offset, pageLength);
    offset += pageLength;
    await writeFile(checkpointPath, JSON.stringify({ offset })).catch(() => {});
    if (rows.length === 0) {
      consecutiveEmpty += 1;
      continue;
    }
    consecutiveEmpty = 0;
    const candidates = rows.map((item) => item.row[specification.imageColumn]).filter(Boolean);
    for (let cursor = 0; cursor < candidates.length && newlyAdded < remaining; cursor += concurrency) {
      const batch = candidates.slice(cursor, cursor + concurrency);
      const downloaded = await Promise.all(batch.map((candidate) => downloadCandidate(candidate, specification.minDimension).catch(() => null)));
      for (const item of downloaded) {
        if (!item || newlyAdded >= remaining || exactHashes.has(item.sha256)) continue;
        exactHashes.add(item.sha256);
        const fileName = `${item.sha256.slice(0, 16)}.${item.extension}`;
        const record = {
          label: 'real',
          domain: specification.key,
          fileName,
          sourceDataset: specification.dataset,
          sourceLicense: specification.license,
          sourceSplit: specification.split,
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
    process.stderr.write(`\r${specification.key}: ${existingFileNames.size}/${specification.count} (offset=${offset})`);
  }
  process.stderr.write('\n');
  const finalCount = (await readdir(directory)).length;
  if (finalCount < specification.count) {
    throw new Error(`Only staged ${finalCount}/${specification.count} images for ${specification.key} from ${specification.dataset}`);
  }
  console.log(`${specification.key}: staged ${finalCount} images from ${specification.dataset}`);
}

await writeFile(`${output}/real-diversity-pull-manifest.json`, `${JSON.stringify({ generatedAt: new Date().toISOString(), rows: manifest }, null, 2)}\n`);
console.log(`Prepared ${manifest.length} licensed real-diversity images in ${output}`);

async function fetchRows(specification, offset, length) {
  const parameters = new URLSearchParams({ dataset: specification.dataset, config: 'default', split: specification.split, offset: String(offset), length: String(length) });
  const response = await fetchWithRetry(`https://datasets-server.huggingface.co/rows?${parameters}`);
  const payload = await response.json();
  return payload.rows || [];
}

async function downloadCandidate(imageField, minDimension) {
  const response = await fetchWithRetry(imageField.src);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < 1_000 || bytes.length > 25_000_000) return null;
  const metadata = await sharp(bytes).metadata();
  if (!metadata.width || !metadata.height || Math.min(metadata.width, metadata.height) < minDimension) return null;
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
    return Number(parsed.offset) || 0;
  } catch {
    return 0;
  }
}
