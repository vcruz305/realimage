import { createHash } from 'node:crypto';
import { link, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { basename, extname, join, resolve } from 'node:path';
import sharp from 'sharp';

const sourcesRoot = resolve(process.argv[2] || '/private/tmp/proofmark-webwild/sources');
const tinyRoot = resolve(process.argv[3] || '/private/tmp/proofmark-webwild/tiny-validation-shard-1');
const output = resolve(process.argv[4] || '/private/tmp/proofmark-webwild/benchmark');
const exclusionPaths = (process.env.PROOFMARK_EXCLUDE_MANIFESTS || '').split(',').filter(Boolean).map((path) => resolve(path));
const sourceManifest = JSON.parse(await readFile(`${sourcesRoot}/manifest.json`, 'utf8'));
const sourceRows = sourceManifest.rows.map((row) => ({ ...row, absolutePath: `${sourcesRoot}/${row.relativePath}`, origin: 'real-vs-ai-corpus' }));
const tinyRows = await collectTinyRows(tinyRoot);
const exclusions = (await Promise.all(exclusionPaths.map(async (path) => JSON.parse(await readFile(path, 'utf8')).rows)))
  .flat()
  .map((row) => ({ ...row, absolutePath: '', origin: 'exclusion' }));
const deduplicated = deduplicate([...exclusions, ...sourceRows, ...tinyRows]).filter((row) => row.origin !== 'exclusion');

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

const sourcePartitions = stratifiedPartitions(deduplicated.filter((row) => row.origin === 'real-vs-ai-corpus'));
const rows = [
  ...sourcePartitions,
  ...deduplicated.filter((row) => row.origin === 'tiny-genimage').map((row) => ({ ...row, split: 'sealed' }))
];
const variants = [];

for (const row of rows) {
  const sourceName = basename(row.absolutePath);
  const relativePath = `${row.split}/original/${row.label}/${safeName(row.domain)}/${row.origin}-${sourceName}`;
  const destination = `${output}/${relativePath}`;
  await mkdir(resolve(destination, '..'), { recursive: true });
  await link(row.absolutePath, destination);
  row.relativePath = relativePath;
  delete row.absolutePath;

  if (row.split === 'calibration' || row.split === 'sealed') {
    const variant = await createWebVariant(output, row);
    variants.push(variant);
  }
}

const counts = summarize(rows);
await writeFile(`${output}/manifest.json`, `${JSON.stringify({
  version: 1,
  seed: 'proofmark-webwild-v1',
  generatedAt: new Date().toISOString(),
  policy: {
    realVsAiCorpus: '50% train / 25% calibration / 25% sealed, stratified by label and source domain',
    tinyGenImage: '100% sealed evaluation; CC-BY-NC-SA images are never used to fit weights',
    duplicateRules: 'exact SHA-256 and near duplicate dHash distance <= 2 with matching aspect ratio'
  },
  counts,
  rows,
  variants
}, null, 2)}\n`);
await writeFile(`${output}/ATTRIBUTION.md`, `# WebWild benchmark attribution\n\n- Real vs AI Corpus, Zitacron, 2026 (CC BY 4.0), retaining each row's upstream source and license.\n- Tiny-GenImage, TheKernel01 (CC BY-NC-SA 4.0), evaluation only.\n\nThe image files remain outside the repository and are not redistributed in the extension package.\n`);
console.log(JSON.stringify(counts, null, 2));
console.log(`Assembled ${rows.length} originals and ${variants.length} web variants in ${output}`);

async function collectTinyRows(root) {
  const files = await walk(root);
  const rows = [];
  for (const absolutePath of files.filter((path) => /\.(?:png|jpe?g|webp|avif)$/i.test(path))) {
    const label = absolutePath.split(/[\\/]/).includes('ai') ? 'ai' : 'real';
    const generator = basename(absolutePath).match(/generator-([^.]+)/)?.[1] || (label === 'real' ? 'imagenet-real' : 'unknown');
    const bytes = await readFile(absolutePath);
    const metadata = await sharp(bytes).metadata();
    rows.push({
      label,
      domain: generator,
      sourceDataset: 'TheKernel01/Tiny-GenImage',
      sourceLicense: 'cc-by-nc-sa-4.0',
      sourceRow: basename(absolutePath).match(/source-(\d+)/)?.[1],
      width: metadata.width,
      height: metadata.height,
      format: metadata.format,
      sha256: createHash('sha256').update(bytes).digest('hex'),
      dHash: await differenceHash(bytes),
      absolutePath,
      origin: 'tiny-genimage'
    });
  }
  return rows;
}

function deduplicate(input) {
  const exact = new Map();
  const accepted = [];
  for (const row of input) {
    const duplicate = exact.get(row.sha256);
    if (duplicate) {
      if (duplicate.label !== row.label) throw new Error(`Exact cross-label collision: ${row.sha256}`);
      continue;
    }
    const near = accepted.find((candidate) =>
      Math.abs(candidate.width / candidate.height - row.width / row.height) < 0.01 &&
      hamming(candidate.dHash, row.dHash) <= 2
    );
    if (near) {
      if (near.label !== row.label) {
        throw new Error(`Near-duplicate cross-label collision: ${near.sha256} / ${row.sha256}`);
      }
      continue;
    }
    exact.set(row.sha256, row);
    accepted.push(row);
  }
  console.log(`Deduplicated ${input.length} inputs to ${accepted.length} image families.`);
  return accepted;
}

function stratifiedPartitions(input) {
  const groups = Map.groupBy(input, (row) => `${row.label}:${row.domain}`);
  const output = [];
  for (const group of groups.values()) {
    group.sort((a, b) => partitionKey(a).localeCompare(partitionKey(b)));
    const trainEnd = Math.floor(group.length * 0.5);
    const calibrationEnd = trainEnd + Math.floor(group.length * 0.25);
    group.forEach((row, index) => output.push({
      ...row,
      split: index < trainEnd ? 'train' : index < calibrationEnd ? 'calibration' : 'sealed'
    }));
  }
  return output;
}

async function createWebVariant(root, row) {
  const source = `${root}/${row.relativePath}`;
  const selector = Number.parseInt(row.sha256.slice(0, 2), 16) % 4;
  const stem = basename(row.relativePath, extname(row.relativePath));
  const base = sharp(source).rotate();
  let pipeline;
  let extension;
  let transformation;
  if (selector === 0) {
    pipeline = base.jpeg({ quality: 68, chromaSubsampling: '4:2:0' });
    extension = 'jpg';
    transformation = 'jpeg-q68';
  } else if (selector === 1) {
    pipeline = base.resize({ width: Math.max(96, Math.round(row.width * 0.62)), withoutEnlargement: true }).webp({ quality: 72 });
    extension = 'webp';
    transformation = 'downscale-webp-q72';
  } else if (selector === 2) {
    const cropWidth = Math.max(96, Math.floor(row.width * 0.88));
    const cropHeight = Math.max(96, Math.floor(row.height * 0.88));
    pipeline = base.extract({
      left: Math.max(0, Math.floor((row.width - cropWidth) / 2)),
      top: Math.max(0, Math.floor((row.height - cropHeight) / 2)),
      width: Math.min(row.width, cropWidth),
      height: Math.min(row.height, cropHeight)
    }).jpeg({ quality: 80 });
    extension = 'jpg';
    transformation = 'center-crop-88-jpeg-q80';
  } else {
    pipeline = base.blur(0.45).jpeg({ quality: 74, chromaSubsampling: '4:2:0' });
    extension = 'jpg';
    transformation = 'light-blur-jpeg-q74';
  }
  const relativePath = `${row.split}/web/${row.label}/${safeName(row.domain)}/${stem}-${transformation}.${extension}`;
  const destination = `${root}/${relativePath}`;
  await mkdir(resolve(destination, '..'), { recursive: true });
  await pipeline.toFile(destination);
  return { baseSha256: row.sha256, split: row.split, label: row.label, domain: row.domain, transformation, relativePath };
}

function summarize(rows) {
  const summary = {};
  for (const row of rows) {
    const key = `${row.split}:${row.label}:${row.domain}`;
    summary[key] = (summary[key] || 0) + 1;
  }
  return summary;
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

function hamming(left, right) {
  let value = BigInt(`0x${left}`) ^ BigInt(`0x${right}`);
  let distance = 0;
  while (value) {
    distance += Number(value & 1n);
    value >>= 1n;
  }
  return distance;
}

function partitionKey(row) {
  return createHash('sha256').update(`proofmark-webwild-v1:${row.sha256}`).digest('hex');
}

function safeName(value) {
  return String(value).replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase();
}

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const output = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    output.push(...(entry.isDirectory() ? await walk(path) : [path]));
  }
  return output;
}
