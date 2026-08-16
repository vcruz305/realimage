import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import sharp from 'sharp';

const newManifestPath = resolve(process.argv[2]);
const priorManifestPaths = process.argv.filter((value) => value.startsWith('--manifest=')).map((value) => resolve(value.slice(11)));
const priorRoots = process.argv.filter((value) => value.startsWith('--root=')).map((value) => resolve(value.slice(7)));
const next = JSON.parse(await readFile(newManifestPath, 'utf8')).rows;
const prior = [];

for (const path of priorManifestPaths) {
  const document = JSON.parse(await readFile(path, 'utf8'));
  prior.push(...document.rows.map((row) => ({ ...row, source: path })));
}
for (const root of priorRoots) {
  for (const path of (await walk(root)).filter((value) => /\.(?:png|jpe?g|webp|avif)$/i.test(value))) {
    const bytes = await readFile(path);
    const metadata = await sharp(bytes).metadata();
    prior.push({
      source: path,
      sha256: createHash('sha256').update(bytes).digest('hex'),
      dHash: await differenceHash(bytes),
      width: metadata.width,
      height: metadata.height
    });
  }
}

const priorSha = new Set(prior.map((row) => row.sha256));
const exact = next.filter((row) => priorSha.has(row.sha256));
const near = [];
for (const row of next) {
  const candidate = prior.find((old) =>
    Math.abs(old.width / old.height - row.width / row.height) < 0.01 &&
    hamming(old.dHash, row.dHash) <= 2
  );
  if (candidate) near.push({ new: row.relativePath, old: candidate.relativePath || candidate.source });
}

const result = { newRows: next.length, priorRows: prior.length, exactOverlap: exact.length, nearOverlap: near.length, nearExamples: near.slice(0, 10) };
console.log(JSON.stringify(result, null, 2));
if (exact.length || near.length) process.exitCode = 2;

async function differenceHash(bytes) {
  const pixels = await sharp(bytes).resize(9, 8, { fit: 'fill' }).greyscale().raw().toBuffer();
  let value = 0n;
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) value = (value << 1n) | BigInt(pixels[y * 9 + x] > pixels[y * 9 + x + 1]);
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

async function walk(directory) {
  const output = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    output.push(...(entry.isDirectory() ? await walk(path) : [path]));
  }
  return output;
}
