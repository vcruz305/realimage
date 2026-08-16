import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';
import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet';

const GENERATOR_NAMES = ['Real', 'ADM', 'BigGAN', 'GLIDE', 'Midjourney', 'SD14', 'SD15', 'VQDM', 'Wukong'];

const source = resolve(process.argv[2] || '/private/tmp/proofmark-tiny-genimage/validation-00000.parquet');
const output = resolve(process.argv[3] || '/private/tmp/proofmark-tiny-genimage/sample-1000');
const perClass = Number(process.argv[4] || 500);
const excludeRoot = process.argv[5] ? resolve(process.argv[5]) : null;
if (!Number.isInteger(perClass) || perClass < 1) {
  throw new Error('per-class sample size must be a positive integer.');
}

const file = await asyncBufferFromFile(source);
const rows = await parquetReadObjects({ file, utf8: false });
const pools = { ai: [], real: [] };
const excluded = excludeRoot ? await collectExcludedSourceIds(excludeRoot) : new Set();

for (let index = 0; index < rows.length; index += 1) {
  const row = rows[index];
  const label = Number(row.label);
  // Tiny-GenImage defines 0 as real and 1 as generated.
  const className = label === 1 ? 'ai' : label === 0 ? 'real' : null;
  if (!className || !row.image?.bytes) continue;
  if (excluded.has(`${className}:${index}`)) continue;
  pools[className].push({
    index,
    bytes: row.image.bytes,
    extension: safeExtension(row.image.path),
    generator: generatorName(row.generator ?? row.model ?? row.source ?? row.class_name)
  });
}

const selected = {
  ai: seededSample(pools.ai, perClass, 323),
  real: seededSample(pools.real, perClass, 324)
};

await rm(output, { recursive: true, force: true });
await Promise.all([mkdir(`${output}/ai`, { recursive: true }), mkdir(`${output}/real`, { recursive: true })]);

for (const [className, items] of Object.entries(selected)) {
  await Promise.all(
    items.map((item, position) =>
      writeFile(
        `${output}/${className}/${String(position).padStart(4, '0')}-source-${item.index}-generator-${safeName(item.generator)}${item.extension}`,
        item.bytes
      )
    )
  );
}

const generatorCounts = selected.ai.reduce((counts, item) => {
  const key = item.generator || 'unspecified';
  counts[key] = (counts[key] || 0) + 1;
  return counts;
}, {});

console.log(`Parquet rows: ${rows.length}; labels: ${pools.ai.length} generated, ${pools.real.length} real.`);
if (excludeRoot) console.log(`Excluded ${excluded.size} prior source rows from ${excludeRoot}`);
console.log(`Prepared ${selected.ai.length} generated and ${selected.real.length} real images in ${output}`);
console.log(`Generated-image sources: ${JSON.stringify(generatorCounts)}`);

function safeExtension(path) {
  const extension = extname(printable(path)).toLowerCase();
  return ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.avif'].includes(extension) ? extension : '.jpg';
}

function printable(value) {
  if (typeof value === 'string') return value;
  if (value instanceof Uint8Array) return new TextDecoder().decode(value);
  return value == null ? '' : String(value);
}

function generatorName(value) {
  const printableValue = printable(value);
  const numeric = Number(printableValue);
  return Number.isInteger(numeric) && GENERATOR_NAMES[numeric] ? GENERATOR_NAMES[numeric] : printableValue;
}

function safeName(value) {
  return (value || 'unspecified').replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase();
}

function seededSample(items, count, seed) {
  if (items.length < count) throw new Error(`Only ${items.length} samples are available for a requested class.`);
  const shuffled = [...items];
  let state = seed >>> 0;
  const random = () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swap]] = [shuffled[swap], shuffled[index]];
  }
  return shuffled.slice(0, count);
}

async function collectExcludedSourceIds(directory) {
  const ids = new Set();
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) {
      for (const id of await collectExcludedSourceIds(path)) ids.add(id);
      continue;
    }
    const match = entry.name.match(/source-(\d+)/);
    const label = directory.split(/[\\/]/).at(-1);
    if (match && (label === 'ai' || label === 'real')) ids.add(`${label}:${match[1]}`);
  }
  return ids;
}
