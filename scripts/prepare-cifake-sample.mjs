import { mkdir, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { asyncBufferFromFile, parquetReadObjects } from 'hyparquet';

const source = resolve(process.argv[2] || '/private/tmp/proofmark-cifake/test.parquet');
const output = resolve(process.argv[3] || '/private/tmp/proofmark-cifake/sample-1000');
const perClass = Number(process.argv[4] || 500);
if (!Number.isInteger(perClass) || perClass < 1 || perClass > 10_000) {
  throw new Error('per-class sample size must be an integer from 1 to 10,000.');
}

const file = await asyncBufferFromFile(source);
const rows = await parquetReadObjects({ file, utf8: false });
const pools = { 0: [], 1: [] };
for (let index = 0; index < rows.length; index += 1) {
  const label = Number(rows[index].label);
  if (label in pools) pools[label].push({ index, bytes: rows[index].image.bytes });
}

const selected = {
  0: seededSample(pools[0], perClass, 323),
  1: seededSample(pools[1], perClass, 324)
};
await rm(output, { recursive: true, force: true });
await Promise.all([mkdir(`${output}/ai`, { recursive: true }), mkdir(`${output}/real`, { recursive: true })]);

for (const [label, items] of Object.entries(selected)) {
  const folder = label === '0' ? 'ai' : 'real';
  await Promise.all(items.map((item, position) => writeFile(`${output}/${folder}/${String(position).padStart(4, '0')}-source-${item.index}.jpg`, item.bytes)));
}

console.log(`Prepared ${selected[0].length} AI and ${selected[1].length} real CIFAKE test images in ${output}`);

function seededSample(items, count, seed) {
  if (items.length < count) throw new Error(`Only ${items.length} samples available for label.`);
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
