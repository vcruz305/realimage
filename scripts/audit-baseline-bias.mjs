import { readFile } from 'node:fs/promises';
import { extname } from 'node:path';

const manifest = JSON.parse(await readFile('benchmark-results/upstream/webwild-v3-locked-manifest.json', 'utf8'));
const clean = JSON.parse(await readFile('benchmark-results/upstream/webwild-v3-sealed-original.json', 'utf8'));
const web = JSON.parse(await readFile('benchmark-results/upstream/webwild-v3-sealed-web.json', 'utf8'));

const originalsByPath = new Map(manifest.rows.map((row) => [stripPrefix(row.relativePath, 'sealed/original/'), row]));
const originalsByHash = new Map(manifest.rows.map((row) => [row.sha256, row]));
const variantsByPath = new Map(manifest.variants.map((row) => [stripPrefix(row.relativePath, 'sealed/web/'), row]));

const cleanRows = clean.rows.map((row) => ({ ...row, metadata: requireRow(originalsByPath, row.file) }));
const webRows = web.rows.map((row) => {
  const variant = requireRow(variantsByPath, row.file);
  return { ...row, variant, metadata: requireRow(originalsByHash, variant.baseSha256) };
});

console.log('Baseline benchmark-bias audit');
console.log('Clean format counts:', JSON.stringify(groupCounts(cleanRows, (row) => `${row.expected}:${row.metadata.format}`)));
console.log('Clean shape counts:', JSON.stringify(groupCounts(cleanRows, (row) => `${row.expected}:${isSquare(row.metadata) ? 'square' : 'non-square'}`)));
console.log('Trivial PNG=>AI balanced accuracy:', percent(balancedAccuracy(cleanRows, (row) => row.metadata.format === 'png' ? 'ai' : 'real')));
console.log('Trivial square=>AI balanced accuracy:', percent(balancedAccuracy(cleanRows, (row) => isSquare(row.metadata) ? 'ai' : 'real')));

for (const [label, rows] of Object.entries(groupRows(cleanRows, (row) => row.expected === 'real' ? (isSquare(row.metadata) ? 'real-square' : 'real-non-square') : 'ai'))) {
  if (label === 'ai') continue;
  console.log(`${label} recall:`, percent(recall(rows)));
}

for (const [transform, rows] of Object.entries(groupRows(webRows, (row) => row.variant.transformation))) {
  console.log(`${transform} balanced accuracy:`, percent(balancedAccuracy(rows, (row) => row.predicted)));
}

function requireRow(index, key) {
  const row = index.get(key);
  if (!row) throw new Error(`Manifest row not found for ${key}`);
  return row;
}

function stripPrefix(value, prefix) {
  return value.startsWith(prefix) ? value.slice(prefix.length) : value;
}

function isSquare(row) {
  return row.width === row.height;
}

function groupCounts(rows, keyFor) {
  return Object.fromEntries(Object.entries(groupRows(rows, keyFor)).map(([key, values]) => [key, values.length]));
}

function groupRows(rows, keyFor) {
  const groups = {};
  for (const row of rows) (groups[keyFor(row)] ??= []).push(row);
  return groups;
}

function balancedAccuracy(rows, predict) {
  const ai = rows.filter((row) => row.expected === 'ai');
  const real = rows.filter((row) => row.expected === 'real');
  const aiRecall = ai.filter((row) => predict(row) === 'ai').length / ai.length;
  const realRecall = real.filter((row) => predict(row) === 'real').length / real.length;
  return (aiRecall + realRecall) / 2;
}

function recall(rows) {
  return rows.filter((row) => row.predicted === row.expected).length / rows.length;
}

function percent(value) {
  return `${(value * 100).toFixed(2)}%`;
}
