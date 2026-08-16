import { readFile, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';

const diagnosticRoot = resolve(process.argv[2] || '/private/tmp/proofmark-tiny-genimage/sample-200');
const resultsPath = resolve(process.argv[3] || 'benchmark-results/upstream/tiny-genimage-production-1000.json');
const diagnosticKeys = new Set(await collectSourceKeys(diagnosticRoot));
const results = JSON.parse(await readFile(resultsPath, 'utf8'));
const rows = results.rows || [];
const holdout = rows.filter((row) => !diagnosticKeys.has(sourceKey(row.expected, row.file)));
const overlap = rows.length - holdout.length;

if (!rows.length || overlap !== diagnosticKeys.size) {
  throw new Error(`Split audit failed: ${rows.length} results, ${diagnosticKeys.size} diagnostic keys, ${overlap} overlaps.`);
}

const metrics = Object.fromEntries(['ai', 'real'].map((label) => {
  const classRows = holdout.filter((row) => row.expected === label);
  const correct = classRows.filter((row) => row.predicted === label).length;
  return [label, { correct, total: classRows.length, recall: correct / classRows.length }];
}));
const balancedAccuracy = (metrics.ai.recall + metrics.real.recall) / 2;

console.log('Proofmark disjoint holdout audit');
console.log(`Diagnostic overlap removed: ${overlap}`);
console.log(`Locked holdout rows:       ${holdout.length}`);
console.log(`AI recall:                ${(metrics.ai.recall * 100).toFixed(2)}% (${metrics.ai.correct}/${metrics.ai.total})`);
console.log(`Real recall:              ${(metrics.real.recall * 100).toFixed(2)}% (${metrics.real.correct}/${metrics.real.total})`);
console.log(`Balanced accuracy:        ${(balancedAccuracy * 100).toFixed(2)}%`);
process.exitCode = balancedAccuracy >= 0.8 ? 0 : 2;

async function collectSourceKeys(directory) {
  const keys = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) keys.push(...await collectSourceKeys(path));
    else {
      const label = path.split(/[\\/]/).at(-2);
      if (label === 'ai' || label === 'real') keys.push(sourceKey(label, entry.name));
    }
  }
  return keys;
}

function sourceKey(label, file) {
  const match = file.match(/source-(\d+)/);
  if (!match) throw new Error(`Missing source row ID in ${file}`);
  return `${label}:${match[1]}`;
}
