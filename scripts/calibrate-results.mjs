import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const inputs = process.argv.slice(2).filter((value) => !value.startsWith('--'));
const outputFlag = process.argv.find((value) => value.startsWith('--output='));
if (!inputs.length) throw new Error('Pass one or more benchmark result JSON files.');

const documents = await Promise.all(inputs.map(async (path) => JSON.parse(await readFile(resolve(path), 'utf8'))));
const rows = documents.flatMap((document) => document.rows.map((row) => ({ ...row, sourceResult: document.view || 'unknown' })));
const candidates = uniqueThresholds(rows.map((row) => row.score));
const evaluations = candidates.map((threshold) => evaluate(rows, threshold));
const eligible = evaluations.filter((item) => item.realRecall >= 0.9);
const best = (eligible.length ? eligible : evaluations).sort(compareEvaluations)[0];
const result = {
  generatedAt: new Date().toISOString(),
  inputs: inputs.map((path) => resolve(path)),
  rowCount: rows.length,
  requirement: { balancedAccuracy: 0.8, aiRecall: 0.8, realRecall: 0.9 },
  selected: best,
  passes: {
    balancedAccuracy: best.balancedAccuracy >= 0.8,
    aiRecall: best.aiRecall >= 0.8,
    realRecall: best.realRecall >= 0.9,
    lower95BalancedAccuracy: best.balancedAccuracyLower95 >= 0.8
  }
};

console.log(JSON.stringify(result, null, 2));
if (outputFlag) await writeFile(resolve(outputFlag.slice('--output='.length)), `${JSON.stringify(result, null, 2)}\n`);

function uniqueThresholds(scores) {
  const sorted = [...new Set(scores.map(Number).filter(Number.isFinite))].sort((a, b) => a - b);
  const output = [0.001];
  for (let index = 0; index < sorted.length - 1; index += 1) output.push((sorted[index] + sorted[index + 1]) / 2);
  output.push(0.999);
  return output;
}

function evaluate(rows, threshold) {
  const ai = rows.filter((row) => row.expected === 'ai');
  const real = rows.filter((row) => row.expected === 'real');
  const aiCorrect = ai.filter((row) => row.score >= threshold).length;
  const realCorrect = real.filter((row) => row.score < threshold).length;
  const aiRecall = aiCorrect / ai.length;
  const realRecall = realCorrect / real.length;
  const balancedAccuracy = (aiRecall + realRecall) / 2;
  const [aiLower] = wilson(aiCorrect, ai.length);
  const [realLower] = wilson(realCorrect, real.length);
  const domains = [...Map.groupBy(rows, (row) => `${row.expected}:${row.domain}`).values()].map((group) => {
    const correct = group.filter((row) => row.expected === 'ai' ? row.score >= threshold : row.score < threshold).length;
    return { label: group[0].expected, domain: group[0].domain, correct, total: group.length, recall: correct / group.length };
  }).sort((left, right) => left.label.localeCompare(right.label) || left.domain.localeCompare(right.domain));
  return {
    threshold,
    balancedAccuracy,
    balancedAccuracyLower95: (aiLower + realLower) / 2,
    aiRecall,
    realRecall,
    aiCorrect,
    aiTotal: ai.length,
    realCorrect,
    realTotal: real.length,
    minimumDomainRecall: Math.min(...domains.map((domain) => domain.recall)),
    domains
  };
}

function compareEvaluations(left, right) {
  return right.minimumDomainRecall - left.minimumDomainRecall
    || right.balancedAccuracy - left.balancedAccuracy
    || right.aiRecall - left.aiRecall
    || left.threshold - right.threshold;
}

function wilson(successes, total, z = 1.959963984540054) {
  if (!total) return [0, 0];
  const probability = successes / total;
  const denominator = 1 + z * z / total;
  const center = (probability + z * z / (2 * total)) / denominator;
  const margin = z * Math.sqrt((probability * (1 - probability) + z * z / (4 * total)) / total) / denominator;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}
