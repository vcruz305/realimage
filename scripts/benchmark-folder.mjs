import { readdir, readFile, writeFile } from 'node:fs/promises';
import { extname, join, relative, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { AutoImageProcessor, AutoModelForImageClassification, env, RawImage } from '@huggingface/transformers';
import { calibrateDecisionScore, inspectEncodedImage, fuseEvidence } from '../src/analysis/forensics.js';
import { MODEL } from '../src/shared/constants.js';

const root = resolve(process.argv[2] || 'benchmark');
const threshold = Number(process.env.PROOFMARK_THRESHOLD || 0.65);
const requiredAccuracy = Number(process.env.PROOFMARK_REQUIRED_ACCURACY || 0.8);
const quiet = process.argv.includes('--quiet');
const benchmarkModelId = process.env.PROOFMARK_MODEL_ID || MODEL.id;
const benchmarkDtype = process.env.PROOFMARK_DTYPE || MODEL.dtype;
const benchmarkModelRoot = resolve(process.env.PROOFMARK_LOCAL_MODEL_PATH || 'public/models');
const resultsPath = process.env.PROOFMARK_RESULTS_PATH;
const samples = await collectSamples(root);
if (!samples.length) {
  console.error('No images found. Expected labeled folders such as benchmark/ai and benchmark/real.');
  process.exit(1);
}

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = benchmarkModelRoot + '/';
const [processor, detector] = await Promise.all([
  AutoImageProcessor.from_pretrained(benchmarkModelId),
  AutoModelForImageClassification.from_pretrained(benchmarkModelId, { dtype: benchmarkDtype, device: 'cpu' })
]);
const totals = { ai: 0, real: 0, aiCorrect: 0, realCorrect: 0 };
const rows = [];

for (const sample of samples) {
  const startedAt = performance.now();
  const image = await RawImage.read(sample.path);
  const output = await detector(await processor(image));
  const modelScore = sigmoid(Number(output.logits.data[0]));
  const secondaryModelScore = output.logits.data.length > 1 ? sigmoid(Number(output.logits.data[1])) : null;
  const bytes = await readFile(sample.path);
  const encoded = inspectEncodedImage(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), mimeFromPath(sample.path));
  const uncalibratedScore = benchmarkModelId === MODEL.id && MODEL.decisionPolicy === 'model-only'
    ? modelScore
    : fuseEvidence(modelScore, encoded);
  const score = benchmarkModelId === MODEL.id
    ? calibrateDecisionScore(uncalibratedScore, MODEL.calibration.rawThreshold, MODEL.calibration.displayThreshold)
    : uncalibratedScore;
  const predicted = score >= threshold ? 'ai' : 'real';
  totals[sample.label] += 1;
  if (predicted === sample.label) totals[`${sample.label}Correct`] += 1;
  rows.push({
    file: relative(root, sample.path),
    domain: sample.domain,
    expected: sample.label,
    predicted,
    modelScore,
    secondaryModelScore,
    pixelAdjustment: 0,
    metadataEvidence: encoded.evidence.length,
    score,
    ai: `${(score * 100).toFixed(1)}%`,
    ms: Math.round(performance.now() - startedAt)
  });
  if (!quiet) console.log(`${predicted === sample.label ? '✓' : '✗'} ${rows.at(-1).ai.padStart(6)} ${sample.label.padEnd(4)} ${relative(root, sample.path)}`);
}

const sensitivity = totals.ai ? totals.aiCorrect / totals.ai : 0;
const specificity = totals.real ? totals.realCorrect / totals.real : 0;
const balancedAccuracy = (sensitivity + specificity) / 2;
const domains = summarizeDomains(rows);
console.log('\nRealImage Node diagnostic (not the installed-Chrome path)');
console.log(`Model:              ${benchmarkModelId} (${benchmarkDtype})`);
console.log(`Local model root:   ${benchmarkModelRoot}`);
console.log(`Threshold:          ${(threshold * 100).toFixed(1)}%`);
console.log(`AI recall:          ${(sensitivity * 100).toFixed(1)}% (${totals.aiCorrect}/${totals.ai})`);
console.log(`Real recall:        ${(specificity * 100).toFixed(1)}% (${totals.realCorrect}/${totals.real})`);
console.log(`Balanced accuracy:  ${(balancedAccuracy * 100).toFixed(1)}%`);
console.log(`Mean sample time:   ${Math.round(rows.reduce((sum, row) => sum + row.ms, 0) / rows.length)} ms (decode + preprocess + native ORT + evidence)`);
console.log(`Required accuracy:  ${(requiredAccuracy * 100).toFixed(1)}%`);
for (const domain of domains) {
  console.log(`  ${domain.domain.padEnd(24)} ${(domain.recall * 100).toFixed(1)}% (${domain.correct}/${domain.total}, ${domain.label})`);
}
if (resultsPath) {
  await writeFile(resultsPath, `${JSON.stringify({
    model: benchmarkModelId,
    localModelRoot: benchmarkModelRoot,
    dtype: benchmarkDtype,
    runtime: 'node-native-onnxruntime',
    timingScope: 'decode+preprocess+inference+encoded-evidence',
    threshold,
    totals,
    metrics: { sensitivity, specificity, balancedAccuracy },
    domains,
    rows
  }, null, 2)}\n`);
  console.log(`Detailed results:   ${resultsPath}`);
}
process.exitCode = balancedAccuracy >= requiredAccuracy ? 0 : 2;

async function collectSamples(directory) {
  const entries = await readdir(directory, { withFileTypes: true }).catch(() => []);
  const output = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) output.push(...await collectSamples(path));
    else if (/\.(?:png|jpe?g|webp|avif)$/i.test(extname(entry.name))) {
      const folders = path.toLowerCase().split(/[\\/]/);
      const label = folders.some((part) => /^(?:ai|fake|synthetic|generated)$/.test(part)) ? 'ai'
        : folders.some((part) => /^(?:real|authentic|human)$/.test(part)) ? 'real' : null;
      if (label) {
        const labelIndex = folders.lastIndexOf(label);
        const generator = entry.name.match(/generator-([a-z0-9]+)/i)?.[1]?.toLowerCase();
        const domain = generator || (labelIndex >= 0 && labelIndex + 2 < folders.length ? folders[labelIndex + 1] : 'unknown');
        output.push({ path, label, domain });
      }
    }
  }
  return output;
}

function summarizeDomains(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.expected}:${row.domain}`;
    const current = groups.get(key) || { domain: row.domain, label: row.expected, total: 0, correct: 0 };
    current.total += 1;
    current.correct += Number(row.expected === row.predicted);
    groups.set(key, current);
  }
  return [...groups.values()]
    .map((group) => ({ ...group, recall: group.correct / group.total }))
    .sort((left, right) => left.label.localeCompare(right.label) || left.domain.localeCompare(right.domain));
}

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function mimeFromPath(path) {
  if (/\.png$/i.test(path)) return 'image/png';
  if (/\.webp$/i.test(path)) return 'image/webp';
  if (/\.avif$/i.test(path)) return 'image/avif';
  return 'image/jpeg';
}

