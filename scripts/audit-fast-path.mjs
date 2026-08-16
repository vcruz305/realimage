import { readFile } from 'node:fs/promises';

const RAW_THRESHOLD = 0.020887045509173325;
const reports = [
  'benchmark-results/upstream/webwild-v3-sealed-original.json',
  'benchmark-results/upstream/webwild-v3-sealed-web.json'
];

for (const reportPath of reports) {
  const report = JSON.parse(await readFile(reportPath, 'utf8'));
  let changedDecisions = 0;

  for (const row of report.rows) {
    const metadataAdjustedScore = row.metadataEvidence > 0
      ? Math.max(row.modelScore, 0.985)
      : row.modelScore;
    const fastPrediction = metadataAdjustedScore >= RAW_THRESHOLD ? 'ai' : 'real';
    if (fastPrediction !== row.predicted) changedDecisions += 1;
  }

  if (changedDecisions) {
    throw new Error(`${reportPath}: removing the duplicate pixel pass changes ${changedDecisions} sealed decisions.`);
  }
  console.log(`${reportPath}: fast path preserves all ${report.rows.length} sealed decisions.`);
}
