import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import {
  MODEL_CONFIG_SHA256,
  MODEL_FILES,
  MODEL_ID,
  MODEL_PREPROCESSOR_SHA256,
  MODEL_ROOT,
  MODEL_WEIGHT_FILE,
  MODEL_WEIGHT_SHA256,
  RUNTIME_DESTINATION_ROOT,
  RUNTIME_FILES,
  RUNTIME_SHA256
} from './model-files.mjs';

const missing = [];
for (const relativePath of MODEL_FILES) {
  const url = new URL(`${MODEL_ID}/${relativePath}`, MODEL_ROOT);
  try {
    const info = await stat(fileURLToPath(url));
    if (info.size < 100) missing.push(relativePath);
  } catch {
    missing.push(relativePath);
  }
}

if (missing.length) {
  console.error(`Missing local model assets:\n${missing.map((file) => `  - ${file}`).join('\n')}`);
  console.error('\nRun `npm run model:download` once, then build again.');
  process.exit(1);
}

const modelWeight = fileURLToPath(new URL(`${MODEL_ID}/${MODEL_WEIGHT_FILE}`, MODEL_ROOT));
const actualDigest = await fileSha256(modelWeight);
if (actualDigest !== MODEL_WEIGHT_SHA256) {
  console.error(`Local model checksum mismatch. Expected ${MODEL_WEIGHT_SHA256}; received ${actualDigest}.`);
  process.exit(1);
}

for (const [relativePath, expectedDigest] of [
  ['config.json', MODEL_CONFIG_SHA256],
  ['preprocessor_config.json', MODEL_PREPROCESSOR_SHA256]
]) {
  const path = fileURLToPath(new URL(`${MODEL_ID}/${relativePath}`, MODEL_ROOT));
  const digest = await fileSha256(path);
  if (digest !== expectedDigest) {
    console.error(`Local model ${relativePath} checksum mismatch. Expected ${expectedDigest}; received ${digest}.`);
    process.exit(1);
  }
}

for (const runtimeFile of RUNTIME_FILES) {
  try {
    const runtimePath = fileURLToPath(new URL(runtimeFile, RUNTIME_DESTINATION_ROOT));
    const runtimeInfo = await stat(runtimePath);
    if (runtimeInfo.size < 1_000) throw new Error('invalid runtime');
    const digest = await fileSha256(runtimePath);
    if (digest !== RUNTIME_SHA256[runtimeFile]) throw new Error(`checksum mismatch (${digest})`);
  } catch {
    console.error(`Missing or mismatched local runtime ${runtimeFile}. Run \`npm run model:download\` once.`);
    process.exit(1);
  }
}

console.log(`Local model verified: ${MODEL_ID} (${actualDigest})`);

async function fileSha256(path) {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest('hex');
}
