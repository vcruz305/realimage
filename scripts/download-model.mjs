import { createHash } from 'node:crypto';
import { copyFile, mkdir, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { MODEL_FILES, MODEL_ID, MODEL_ROOT, MODEL_WEIGHT_FILE, MODEL_WEIGHT_SHA256, RUNTIME_DESTINATION_ROOT, RUNTIME_FILES, RUNTIME_SOURCE_ROOT } from './model-files.mjs';

for (const relativePath of MODEL_FILES) {
  const destinationUrl = new URL(`${MODEL_ID}/${relativePath}`, MODEL_ROOT);
  const destination = fileURLToPath(destinationUrl);
  try {
    const existing = await stat(destination);
    if (existing.size > 100) {
      console.log(`bundled  ${relativePath} (${formatBytes(existing.size)})`);
      continue;
    }
  } catch {
    throw new Error(
      `Missing frozen candidate asset ${relativePath}. Restore the exact pinned FP32 candidate bundle; `
      + 'this command never downloads or substitutes model weights.'
    );
  }
}

const weights = fileURLToPath(new URL(`${MODEL_ID}/${MODEL_WEIGHT_FILE}`, MODEL_ROOT));
const digest = createHash('sha256');
for await (const chunk of createReadStream(weights)) {
  digest.update(chunk);
}
const actualDigest = digest.digest('hex');
if (actualDigest !== MODEL_WEIGHT_SHA256) {
  throw new Error(`Bundled model checksum mismatch. Expected ${MODEL_WEIGHT_SHA256}; received ${actualDigest}.`);
}
console.log(`sha256   ${actualDigest}`);

await mkdir(RUNTIME_DESTINATION_ROOT, { recursive: true });
for (const runtimeFile of RUNTIME_FILES) {
  await copyFile(new URL(runtimeFile, RUNTIME_SOURCE_ROOT), new URL(runtimeFile, RUNTIME_DESTINATION_ROOT));
  const runtimeInfo = await stat(new URL(runtimeFile, RUNTIME_DESTINATION_ROOT));
  console.log(`runtime  ${runtimeFile} (${formatBytes(runtimeInfo.size)})`);
}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
