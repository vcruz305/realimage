import { createHash } from 'node:crypto';
import { readFile, readdir, stat } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
import sharp from 'sharp';
import { LOCAL_RUNTIME, MODEL, RELEASE } from '../src/shared/constants.js';

const projectRoot = resolve(import.meta.dirname, '..');
const distRoot = resolve(projectRoot, 'dist');
const requiredDocuments = ['LICENSE', 'NOTICE.md', 'PRIVACY.md'];
const expectedRuntime = [
  `wasm/${LOCAL_RUNTIME.wasmLoader}`,
  `wasm/${LOCAL_RUNTIME.wasmModule}`
].sort();
const expectedModelFiles = [
  `models/${MODEL.id}/config.json`,
  `models/${MODEL.id}/preprocessor_config.json`,
  `models/${MODEL.id}/${MODEL.weightFile}`
].sort();
const expectedIcons = [16, 32, 48, 128].map((size) => `icons/icon-${size}.png`).sort();

const files = await listFiles(distRoot);
const normalized = files.map((path) => relative(distRoot, path).replaceAll('\\', '/')).sort();

for (const document of requiredDocuments) {
  if (!normalized.includes(document)) throw new Error(`Release build is missing ${document}.`);
}

const sourceMaps = normalized.filter((path) => path.endsWith('.map'));
if (sourceMaps.length) throw new Error(`Release build contains source maps: ${sourceMaps.join(', ')}`);

const runtimeFiles = normalized.filter((path) => /^wasm\//.test(path));
if (JSON.stringify(runtimeFiles) !== JSON.stringify(expectedRuntime)) {
  throw new Error(`Expected only ${expectedRuntime.join(', ')}; found ${runtimeFiles.join(', ') || 'none'}.`);
}

const packagedModelFiles = normalized.filter((path) => /^models\//.test(path));
if (JSON.stringify(packagedModelFiles) !== JSON.stringify(expectedModelFiles)) {
  throw new Error(`Expected only the frozen candidate bundle; found ${packagedModelFiles.join(', ') || 'none'}.`);
}

const manifest = JSON.parse(await readFile(resolve(distRoot, 'manifest.json'), 'utf8'));
if (manifest.name !== RELEASE.name || manifest.short_name !== RELEASE.shortName || manifest.version !== RELEASE.version) {
  throw new Error(`Unexpected release manifest name: ${manifest.name}`);
}
if (Object.hasOwn(manifest, 'message_serialization')) {
  throw new Error('Release manifest must use Stable Chrome JSON messaging without the channel-gated message_serialization key.');
}
if (JSON.stringify(manifest.icons) !== JSON.stringify({
  16: 'icons/icon-16.png',
  32: 'icons/icon-32.png',
  48: 'icons/icon-48.png',
  128: 'icons/icon-128.png'
})) {
  throw new Error('Release manifest does not declare the exact RealImage icon family.');
}
const packagedIcons = normalized.filter((path) => /^icons\//.test(path));
if (JSON.stringify(packagedIcons) !== JSON.stringify(expectedIcons)) {
  throw new Error(`Expected only ${expectedIcons.join(', ')}; found ${packagedIcons.join(', ') || 'none'}.`);
}
for (const size of [16, 32, 48, 128]) {
  const icon = sharp(resolve(distRoot, `icons/icon-${size}.png`));
  const [metadata, stats] = await Promise.all([icon.metadata(), icon.stats()]);
  if (metadata.width !== size || metadata.height !== size || !metadata.hasAlpha || stats.isOpaque) {
    throw new Error(`RealImage icon-${size}.png must be ${size}×${size} with genuine transparency.`);
  }
}

const offscreenHtml = await readFile(resolve(distRoot, 'src/offscreen/offscreen.html'), 'utf8');
const offscreenEntryMatch = /src="\/(assets\/offscreen-[^"]+\.js)"/.exec(offscreenHtml);
if (!offscreenEntryMatch) throw new Error('Release build is missing the hashed offscreen listener entry.');
const offscreenEntry = offscreenEntryMatch[1];
const transformerChunks = normalized.filter((path) => /^assets\/transformers\.web-[^/]+\.js$/.test(path));
if (transformerChunks.length !== 1) {
  throw new Error(`Expected exactly one local dynamic Transformers.js chunk; found ${transformerChunks.join(', ') || 'none'}.`);
}
const offscreenEntryBytes = (await stat(resolve(distRoot, offscreenEntry))).size;
if (offscreenEntryBytes > 100_000) {
  throw new Error(`Offscreen listener entry is ${offscreenEntryBytes} bytes; heavy runtime code is blocking receiver registration.`);
}
const offscreenEntrySource = await readFile(resolve(distRoot, offscreenEntry), 'utf8');
const transformerBasename = transformerChunks[0].slice('assets/'.length);
if (!offscreenEntrySource.includes(`import("./${transformerBasename}")`)) {
  throw new Error('Transformers.js must be loaded from the offscreen listener through a bundled local dynamic import.');
}
if (offscreenHtml.includes(transformerBasename)) {
  throw new Error('The offscreen HTML must not preload Transformers.js before its message receiver is registered.');
}

for (const [relativePath, expectedDigest] of [
  [`models/${MODEL.id}/config.json`, MODEL.configSha256],
  [`models/${MODEL.id}/preprocessor_config.json`, MODEL.preprocessorConfigSha256],
  [`models/${MODEL.id}/${MODEL.weightFile}`, MODEL.weightSha256],
  [`wasm/${LOCAL_RUNTIME.wasmLoader}`, LOCAL_RUNTIME.wasmLoaderSha256],
  [`wasm/${LOCAL_RUNTIME.wasmModule}`, LOCAL_RUNTIME.wasmSha256]
]) {
  const digest = createHash('sha256').update(await readFile(resolve(distRoot, relativePath))).digest('hex');
  if (digest !== expectedDigest) throw new Error(`Packaged checksum mismatch for ${relativePath}: ${digest}.`);
}

const totalBytes = (await Promise.all(files.map(async (path) => (await stat(path)).size)))
  .reduce((sum, size) => sum + size, 0);
const digest = createHash('sha256');
for (const path of normalized) {
  const bytes = await readFile(resolve(distRoot, path));
  digest.update(path).update('\0').update(bytes).update('\0');
}

console.log(`Release build verified: ${normalized.length} files, ${totalBytes} bytes, tree ${digest.digest('hex')}`);

async function listFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const output = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) output.push(...await listFiles(path));
    else if (entry.isFile()) output.push(path);
  }
  return output;
}
