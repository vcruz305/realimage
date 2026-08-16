import { createHash } from 'node:crypto';
import { access, copyFile, mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, isAbsolute, relative, resolve } from 'node:path';
import sharp from 'sharp';

const SOURCE_SCHEMA = 'proofmark-modern-holdout-source-v1';
const OUTPUT_SCHEMA = 'proofmark-modern-holdout-matrix-v1';
const DISJOINT_POLICY = 'source-group+source-row+sha256+dhash-v1';
const IMAGE_PATTERN = /\.(?:png|jpe?g|webp|avif)$/i;
const TRANSFORMS = Object.freeze([
  Object.freeze({
    id: 'double-jpeg-q85-q45',
    extension: 'jpg',
    description: 'Normalize orientation, encode JPEG q85 4:2:0, then JPEG q45 4:2:0.'
  }),
  Object.freeze({
    id: 'thumbnail-256-webp-q60-upscale',
    extension: 'webp',
    description: 'Resize the long edge to 256 px, encode WebP q60, then upscale to the original dimensions and encode WebP q60.'
  }),
  Object.freeze({
    id: 'offcenter-crop-75-jpeg-q70',
    extension: 'jpg',
    description: 'Take a deterministic label-blind off-center 75% crop and encode JPEG q70 4:2:0.'
  }),
  Object.freeze({
    id: 'screenshot-social-jpeg-q68',
    extension: 'jpg',
    description: 'Flatten to sRGB, shrink into a light screenshot canvas, add geometric UI chrome, pass through PNG, and encode JPEG q68.'
  })
]);

await main().catch((error) => {
  console.error(`Stress-matrix build failed: ${error.message}`);
  process.exitCode = 1;
});

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const inputRoot = resolve(options.positionals[0] || '.bench-data/modern-v1');
  const outputRoot = resolve(options.positionals[1] || '.bench-data/modern-v1-matrix');
  const manifestPath = resolve(options.manifest || `${inputRoot}/manifest.json`);
  if (inputRoot === outputRoot) throw new Error('Input and output roots must be different.');
  await assertMissing(outputRoot);

  const manifestBytes = await readFile(manifestPath);
  const manifest = parseJson(manifestBytes, manifestPath);
  const seed = options.seed === undefined ? manifest.seed : parseInteger(options.seed, '--seed');
  validateSourceManifest(manifest, seed);

  const validatedRows = await validateSourceRows(manifest.rows, inputRoot);
  await validateDisjointness(manifest.sourceDisjointness, validatedRows);

  const partialRoot = `${outputRoot}.partial-${process.pid}`;
  await assertMissing(partialRoot);
  await mkdir(dirname(outputRoot), { recursive: true });
  await mkdir(partialRoot, { recursive: true });

  try {
    const outputRows = [];
    const destinations = new Set();
    for (const row of validatedRows.toSorted((left, right) => left.familyId.localeCompare(right.familyId))) {
      const bucket = safeName(row.label === 'ai' ? row.domain : row.slice);
      const stem = `${safeName(row.familyId)}-${row.sha256.slice(0, 10)}`;
      const sourceExtension = normalizedExtension(row.absolutePath, row.format);
      const cleanPath = `clean/${row.label}/${bucket}/${stem}-original.${sourceExtension}`;
      assertUniqueDestination(destinations, cleanPath);
      await mkdir(resolve(partialRoot, dirname(cleanPath)), { recursive: true });
      await copyFile(row.absolutePath, resolve(partialRoot, cleanPath));
      outputRows.push(toOutputRow(row, {
        view: 'clean',
        transformId: 'original',
        relativePath: cleanPath,
        benchmarkPath: cleanPath.slice('clean/'.length),
        sha256: row.sha256,
        dHash: row.dHash,
        width: row.width,
        height: row.height,
        format: row.format
      }));

      const sourceBytes = await readFile(row.absolutePath);
      for (const transform of TRANSFORMS) {
        const transformed = await applyTransform(transform.id, sourceBytes, row, seed);
        const stressPath = `stress/${transform.id}/${row.label}/${bucket}/${stem}-${transform.id}.${transform.extension}`;
        assertUniqueDestination(destinations, stressPath);
        await mkdir(resolve(partialRoot, dirname(stressPath)), { recursive: true });
        await writeFile(resolve(partialRoot, stressPath), transformed);
        const metadata = await sharp(transformed, { failOn: 'warning' }).metadata();
        outputRows.push(toOutputRow(row, {
          view: 'stress',
          transformId: transform.id,
          relativePath: stressPath,
          benchmarkPath: stressPath.slice('stress/'.length),
          sha256: sha256(transformed),
          dHash: await differenceHash(transformed),
          width: metadata.width,
          height: metadata.height,
          format: metadata.format
        }));
      }
    }

    assertLabelBlindMatrix(outputRows);
    const sourceManifestSha256 = sha256(manifestBytes);
    const outputManifest = {
      schema: OUTPUT_SCHEMA,
      benchmarkId: manifest.benchmarkId,
      seed,
      frozenAt: manifest.frozenAt,
      sourceManifest: {
        path: relative(process.cwd(), manifestPath).replaceAll('\\', '/'),
        sha256: sourceManifestSha256,
        schema: SOURCE_SCHEMA
      },
      sourceDisjointness: manifest.sourceDisjointness,
      transformationPolicy: {
        labelBlind: true,
        everyTransformAppliedToEveryOriginal: true,
        originalClusterKey: 'familyId',
        transforms: TRANSFORMS
      },
      toolchain: {
        node: process.version,
        sharp: sharp.versions.sharp,
        libvips: sharp.versions.vips
      },
      counts: summarizeCounts(outputRows),
      rows: outputRows
    };
    await writeFile(resolve(partialRoot, 'manifest.json'), `${JSON.stringify(outputManifest, null, 2)}\n`);
    await rename(partialRoot, outputRoot);
    console.log(JSON.stringify(outputManifest.counts, null, 2));
    console.log(`Wrote deterministic holdout matrix to ${outputRoot}`);
  } catch (error) {
    await rm(partialRoot, { recursive: true, force: true });
    throw error;
  }
}

function parseArgs(argv) {
  const options = { positionals: [] };
  for (const value of argv) {
    if (value === '--help' || value === '-h') options.help = true;
    else if (value.startsWith('--manifest=')) options.manifest = value.slice('--manifest='.length);
    else if (value.startsWith('--seed=')) options.seed = value.slice('--seed='.length);
    else if (value.startsWith('--')) throw new Error(`Unknown option: ${value}`);
    else options.positionals.push(value);
  }
  if (options.positionals.length > 2) throw new Error('Expected at most INPUT_ROOT and OUTPUT_ROOT.');
  return options;
}

function printUsage() {
  console.log('Usage: node scripts/make-stress-matrix.mjs [INPUT_ROOT] [OUTPUT_ROOT] [--manifest=PATH] [--seed=323]');
  console.log('Reads local originals only; it never downloads data. OUTPUT_ROOT must not already exist.');
}

function validateSourceManifest(manifest, seed) {
  if (!manifest || manifest.schema !== SOURCE_SCHEMA) throw new Error(`Manifest schema must be ${SOURCE_SCHEMA}.`);
  requiredString(manifest.benchmarkId, 'benchmarkId');
  requiredString(manifest.frozenAt, 'frozenAt');
  if (!Number.isInteger(manifest.seed)) throw new Error('Manifest seed must be an integer.');
  if (manifest.seed !== seed) throw new Error(`Requested seed ${seed} does not match frozen manifest seed ${manifest.seed}.`);
  if (!Array.isArray(manifest.rows) || manifest.rows.length === 0) throw new Error('Manifest rows must be a non-empty array.');
  const audit = manifest.sourceDisjointness;
  if (!audit || audit.confirmed !== true || audit.policy !== DISJOINT_POLICY) {
    throw new Error(`sourceDisjointness must be confirmed under ${DISJOINT_POLICY}.`);
  }
  if (!Array.isArray(audit.checkedAgainst) || audit.checkedAgainst.length === 0) {
    throw new Error('sourceDisjointness.checkedAgainst must contain at least one frozen prior manifest.');
  }
}

async function validateSourceRows(rows, inputRoot) {
  const validated = [];
  const familyIds = new Set();
  const exactHashes = new Set();
  const sourceRows = new Set();

  for (const [index, source] of rows.entries()) {
    const prefix = `rows[${index}]`;
    const familyId = requiredString(source.familyId, `${prefix}.familyId`);
    if (!/^[a-z0-9][a-z0-9._:-]*$/i.test(familyId)) throw new Error(`${prefix}.familyId contains unsafe characters.`);
    if (familyIds.has(normalize(familyId))) throw new Error(`Duplicate familyId: ${familyId}`);
    familyIds.add(normalize(familyId));

    if (source.label !== 'ai' && source.label !== 'real') throw new Error(`${prefix}.label must be ai or real.`);
    const domain = requiredString(source.domain, `${prefix}.domain`);
    const slice = requiredString(source.slice, `${prefix}.slice`);
    const sourceDataset = requiredString(source.sourceDataset, `${prefix}.sourceDataset`);
    const sourceRow = requiredString(String(source.sourceRow ?? ''), `${prefix}.sourceRow`);
    const sourceGroup = requiredString(source.sourceGroup, `${prefix}.sourceGroup`);
    if (normalize(domain) === 'unknown' || normalize(slice) === 'unknown' || normalize(sourceGroup) === 'unknown') {
      throw new Error(`${prefix} may not use an unknown domain, slice, or sourceGroup.`);
    }

    const sourceIdentity = `${normalize(sourceDataset)}\u0000${normalize(sourceRow)}`;
    if (sourceRows.has(sourceIdentity)) throw new Error(`Duplicate immutable source row: ${sourceDataset}/${sourceRow}`);
    sourceRows.add(sourceIdentity);

    validateRights(source, prefix);
    const relativePath = requiredString(source.relativePath, `${prefix}.relativePath`).replaceAll('\\', '/');
    if (isAbsolute(relativePath) || relativePath.split('/').includes('..') || !IMAGE_PATTERN.test(relativePath)) {
      throw new Error(`${prefix}.relativePath must be a contained local image path.`);
    }
    const absolutePath = resolve(inputRoot, relativePath);
    const containment = relative(inputRoot, absolutePath);
    if (!containment || containment.startsWith('..') || isAbsolute(containment)) {
      throw new Error(`${prefix}.relativePath escapes the input root.`);
    }

    const bytes = await readFile(absolutePath).catch((error) => {
      throw new Error(`${prefix} cannot read ${relativePath}: ${error.message}`);
    });
    const digest = sha256(bytes);
    if (!/^[a-f0-9]{64}$/.test(source.sha256) || digest !== source.sha256) {
      throw new Error(`${prefix}.sha256 mismatch for ${relativePath}.`);
    }
    if (exactHashes.has(digest)) throw new Error(`Duplicate image bytes in source manifest: ${digest}`);
    exactHashes.add(digest);

    const metadata = await sharp(bytes, { failOn: 'warning' }).metadata();
    if (!metadata.width || !metadata.height || !metadata.format) throw new Error(`${prefix} has incomplete image metadata.`);
    if ((metadata.pages || 1) !== 1) throw new Error(`${prefix} must be a single-frame image.`);
    if (metadata.orientation && metadata.orientation !== 1) throw new Error(`${prefix} must be normalized to orientation 1 before freezing.`);
    if (metadata.width < 384 || metadata.height < 384) throw new Error(`${prefix} must be at least 384x384.`);
    if (source.width !== metadata.width || source.height !== metadata.height || normalize(source.format) !== normalize(metadata.format)) {
      throw new Error(`${prefix} width, height, or format does not match the encoded image.`);
    }
    const dHash = await differenceHash(bytes);
    if (!/^[a-f0-9]{16}$/.test(source.dHash) || dHash !== source.dHash) {
      throw new Error(`${prefix}.dHash mismatch for ${relativePath}.`);
    }

    validated.push({
      ...source,
      familyId,
      domain,
      slice,
      sourceDataset,
      sourceRow,
      sourceGroup,
      relativePath,
      absolutePath,
      sha256: digest,
      dHash,
      width: metadata.width,
      height: metadata.height,
      format: metadata.format
    });
  }

  for (let leftIndex = 0; leftIndex < validated.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < validated.length; rightIndex += 1) {
      const left = validated[leftIndex];
      const right = validated[rightIndex];
      if (sameAspect(left, right) && hamming(left.dHash, right.dHash) <= 2) {
        throw new Error(`Near-duplicate source families: ${left.familyId} and ${right.familyId}.`);
      }
    }
  }
  return validated;
}

function validateRights(row, prefix) {
  const license = row.license;
  if (!license || license.status !== 'approved') throw new Error(`${prefix}.license.status must be approved.`);
  requiredString(license.identifier, `${prefix}.license.identifier`);
  requiredHttpsUrl(license.url, `${prefix}.license.url`);
  requiredIsoDate(license.verifiedAt, `${prefix}.license.verifiedAt`);
  for (const field of ['commercialUse', 'evaluationUse', 'derivatives', 'localCopy']) {
    if (license[field] !== true) throw new Error(`${prefix}.license.${field} must be explicitly true.`);
  }
  requiredString(license.attribution, `${prefix}.license.attribution`);

  const provenance = row.provenance;
  if (!provenance || provenance.status !== 'verified') throw new Error(`${prefix}.provenance.status must be verified.`);
  requiredHttpsUrl(provenance.sourceUrl, `${prefix}.provenance.sourceUrl`);
  requiredString(provenance.immutableRevision, `${prefix}.provenance.immutableRevision`);
  requiredString(provenance.creator, `${prefix}.provenance.creator`);
  requiredString(provenance.labelMethod, `${prefix}.provenance.labelMethod`);
  requiredIsoDate(provenance.retrievedAt, `${prefix}.provenance.retrievedAt`);
}

async function validateDisjointness(audit, rows) {
  const seenPaths = new Set();
  const newHashes = new Set(rows.map((row) => row.sha256));
  const newFamilies = new Set(rows.map((row) => normalize(row.familyId)));
  const newSourceRows = new Set(rows.map((row) => `${normalize(row.sourceDataset)}\u0000${normalize(row.sourceRow)}`));
  const newSourceGroups = new Set(rows.map((row) => normalize(row.sourceGroup)));
  const declaredPaths = new Set(audit.checkedAgainst.map((entry) => normalize(resolve(entry.manifest || ''))));
  const lockedManifests = await readdir(resolve('benchmark-results'), { withFileTypes: true })
    .then((entries) => entries
      .filter((entry) => entry.isFile() && /-locked-manifest\.json$/i.test(entry.name))
      .map((entry) => resolve('benchmark-results', entry.name)), () => []);
  for (const path of lockedManifests) {
    if (!declaredPaths.has(normalize(path))) {
      throw new Error(`sourceDisjointness.checkedAgainst omits frozen repository manifest ${path}.`);
    }
  }

  for (const [index, checked] of audit.checkedAgainst.entries()) {
    const prefix = `sourceDisjointness.checkedAgainst[${index}]`;
    const manifestPath = resolve(requiredString(checked.manifest, `${prefix}.manifest`));
    if (seenPaths.has(normalize(manifestPath))) throw new Error(`Duplicate exclusion manifest: ${manifestPath}`);
    seenPaths.add(normalize(manifestPath));
    const bytes = await readFile(manifestPath).catch((error) => {
      throw new Error(`${prefix} cannot read ${manifestPath}: ${error.message}`);
    });
    const digest = sha256(bytes);
    if (!/^[a-f0-9]{64}$/.test(checked.sha256) || checked.sha256 !== digest) {
      throw new Error(`${prefix}.sha256 does not match ${manifestPath}.`);
    }
    const prior = parseJson(bytes, manifestPath);
    if (!Array.isArray(prior.rows)) throw new Error(`${manifestPath} does not contain rows.`);

    for (const old of prior.rows) {
      if (old.sha256 && newHashes.has(normalize(old.sha256))) {
        throw new Error(`Exact overlap with ${manifestPath}: ${old.sha256}`);
      }
      const oldFamily = old.familyId || old.sourceFamilyId;
      if (oldFamily && newFamilies.has(normalize(oldFamily))) {
        throw new Error(`Family overlap with ${manifestPath}: ${oldFamily}`);
      }
      if (old.sourceDataset !== undefined && old.sourceRow !== undefined) {
        const sourceIdentity = `${normalize(old.sourceDataset)}\u0000${normalize(String(old.sourceRow))}`;
        if (newSourceRows.has(sourceIdentity)) throw new Error(`Source-row overlap with ${manifestPath}: ${old.sourceDataset}/${old.sourceRow}`);
      }
      const oldGroup = old.sourceGroup || old.sourceDataset;
      if (oldGroup && newSourceGroups.has(normalize(oldGroup))) {
        throw new Error(`Source-group overlap with ${manifestPath}: ${oldGroup}`);
      }
      if (/^[a-f0-9]{16}$/i.test(old.dHash || '') && Number.isFinite(old.width) && Number.isFinite(old.height)) {
        const near = rows.find((row) => sameAspect(row, old) && hamming(row.dHash, old.dHash) <= 2);
        if (near) throw new Error(`Near-duplicate overlap with ${manifestPath}: ${near.familyId}`);
      }
    }
  }
}

async function applyTransform(id, bytes, row, seed) {
  if (id === 'double-jpeg-q85-q45') {
    const firstPass = await sharp(bytes, { failOn: 'warning' })
      .rotate()
      .flatten({ background: '#f5f5f5' })
      .jpeg({ quality: 85, chromaSubsampling: '4:2:0', mozjpeg: false })
      .toBuffer();
    return sharp(firstPass, { failOn: 'warning' })
      .jpeg({ quality: 45, chromaSubsampling: '4:2:0', mozjpeg: false })
      .toBuffer();
  }

  if (id === 'thumbnail-256-webp-q60-upscale') {
    const scale = 256 / Math.max(row.width, row.height);
    const width = Math.max(1, Math.round(row.width * scale));
    const height = Math.max(1, Math.round(row.height * scale));
    const thumbnail = await sharp(bytes, { failOn: 'warning' })
      .rotate()
      .flatten({ background: '#f5f5f5' })
      .resize(width, height, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
      .webp({ quality: 60, effort: 6 })
      .toBuffer();
    return sharp(thumbnail, { failOn: 'warning' })
      .resize(row.width, row.height, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
      .webp({ quality: 60, effort: 6 })
      .toBuffer();
  }

  if (id === 'offcenter-crop-75-jpeg-q70') {
    const width = Math.max(1, Math.floor(row.width * 0.75));
    const height = Math.max(1, Math.floor(row.height * 0.75));
    const availableX = row.width - width;
    const availableY = row.height - height;
    const selector = createHash('sha256').update(`${seed}:${row.sha256}:${id}`).digest();
    const left = Math.round(availableX * (selector[0] & 1 ? 0.9 : 0.1));
    const top = Math.round(availableY * (selector[1] & 1 ? 0.9 : 0.1));
    return sharp(bytes, { failOn: 'warning' })
      .rotate()
      .flatten({ background: '#f5f5f5' })
      .extract({ left, top, width, height })
      .jpeg({ quality: 70, chromaSubsampling: '4:2:0', mozjpeg: false })
      .toBuffer();
  }

  if (id === 'screenshot-social-jpeg-q68') {
    const innerWidth = Math.max(1, Math.floor(row.width * 0.82));
    const innerHeight = Math.max(1, Math.floor(row.height * 0.76));
    const left = Math.floor((row.width - innerWidth) / 2);
    const top = Math.floor(row.height * 0.08);
    const normalized = await sharp(bytes, { failOn: 'warning' })
      .rotate()
      .flatten({ background: '#f5f5f5' })
      .toColourspace('srgb')
      .resize(innerWidth, innerHeight, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
      .png({ compressionLevel: 9 })
      .toBuffer();
    const bandTop = Math.floor(row.height * 0.89);
    const overlay = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${row.width}" height="${row.height}"><rect x="0" y="${bandTop}" width="${row.width}" height="${row.height - bandTop}" fill="#202124" fill-opacity="0.84"/><circle cx="${Math.floor(row.width * 0.08)}" cy="${Math.floor(row.height * 0.945)}" r="${Math.max(4, Math.floor(row.height * 0.018))}" fill="#ffffff"/><rect x="${Math.floor(row.width * 0.14)}" y="${Math.floor(row.height * 0.925)}" width="${Math.floor(row.width * 0.28)}" height="${Math.max(3, Math.floor(row.height * 0.012))}" rx="2" fill="#ffffff" fill-opacity="0.9"/><rect x="${Math.floor(row.width * 0.14)}" y="${Math.floor(row.height * 0.952)}" width="${Math.floor(row.width * 0.18)}" height="${Math.max(2, Math.floor(row.height * 0.009))}" rx="2" fill="#ffffff" fill-opacity="0.55"/></svg>`);
    const screenshot = await sharp({
      create: { width: row.width, height: row.height, channels: 3, background: '#eceff1' }
    }).composite([
      { input: normalized, left, top },
      { input: overlay, left: 0, top: 0 }
    ]).png({ compressionLevel: 9 }).toBuffer();
    return sharp(screenshot, { failOn: 'warning' })
      .jpeg({ quality: 68, chromaSubsampling: '4:2:0', mozjpeg: false })
      .toBuffer();
  }

  throw new Error(`Unknown transform: ${id}`);
}

function toOutputRow(row, generated) {
  return {
    sampleId: `${row.familyId}:${generated.transformId}`,
    familyId: row.familyId,
    label: row.label,
    domain: row.domain,
    slice: row.slice,
    sourceDataset: row.sourceDataset,
    sourceRow: row.sourceRow,
    sourceGroup: row.sourceGroup,
    license: row.license,
    provenance: row.provenance,
    ...generated
  };
}

function assertLabelBlindMatrix(rows) {
  const expected = new Set(['original', ...TRANSFORMS.map((item) => item.id)]);
  const byFamily = groupBy(rows, (row) => row.familyId);
  for (const [familyId, familyRows] of byFamily) {
    const actual = new Set(familyRows.map((row) => row.transformId));
    if (!sameSet(actual, expected)) throw new Error(`Incomplete transform matrix for ${familyId}.`);
  }
  const byLabel = groupBy(rows, (row) => row.label);
  if (!byLabel.has('ai') || !byLabel.has('real')) throw new Error('Both ai and real labels are required.');
  const labelSets = [...byLabel.values()].map((items) => new Set(items.map((row) => row.transformId)));
  if (!sameSet(labelSets[0], labelSets[1])) throw new Error('Transform selection differs by label.');
}

function summarizeCounts(rows) {
  const families = new Map();
  for (const row of rows) families.set(row.familyId, row.label);
  return {
    originalFamilies: families.size,
    aiFamilies: [...families.values()].filter((label) => label === 'ai').length,
    realFamilies: [...families.values()].filter((label) => label === 'real').length,
    cleanSamples: rows.filter((row) => row.view === 'clean').length,
    stressSamples: rows.filter((row) => row.view === 'stress').length,
    totalSamples: rows.length
  };
}

async function differenceHash(bytes) {
  const pixels = await sharp(bytes, { failOn: 'warning' }).resize(9, 8, { fit: 'fill' }).greyscale().raw().toBuffer();
  let value = 0n;
  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      value = (value << 1n) | BigInt(pixels[y * 9 + x] > pixels[y * 9 + x + 1]);
    }
  }
  return value.toString(16).padStart(16, '0');
}

function hamming(left, right) {
  let value = BigInt(`0x${left}`) ^ BigInt(`0x${right}`);
  let distance = 0;
  while (value) {
    distance += Number(value & 1n);
    value >>= 1n;
  }
  return distance;
}

function sameAspect(left, right) {
  return Math.abs(left.width / left.height - right.width / right.height) < 0.01;
}

function normalizedExtension(path, format) {
  const extension = extname(path).toLowerCase().slice(1);
  if (extension === 'jpeg') return 'jpg';
  if (IMAGE_PATTERN.test(`.${extension}`)) return extension;
  return format === 'jpeg' ? 'jpg' : format;
}

function assertUniqueDestination(destinations, path) {
  const key = normalize(path);
  if (destinations.has(key)) throw new Error(`Output path collision: ${path}`);
  destinations.add(key);
}

async function assertMissing(path) {
  await access(path).then(
    () => { throw new Error(`Refusing to overwrite existing path: ${path}`); },
    (error) => {
      if (error.code !== 'ENOENT') throw error;
    }
  );
}

function requiredString(value, field) {
  if (typeof value !== 'string' || value.trim() === '') throw new Error(`${field} must be a non-empty string.`);
  return value.trim();
}

function requiredHttpsUrl(value, field) {
  requiredString(value, field);
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${field} must be a valid URL.`);
  }
  if (parsed.protocol !== 'https:') throw new Error(`${field} must use HTTPS.`);
}

function requiredIsoDate(value, field) {
  requiredString(value, field);
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value) || Number.isNaN(Date.parse(value))) {
    throw new Error(`${field} must be an ISO-8601 UTC timestamp.`);
  }
}

function parseInteger(value, field) {
  const result = Number(value);
  if (!Number.isSafeInteger(result)) throw new Error(`${field} must be an integer.`);
  return result;
}

function parseJson(bytes, path) {
  try {
    return JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Invalid JSON in ${path}: ${error.message}`);
  }
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function normalize(value) {
  return String(value).trim().toLowerCase();
}

function safeName(value) {
  const result = String(value).normalize('NFKD').replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase();
  if (!result) throw new Error(`Cannot derive a safe path component from ${value}.`);
  return result;
}

function groupBy(values, key) {
  const groups = new Map();
  for (const value of values) {
    const id = key(value);
    const group = groups.get(id) || [];
    group.push(value);
    groups.set(id, group);
  }
  return groups;
}

function sameSet(left, right) {
  return left.size === right.size && [...left].every((value) => right.has(value));
}
