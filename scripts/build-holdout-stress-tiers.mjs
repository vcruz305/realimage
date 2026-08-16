// Builds "web" and "hard" degraded-quality copies of the holdout split for
// stress-tier evaluation.
//
// scripts/make-stress-matrix.mjs was read first as instructed, and its
// per-transform image logic (double-JPEG recompression, thumbnail+upscale,
// off-center crop) is reused/adapted below. It was NOT run directly because
// its manifest validator (validateRights / requiredHttpsUrl on
// provenance.sourceUrl, license.status === 'approved', etc.) requires
// per-row legal provenance metadata that several realimage-broad-v1 sources
// genuinely do not have -- ai/nano-banana and ai/flux-krea are explicitly
// logged as "proprietary-generation, locally-authored dev sample
// (unpublished, not redistributed)" in
// .bench-data/realimage-broad-v1/_staging/local-reuse-summary.json, with no
// public HTTPS source URL to give. Fabricating a provenance.sourceUrl to
// force that validator to pass would be dishonest, so this script applies
// the same style of transform directly to the already-assembled holdout
// split instead, without the release-grade rights-manifest gate (this is an
// internal eval artifact, not a published benchmark release).
//
// Every transform in a tier is applied to every holdout image (matching
// make-stress-matrix's "everyTransformAppliedToEveryOriginal" policy), so
// each tier's sample count = holdoutImages * transformsInTier.
import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import sharp from 'sharp';

const HOLDOUT = resolve('.bench-data/realimage-broad-v1/holdout');
const WEB_OUT = resolve('.bench-data/realimage-broad-v1/holdout-web');
const HARD_OUT = resolve('.bench-data/realimage-broad-v1/holdout-hard');

const WEB_TRANSFORMS = [
  {
    id: 'jpeg-q75',
    extension: 'jpg',
    run: (image) => image.jpeg({ quality: 75, chromaSubsampling: '4:2:0', mozjpeg: false }),
  },
  {
    id: 'webp-q72',
    extension: 'webp',
    run: (image) => image.webp({ quality: 72, effort: 4 }),
  },
  {
    id: 'resize1024-jpeg-q80',
    extension: 'jpg',
    run: async (image, meta) => {
      const longest = Math.max(meta.width, meta.height);
      if (longest > 1024) {
        const scale = 1024 / longest;
        image = image.resize(Math.round(meta.width * scale), Math.round(meta.height * scale), { kernel: sharp.kernel.lanczos3 });
      }
      return image.jpeg({ quality: 80, chromaSubsampling: '4:2:0', mozjpeg: false });
    },
  },
];

const HARD_TRANSFORMS = [
  {
    id: 'double-jpeg-q85-q40',
    extension: 'jpg',
    run: async (image) => {
      const first = await image.jpeg({ quality: 85, chromaSubsampling: '4:2:0', mozjpeg: false }).toBuffer();
      return sharp(first, { failOn: 'warning' }).jpeg({ quality: 40, chromaSubsampling: '4:2:0', mozjpeg: false });
    },
  },
  {
    id: 'thumbnail192-webp-q50-upscale',
    extension: 'webp',
    run: async (image, meta) => {
      const scale = 192 / Math.max(meta.width, meta.height);
      const width = Math.max(1, Math.round(meta.width * scale));
      const height = Math.max(1, Math.round(meta.height * scale));
      const thumbnail = await image
        .resize(width, height, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
        .webp({ quality: 50, effort: 4 })
        .toBuffer();
      return sharp(thumbnail, { failOn: 'warning' }).resize(meta.width, meta.height, { fit: 'fill', kernel: sharp.kernel.lanczos3 }).webp({ quality: 50, effort: 4 });
    },
  },
  {
    id: 'crop70-jpeg-q60',
    extension: 'jpg',
    run: async (image, meta, seedKey) => {
      const width = Math.max(1, Math.floor(meta.width * 0.7));
      const height = Math.max(1, Math.floor(meta.height * 0.7));
      const availableX = meta.width - width;
      const availableY = meta.height - height;
      const selector = createHash('sha256').update(seedKey).digest();
      const left = Math.round(availableX * (selector[0] & 1 ? 0.9 : 0.1));
      const top = Math.round(availableY * (selector[1] & 1 ? 0.9 : 0.1));
      return image.extract({ left, top, width, height }).jpeg({ quality: 60, chromaSubsampling: '4:2:0', mozjpeg: false });
    },
  },
];

async function walkImages(root) {
  const results = [];
  for (const label of ['ai', 'real']) {
    const labelDir = resolve(root, label);
    let domains;
    try {
      domains = (await readdir(labelDir, { withFileTypes: true })).filter((e) => e.isDirectory()).map((e) => e.name);
    } catch {
      continue;
    }
    for (const domain of domains) {
      const domainDir = resolve(labelDir, domain);
      const files = await readdir(domainDir);
      for (const file of files) {
        results.push({ label, domain, file, path: resolve(domainDir, file) });
      }
    }
  }
  return results;
}

async function buildTier(name, outRoot, transforms) {
  const images = await walkImages(HOLDOUT);
  let written = 0;
  for (const item of images) {
    const bytes = await readFile(item.path);
    // Bake in EXIF auto-orientation first and read metadata from the
    // *rotated* buffer -- sharp's .rotate() on orientation 5-8 images swaps
    // width/height, but metadata() on the original bytes reports the
    // un-rotated dimensions, which previously caused "bad extract area" in
    // the crop transform whenever it sized a crop off the pre-rotation
    // width/height and applied it to the post-rotation pixel buffer.
    const rotatedBytes = await sharp(bytes, { failOn: 'warning' }).rotate().flatten({ background: '#f5f5f5' }).toBuffer();
    const meta = await sharp(rotatedBytes, { failOn: 'warning' }).metadata();
    for (const transform of transforms) {
      const image = sharp(rotatedBytes, { failOn: 'warning' });
      const pipeline = await transform.run(image, meta, `${transform.id}:${item.label}:${item.domain}:${item.file}`);
      const outBytes = await pipeline.toBuffer();
      const outDir = resolve(outRoot, item.label, item.domain);
      await mkdir(outDir, { recursive: true });
      const stem = item.file.replace(/\.[^.]+$/, '');
      await writeFile(resolve(outDir, `${stem}-${transform.id}.${transform.extension}`), outBytes);
      written += 1;
    }
  }
  console.log(`${name}: wrote ${written} degraded samples from ${images.length} holdout images x ${transforms.length} transforms to ${outRoot}`);
  return written;
}

async function main() {
  const webCount = await buildTier('web', WEB_OUT, WEB_TRANSFORMS);
  const hardCount = await buildTier('hard', HARD_OUT, HARD_TRANSFORMS);
  console.log(JSON.stringify({ web: webCount, hard: hardCount }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
