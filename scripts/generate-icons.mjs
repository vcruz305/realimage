import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import sharp from 'sharp';

const projectRoot = resolve(import.meta.dirname, '..');
const masterPath = resolve(projectRoot, 'icons/realimage-master.png');
const sizes = [16, 32, 48, 128];
const NAVY = [8, 42, 80];
const CORAL = [240, 68, 56];

const masterBytes = await readFile(masterPath);
const master = sharp(masterBytes).ensureAlpha();
const metadata = await master.metadata();
if (!metadata.hasAlpha || metadata.width < 1024 || metadata.height < 1024) {
  throw new Error('RealImage icon master must be a transparent PNG at least 1024×1024.');
}

const { data, info } = await master.raw().toBuffer({ resolveWithObject: true });
let transparentPixels = 0;
let opaquePixels = 0;
for (let offset = 0; offset < data.length; offset += 4) {
  const alpha = data[offset + 3];
  if (alpha === 0) transparentPixels += 1;
  if (alpha >= 240) opaquePixels += 1;
  if (alpha === 0) continue;
  const isCoral = data[offset] > data[offset + 2] + 40 && data[offset] > data[offset + 1] + 20;
  const color = isCoral ? CORAL : NAVY;
  data[offset] = color[0];
  data[offset + 1] = color[1];
  data[offset + 2] = color[2];
}

const pixelCount = info.width * info.height;
if (transparentPixels < pixelCount * 0.3 || opaquePixels < pixelCount * 0.1) {
  throw new Error('RealImage icon master must contain meaningful transparent and opaque regions.');
}

await Promise.all(sizes.map(async (size) => {
  const destination = resolve(projectRoot, `icons/icon-${size}.png`);
  const bytes = await sharp(data, {
    raw: { width: info.width, height: info.height, channels: 4 }
  })
    .resize(size, size, {
      fit: 'contain',
      background: { r: 0, g: 0, b: 0, alpha: 0 },
      kernel: sharp.kernel.lanczos3
    })
    .png({ compressionLevel: 9, adaptiveFiltering: false, palette: false })
    .toBuffer();
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, bytes);
}));

console.log(`Generated RealImage icons: ${sizes.map((size) => `${size}×${size}`).join(', ')}`);
