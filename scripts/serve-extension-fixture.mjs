import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';
import sharp from 'sharp';

const fixturePath = resolve('tests/fixture');
const requestedPort = Number(process.argv[2] || 4173);
const markedFixture = insertPngText(
  await sharp(resolve(fixturePath, 'generator-marked.svg')).png().toBuffer(),
  'parameters',
  'a misty futuristic valley\nSteps: 30, Sampler: Euler, Seed: 42, Model hash: proofmark-fixture, ComfyUI workflow, stable diffusion'
);
const unmarkedFixture = await sharp(resolve(fixturePath, 'unmarked.svg')).png().toBuffer();

const server = createServer(async (request, response) => {
  const pathname = request.url === '/' ? '/index.html' : request.url.split('?')[0];
  const headers = {
    'access-control-allow-origin': '*',
    'cache-control': 'no-store'
  };

  if (pathname === '/generator-marked.png' || pathname === '/unmarked.png') {
    response.writeHead(200, { ...headers, 'content-type': 'image/png' });
    response.end(pathname === '/generator-marked.png' ? markedFixture : unmarkedFixture);
    return;
  }

  const path = resolve(fixturePath, `.${pathname}`);
  if (!path.startsWith(fixturePath)) {
    response.writeHead(403, headers).end('Forbidden');
    return;
  }

  try {
    const body = await readFile(path);
    response.writeHead(200, { ...headers, 'content-type': mimeType(path) });
    response.end(body);
  } catch {
    response.writeHead(404, headers).end('Not found');
  }
});

server.listen(requestedPort, '127.0.0.1', () => {
  console.log(`Extension fixture ready at http://127.0.0.1:${requestedPort}/`);
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => server.close(() => process.exit(0)));
}

function mimeType(path) {
  const extension = extname(path).toLowerCase();
  if (extension === '.html') return 'text/html; charset=utf-8';
  if (extension === '.svg') return 'image/svg+xml';
  if (extension === '.png') return 'image/png';
  return 'application/octet-stream';
}

function insertPngText(png, keyword, text) {
  const payload = Buffer.from(`${keyword}\0${text}`);
  const type = Buffer.from('tEXt');
  const chunk = Buffer.alloc(12 + payload.length);
  chunk.writeUInt32BE(payload.length, 0);
  type.copy(chunk, 4);
  payload.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(Buffer.concat([type, payload])), 8 + payload.length);
  return Buffer.concat([png.subarray(0, 33), chunk, png.subarray(33)]);
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
