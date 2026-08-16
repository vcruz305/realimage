import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { extname, join, resolve } from 'node:path';
import { chromium } from 'playwright-core';
import sharp from 'sharp';

const extensionPath = resolve('dist');
const fixturePath = resolve('tests/fixture');
const profilePath = await mkdtemp(join(tmpdir(), 'proofmark-chrome-'));
const chromePath = process.env.PROOFMARK_CHROME;
const markedFixture = insertPngText(
  await sharp(join(fixturePath, 'generator-marked.svg')).png().toBuffer(),
  'parameters',
  'a misty futuristic valley\nSteps: 30, Sampler: Euler, Seed: 42, Model hash: proofmark-fixture, ComfyUI workflow, stable diffusion'
);
const unmarkedFixture = await sharp(join(fixturePath, 'unmarked.svg')).png().toBuffer();
const server = createServer(async (request, response) => {
  const pathname = request.url === '/' ? '/index.html' : request.url.split('?')[0];
  if (pathname === '/generator-marked.png' || pathname === '/unmarked.png') {
    response.writeHead(200, { 'content-type': 'image/png', 'cache-control': 'no-store' });
    response.end(pathname === '/generator-marked.png' ? markedFixture : unmarkedFixture);
    return;
  }
  const path = resolve(fixturePath, `.${pathname}`);
  if (!path.startsWith(fixturePath)) {
    response.writeHead(403).end('Forbidden');
    return;
  }
  try {
    const body = await readFile(path);
    response.writeHead(200, { 'content-type': mimeType(path), 'cache-control': 'no-store' });
    response.end(body);
  } catch {
    response.writeHead(404).end('Not found');
  }
});

await new Promise((resolveReady) => server.listen(0, '127.0.0.1', resolveReady));
const { port } = server.address();
let context;

try {
  context = await chromium.launchPersistentContext(profilePath, {
    ...(chromePath ? { executablePath: chromePath } : { channel: 'chromium' }),
    headless: true,
    // Playwright's default Chromium arguments include --disable-extensions.
    // Leaving that flag in place makes the load-extension arguments below a
    // no-op and turns the service-worker wait into a misleading registration
    // timeout, especially with current branded Chrome builds.
    ignoreDefaultArgs: ['--disable-extensions'],
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      '--no-first-run',
      '--disable-default-apps'
    ]
  });

  let worker = context.serviceWorkers()[0];
  if (!worker) worker = await context.waitForEvent('serviceworker', { timeout: 15_000 });
  const extensionId = new URL(worker.url()).host;
  console.log(`Loaded MV3 extension ${extensionId}`);

  const setup = await context.newPage();
  const setupLogs = [];
  setup.on('console', (message) => setupLogs.push(`[${message.type()}] ${message.text()}`));
  setup.on('pageerror', (error) => setupLogs.push(`[pageerror] ${error.message}`));
  await setup.goto(`chrome-extension://${extensionId}/src/options/options.html`);
  await setup.getByRole('button', { name: 'Run local readiness check' }).click();
  try {
    await setup.getByText(/Ready · WebAssembly/).waitFor({ timeout: 60_000 });
  } catch (error) {
    console.error(`Readiness UI: ${await setup.locator('#model-status').innerText().catch(() => 'missing')}`);
    console.error(setupLogs.join('\n'));
    throw error;
  }
  console.log('✓ bundled model initialized in Chrome');

  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`);
  try {
    await page.locator('.proofmark-badge--ai').first().waitFor({ timeout: 60_000 });
  } catch (error) {
    const diagnostics = await page.locator('.proofmark-badge').evaluateAll((nodes) => nodes.map((node) => ({ text: node.textContent, className: node.className, title: node.title })));
    console.error(`Overlay diagnostics: ${JSON.stringify(diagnostics)}`);
    console.error(`Page errors: ${pageErrors.join('; ') || 'none'}`);
    throw error;
  }
  const badges = await page.locator('.proofmark-badge').allTextContents();
  const aiBadge = badges.find((value) => /(?:9[89]|100)% AI/.test(value.replace(/\s+/g, ' ')));
  if (!aiBadge) throw new Error(`Expected explicit metadata to produce a high AI score. Badges: ${badges.join(', ')}`);
  if (pageErrors.length) throw new Error(`Fixture page errors: ${pageErrors.join('; ')}`);
  console.log(`✓ automatic page analysis rendered ${badges.length} score badge(s)`);
  console.log(`✓ generator metadata override surfaced as ${aiBadge.replace(/\s+/g, ' ').trim()}`);

  const markedImage = page.getByAltText('Abstract generated landscape fixture');
  if (await markedImage.getAttribute('data-proofmark-treatment') !== 'blur') {
    throw new Error('Expected the likely-AI fixture to be blurred by default.');
  }
  const filter = await markedImage.evaluate((image) => getComputedStyle(image).filter);
  if (!filter.includes('blur(18px)')) throw new Error(`Expected a local blur filter; received ${filter}.`);
  console.log('✓ likely-AI image blurred automatically at the fixed threshold');

  await page.locator('.proofmark-badge--ai').first().click();
  await page.getByText('Embedded generation parameters').waitFor({ timeout: 5_000 });
  await page.getByText('Pixels stayed in this browser.').waitFor({ timeout: 5_000 });
  console.log('✓ evidence panel exposes metadata and privacy disclosure');

  await page.getByRole('button', { name: 'Reveal this image' }).click();
  if (await markedImage.getAttribute('data-proofmark-treatment')) throw new Error('Reveal control did not restore the image.');
  await page.getByRole('button', { name: 'Reapply blur' }).click();
  if (await markedImage.getAttribute('data-proofmark-treatment') !== 'blur') throw new Error('Reapply control did not restore the blur.');
  console.log('✓ reveal and reapply controls work without removing evidence access');

  await setup.locator('#ai-image-action').selectOption('hide');
  await page.waitForFunction(() => document.querySelector('img[alt="Abstract generated landscape fixture"]')?.dataset.proofmarkTreatment === 'hide');
  if (await markedImage.evaluate((image) => getComputedStyle(image).visibility) !== 'hidden') {
    throw new Error('Hide mode did not conceal the likely-AI fixture.');
  }
  console.log('✓ hide mode removes likely-AI imagery from view');
  await setup.locator('#ai-image-action').selectOption('blur');
} finally {
  if (context) await context.close();
  await new Promise((resolveClosed) => server.close(resolveClosed));
  await rm(profilePath, { recursive: true, force: true });
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
