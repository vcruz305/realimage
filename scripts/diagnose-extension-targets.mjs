import { spawn } from 'node:child_process';
import { mkdtemp, cp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';

const executablePath = resolve(process.env.REALIMAGE_DIAGNOSTIC_CHROME || '');
if (!process.env.REALIMAGE_DIAGNOSTIC_CHROME) {
  throw new Error('REALIMAGE_DIAGNOSTIC_CHROME must name a local Chromium/Chrome-for-Testing executable.');
}

const projectRoot = resolve(import.meta.dirname, '..');
const sourceExtension = resolve(projectRoot, 'dist');
const scratchRoot = await mkdtemp(join(tmpdir(), 'realimage-target-diagnostic-'));
const profilePath = join(scratchRoot, 'profile');
const extensionPath = join(scratchRoot, 'extension');
const stderr = [];
let chromeProcess;
let socket;

try {
  await cp(sourceExtension, extensionPath, { recursive: true, force: false, errorOnExist: true });
  const manifestPath = join(extensionPath, 'manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const browserMajor = Number((await fileVersion(executablePath)).split('.')[0]);
  const originalMinimum = Number(manifest.minimum_chrome_version || 0);
  let diagnosticManifestDelta = null;
  if (browserMajor < originalMinimum) {
    manifest.minimum_chrome_version = String(browserMajor);
    diagnosticManifestDelta = {
      field: 'minimum_chrome_version',
      from: String(originalMinimum),
      to: String(browserMajor),
      reason: 'Permit an already-installed older Chromium to execute the otherwise byte-identical diagnostic copy.'
    };
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  }

  chromeProcess = spawn(executablePath, [
    '--headless=new',
    '--no-first-run',
    '--disable-default-apps',
    '--disable-background-networking',
    '--disable-component-update',
    '--disable-sync',
    '--no-sandbox',
    '--no-default-browser-check',
    '--remote-debugging-port=0',
    `--user-data-dir=${profilePath}`,
    `--disable-extensions-except=${extensionPath}`,
    `--load-extension=${extensionPath}`,
    'about:blank'
  ], { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true });
  chromeProcess.stderr.setEncoding('utf8');
  chromeProcess.stderr.on('data', (chunk) => stderr.push(...String(chunk).split(/\r?\n/).filter(Boolean)));

  const { port, browserWebSocketUrl } = await waitForDevTools(profilePath, chromeProcess);
  const cdp = await connectCdp(browserWebSocketUrl);
  socket = cdp.socket;
  const targets = new Map();
  const sessions = new Map();
  const executionContexts = new Map();
  const events = [];

  cdp.on('Target.targetCreated', ({ targetInfo }) => targets.set(targetInfo.targetId, targetInfo));
  cdp.on('Target.targetInfoChanged', ({ targetInfo }) => targets.set(targetInfo.targetId, targetInfo));
  cdp.on('Target.attachedToTarget', async ({ sessionId, targetInfo, waitingForDebugger }) => {
    targets.set(targetInfo.targetId, targetInfo);
    sessions.set(sessionId, targetInfo);
    for (const method of ['Runtime.enable', 'Log.enable']) {
      await cdp.send(method, {}, sessionId).catch((error) => events.push(event('protocol-error', targetInfo, { method, message: error.message })));
    }
    if (waitingForDebugger) {
      await cdp.send('Runtime.runIfWaitingForDebugger', {}, sessionId).catch((error) => events.push(event('protocol-error', targetInfo, { method: 'Runtime.runIfWaitingForDebugger', message: error.message })));
    }
  });
  cdp.on('Runtime.exceptionThrown', (params, sessionId) => {
    events.push(event('exception', sessions.get(sessionId), simplifyException(params.exceptionDetails)));
  });
  cdp.on('Runtime.executionContextCreated', ({ context }, sessionId) => {
    const byId = executionContexts.get(sessionId) || new Map();
    byId.set(context.id, context);
    executionContexts.set(sessionId, byId);
  });
  cdp.on('Runtime.consoleAPICalled', (params, sessionId) => {
    events.push(event('console', sessions.get(sessionId), {
      level: params.type,
      values: params.args.map((item) => item.value ?? item.description ?? item.type)
    }));
  });
  cdp.on('Log.entryAdded', ({ entry }, sessionId) => {
    events.push(event('log', sessions.get(sessionId), {
      level: entry.level,
      source: entry.source,
      text: entry.text,
      url: entry.url,
      lineNumber: entry.lineNumber
    }));
  });

  await cdp.send('Target.setDiscoverTargets', { discover: true });
  await cdp.send('Target.setAutoAttach', {
    autoAttach: true,
    waitForDebuggerOnStart: process.env.REALIMAGE_PAUSE_TARGETS === '1',
    flatten: true,
    filter: [{ type: 'browser', exclude: true }, { type: 'tab', exclude: true }, {}]
  });

  // Attach to targets that were created before auto-attach was enabled.
  const initialTargets = await cdp.send('Target.getTargets');
  for (const targetInfo of initialTargets.targetInfos) {
    targets.set(targetInfo.targetId, targetInfo);
    if (targetInfo.type === 'browser' || [...sessions.values()].some((item) => item.targetId === targetInfo.targetId)) continue;
    await cdp.send('Target.attachToTarget', { targetId: targetInfo.targetId, flatten: true }).catch(() => {});
  }

  const fixtureTarget = await cdp.send('Target.createTarget', { url: 'http://127.0.0.1:4173/' });
  const fixtureAttached = await waitFor(() => [...sessions.entries()].find(([, info]) => info.targetId === fixtureTarget.targetId), 5_000);
  if (fixtureAttached) {
    await cdp.send('Page.enable', {}, fixtureAttached[0]).catch(() => {});
    await cdp.send('Page.bringToFront', {}, fixtureAttached[0]).catch(() => {});
    for (const target of targets.values()) {
      if (target.type === 'page' && target.targetId !== fixtureTarget.targetId) {
        await cdp.send('Target.closeTarget', { targetId: target.targetId }).catch(() => {});
      }
    }
    await delay(1_000);
    await cdp.send('Runtime.evaluate', {
      expression: `window.scrollTo(0, 1); window.dispatchEvent(new Event('resize')); document.images.forEach((image) => image.dispatchEvent(new Event('load')))`
    }, fixtureAttached[0]).catch(() => {});
  }
  let directMessageProbe;
  if (fixtureAttached) {
    const extensionId = [...targets.values()]
      .map((item) => /^chrome-extension:\/\/([a-p]{32})\/service-worker-loader\.js$/.exec(item.url)?.[1])
      .find(Boolean);
    const isolatedContext = await waitFor(async () => {
      for (const context of executionContexts.get(fixtureAttached[0])?.values() || []) {
        const probe = await cdp.send('Runtime.evaluate', {
          expression: 'globalThis.chrome?.runtime?.id || null',
          contextId: context.id,
          returnByValue: true
        }, fixtureAttached[0]).catch(() => undefined);
        if (probe?.result?.value === extensionId) return context;
      }
      return undefined;
    }, 5_000);
    if (isolatedContext) {
      const result = await cdp.send('Runtime.evaluate', {
        expression: `(async () => {
          try {
            const value = await chrome.runtime.sendMessage({
              type: 'proofmark/analyze-image',
              payload: {
                requestId: 'target-diagnostic-1',
                source: 'http://127.0.0.1:4173/unmarked.png',
                naturalWidth: 640,
                naturalHeight: 420,
                pageUrl: 'http://127.0.0.1:4173/'
              }
            });
            return { resolved: true, value };
          } catch (error) {
            return { resolved: false, error: error instanceof Error ? error.message : String(error) };
          }
        })()`,
        contextId: isolatedContext.id,
        awaitPromise: true,
        returnByValue: true
      }, fixtureAttached[0]);
      directMessageProbe = result.result?.value;
    } else {
      directMessageProbe = { resolved: false, error: 'RealImage isolated execution context was not found.' };
    }
  }
  await delay(20_000);

  let fixtureProbe;
  const fixtureSession = [...sessions.entries()].find(([, info]) => info.type === 'page' && info.url === 'http://127.0.0.1:4173/');
  if (fixtureSession) {
    const result = await cdp.send('Runtime.evaluate', {
      expression: `JSON.stringify({
        title: document.title,
        visibilityState: document.visibilityState,
        viewport: { width: innerWidth, height: innerHeight },
        overlays: document.querySelectorAll('#proofmark-overlay-layer').length,
        images: [...document.images].map((image) => ({
          complete: image.complete,
          naturalWidth: image.naturalWidth,
          naturalHeight: image.naturalHeight,
          source: image.currentSrc || image.src,
          rect: ((rect) => ({ top: rect.top, left: rect.left, width: rect.width, height: rect.height }))(image.getBoundingClientRect())
        })),
        badges: [...document.querySelectorAll('.proofmark-badge')].map((node) => ({
          text: node.textContent.replace(/\\s+/g, ' ').trim(),
          className: node.className,
          ariaLabel: node.getAttribute('aria-label')
        }))
      })`,
      returnByValue: true
    }, fixtureSession[0]);
    fixtureProbe = JSON.parse(result.result.value);
  }

  const contextProbes = [];
  for (const [sessionId, targetInfo] of sessions) {
    if (!['service_worker', 'other'].includes(targetInfo.type)) continue;
    const result = await cdp.send('Runtime.evaluate', {
      expression: `(async () => ({
        href: globalThis.location?.href || '',
        hasDocument: typeof document !== 'undefined',
        readyState: typeof document === 'undefined' ? null : document.readyState,
        crossOriginIsolated: Boolean(globalThis.crossOriginIsolated),
        contexts: typeof chrome?.runtime?.getContexts === 'function'
          ? (await chrome.runtime.getContexts({})).map((item) => ({ contextType: item.contextType, documentUrl: item.documentUrl }))
          : null
      }))()`,
      awaitPromise: true,
      returnByValue: true
    }, sessionId).catch((error) => ({ exceptionDetails: { text: error.message } }));
    contextProbes.push({
      targetType: targetInfo.type,
      targetUrl: redactExtensionId(targetInfo.url),
      value: result.result?.value,
      exception: result.exceptionDetails ? simplifyException(result.exceptionDetails) : undefined
    });
  }

  const report = {
    schema: 'realimage-extension-target-diagnostic-v1',
    browser: { executablePath, version: await fileVersion(executablePath), devToolsPort: port },
    extension: {
      sourcePath: sourceExtension,
      diagnosticCopyPath: extensionPath,
      manifestDelta: diagnosticManifestDelta
    },
    targets: [...targets.values()].map((item) => ({
      type: item.type,
      url: redactExtensionId(item.url),
      title: item.title,
      attached: item.attached
    })).sort((left, right) => `${left.type}\0${left.url}`.localeCompare(`${right.type}\0${right.url}`)),
    fixtureProbe,
    directMessageProbe,
    contextProbes,
    events: events.map((item) => ({ ...item, url: redactExtensionId(item.url) })),
    chromeStderr: stderr.filter((line) => /extension|service.?worker|offscreen|manifest|error/i.test(line)).slice(-200)
  };
  console.log(JSON.stringify(report, null, 2));
} finally {
  try { socket?.close(); } catch {}
  if (chromeProcess && chromeProcess.exitCode == null) {
    chromeProcess.kill();
    await Promise.race([new Promise((resolveExit) => chromeProcess.once('exit', resolveExit)), delay(5_000)]);
  }
  const resolvedScratch = resolve(scratchRoot);
  const resolvedTemp = resolve(tmpdir());
  if (dirname(resolvedScratch) !== resolvedTemp || basename(resolvedScratch).startsWith('realimage-target-diagnostic-') === false) {
    throw new Error(`Refusing to remove unexpected diagnostic path: ${resolvedScratch}`);
  }
  await rm(resolvedScratch, { recursive: true, force: true });
}

async function waitForDevTools(profile, child) {
  const path = join(profile, 'DevToolsActivePort');
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (child.exitCode != null) throw new Error(`Chrome exited before DevTools became ready (${child.exitCode}).`);
    try {
      const [portLine, pathLine] = (await readFile(path, 'utf8')).trim().split(/\r?\n/);
      const port = Number(portLine);
      if (Number.isInteger(port) && pathLine) {
        return { port, browserWebSocketUrl: `ws://127.0.0.1:${port}${pathLine}` };
      }
    } catch {}
    await delay(50);
  }
  throw new Error('Timed out waiting for Chrome DevToolsActivePort.');
}

async function connectCdp(url) {
  const socket = new WebSocket(url);
  await new Promise((resolveOpen, rejectOpen) => {
    socket.addEventListener('open', resolveOpen, { once: true });
    socket.addEventListener('error', rejectOpen, { once: true });
  });
  let nextId = 0;
  const pending = new Map();
  const listeners = new Map();
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(String(data));
    if (message.id) {
      const waiter = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) waiter?.reject(new Error(message.error.message));
      else waiter?.resolve(message.result || {});
      return;
    }
    for (const listener of listeners.get(message.method) || []) listener(message.params || {}, message.sessionId);
  });
  return {
    socket,
    on(method, listener) {
      const registered = listeners.get(method) || [];
      registered.push(listener);
      listeners.set(method, registered);
    },
    send(method, params = {}, sessionId = undefined) {
      const id = ++nextId;
      socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
      return new Promise((resolveResult, rejectResult) => pending.set(id, { resolve: resolveResult, reject: rejectResult }));
    }
  };
}

function simplifyException(details) {
  return {
    text: details.text,
    url: details.url,
    lineNumber: details.lineNumber,
    columnNumber: details.columnNumber,
    exception: details.exception?.description || details.exception?.value
  };
}

function event(kind, target, details) {
  return {
    kind,
    targetType: target?.type || 'unknown',
    url: target?.url || details?.url || '',
    ...details
  };
}

function redactExtensionId(value) {
  return String(value || '').replace(/chrome-extension:\/\/[a-p]{32}/g, 'chrome-extension://<extension-id>');
}

async function fileVersion(path) {
  if (process.platform !== 'win32') return 'unknown';
  const escaped = path.replaceAll("'", "''");
  const child = spawn('powershell.exe', ['-NoProfile', '-Command', `(Get-Item -LiteralPath '${escaped}').VersionInfo.FileVersion`], {
    stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true
  });
  let output = '';
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => { output += chunk; });
  const exitCode = await new Promise((resolveExit) => child.once('exit', resolveExit));
  if (exitCode !== 0) throw new Error(`Could not read browser version for ${path}.`);
  return output.trim();
}

function delay(ms) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, ms));
}

async function waitFor(probe, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await probe();
    if (result) return result;
    await delay(25);
  }
  return undefined;
}
