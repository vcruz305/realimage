import { MESSAGE } from '../shared/constants.js';

const DEFAULT_ATTEMPTS = 12;
const DEFAULT_INITIAL_DELAY_MS = 25;
const DEFAULT_MAX_DELAY_MS = 500;

/**
 * Wait until the offscreen document has installed its runtime message listener.
 * Creating an offscreen context is not sufficient: Chrome may resolve
 * createDocument() before the document's module graph has evaluated.
 */
export async function waitForOffscreenReceiver({
  sendMessage,
  nonceFactory = () => crypto.randomUUID(),
  sleep = delay,
  attempts = DEFAULT_ATTEMPTS,
  initialDelayMs = DEFAULT_INITIAL_DELAY_MS,
  maxDelayMs = DEFAULT_MAX_DELAY_MS
}) {
  if (typeof sendMessage !== 'function') throw new TypeError('sendMessage must be a function.');
  if (!Number.isInteger(attempts) || attempts < 1) throw new TypeError('attempts must be a positive integer.');

  const nonce = nonceFactory();
  let lastError;
  let waitMs = initialDelayMs;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await sendMessage({ type: MESSAGE.OFFSCREEN_PING, nonce });
      if (response?.ok === true && response.receiver === 'offscreen' && response.nonce === nonce) return;
      lastError = new Error('The offscreen document returned an invalid readiness acknowledgement.');
    } catch (error) {
      lastError = error;
    }

    if (attempt + 1 < attempts) {
      await sleep(waitMs);
      waitMs = Math.min(maxDelayMs, Math.max(waitMs + 1, waitMs * 2));
    }
  }

  throw startupError(
    'OFFSCREEN_RECEIVER_MISSING',
    'The local inference document started but did not register its message receiver.',
    lastError
  );
}

export function startupError(code, message, cause) {
  const error = new Error(message, cause ? { cause } : undefined);
  error.name = 'OffscreenStartupError';
  error.code = code;
  return error;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
