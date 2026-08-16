import { MESSAGE } from '../shared/constants.js';

export const DEFAULT_HEAVY_IMAGE_ADMISSION_RETRY_POLICY = Object.freeze({
  maxAttempts: 120,
  initialDelayMs: 80,
  maxDelayMs: 2_500,
  jitterRatio: 0.25
});
export const HEAVY_IMAGE_HEARTBEAT_INTERVAL_MS = 5_000;

/**
 * Reserve one of two background-owned slots before actively preparing a blob
 * fallback or sending a heavyweight data URL. Control messages contain only
 * IDs/tokens; `prepare` and `sendWithToken` cannot start while queued. The
 * bounded claim is admission/IPC concurrency, not page-owned DOM memory in a
 * renderer Chrome has suspended after its renewable lease was lost.
 */
export async function withHeavyImageAdmission({
  requestId,
  sendControlMessage,
  prepare = async () => undefined,
  sendWithToken,
  shouldContinue = () => true,
  onAdmissionReserved = () => {},
  wait = delay,
  random = Math.random,
  policy = DEFAULT_HEAVY_IMAGE_ADMISSION_RETRY_POLICY,
  heartbeatIntervalMs = HEAVY_IMAGE_HEARTBEAT_INTERVAL_MS,
  setHeartbeatTimer = setTimeout,
  clearHeartbeatTimer = clearTimeout
}) {
  if (typeof requestId !== 'string' || !requestId) throw new TypeError('requestId is required.');
  for (const [name, value] of Object.entries({ sendControlMessage, prepare, sendWithToken, shouldContinue, onAdmissionReserved })) {
    if (typeof value !== 'function') throw new TypeError(`${name} must be a function.`);
  }
  validatePolicy(policy);

  const admissionToken = await reserveHeavyImageWithRetry({
    requestId,
    sendControlMessage,
    shouldContinue,
    wait,
    random,
    policy
  });
  if (!admissionToken) return undefined;

  let prepared;
  let heartbeat;
  let releasePromise;
  const releaseAdmission = () => {
    if (!releasePromise) {
      try {
        releasePromise = Promise.resolve(sendControlMessage({
          type: MESSAGE.RELEASE_HEAVY_IMAGE,
          requestId,
          admissionToken
        })).catch(() => undefined);
      } catch {
        releasePromise = Promise.resolve(undefined);
      }
    }
    return releasePromise;
  };
  try {
    onAdmissionReserved(releaseAdmission);
    if (!shouldContinue()) return undefined;
    await activateAdmission({ requestId, admissionToken, sendControlMessage });
    heartbeat = startHeavyImageHeartbeat({
      requestId,
      admissionToken,
      sendControlMessage,
      intervalMs: heartbeatIntervalMs,
      setTimer: setHeartbeatTimer,
      clearTimer: clearHeartbeatTimer
    });
    if (!shouldContinue()) return undefined;
    prepared = await prepare();
    if (!shouldContinue()) return undefined;
    // Revalidate with a small hop after potentially slow preparation. If the
    // MV3 worker restarted, it rejects the old token before any large message.
    await activateAdmission({ requestId, admissionToken, sendControlMessage });
    heartbeat.confirm();
    if (!shouldContinue()) return undefined;
    return await sendWithToken({
      admissionToken,
      prepared,
      // This authoritative small hop belongs inside every queue-retry send
      // callback. A heartbeat result cached before suspension/restart must
      // never authorize one stale heavyweight IPC message after thaw.
      activateBeforeSend: async () => {
        await activateAdmission({ requestId, admissionToken, sendControlMessage });
        heartbeat.confirm();
      }
    });
  } finally {
    heartbeat?.stop();
    // Drop this content-context reference before returning the global slot.
    // New reservations therefore cannot overlap a body we have finished with.
    prepared = undefined;
    // Best effort and deliberately small. Background also revokes on terminal
    // analysis outcomes against replay; this release handles every completion,
    // stale/cancelled path, and exhausted queue retry.
    // A visibility cancellation can request this release while an analysis
    // message is already in flight. The background defers deletion until that
    // attempt settles; this idempotent await avoids a second release message.
    await releaseAdmission();
  }
}

export function startHeavyImageHeartbeat({
  requestId,
  admissionToken,
  sendControlMessage,
  intervalMs = HEAVY_IMAGE_HEARTBEAT_INTERVAL_MS,
  setTimer = setTimeout,
  clearTimer = clearTimeout
}) {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) throw new TypeError('heartbeat interval must be positive.');
  let stopped = false;
  let valid = true;
  let timer;

  const schedule = () => {
    if (stopped || timer !== undefined) return;
    timer = setTimer(() => {
      timer = undefined;
      void pulse();
    }, intervalMs);
  };
  const pulse = async () => {
    try {
      await activateAdmission({ requestId, admissionToken, sendControlMessage });
      valid = true;
    } catch {
      valid = false;
    }
    if (valid) schedule();
  };
  schedule();

  return {
    isValid: () => valid,
    confirm() {
      if (stopped) return;
      valid = true;
      schedule();
    },
    stop() {
      stopped = true;
      if (timer !== undefined) clearTimer(timer);
      timer = undefined;
    }
  };
}

async function activateAdmission({ requestId, admissionToken, sendControlMessage }) {
  const response = await sendControlMessage({
    type: MESSAGE.ACTIVATE_HEAVY_IMAGE,
    requestId,
    admissionToken
  });
  if (response?.ok !== true || response.activated !== true) throw responseError(response);
}

export async function reserveHeavyImageWithRetry({
  requestId,
  sendControlMessage,
  shouldContinue = () => true,
  wait = delay,
  random = Math.random,
  policy = DEFAULT_HEAVY_IMAGE_ADMISSION_RETRY_POLICY
}) {
  validatePolicy(policy);
  for (let attempt = 0; attempt < policy.maxAttempts; attempt += 1) {
    if (!shouldContinue()) return undefined;
    const response = await sendControlMessage({ type: MESSAGE.RESERVE_HEAVY_IMAGE, requestId });
    if (response?.ok === true && typeof response.admissionToken === 'string') return response.admissionToken;
    if (!isCapacityResponse(response) || attempt === policy.maxAttempts - 1) {
      throw responseError(response);
    }
    await wait(retryDelayMs(attempt, policy, random));
    if (!shouldContinue()) return undefined;
  }
  return undefined;
}

export function retryDelayMs(attempt, policy = DEFAULT_HEAVY_IMAGE_ADMISSION_RETRY_POLICY, random = Math.random) {
  const exponent = Math.max(0, Math.floor(Number(attempt) || 0));
  const base = Math.min(policy.maxDelayMs, policy.initialDelayMs * (2 ** exponent));
  const unit = Math.min(1, Math.max(0, Number(random()) || 0));
  return Math.min(policy.maxDelayMs, Math.max(0, Math.round(base * (1 + ((unit * 2 - 1) * policy.jitterRatio)))));
}

function isCapacityResponse(response) {
  return response?.ok === false
    && response.retryable === true
    && response.code === 'HEAVY_IMAGE_CAPACITY';
}

function responseError(response) {
  const error = new Error(response?.error || 'The heavy-image reservation failed.');
  if (typeof response?.code === 'string') error.code = response.code;
  return error;
}

function validatePolicy(policy) {
  if (
    !policy
    || !Number.isInteger(policy.maxAttempts)
    || policy.maxAttempts < 1
    || !Number.isFinite(policy.initialDelayMs)
    || policy.initialDelayMs < 0
    || !Number.isFinite(policy.maxDelayMs)
    || policy.maxDelayMs < policy.initialDelayMs
    || !Number.isFinite(policy.jitterRatio)
    || policy.jitterRatio < 0
    || policy.jitterRatio > 1
  ) {
    throw new TypeError('Invalid heavy-image admission retry policy.');
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
