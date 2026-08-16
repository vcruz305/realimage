export const DEFAULT_QUEUE_RETRY_POLICY = Object.freeze({
  maxAttempts: 24,
  initialDelayMs: 80,
  maxDelayMs: 2_500,
  jitterRatio: 0.25
});

/**
 * Retry only explicit offscreen-capacity responses. The offscreen queue remains
 * deliberately small; this caller-side backpressure lets image grids drain
 * without retaining additional encoded inputs in the inference document.
 *
 * `shouldContinue` is checked before every send and after every wait so an
 * image that was removed or recycled never re-enters the queue under a stale
 * generation.
 */
export async function sendAnalysisWithQueueRetry(
  send,
  {
    shouldContinue = () => true,
    wait = delay,
    random = Math.random,
    policy = DEFAULT_QUEUE_RETRY_POLICY
  } = {}
) {
  if (typeof send !== 'function') throw new TypeError('send must be a function.');
  if (typeof shouldContinue !== 'function') throw new TypeError('shouldContinue must be a function.');
  validatePolicy(policy);

  for (let attempt = 0; attempt < policy.maxAttempts; attempt += 1) {
    if (!shouldContinue()) return undefined;
    const result = await send();
    if (!isQueueCapacityResponse(result) || attempt === policy.maxAttempts - 1) return result;

    await wait(retryDelayMs(attempt, policy, random));
    if (!shouldContinue()) return undefined;
  }

  return undefined;
}

export function retryDelayMs(attempt, policy = DEFAULT_QUEUE_RETRY_POLICY, random = Math.random) {
  const exponent = Math.max(0, Math.floor(Number(attempt) || 0));
  const base = Math.min(policy.maxDelayMs, policy.initialDelayMs * (2 ** exponent));
  const unit = clamp(Number(random()), 0, 1);
  const jitter = (unit * 2 - 1) * policy.jitterRatio;
  return Math.min(policy.maxDelayMs, Math.max(0, Math.round(base * (1 + jitter))));
}

function isQueueCapacityResponse(result) {
  return result?.ok === false
    && result.retryable === true
    && result.code === 'OFFSCREEN_QUEUE_FULL';
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
    throw new TypeError('Invalid queue retry policy.');
  }
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function clamp(value, minimum, maximum) {
  if (!Number.isFinite(value)) return minimum;
  return Math.min(maximum, Math.max(minimum, value));
}
