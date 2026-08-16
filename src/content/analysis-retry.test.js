import { describe, expect, it, vi } from 'vitest';
import { retryDelayMs, sendAnalysisWithQueueRetry } from './analysis-retry.js';

const queueFull = Object.freeze({
  ok: false,
  unavailable: true,
  retryable: true,
  code: 'OFFSCREEN_QUEUE_FULL'
});

describe('sendAnalysisWithQueueRetry', () => {
  it('drains a burst larger than the nine-slot offscreen capacity without dropping images', async () => {
    let active = 0;
    let peak = 0;
    const attempts = Array(20).fill(0);
    const send = (index) => async () => {
      attempts[index] += 1;
      if (active >= 9) return queueFull;
      active += 1;
      peak = Math.max(peak, active);
      await new Promise((resolve) => setTimeout(resolve, 3));
      active -= 1;
      return { ok: true, score: index / 20 };
    };

    const results = await Promise.all(attempts.map((_, index) => sendAnalysisWithQueueRetry(send(index), {
      wait: () => new Promise((resolve) => setTimeout(resolve, 4)),
      random: () => 0.5,
      policy: { maxAttempts: 5, initialDelayMs: 1, maxDelayMs: 4, jitterRatio: 0 }
    })));

    expect(results.every((result) => result?.ok)).toBe(true);
    expect(attempts.some((count) => count > 1)).toBe(true);
    expect(peak).toBe(9);
  });

  it('cancels a retry when the image generation becomes stale during backoff', async () => {
    let current = true;
    const send = vi.fn(async () => queueFull);
    const result = await sendAnalysisWithQueueRetry(send, {
      shouldContinue: () => current,
      wait: async () => { current = false; },
      random: () => 0.5
    });

    expect(result).toBeUndefined();
    expect(send).toHaveBeenCalledTimes(1);
  });

  it('does not retry acquisition errors merely because they are transient', async () => {
    const response = { ok: false, retryable: true, code: 'FETCH_TIMEOUT' };
    const send = vi.fn(async () => response);

    expect(await sendAnalysisWithQueueRetry(send)).toBe(response);
    expect(send).toHaveBeenCalledTimes(1);
  });

  it('returns the final queue-full response when the retry budget is exhausted', async () => {
    const send = vi.fn(async () => queueFull);
    const result = await sendAnalysisWithQueueRetry(send, {
      wait: async () => {},
      random: () => 0.5,
      policy: { maxAttempts: 3, initialDelayMs: 1, maxDelayMs: 2, jitterRatio: 0 }
    });

    expect(result).toBe(queueFull);
    expect(send).toHaveBeenCalledTimes(3);
  });
});

describe('retryDelayMs', () => {
  it('uses bounded exponential backoff with symmetric jitter', () => {
    const policy = { maxAttempts: 5, initialDelayMs: 100, maxDelayMs: 250, jitterRatio: 0.25 };
    expect(retryDelayMs(0, policy, () => 0)).toBe(75);
    expect(retryDelayMs(1, policy, () => 0.5)).toBe(200);
    expect(retryDelayMs(8, policy, () => 1)).toBe(250);
  });
});
