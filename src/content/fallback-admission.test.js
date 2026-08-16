import { describe, expect, it, vi } from 'vitest';
import {
  HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS,
  HeavyImageAdmissionController
} from '../background/fallback-admission.js';
import { toUnavailableResponse } from '../background/validation.js';
import { MESSAGE } from '../shared/constants.js';
import { sendAnalysisWithQueueRetry } from './analysis-retry.js';
import {
  HEAVY_IMAGE_HEARTBEAT_INTERVAL_MS,
  withHeavyImageAdmission
} from './fallback-admission.js';

const extensionId = 'abcdefghijklmnopabcdefghijklmnop';

describe('content heavy-image admission', () => {
  it('keeps no more than two live admitted blob preparations across 200 concurrent requests', async () => {
    let activeBodies = 0;
    let peakBodies = 0;
    let captures = 0;
    let sends = 0;
    const waiters = [];
    const wakeWaiters = () => {
      for (const resolve of waiters.splice(0)) queueMicrotask(resolve);
    };
    const harness = admissionHarness({ onRelease: wakeWaiters });

    const results = await Promise.all(Array.from({ length: 200 }, (_, index) => {
      const requestId = `request-${index}`;
      return withHeavyImageAdmission({
        requestId,
        sendControlMessage: harness.sendControlMessage,
        prepare: () => {
          captures += 1;
          activeBodies += 1;
          peakBodies = Math.max(peakBodies, activeBodies);
          return 'data:image/jpeg;base64,/9j/';
        },
        sendWithToken: async ({ admissionToken, activateBeforeSend }) => {
          await activateBeforeSend();
          sends += 1;
          const attempt = harness.controller.beginAttempt(admissionToken, harness.owner(requestId));
          await Promise.resolve();
          activeBodies -= 1;
          harness.controller.finishAttempt(attempt, { terminal: true });
          wakeWaiters();
          return { ok: true };
        },
        wait: () => new Promise((resolve) => waiters.push(resolve)),
        random: () => 0.5,
        policy: { maxAttempts: 201, initialDelayMs: 0, maxDelayMs: 0, jitterRatio: 0 }
      });
    }));

    expect(results.every((result) => result?.ok)).toBe(true);
    expect(captures).toBe(200);
    expect(sends).toBe(200);
    expect(peakBodies).toBe(2);
    expect(harness.controller.size).toBe(0);
    expect(harness.controlMessages.every((message) => !('fallbackDataUrl' in message))).toBe(true);
  });

  it('keeps no more than two data-URL analysis sends across 200 concurrent requests', async () => {
    let activeSends = 0;
    let peakSends = 0;
    const waiters = [];
    const wakeWaiters = () => {
      for (const resolve of waiters.splice(0)) queueMicrotask(resolve);
    };
    const harness = admissionHarness({ onRelease: wakeWaiters });
    const dataSource = 'data:image/jpeg;base64,/9j/';

    const results = await Promise.all(Array.from({ length: 200 }, (_, index) => {
      const requestId = `data-${index}`;
      return withHeavyImageAdmission({
        requestId,
        sendControlMessage: harness.sendControlMessage,
        sendWithToken: async ({ admissionToken, prepared, activateBeforeSend }) => {
          await activateBeforeSend();
          expect(prepared).toBeUndefined();
          expect(dataSource.startsWith('data:')).toBe(true);
          const attempt = harness.controller.beginAttempt(admissionToken, harness.owner(requestId));
          activeSends += 1;
          peakSends = Math.max(peakSends, activeSends);
          await Promise.resolve();
          activeSends -= 1;
          harness.controller.finishAttempt(attempt, { terminal: true });
          return { ok: true };
        },
        wait: () => new Promise((resolve) => waiters.push(resolve)),
        random: () => 0.5,
        policy: { maxAttempts: 201, initialDelayMs: 0, maxDelayMs: 0, jitterRatio: 0 }
      });
    }));

    expect(results.every((result) => result?.ok)).toBe(true);
    expect(peakSends).toBe(2);
    expect(harness.controller.size).toBe(0);
  });

  it('cancels a stale queued data-URL request before prepare or analysis send', async () => {
    const harness = admissionHarness();
    harness.controller.reserve(harness.owner('holder-one'));
    harness.controller.reserve(harness.owner('holder-two'));
    let enabled = true;
    const capture = vi.fn();
    const sendWithToken = vi.fn();

    const result = await withHeavyImageAdmission({
      requestId: 'stale',
      sendControlMessage: harness.sendControlMessage,
      prepare: capture,
      sendWithToken,
      shouldContinue: () => enabled,
      wait: async () => { enabled = false; },
      random: () => 0.5,
      policy: { maxAttempts: 2, initialDelayMs: 0, maxDelayMs: 0, jitterRatio: 0 }
    });

    expect(result).toBeUndefined();
    expect(capture).not.toHaveBeenCalled();
    expect(sendWithToken).not.toHaveBeenCalled();
  });

  it('releases without sending when the user disables analysis during preparation', async () => {
    const harness = admissionHarness();
    let enabled = true;
    const sendWithToken = vi.fn();
    const result = await withHeavyImageAdmission({
      requestId: 'disabled-during-prepare',
      sendControlMessage: harness.sendControlMessage,
      prepare: async () => {
        enabled = false;
        return 'data:image/jpeg;base64,/9j/';
      },
      sendWithToken,
      shouldContinue: () => enabled
    });

    expect(result).toBeUndefined();
    expect(sendWithToken).not.toHaveBeenCalled();
    expect(harness.controller.size).toBe(0);
  });

  it('requests one immediate release when visibility cancellation races an active heavy analysis', async () => {
    const harness = admissionHarness();
    let releaseAdmission;
    let attempt;
    let finishSend;
    let releaseMessages = 0;
    const sendControlMessage = async (message) => {
      if (message.type === MESSAGE.RELEASE_HEAVY_IMAGE) releaseMessages += 1;
      return harness.sendControlMessage(message);
    };
    const task = withHeavyImageAdmission({
      requestId: 'hidden-in-flight',
      sendControlMessage,
      prepare: () => 'data:image/jpeg;base64,/9j/',
      onAdmissionReserved(release) {
        releaseAdmission = release;
      },
      sendWithToken: async ({ admissionToken, activateBeforeSend }) => {
        await activateBeforeSend();
        attempt = harness.controller.beginAttempt(admissionToken, harness.owner('hidden-in-flight'));
        return new Promise((resolve) => { finishSend = resolve; });
      }
    });
    for (let index = 0; index < 8 && !attempt; index += 1) await flushMicrotasks();
    expect(attempt).toBeDefined();

    await releaseAdmission();
    expect(releaseMessages).toBe(1);
    // Background keeps an already-delivered body counted until its attempt
    // settles, but the cancellation has marked the token for deletion.
    expect(harness.controller.size).toBe(1);

    harness.controller.finishAttempt(attempt, { terminal: true });
    finishSend({ ok: true, score: 0.5 });
    await expect(task).resolves.toMatchObject({ ok: true });
    expect(releaseMessages).toBe(1);
    expect(harness.controller.size).toBe(0);
  });

  it('holds one token across OFFSCREEN_QUEUE_FULL retries and releases it on the terminal result', async () => {
    const harness = admissionHarness();
    const seenTokens = [];
    let attempts = 0;
    const result = await withHeavyImageAdmission({
      requestId: 'queue-retry',
      sendControlMessage: harness.sendControlMessage,
      prepare: () => 'data:image/jpeg;base64,/9j/',
      sendWithToken: ({ admissionToken, activateBeforeSend }) => sendAnalysisWithQueueRetry(async () => {
        await activateBeforeSend();
        seenTokens.push(admissionToken);
        const active = harness.controller.beginAttempt(admissionToken, harness.owner('queue-retry'));
        attempts += 1;
        const response = attempts < 3
          ? { ok: false, unavailable: true, retryable: true, code: 'OFFSCREEN_QUEUE_FULL' }
          : { ok: true, score: 0.5 };
        harness.controller.finishAttempt(active, { terminal: response.ok });
        return response;
      }, {
        wait: async () => {},
        random: () => 0.5,
        policy: { maxAttempts: 3, initialDelayMs: 0, maxDelayMs: 0, jitterRatio: 0 }
      })
    });

    expect(result).toMatchObject({ ok: true });
    expect(attempts).toBe(3);
    expect(new Set(seenTokens).size).toBe(1);
    expect(harness.controller.size).toBe(0);
  });

  it('releases a reservation when capture or send fails', async () => {
    const harness = admissionHarness();
    await expect(withHeavyImageAdmission({
      requestId: 'capture-fails',
      sendControlMessage: harness.sendControlMessage,
      prepare: async () => { throw new Error('capture failed'); },
      sendWithToken: vi.fn()
    })).rejects.toThrow('capture failed');
    expect(harness.controller.size).toBe(0);
  });

  it('renews the active lease while asynchronous preparation is still pending', async () => {
    expect(HEAVY_IMAGE_HEARTBEAT_INTERVAL_MS * 2).toBeLessThan(HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS);
    const harness = admissionHarness();
    const timers = [];
    let finishPreparation;
    const task = withHeavyImageAdmission({
      requestId: 'slow-prepare',
      sendControlMessage: harness.sendControlMessage,
      prepare: () => new Promise((resolve) => { finishPreparation = resolve; }),
      sendWithToken: async ({ admissionToken, activateBeforeSend }) => {
        await activateBeforeSend();
        const attempt = harness.controller.beginAttempt(admissionToken, harness.owner('slow-prepare'));
        harness.controller.finishAttempt(attempt, { terminal: true });
        return { ok: true };
      },
      setHeartbeatTimer(callback, milliseconds) {
        const timer = { callback, milliseconds, cancelled: false };
        timers.push(timer);
        return timer;
      },
      clearHeartbeatTimer(timer) {
        timer.cancelled = true;
      }
    });

    await flushMicrotasks();
    expect(typeof finishPreparation).toBe('function');
    expect(timers[0]).toMatchObject({ milliseconds: HEAVY_IMAGE_HEARTBEAT_INTERVAL_MS, cancelled: false });
    const activationsBeforeHeartbeat = harness.controlMessages.filter((item) => item.type === MESSAGE.ACTIVATE_HEAVY_IMAGE).length;
    timers[0].callback();
    await flushMicrotasks();
    expect(harness.controlMessages.filter((item) => item.type === MESSAGE.ACTIVATE_HEAVY_IMAGE)).toHaveLength(
      activationsBeforeHeartbeat + 1
    );

    finishPreparation('data:image/jpeg;base64,/9j/');
    await expect(task).resolves.toMatchObject({ ok: true });
    expect(harness.controller.size).toBe(0);
  });

  it('preflights a thawed queue retry and sends no second heavy body after lease replacement', async () => {
    let clock = 0;
    const harness = admissionHarness({ now: () => clock });
    let largeSends = 0;
    const queueFull = { ok: false, unavailable: true, retryable: true, code: 'OFFSCREEN_QUEUE_FULL' };

    await expect(withHeavyImageAdmission({
      requestId: 'thawed-retry',
      sendControlMessage: harness.sendControlMessage,
      prepare: () => 'data:image/jpeg;base64,/9j/',
      sendWithToken: ({ admissionToken, activateBeforeSend }) => sendAnalysisWithQueueRetry(async () => {
        await activateBeforeSend();
        largeSends += 1;
        const attempt = harness.controller.beginAttempt(admissionToken, harness.owner('thawed-retry'));
        harness.controller.finishAttempt(attempt, { terminal: false });
        return queueFull;
      }, {
        wait: async () => {
          clock += HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS;
          harness.controller.reserve(harness.owner('replacement-one'));
          harness.controller.reserve(harness.owner('replacement-two'));
        },
        random: () => 0.5,
        policy: { maxAttempts: 2, initialDelayMs: 0, maxDelayMs: 0, jitterRatio: 0 }
      }),
      setHeartbeatTimer: () => ({ pending: true }),
      clearHeartbeatTimer: () => {}
    })).rejects.toThrow('missing, expired, or from an earlier worker');

    expect(largeSends).toBe(1);
    expect(harness.controller.size).toBe(2);
  });
});

function admissionHarness({ onRelease = () => {}, now } = {}) {
  let tokenSequence = 0;
  const controller = new HeavyImageAdmissionController({
    createToken: () => (++tokenSequence).toString(16).padStart(32, '0'),
    ...(now ? { now } : {})
  });
  const controlMessages = [];
  const owner = (requestId) => ({
    extensionId,
    tabId: 7,
    frameId: 0,
    documentId: 'document-one',
    requestId
  });
  const sendControlMessage = async (message) => {
    controlMessages.push(message);
    try {
      if (message.type === MESSAGE.RESERVE_HEAVY_IMAGE) {
        return { ok: true, admissionToken: controller.reserve(owner(message.requestId)) };
      }
      if (message.type === MESSAGE.ACTIVATE_HEAVY_IMAGE) {
        return {
          ok: true,
          activated: controller.activate(message.admissionToken, owner(message.requestId))
        };
      }
      if (message.type === MESSAGE.RELEASE_HEAVY_IMAGE) {
        const released = controller.release(message.admissionToken, owner(message.requestId));
        onRelease();
        return { ok: true, released };
      }
      throw new Error('unexpected control message');
    } catch (error) {
      return toUnavailableResponse(error, message.requestId);
    }
  };
  return { controller, controlMessages, owner, sendControlMessage };
}

async function flushMicrotasks() {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}
