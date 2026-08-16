import { describe, expect, it, vi } from 'vitest';
import { SerialInferenceQueue } from './inference-queue.js';
import { createUnavailableResponse } from './unavailable-response.js';

describe('SerialInferenceQueue', () => {
  it('runs exactly one model task at a time in submission order', async () => {
    const queue = new SerialInferenceQueue({ maxQueued: 3 });
    let active = 0;
    let maximumActive = 0;
    const releases = [];
    const makeTask = (value) => vi.fn(async () => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      await new Promise((resolve) => releases.push(resolve));
      active -= 1;
      return value;
    });
    const firstTask = makeTask('first');
    const secondTask = makeTask('second');
    const first = queue.run(firstTask);
    const second = queue.run(secondTask);

    await vi.waitFor(() => expect(firstTask).toHaveBeenCalledTimes(1));
    expect(secondTask).not.toHaveBeenCalled();
    releases.shift()();
    await vi.waitFor(() => expect(secondTask).toHaveBeenCalledTimes(1));
    releases.shift()();

    await expect(Promise.all([first, second])).resolves.toEqual(['first', 'second']);
    expect(maximumActive).toBe(1);
  });

  it('rejects overflow deterministically without running the extra task', async () => {
    const queue = new SerialInferenceQueue({ maxQueued: 1 });
    let release;
    const first = queue.run(() => new Promise((resolve) => { release = resolve; }));
    await vi.waitFor(() => expect(queue.active).toBe(1));
    const second = queue.run(async () => 'second');
    const overflowTask = vi.fn(async () => 'overflow');

    const overflow = queue.run(overflowTask);
    await expect(overflow).rejects.toMatchObject({
      code: 'OFFSCREEN_QUEUE_FULL',
      retryable: true
    });
    const error = await overflow.catch((reason) => reason);
    const response = createUnavailableResponse(error, 'pm-overflow');
    expect(response).toMatchObject({
      ok: false,
      unavailable: true,
      code: 'OFFSCREEN_QUEUE_FULL',
      requestId: 'pm-overflow'
    });
    expect(response).not.toHaveProperty('score');
    expect(overflowTask).not.toHaveBeenCalled();
    release('first');
    await expect(Promise.all([first, second])).resolves.toEqual(['first', 'second']);
  });

  it('continues after a failed task', async () => {
    const queue = new SerialInferenceQueue({ maxQueued: 1 });
    const failed = queue.run(async () => { throw new Error('model failure'); });
    const next = queue.run(async () => 'ok');
    await expect(failed).rejects.toThrow('model failure');
    await expect(next).resolves.toBe('ok');
  });
});
