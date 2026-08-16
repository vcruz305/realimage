import { describe, expect, it, vi } from 'vitest';
import { InferenceScheduler, LruCache, TabBadgeTracker } from './scheduler.js';

describe('InferenceScheduler', () => {
  it('coalesces identical in-flight work while preserving each requestId', async () => {
    let resolveInference;
    const infer = vi.fn(() => new Promise((resolve) => { resolveInference = resolve; }));
    const scheduler = new InferenceScheduler();
    const image = { source: 'data:image/png;base64,AAAA', naturalWidth: 512, naturalHeight: 512 };

    const first = scheduler.run({ ...image, requestId: 'first' }, infer);
    const second = scheduler.run({ ...image, requestId: 'second' }, infer);
    await vi.waitFor(() => expect(infer).toHaveBeenCalledTimes(1));
    resolveInference({ ok: true, requestId: 'offscreen-first', score: 0.91 });

    const [firstResult, secondResult] = await Promise.all([first, second]);
    expect(firstResult).toMatchObject({ ok: true, requestId: 'first', cached: false });
    expect(secondResult).toMatchObject({ ok: true, requestId: 'second', cached: false });
    // Cache-key hashing is asynchronous, so either same-image caller may reach
    // the in-flight map first. The contract is exactly one owner, one coalesced
    // caller, one inference, and caller-specific request IDs—not arrival-order
    // ownership.
    expect([firstResult.coalesced, secondResult.coalesced].sort()).toEqual([false, true]);
    await expect(scheduler.run({ ...image, requestId: 'third' }, infer)).resolves.toMatchObject({
      ok: true,
      requestId: 'third',
      cached: true
    });
    expect(infer).toHaveBeenCalledTimes(1);
  });

  it('coalesces concurrent HTTP work but infers the same URL again sequentially', async () => {
    let resolveInference;
    const infer = vi.fn(() => new Promise((resolve) => { resolveInference = resolve; }));
    const scheduler = new InferenceScheduler();
    const image = {
      source: 'https://example.test/changeable.png?token=secret',
      naturalWidth: 512,
      naturalHeight: 512
    };

    const first = scheduler.run({ ...image, requestId: 'first' }, infer);
    const concurrent = scheduler.run({ ...image, requestId: 'concurrent' }, infer);
    await vi.waitFor(() => expect(infer).toHaveBeenCalledTimes(1));
    resolveInference({ ok: true, score: 0.91 });
    await expect(Promise.all([first, concurrent])).resolves.toEqual(expect.arrayContaining([
      expect.objectContaining({ requestId: 'first', cached: false }),
      expect.objectContaining({ requestId: 'concurrent', cached: false })
    ]));

    const sequential = scheduler.run({ ...image, requestId: 'sequential' }, infer);
    await vi.waitFor(() => expect(infer).toHaveBeenCalledTimes(2));
    resolveInference({ ok: true, score: 0.12 });
    await expect(sequential).resolves.toMatchObject({
      ok: true,
      requestId: 'sequential',
      cached: false,
      coalesced: false,
      score: 0.12
    });
    expect(scheduler.cache.size).toBe(0);
  });

  it('clears rejected and unsuccessful work from the in-flight map', async () => {
    const scheduler = new InferenceScheduler();
    const payload = { source: 'https://example.test/failure.png', naturalWidth: 128, naturalHeight: 128, requestId: 'a' };
    const rejected = vi.fn().mockImplementationOnce(() => { throw new Error('worker stopped'); });
    await expect(scheduler.run(payload, rejected)).rejects.toThrow('worker stopped');
    expect(scheduler.inFlight.size).toBe(0);

    const unsuccessful = vi.fn().mockResolvedValueOnce({ ok: false, error: 'decode failed' });
    await expect(scheduler.run(payload, unsuccessful)).resolves.toMatchObject({ ok: false, requestId: 'a' });
    expect(scheduler.inFlight.size).toBe(0);

    const retry = vi.fn().mockResolvedValueOnce({ ok: true, score: 0.2 });
    await scheduler.run(payload, retry);
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it('stores only a fixed-size digest key for giant data URLs, fallbacks, and signed secrets', async () => {
    const secret = 'signed-token-that-must-not-be-retained';
    const giant = `data:image/png;base64,${'A'.repeat(20_000)}${secret}`;
    const fallbackDataUrl = `data:image/jpeg;base64,${Buffer.from(secret).toString('base64')}`;
    const payload = { source: giant, fallbackDataUrl, naturalWidth: 400, naturalHeight: 300, requestId: 'giant' };
    const scheduler = new InferenceScheduler();
    await scheduler.run(payload, async () => ({ ok: true, score: 0.4 }));
    const [key] = scheduler.cache.entries.keys();

    expect(key).toHaveLength('image-v3:'.length + 64);
    expect(key).not.toContain(secret);
    expect(key).not.toContain('data:image');
  });

  it('does not coalesce equal-length blob fallbacks with different content', async () => {
    const scheduler = new InferenceScheduler();
    const infer = vi.fn().mockResolvedValue({ ok: true, score: 0.4 });
    const payload = {
      source: 'blob:https://example.test/90f78da0-f652-4ed8-b9b3-91350d747531',
      naturalWidth: 400,
      naturalHeight: 300
    };
    await scheduler.run({ ...payload, requestId: 'a', fallbackDataUrl: 'data:image/jpeg;base64,AAAA' }, infer);
    await scheduler.run({ ...payload, requestId: 'b', fallbackDataUrl: 'data:image/jpeg;base64,AAAB' }, infer);
    expect(infer).toHaveBeenCalledTimes(2);
    expect([...scheduler.cache.entries.keys()]).toHaveLength(2);
  });

  it('treats a fresh scheduler as a cache miss after an MV3-style restart', async () => {
    const payload = { source: 'https://example.test/restart.jpg', naturalWidth: 200, naturalHeight: 200, requestId: 'one' };
    const firstInfer = vi.fn().mockResolvedValue({ ok: true, score: 0.1 });
    await new InferenceScheduler().run(payload, firstInfer);

    const restartedInfer = vi.fn().mockResolvedValue({ ok: true, score: 0.1 });
    await new InferenceScheduler().run({ ...payload, requestId: 'two' }, restartedInfer);
    expect(restartedInfer).toHaveBeenCalledTimes(1);
  });
});

describe('LruCache', () => {
  it('evicts the least recently used entry and remains bounded', () => {
    const cache = new LruCache(2);
    cache.set('a', 1);
    cache.set('b', 2);
    expect(cache.get('a')).toBe(1);
    cache.set('c', 3);
    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('a')).toBe(1);
    expect(cache.get('c')).toBe(3);
    expect(cache.size).toBe(2);
  });
});

describe('TabBadgeTracker', () => {
  it('keeps 0.649999 below the fixed line and includes exactly 0.65', async () => {
    const below = new TabBadgeTracker({ threshold: 0.65 });
    const belowToken = await below.begin(5);
    expect(below.settle(belowToken, { ok: true, score: 0.649999 })).toMatchObject({ text: '' });

    const at = new TabBadgeTracker({ threshold: 0.65 });
    const atToken = await at.begin(6);
    expect(at.settle(atToken, { ok: true, score: 0.65 })).toMatchObject({ text: 'AI' });
  });

  it('does not let a real completion clear a pending or known AI badge', async () => {
    const tracker = new TabBadgeTracker();
    const first = await tracker.begin(7);
    const second = await tracker.begin(7);

    expect(tracker.settle(first, { ok: true, score: 0.1 })).toBeUndefined();
    expect(tracker.settle(second, { ok: true, score: 0.9 })).toMatchObject({ text: 'AI' });

    const third = await tracker.begin(7);
    expect(tracker.settle(third, { ok: true, score: 0.05 })).toMatchObject({ text: 'AI' });
  });

  it('hydrates a browser-held AI badge after a worker restart', async () => {
    const tracker = new TabBadgeTracker();
    const token = await tracker.begin(9, async () => 'AI');
    expect(tracker.settle(token, { ok: true, score: 0.01 })).toMatchObject({ text: 'AI' });
  });

  it('ignores a stale result after navigation resets the tab generation', async () => {
    const tracker = new TabBadgeTracker();
    const oldPage = await tracker.begin(11);
    tracker.reset(11);
    const newPage = await tracker.begin(11);

    expect(tracker.settle(oldPage, { ok: true, score: 0.99 })).toBeUndefined();
    expect(tracker.settle(newPage, { ok: true, score: 0.1 })).toMatchObject({ text: '' });
  });
});
