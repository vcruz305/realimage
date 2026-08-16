import { describe, expect, it, vi } from 'vitest';
import { waitForOffscreenReceiver } from './offscreen-readiness.js';

describe('waitForOffscreenReceiver', () => {
  it('accepts only the nonce-matched offscreen acknowledgement', async () => {
    const sendMessage = vi.fn()
      .mockResolvedValueOnce(undefined)
      .mockResolvedValueOnce({ ok: true, receiver: 'offscreen', nonce: 'wrong' })
      .mockResolvedValueOnce({ ok: true, receiver: 'offscreen', nonce: 'nonce-1' });
    const sleep = vi.fn().mockResolvedValue(undefined);

    await waitForOffscreenReceiver({
      sendMessage,
      nonceFactory: () => 'nonce-1',
      sleep,
      attempts: 3,
      initialDelayMs: 1,
      maxDelayMs: 2
    });

    expect(sendMessage).toHaveBeenCalledTimes(3);
    expect(sendMessage).toHaveBeenLastCalledWith({
      type: 'proofmark/offscreen-ping',
      nonce: 'nonce-1'
    });
    expect(sleep).toHaveBeenCalledTimes(2);
  });

  it('retries a missing receiver without sending an analysis message', async () => {
    const sendMessage = vi.fn().mockRejectedValue(new Error('Receiving end does not exist.'));

    await expect(waitForOffscreenReceiver({
      sendMessage,
      nonceFactory: () => 'nonce-2',
      sleep: vi.fn().mockResolvedValue(undefined),
      attempts: 2,
      initialDelayMs: 1,
      maxDelayMs: 1
    })).rejects.toMatchObject({
      name: 'OffscreenStartupError',
      code: 'OFFSCREEN_RECEIVER_MISSING'
    });

    expect(sendMessage).toHaveBeenCalledTimes(2);
    expect(sendMessage.mock.calls.every(([message]) => message.type === 'proofmark/offscreen-ping')).toBe(true);
  });
});
