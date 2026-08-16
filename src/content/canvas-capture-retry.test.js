import { describe, expect, it } from 'vitest';
import { CANVAS_CAPTURE_RETRY_CODES, isCanvasCaptureRetryable } from './canvas-capture-retry.js';

describe('isCanvasCaptureRetryable', () => {
  it('retries every code that means the background could not fetch the image itself', () => {
    for (const code of CANVAS_CAPTURE_RETRY_CODES) {
      expect(isCanvasCaptureRetryable({ ok: false, unavailable: true, code })).toBe(true);
    }
  });

  it('does not retry a code a canvas capture cannot fix', () => {
    for (const code of ['IMAGE_TOO_LARGE', 'UNSUPPORTED_MIME', 'IMAGE_DIMENSIONS_EXCEEDED', 'INVALID_SOURCE']) {
      expect(isCanvasCaptureRetryable({ ok: false, unavailable: true, code })).toBe(false);
    }
  });

  it('does not retry a successful result', () => {
    expect(isCanvasCaptureRetryable({ ok: true, score: 0.2 })).toBe(false);
  });

  it('does not retry a missing/stale result', () => {
    expect(isCanvasCaptureRetryable(undefined)).toBe(false);
    expect(isCanvasCaptureRetryable(null)).toBe(false);
  });
});
