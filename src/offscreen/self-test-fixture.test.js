import { describe, expect, it } from 'vitest';
import { createSelfTestPixelData, SELF_TEST_IMAGE_SIZE, SELF_TEST_REFERENCE_LOGIT } from './self-test-fixture.js';

describe('self-test fixture', () => {
  it('is fully deterministic across calls', () => {
    const first = createSelfTestPixelData();
    const second = createSelfTestPixelData();
    expect([...first]).toEqual([...second]);
  });

  it('produces exactly width * height * 3 RGB bytes for the default size', () => {
    const data = createSelfTestPixelData();
    expect(data).toBeInstanceOf(Uint8Array);
    expect(data.length).toBe(SELF_TEST_IMAGE_SIZE * SELF_TEST_IMAGE_SIZE * 3);
  });

  it('supports a custom size', () => {
    const data = createSelfTestPixelData(8);
    expect(data.length).toBe(8 * 8 * 3);
  });

  it('stores a finite empirically measured reference logit', () => {
    expect(typeof SELF_TEST_REFERENCE_LOGIT).toBe('number');
    expect(Number.isFinite(SELF_TEST_REFERENCE_LOGIT)).toBe(true);
  });
});
