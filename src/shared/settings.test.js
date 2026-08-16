import { describe, expect, it } from 'vitest';
import { sanitizeSettings } from './settings.js';

describe('settings sanitization', () => {
  it('uses the bounty threshold by default', () => {
    expect(sanitizeSettings().threshold).toBe(0.65);
    expect(sanitizeSettings().aiImageAction).toBe('blur');
  });

  it('bounds untrusted stored values', () => {
    const settings = sanitizeSettings({ threshold: 9, minimumDimension: 1, maxImagesPerPage: 999 });
    expect(settings.threshold).toBe(0.95);
    expect(settings.minimumDimension).toBe(48);
    expect(settings.maxImagesPerPage).toBe(200);
  });

  it('accepts only supported AI-image actions', () => {
    expect(sanitizeSettings({ aiImageAction: 'hide' }).aiImageAction).toBe('hide');
    expect(sanitizeSettings({ aiImageAction: 'remove-from-dom' }).aiImageAction).toBe('blur');
  });
});
