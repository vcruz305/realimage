import { describe, expect, it } from 'vitest';
import {
  ImageUnavailableError,
  FALLBACK_DATA_URL_PREFIX,
  MAX_DATA_URL_CHARS,
  MAX_FALLBACK_BYTES,
  MAX_FALLBACK_DATA_URL_CHARS,
  MAX_REMOTE_SOURCE_CHARS,
  assertRemoteSourceAllowed,
  classifyHostname,
  inspectFallbackDataUrl,
  parseImageSource,
  validateDecodedDimensions
} from './source-policy.js';

describe('remote source policy', () => {
  it('blocks private, metadata, integer-loopback, and non-public host targets', () => {
    const targets = [
      'http://192.168.1.4/image.png',
      'http://169.254.169.254/latest/meta-data',
      'http://2130706433/image.png',
      'http://router.local/image.png',
      'http://[::1]/image.png'
    ];
    for (const target of targets) {
      expect(() => assertRemoteSourceAllowed(parseImageSource(target), 'https://public.example/page')).toThrow(ImageUnavailableError);
    }
  });

  it('allows a same-kind local page to reach a loopback fixture on a different port', () => {
    // The exact-origin requirement is gone: any loopback page may reach any
    // loopback target, not only one on the identical origin.
    const source = parseImageSource('http://127.0.0.1:4173/fixture.png');
    expect(() => assertRemoteSourceAllowed(source, 'http://127.0.0.1:4173/index.html')).not.toThrow();
    expect(() => assertRemoteSourceAllowed(source, 'http://127.0.0.1:5173/index.html')).not.toThrow();
    expect(() => assertRemoteSourceAllowed(source, 'http://localhost:5173/index.html')).not.toThrow();
  });

  it('allows a localhost page to reach a 127.0.0.1 target on the same or a different port', () => {
    const samePort = parseImageSource('http://127.0.0.1:4173/fixture.png');
    const otherPort = parseImageSource('http://127.0.0.1:5173/fixture.png');
    expect(() => assertRemoteSourceAllowed(samePort, 'http://localhost:4173/gallery.html')).not.toThrow();
    expect(() => assertRemoteSourceAllowed(otherPort, 'http://localhost:4173/gallery.html')).not.toThrow();
  });

  it('allows a private-network page to reach a different private-network image host', () => {
    const source = parseImageSource('http://192.168.1.51:9000/photo.png');
    expect(() => assertRemoteSourceAllowed(source, 'http://192.168.1.50:8000/gallery.html')).not.toThrow();
  });

  it('allows a file: page to reach any local (loopback or private-network) image target', () => {
    const loopback = parseImageSource('http://127.0.0.1:4173/fixture.png');
    const privateNetwork = parseImageSource('http://192.168.1.51:9000/photo.png');
    expect(() => assertRemoteSourceAllowed(loopback, 'file:///C:/Users/me/gallery/index.html')).not.toThrow();
    expect(() => assertRemoteSourceAllowed(privateNetwork, 'file:///C:/Users/me/gallery/index.html')).not.toThrow();
    // A bare file:// URL has an empty hostname; it must still count as local.
    expect(() => assertRemoteSourceAllowed(loopback, 'file:///index.html')).not.toThrow();
  });

  it('still blocks a genuinely public page from reaching a loopback or private-network target', () => {
    const loopback = parseImageSource('http://127.0.0.1:4173/fixture.png');
    const privateNetwork = parseImageSource('http://192.168.1.51:9000/photo.png');
    expect(() => assertRemoteSourceAllowed(loopback, 'https://public.example/page')).toThrowError(
      expect.objectContaining({ code: 'LOCAL_TARGET_BLOCKED' })
    );
    expect(() => assertRemoteSourceAllowed(privateNetwork, 'https://public.example/page')).toThrowError(
      expect.objectContaining({ code: 'PRIVATE_TARGET_BLOCKED' })
    );
  });

  it('recognizes public addresses and rejects unsafe protocols and credentials', () => {
    expect(classifyHostname('8.8.8.8')).toBe('public');
    expect(classifyHostname('images.example.com')).toBe('public');
    // file: is now a supported wire-level source protocol (it can only ever
    // reach analysis through the canvas-capture fallback path), but other
    // unsupported schemes are still rejected outright.
    expect(parseImageSource('file:///secret.png').protocol).toBe('file:');
    expect(() => parseImageSource('ftp://example.com/secret.png')).toThrowError(
      expect.objectContaining({ code: 'UNSUPPORTED_PROTOCOL' })
    );
    expect(() => parseImageSource('https://user:pass@example.com/image.png')).toThrowError(
      expect.objectContaining({ code: 'URL_CREDENTIALS_BLOCKED' })
    );
  });

  it('rejects page-controlled sources before they can exceed Chrome JSON messaging limits', () => {
    const remote = `https://images.example.test/${'a'.repeat(MAX_REMOTE_SOURCE_CHARS)}`;
    expect(remote.length).toBeGreaterThan(MAX_REMOTE_SOURCE_CHARS);
    expect(() => parseImageSource(remote)).toThrowError(expect.objectContaining({ code: 'SOURCE_TOO_LONG' }));

    const embedded = `data:image/jpeg;base64,${'A'.repeat(MAX_DATA_URL_CHARS)}`;
    expect(embedded.length).toBeGreaterThan(MAX_DATA_URL_CHARS);
    expect(() => parseImageSource(embedded)).toThrowError(expect.objectContaining({ code: 'IMAGE_TOO_LARGE' }));
  });

  it('accepts an embedded source at the exact 2 MiB wire cap and rejects one extra character', () => {
    expect(MAX_DATA_URL_CHARS).toBe(2 * 1024 * 1024);
    const prefix = 'data:image/jpeg;base64,';
    const exact = `${prefix}${'A'.repeat(MAX_DATA_URL_CHARS - prefix.length)}`;
    expect(exact.length).toBe(MAX_DATA_URL_CHARS);
    expect(parseImageSource(exact).protocol).toBe('data:');
    expect(() => parseImageSource(`${exact}A`)).toThrowError(
      expect.objectContaining({ code: 'IMAGE_TOO_LARGE' })
    );
  });
});

describe('Stable Chrome blob fallback wire policy', () => {
  it('accepts only canonical bounded JPEG base64', () => {
    expect(inspectFallbackDataUrl(`${FALLBACK_DATA_URL_PREFIX}/9j/`)).toEqual({
      dataUrl: `${FALLBACK_DATA_URL_PREFIX}/9j/`,
      mimeType: 'image/jpeg',
      byteLength: 3
    });
    for (const value of [
      `${FALLBACK_DATA_URL_PREFIX}`,
      `${FALLBACK_DATA_URL_PREFIX}%%%a`,
      `${FALLBACK_DATA_URL_PREFIX}AB==`,
      `${FALLBACK_DATA_URL_PREFIX}AAA`,
      'data:image/png;base64,iVBORw=='
    ]) {
      expect(() => inspectFallbackDataUrl(value)).toThrow(ImageUnavailableError);
    }
  });

  it('rejects the wire representation before it can exceed the fallback byte cap', () => {
    const oversized = `${FALLBACK_DATA_URL_PREFIX}${'A'.repeat(MAX_FALLBACK_DATA_URL_CHARS - FALLBACK_DATA_URL_PREFIX.length + 1)}`;
    expect(oversized.length).toBe(MAX_FALLBACK_DATA_URL_CHARS + 1);
    expect(() => inspectFallbackDataUrl(oversized)).toThrowError(
      expect.objectContaining({ code: 'IMAGE_TOO_LARGE' })
    );
  });

  it('accepts exactly 2 MiB and rejects 2 MiB plus one byte', () => {
    expect(MAX_FALLBACK_BYTES).toBe(2 * 1024 * 1024);
    const exact = `${FALLBACK_DATA_URL_PREFIX}${zeroBytesBase64(MAX_FALLBACK_BYTES)}`;
    const plusOne = `${FALLBACK_DATA_URL_PREFIX}${zeroBytesBase64(MAX_FALLBACK_BYTES + 1)}`;
    // Both encodings have the same wire length. The decoded-byte check—not a
    // character estimate—must enforce the exact boundary.
    expect(plusOne.length).toBe(exact.length);
    expect(inspectFallbackDataUrl(exact).byteLength).toBe(MAX_FALLBACK_BYTES);
    expect(() => inspectFallbackDataUrl(plusOne)).toThrowError(
      expect.objectContaining({ code: 'IMAGE_TOO_LARGE' })
    );
  });
});

describe('decoded dimension policy', () => {
  it('caps individual sides and total decoded pixels', () => {
    expect(validateDecodedDimensions(4_000, 4_000)).toEqual({ width: 4_000, height: 4_000 });
    expect(() => validateDecodedDimensions(16_385, 1)).toThrowError(
      expect.objectContaining({ code: 'IMAGE_DIMENSIONS_EXCEEDED' })
    );
    expect(() => validateDecodedDimensions(10_000, 5_000)).toThrowError(
      expect.objectContaining({ code: 'IMAGE_DIMENSIONS_EXCEEDED' })
    );
  });
});

function zeroBytesBase64(byteLength) {
  const completeTriples = Math.floor(byteLength / 3);
  const remainder = byteLength % 3;
  return 'A'.repeat(completeTriples * 4)
    + (remainder === 1 ? 'AA==' : remainder === 2 ? 'AAA=' : '');
}
