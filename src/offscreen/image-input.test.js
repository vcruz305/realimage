import { describe, expect, it, vi } from 'vitest';
import { acquireImageInput, fetchEncodedImage, inspectImageEnvelope, readBoundedStream } from './image-input.js';
import { parseImageSource } from './source-policy.js';

describe('bounded image acquisition', () => {
  it('decodes a JSON-safe blob fallback while preserving the original blob source', async () => {
    const bytes = jpegHeader(320, 200);
    const fallbackDataUrl = jpegDataUrl(bytes);
    const payload = JSON.parse(JSON.stringify({
      source: 'blob:https://page.example.test/90f78da0-f652-4ed8-b9b3-91350d747531',
      fallbackDataUrl,
      pageUrl: 'https://page.example.test/'
    }));
    const fetchImpl = vi.fn().mockResolvedValue(new Response(bytes, {
      headers: { 'content-type': 'image/jpeg' }
    }));

    const result = await acquireImageInput(payload, { fetchImpl });

    expect(result).toMatchObject({ format: 'jpeg', width: 320, height: 200, mimeType: 'image/jpeg' });
    expect(fetchImpl).toHaveBeenCalledWith(fallbackDataUrl, expect.objectContaining({
      credentials: 'omit',
      redirect: 'manual',
      referrerPolicy: 'no-referrer'
    }));
    expect(fetchImpl.mock.calls[0][0]).not.toContain(payload.source);
  });

  it('rejects malformed or unexpected fallback strings before fetching', async () => {
    const fetchImpl = vi.fn();
    await expect(acquireImageInput({
      source: 'blob:https://page.example.test/90f78da0-f652-4ed8-b9b3-91350d747531',
      fallbackDataUrl: 'data:image/jpeg;base64,not canonical',
      pageUrl: 'https://page.example.test/'
    }, { fetchImpl })).rejects.toMatchObject({ code: 'INVALID_FALLBACK_ENCODING' });
    // A data: source already carries its own bytes as the source itself; a
    // fallback body alongside it is never expected, unlike blob:/http(s)/file:.
    await expect(acquireImageInput({
      source: 'data:image/jpeg;base64,/9j/',
      fallbackDataUrl: 'data:image/jpeg;base64,/9j/',
      pageUrl: 'https://page.example.test/'
    }, { fetchImpl })).rejects.toMatchObject({ code: 'UNEXPECTED_FALLBACK' });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('uses a captured fallback body instead of a network fetch for an http(s) or file source', async () => {
    // Mirrors the content script's canvas-capture retry: an http(s) source
    // whose direct background fetch already failed, and a file: source that
    // can never be fetched by the background at all, both supply the
    // already-rendered pixels as fallbackDataUrl instead of a URL to fetch.
    const bytes = jpegHeader(200, 100);
    const fallbackDataUrl = jpegDataUrl(bytes);
    // A fresh Response per call -- a Response body stream can only be read once.
    const fetchImpl = vi.fn(() => Promise.resolve(new Response(bytes, { headers: { 'content-type': 'image/jpeg' } })));

    const httpResult = await acquireImageInput({
      source: 'https://images.example.test/photo.png',
      fallbackDataUrl,
      pageUrl: 'https://page.example.test/'
    }, { fetchImpl });
    expect(httpResult).toMatchObject({ format: 'jpeg', width: 200, height: 100 });
    expect(fetchImpl).toHaveBeenCalledWith(fallbackDataUrl, expect.objectContaining({ credentials: 'omit' }));
    expect(fetchImpl.mock.calls[0][0]).not.toContain('images.example.test');

    fetchImpl.mockClear();
    const fileResult = await acquireImageInput({
      source: 'file:///C:/gallery/photo.png',
      fallbackDataUrl,
      pageUrl: 'file:///C:/gallery/index.html'
    }, { fetchImpl });
    expect(fileResult).toMatchObject({ format: 'jpeg', width: 200, height: 100 });
    expect(fetchImpl).toHaveBeenCalledWith(fallbackDataUrl, expect.objectContaining({ credentials: 'omit' }));
  });

  it('omits credentials, exposes redirects for validation, and sends no referrer', async () => {
    const bytes = pngHeader(640, 480);
    const fetchImpl = vi.fn().mockResolvedValue(new Response(bytes, { headers: { 'content-type': 'image/png' } }));
    const result = await acquireImageInput(
      { source: 'https://images.example.test/photo.png', pageUrl: 'https://page.example.test/' },
      { fetchImpl }
    );

    expect(result).toMatchObject({ format: 'png', width: 640, height: 480, mimeType: 'image/png' });
    expect(fetchImpl).toHaveBeenCalledWith('https://images.example.test/photo.png', expect.objectContaining({
      cache: 'default',
      credentials: 'omit',
      redirect: 'manual',
      referrerPolicy: 'no-referrer'
    }));
  });

  it('follows a readable same-origin localhost redirect after validating it', async () => {
    const start = 'http://127.0.0.1:4173/redirect/fixture';
    const final = 'http://127.0.0.1:4173/assets/fixture.png';
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(redirectResponse(start, '/assets/fixture.png'))
      .mockResolvedValueOnce(imageResponse(final, pngHeader(320, 200)));

    const result = await acquireImageInput(
      { source: start, pageUrl: 'http://127.0.0.1:4173/index.html' },
      { fetchImpl }
    );

    expect(result).toMatchObject({ format: 'png', width: 320, height: 200 });
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(fetchImpl.mock.calls.map(([url]) => url)).toEqual([start, final]);
  });

  it('blocks a redirect to localhost before issuing the private request', async () => {
    const start = 'https://images.example.test/redirect';
    const fetchImpl = vi.fn().mockResolvedValue(
      redirectResponse(start, 'http://127.0.0.1:8080/private.png')
    );

    await expect(fetchEncodedImage(parseImageSource(start), {
      fetchImpl,
      pageUrl: 'https://page.example.test/'
    })).rejects.toMatchObject({ code: 'LOCAL_TARGET_BLOCKED' });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('fails closed when Chrome exposes an opaque redirect without Location', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: false,
      status: 0,
      type: 'opaqueredirect',
      url: '',
      headers: new Headers(),
      body: null
    });
    await expect(fetchEncodedImage(parseImageSource('https://images.example.test/opaque'), {
      fetchImpl,
      pageUrl: 'https://page.example.test/'
    })).rejects.toMatchObject({ code: 'OPAQUE_REDIRECT_BLOCKED' });
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('blocks a custom fetch response that changes to a private target', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      type: 'basic',
      url: 'http://127.0.0.1:8080/private.png',
      headers: new Headers({ 'content-type': 'image/png' }),
      body: new Response(pngHeader(10, 10)).body
    });
    await expect(fetchEncodedImage(parseImageSource('https://public.example/image.png'), {
      fetchImpl,
      pageUrl: 'https://public.example/'
    })).rejects.toMatchObject({ code: 'LOCAL_TARGET_BLOCKED' });
  });

  it('enforces content-length and streamed-body limits', async () => {
    const tooLong = vi.fn().mockResolvedValue(new Response(new Uint8Array(1), {
      headers: { 'content-length': '101', 'content-type': 'image/png' }
    }));
    await expect(fetchEncodedImage(parseImageSource('https://images.example.test/a.png'), {
      fetchImpl: tooLong,
      maxBytes: 100,
      pageUrl: 'https://page.example.test/'
    })).rejects.toMatchObject({ code: 'IMAGE_TOO_LARGE' });

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(60));
        controller.enqueue(new Uint8Array(60));
        controller.close();
      }
    });
    await expect(readBoundedStream(stream, { maxBytes: 100 })).rejects.toMatchObject({ code: 'IMAGE_TOO_LARGE' });
  });

  it('aborts a fetch that hangs', async () => {
    const fetchImpl = vi.fn((_url, { signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')), { once: true });
    }));
    await expect(fetchEncodedImage(parseImageSource('https://images.example.test/hang.png'), {
      fetchImpl,
      timeoutMs: 5,
      pageUrl: 'https://page.example.test/'
    })).rejects.toMatchObject({ code: 'FETCH_TIMEOUT', retryable: true });
  });
});

describe('encoded image envelope validation', () => {
  it('rejects unsupported MIME, signature mismatch, and oversized decoded dimensions', () => {
    expect(() => inspectImageEnvelope(pngHeader(20, 20), 'image/svg+xml')).toThrowError(
      expect.objectContaining({ code: 'UNSUPPORTED_MIME' })
    );
    expect(() => inspectImageEnvelope(pngHeader(20, 20), 'image/jpeg')).toThrowError(
      expect.objectContaining({ code: 'MIME_MISMATCH' })
    );
    expect(() => inspectImageEnvelope(pngHeader(10_000, 5_000), 'image/png')).toThrowError(
      expect.objectContaining({ code: 'IMAGE_DIMENSIONS_EXCEEDED' })
    );
    expect(() => inspectImageEnvelope(new TextEncoder().encode('<svg></svg>'), 'image/png')).toThrowError(
      expect.objectContaining({ code: 'UNSUPPORTED_IMAGE' })
    );
  });
});

function pngHeader(width, height) {
  const bytes = new Uint8Array(24);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, width);
  view.setUint32(20, height);
  return bytes;
}

function jpegHeader(width, height) {
  const bytes = new Uint8Array(21);
  bytes.set([0xff, 0xd8, 0xff, 0xc0]);
  const view = new DataView(bytes.buffer);
  view.setUint16(4, 17);
  bytes[6] = 8;
  view.setUint16(7, height);
  view.setUint16(9, width);
  return bytes;
}

function jpegDataUrl(bytes) {
  return `data:image/jpeg;base64,${Buffer.from(bytes).toString('base64')}`;
}

function redirectResponse(url, location) {
  return {
    ok: false,
    status: 302,
    type: 'basic',
    url,
    headers: new Headers({ location }),
    body: null
  };
}

function imageResponse(url, bytes) {
  return {
    ok: true,
    status: 200,
    type: 'basic',
    url,
    headers: new Headers({ 'content-type': 'image/png', 'content-length': String(bytes.byteLength) }),
    body: new Response(bytes).body
  };
}
