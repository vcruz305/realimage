import { describe, expect, it } from 'vitest';
import { sanitizeAnalyzePayload, toUnavailableResponse } from './validation.js';

const extensionId = 'realimage-extension-id';
const sender = {
  id: extensionId,
  url: 'https://frame.example.test/gallery',
  tab: { id: 7, url: 'https://top.example.test/' }
};

describe('background request validation', () => {
  it('overwrites a claimed pageUrl with Chrome MessageSender data', () => {
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-safe-1',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600,
      pageUrl: 'http://127.0.0.1:4173/spoofed'
    }, sender, extensionId);

    expect(result.pageUrl).toBe(sender.url);
    expect(result).not.toHaveProperty('arbitrary');
  });

  it('rejects untrusted senders and malformed dimensions', () => {
    const payload = {
      requestId: 'pm-safe-2',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600
    };
    expect(() => sanitizeAnalyzePayload(payload, { ...sender, id: 'other-extension' }, extensionId)).toThrowError(
      expect.objectContaining({ code: 'UNTRUSTED_SENDER' })
    );
    expect(() => sanitizeAnalyzePayload({ ...payload, naturalWidth: 100_000 }, sender, extensionId)).toThrowError(
      expect.objectContaining({ code: 'INVALID_DIMENSIONS' })
    );
  });

  it('does not let a spoofed payload page authorize localhost access', () => {
    expect(() => sanitizeAnalyzePayload({
      requestId: 'pm-safe-3',
      source: 'http://127.0.0.1:4173/image.png',
      naturalWidth: 100,
      naturalHeight: 100,
      pageUrl: 'http://127.0.0.1:4173/'
    }, sender, extensionId)).toThrowError(expect.objectContaining({ code: 'LOCAL_TARGET_BLOCKED' }));
  });

  it('preserves same-origin localhost fixture requests', () => {
    const localSender = {
      id: extensionId,
      url: 'http://127.0.0.1:4173/index.html',
      tab: { id: 8, url: 'http://127.0.0.1:4173/index.html' }
    };
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-local-1',
      source: 'http://127.0.0.1:4173/fixture.png',
      naturalWidth: 100,
      naturalHeight: 100
    }, localSender, extensionId);
    expect(result.pageUrl).toBe(localSender.url);
  });

  it('allows a localhost page to reach a private-network target on a different host/port', () => {
    const localSender = {
      id: extensionId,
      url: 'http://localhost:4173/index.html',
      tab: { id: 8, url: 'http://localhost:4173/index.html' }
    };
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-local-2',
      source: 'http://192.168.1.50:9000/fixture.png',
      naturalWidth: 100,
      naturalHeight: 100
    }, localSender, extensionId);
    expect(result.pageUrl).toBe(localSender.url);
  });

  it('trusts a file: sender URL for the local-target policy', () => {
    const fileSender = {
      id: extensionId,
      url: 'file:///C:/Users/me/gallery/index.html',
      tab: { id: 9, url: 'file:///C:/Users/me/gallery/index.html' }
    };
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-file-1',
      source: 'http://127.0.0.1:4173/fixture.png',
      naturalWidth: 100,
      naturalHeight: 100
    }, fileSender, extensionId);
    expect(result.pageUrl).toBe(fileSender.url);
  });

  it('still blocks a public sender from reaching a private-network target even with a fallback body attached', () => {
    // validateSource() re-runs the private-network/loopback policy on every
    // attempt, fallback or not, so a public page cannot use the canvas-capture
    // retry path to reach a target the direct-fetch policy already blocks.
    expect(() => sanitizeAnalyzePayload({
      requestId: 'pm-public-retry-attempt',
      source: 'http://192.168.1.50:9000/fixture.png',
      fallbackDataUrl: 'data:image/jpeg;base64,/9j/',
      naturalWidth: 100,
      naturalHeight: 100
    }, sender, extensionId)).toThrowError(expect.objectContaining({ code: 'PRIVATE_TARGET_BLOCKED' }));
  });

  it('never forwards an oversized trusted sender URL to the offscreen message hop', () => {
    const oversizedSender = {
      ...sender,
      url: `https://frame.example.test/${'a'.repeat(40_000)}`
    };
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-safe-tab-url',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600
    }, oversizedSender, extensionId);
    expect(result.pageUrl).toBe(sender.tab.url);

    expect(() => sanitizeAnalyzePayload({
      requestId: 'pm-no-safe-page-url',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600
    }, {
      ...oversizedSender,
      tab: { ...sender.tab, url: `https://top.example.test/${'b'.repeat(40_000)}` }
    }, extensionId)).toThrowError(expect.objectContaining({ code: 'INVALID_SENDER_CONTEXT' }));
  });

  it('accepts a JSON-round-tripped canonical same-origin blob fallback', () => {
    const fallbackDataUrl = 'data:image/jpeg;base64,/9j/';
    const wirePayload = JSON.parse(JSON.stringify({
      requestId: 'pm-blob-1',
      source: 'blob:https://frame.example.test/90f78da0-f652-4ed8-b9b3-91350d747531',
      fallbackDataUrl,
      naturalWidth: 10,
      naturalHeight: 10
    }));
    const result = sanitizeAnalyzePayload(wirePayload, sender, extensionId);
    expect(result.fallbackDataUrl).toBe(fallbackDataUrl);
    expect(JSON.parse(JSON.stringify(result))).toEqual(result);

    expect(() => sanitizeAnalyzePayload({
      requestId: 'pm-blob-2',
      source: 'blob:https://other.example.test/90f78da0-f652-4ed8-b9b3-91350d747531',
      fallbackDataUrl,
      naturalWidth: 10,
      naturalHeight: 10
    }, sender, extensionId)).toThrowError(expect.objectContaining({ code: 'BLOB_ORIGIN_MISMATCH' }));
  });

  it('rejects malformed and unexpected fallback transports', () => {
    const blobPayload = {
      requestId: 'pm-blob-invalid',
      source: 'blob:https://frame.example.test/90f78da0-f652-4ed8-b9b3-91350d747531',
      naturalWidth: 10,
      naturalHeight: 10
    };
    expect(() => sanitizeAnalyzePayload({
      ...blobPayload,
      fallbackDataUrl: 'data:image/png;base64,iVBORw=='
    }, sender, extensionId)).toThrowError(expect.objectContaining({ code: 'UNSUPPORTED_MIME' }));
    // A data: source already carries its own bytes as the source itself, so a
    // fallback body alongside it is still rejected -- unlike blob:/http(s)/file:.
    expect(() => sanitizeAnalyzePayload({
      requestId: 'pm-data-fallback',
      source: 'data:image/jpeg;base64,/9j/',
      fallbackDataUrl: 'data:image/jpeg;base64,/9j/',
      naturalWidth: 10,
      naturalHeight: 10
    }, sender, extensionId)).toThrowError(expect.objectContaining({ code: 'UNEXPECTED_FALLBACK' }));
  });

  it('accepts a canvas-capture fallback body alongside an allowed http(s) or file source', () => {
    const fallbackDataUrl = 'data:image/jpeg;base64,/9j/';
    const httpResult = sanitizeAnalyzePayload({
      requestId: 'pm-http-retry',
      source: 'https://images.example.test/image.jpg',
      fallbackDataUrl,
      naturalWidth: 10,
      naturalHeight: 10
    }, sender, extensionId);
    expect(httpResult.fallbackDataUrl).toBe(fallbackDataUrl);
    expect(httpResult.source).toBe('https://images.example.test/image.jpg');

    const fileSender = {
      id: extensionId,
      url: 'file:///C:/Users/me/gallery/index.html',
      tab: { id: 10, url: 'file:///C:/Users/me/gallery/index.html' }
    };
    const fileResult = sanitizeAnalyzePayload({
      requestId: 'pm-file-retry',
      source: 'file:///C:/Users/me/gallery/photo.png',
      fallbackDataUrl,
      naturalWidth: 10,
      naturalHeight: 10
    }, fileSender, extensionId);
    expect(fileResult.fallbackDataUrl).toBe(fallbackDataUrl);
    expect(fileResult.source).toBe('file:///C:/Users/me/gallery/photo.png');
  });

  it('leaves ordinary HTTP and data-URL sources unchanged without a fallback', () => {
    for (const [requestId, source] of [
      ['pm-http-plain', 'https://images.example.test/image.jpg'],
      ['pm-data-plain', 'data:image/jpeg;base64,/9j/']
    ]) {
      const result = sanitizeAnalyzePayload({
        requestId,
        source,
        naturalWidth: 10,
        naturalHeight: 10
      }, sender, extensionId);
      expect(result.source).toBe(source);
      expect(result).not.toHaveProperty('fallbackDataUrl');
    }
  });

  it('passes a short valid pageContextText through, trimmed', () => {
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-context-1',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600,
      pageContextText: '  AI-generated artwork by a talented model  '
    }, sender, extensionId);
    expect(result.pageContextText).toBe('AI-generated artwork by a talented model');
  });

  it('omits pageContextText entirely when absent, without error', () => {
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-context-2',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600
    }, sender, extensionId);
    expect(result).not.toHaveProperty('pageContextText');
  });

  it('omits pageContextText when it is only whitespace', () => {
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-context-3',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600,
      pageContextText: '   \n\t  '
    }, sender, extensionId);
    expect(result).not.toHaveProperty('pageContextText');
  });

  it('truncates an overlong pageContextText to 500 characters instead of rejecting it', () => {
    const huge = 'a'.repeat(5_000);
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-context-4',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600,
      pageContextText: huge
    }, sender, extensionId);
    expect(result.pageContextText).toHaveLength(500);
    expect(result.pageContextText).toBe('a'.repeat(500));
  });

  it('drops a non-string pageContextText instead of crashing validation', () => {
    for (const badValue of [12345, { alt: 'AI-generated' }, ['AI-generated'], true]) {
      const result = sanitizeAnalyzePayload({
        requestId: 'pm-context-5',
        source: 'https://images.example.test/image.png',
        naturalWidth: 800,
        naturalHeight: 600,
        pageContextText: badValue
      }, sender, extensionId);
      expect(result).not.toHaveProperty('pageContextText');
    }
  });

  it('passes a finite numeric priority through unchanged', () => {
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-priority-1',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600,
      priority: 1234
    }, sender, extensionId);
    expect(result.priority).toBe(1234);
  });

  it('omits priority entirely when absent, without error', () => {
    const result = sanitizeAnalyzePayload({
      requestId: 'pm-priority-2',
      source: 'https://images.example.test/image.png',
      naturalWidth: 800,
      naturalHeight: 600
    }, sender, extensionId);
    expect(result).not.toHaveProperty('priority');
  });

  it('drops a non-finite or non-numeric priority instead of crashing validation', () => {
    for (const badValue of ['top', NaN, Infinity, { top: 0 }, [0], null]) {
      const result = sanitizeAnalyzePayload({
        requestId: 'pm-priority-3',
        source: 'https://images.example.test/image.png',
        naturalWidth: 800,
        naturalHeight: 600,
        priority: badValue
      }, sender, extensionId);
      expect(result).not.toHaveProperty('priority');
    }
  });

  it('serializes terminal unavailable errors without a score', () => {
    const response = toUnavailableResponse(Object.assign(new Error('blocked'), { code: 'PRIVATE_TARGET_BLOCKED' }), 'pm-1');
    expect(response).toMatchObject({ ok: false, unavailable: true, code: 'PRIVATE_TARGET_BLOCKED', requestId: 'pm-1' });
    expect(response).not.toHaveProperty('score');
  });
});
