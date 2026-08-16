import {
  ImageUnavailableError,
  MAX_IMAGE_PIXELS,
  MAX_IMAGE_SIDE,
  MAX_REMOTE_SOURCE_CHARS,
  assertRemoteSourceAllowed,
  inspectFallbackDataUrl,
  parseImageSource
} from '../offscreen/source-policy.js';

const MAX_REQUEST_ID_CHARS = 128;

export function isTrustedExtensionSender(sender, extensionId) {
  return Boolean(extensionId && sender && sender.id === extensionId);
}

export function sanitizeAnalyzePayload(payload, sender, extensionId) {
  if (!isTrustedExtensionSender(sender, extensionId)) {
    throw unavailable('UNTRUSTED_SENDER', 'The analysis request did not come from this extension.');
  }
  if (!Number.isInteger(sender.tab?.id)) {
    throw unavailable('INVALID_SENDER_CONTEXT', 'Image analysis is available only from a browser tab.');
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw unavailable('INVALID_REQUEST', 'The analysis request is malformed.');
  }

  const pageUrl = trustedPageUrl(sender);
  if (!pageUrl) {
    throw unavailable('INVALID_SENDER_CONTEXT', 'The requesting page URL could not be verified.');
  }

  const requestId = validateRequestId(payload.requestId);
  const source = validateSource(payload.source, pageUrl);
  const naturalWidth = validateClaimedDimension(payload.naturalWidth, 'width');
  const naturalHeight = validateClaimedDimension(payload.naturalHeight, 'height');
  if (naturalWidth * naturalHeight > MAX_IMAGE_PIXELS) {
    throw unavailable('IMAGE_DIMENSIONS_EXCEEDED', 'The claimed image dimensions exceed the pixel safety limit.');
  }

  const sourceUrl = parseImageSource(source);
  const fallbackDataUrl = validateFallbackDataUrl(payload.fallbackDataUrl, sourceUrl, pageUrl);

  return {
    requestId,
    source,
    ...(fallbackDataUrl ? { fallbackDataUrl } : {}),
    naturalWidth,
    naturalHeight,
    // Never accept a pageUrl claimed by the content payload. This value comes
    // only from Chrome's MessageSender and is used for localhost policy checks.
    pageUrl
  };
}

export function toUnavailableResponse(error, requestId) {
  const message = error instanceof Error ? error.message : String(error || 'The image could not be analyzed.');
  const safeRequestId = typeof requestId === 'string'
    && requestId.length <= MAX_REQUEST_ID_CHARS
    && /^[A-Za-z0-9._:-]+$/.test(requestId)
    ? requestId
    : undefined;
  return {
    ok: false,
    unavailable: true,
    code: typeof error?.code === 'string' ? error.code : 'ANALYSIS_UNAVAILABLE',
    retryable: Boolean(error?.retryable),
    requestId: safeRequestId,
    error: message
  };
}

function trustedPageUrl(sender) {
  for (const value of [sender.url, sender.tab?.url]) {
    if (typeof value !== 'string' || value.length > MAX_REMOTE_SOURCE_CHARS) continue;
    try {
      const url = new URL(value);
      // file: is accepted so a local HTML page opened directly (no server at
      // all) can pass Chrome-verified sender identity through to the same
      // local-target policy every other page goes through.
      if (url.protocol === 'http:' || url.protocol === 'https:' || url.protocol === 'file:') return url.href;
    } catch {
      // about:blank and transient sender URLs fall back to the trusted tab URL.
    }
  }
  return null;
}

function validateRequestId(value) {
  if (typeof value !== 'string' || value.length < 1 || value.length > MAX_REQUEST_ID_CHARS) {
    throw unavailable('INVALID_REQUEST_ID', 'The analysis request ID is invalid.');
  }
  if (!/^[A-Za-z0-9._:-]+$/.test(value)) {
    throw unavailable('INVALID_REQUEST_ID', 'The analysis request ID contains unsupported characters.');
  }
  return value;
}

function validateSource(value, pageUrl) {
  const sourceUrl = parseImageSource(value);
  if (sourceUrl.protocol === 'http:' || sourceUrl.protocol === 'https:') {
    assertRemoteSourceAllowed(sourceUrl, pageUrl);
  }
  return value;
}

function validateClaimedDimension(value, label) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0 || value > MAX_IMAGE_SIDE) {
    throw unavailable('INVALID_DIMENSIONS', `The claimed image ${label} is invalid.`);
  }
  return value;
}

function validateFallbackDataUrl(value, sourceUrl, pageUrl) {
  if (sourceUrl.protocol === 'blob:') {
    // A blob: URL's origin is the origin that created it; a page cannot
    // supply a fallback claiming one that belongs to a different origin.
    const page = new URL(pageUrl);
    if (sourceUrl.origin === 'null' || sourceUrl.origin !== page.origin) {
      throw unavailable('BLOB_ORIGIN_MISMATCH', 'The page-local image origin could not be verified.');
    }
    return inspectFallbackDataUrl(value).dataUrl;
  }

  // http(s)/file sources may also carry a fallback body: the content script
  // sends one when it already tried (or, for file:, could never attempt) a
  // direct background fetch and captured the on-page pixels instead. These
  // have no page-relative origin to check the way a page-local blob: URL
  // does -- validateSource() already re-ran the private-network/loopback
  // policy on the http(s) source URL itself above, and a file: source has no
  // network origin at all.
  if (['http:', 'https:', 'file:'].includes(sourceUrl.protocol)) {
    return value === undefined ? undefined : inspectFallbackDataUrl(value).dataUrl;
  }

  if (value !== undefined) {
    throw unavailable('UNEXPECTED_FALLBACK', 'Fallback image bytes are accepted only for blob, file, or HTTP(S) sources.');
  }
  return undefined;
}

function unavailable(code, message) {
  return new ImageUnavailableError(code, message);
}
