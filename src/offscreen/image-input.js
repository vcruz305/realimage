import {
  ACQUISITION_TIMEOUT_MS,
  ImageUnavailableError,
  MAX_ENCODED_BYTES,
  MAX_FALLBACK_BYTES,
  assertRemoteSourceAllowed,
  inspectFallbackDataUrl,
  parseImageSource,
  validateDecodedDimensions
} from './source-policy.js';

const MIME_ALIASES = new Map([
  ['image/png', 'png'],
  ['image/jpeg', 'jpeg'],
  ['image/jpg', 'jpeg'],
  ['image/pjpeg', 'jpeg'],
  ['image/webp', 'webp'],
  ['image/avif', 'avif']
]);
const MAX_REDIRECTS = 3;

export async function acquireImageInput(
  { source, fallbackDataUrl, pageUrl },
  {
    fetchImpl = globalThis.fetch,
    maxBytes = MAX_ENCODED_BYTES,
    timeoutMs = ACQUISITION_TIMEOUT_MS
  } = {}
) {
  const sourceUrl = parseImageSource(source);
  let encoded;

  if (sourceUrl.protocol === 'blob:') {
    const fallback = inspectFallbackDataUrl(fallbackDataUrl);
    encoded = await fetchEncodedImage(parseImageSource(fallback.dataUrl), {
      fetchImpl,
      maxBytes: Math.min(maxBytes, MAX_FALLBACK_BYTES),
      timeoutMs,
      pageUrl
    });
  } else if (fallbackDataUrl !== undefined) {
    // An http(s)/file source arriving with a fallback body: the content
    // script already tried (or, for file:, could never attempt) a direct
    // fetch and captured the on-page pixels instead. A background context
    // can never fetch file: at all, and http(s) reaching this branch already
    // failed its own direct attempt, so use the captured bytes instead of
    // trying (or retrying) a network fetch of the original source.
    if (!['http:', 'https:', 'file:'].includes(sourceUrl.protocol)) {
      throw unavailable('UNEXPECTED_FALLBACK', 'A fallback body is allowed only for a page-local blob, file, or HTTP(S) source.');
    }
    const fallback = inspectFallbackDataUrl(fallbackDataUrl);
    encoded = await fetchEncodedImage(parseImageSource(fallback.dataUrl), {
      fetchImpl,
      maxBytes: Math.min(maxBytes, MAX_FALLBACK_BYTES),
      timeoutMs,
      pageUrl
    });
  } else {
    if (sourceUrl.protocol === 'http:' || sourceUrl.protocol === 'https:') {
      assertRemoteSourceAllowed(sourceUrl, pageUrl);
    }
    encoded = await fetchEncodedImage(sourceUrl, { fetchImpl, maxBytes, timeoutMs, pageUrl });
  }

  const envelope = inspectImageEnvelope(encoded.bytes, encoded.mimeType);
  validateDecodedDimensions(envelope.width, envelope.height);
  const blob = new Blob([encoded.bytes], { type: envelope.mimeType });
  return {
    buffer: encoded.bytes.buffer,
    blob,
    mimeType: envelope.mimeType,
    format: envelope.format,
    width: envelope.width,
    height: envelope.height
  };
}

export async function fetchEncodedImage(
  sourceUrl,
  { fetchImpl = globalThis.fetch, maxBytes = MAX_ENCODED_BYTES, timeoutMs = ACQUISITION_TIMEOUT_MS, pageUrl } = {}
) {
  if (!(sourceUrl instanceof URL) || !['http:', 'https:', 'data:'].includes(sourceUrl.protocol)) {
    throw unavailable('UNSUPPORTED_PROTOCOL', 'This image source cannot be fetched.');
  }
  if (sourceUrl.protocol === 'http:' || sourceUrl.protocol === 'https:') {
    assertRemoteSourceAllowed(sourceUrl, pageUrl);
  }
  if (typeof fetchImpl !== 'function') {
    throw unavailable('FETCH_UNAVAILABLE', 'This browser cannot fetch the image.');
  }

  const controller = new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    let response;
    let currentUrl = sourceUrl;
    let redirectsFollowed = 0;
    while (true) {
      try {
        response = await fetchImpl(currentUrl.href, {
          cache: 'default',
          credentials: 'omit',
          redirect: 'manual',
          referrerPolicy: 'no-referrer',
          signal: controller.signal
        });
      } catch (error) {
        if (timedOut || controller.signal.aborted) {
          throw unavailable('FETCH_TIMEOUT', 'The image did not arrive before the acquisition timeout.', { retryable: true });
        }
        throw unavailable(
          'FETCH_BLOCKED',
          'The image fetch failed, so it was not analyzed.',
          { retryable: true, cause: error }
        );
      }

      if (!response) throw unavailable('HTTP_UNAVAILABLE', 'The image server returned no response.', { retryable: true });
      if (response.type === 'opaqueredirect') {
        // Chrome may intentionally hide Location for a manual cross-origin
        // redirect. Following automatically would contact the destination
        // before we can reject a private address, so opaque redirects fail shut.
        throw unavailable('OPAQUE_REDIRECT_BLOCKED', 'The browser hid the redirect target, so the image was not analyzed.');
      }
      if (response.status >= 300 && response.status < 400) {
        if (redirectsFollowed >= MAX_REDIRECTS) {
          throw unavailable('REDIRECT_LIMIT', `The image exceeded the ${MAX_REDIRECTS}-redirect safety limit.`);
        }
        const location = response.headers?.get?.('location');
        if (!location) {
          throw unavailable('OPAQUE_REDIRECT_BLOCKED', 'The redirect target was unavailable, so the image was not analyzed.');
        }
        const nextUrl = parseRedirectTarget(location, currentUrl);
        assertRemoteSourceAllowed(nextUrl, pageUrl);
        if (currentUrl.protocol === 'https:' && nextUrl.protocol !== 'https:') {
          throw unavailable('INSECURE_REDIRECT_BLOCKED', 'A secure image URL redirected to an insecure target.');
        }
        currentUrl = nextUrl;
        redirectsFollowed += 1;
        continue;
      }
      break;
    }

    if (!response.ok) {
      throw unavailable('HTTP_UNAVAILABLE', `The image server returned HTTP ${response.status || 'an error'}.`, { retryable: true });
    }

    // Manual redirects are the primary SSRF boundary. Checking response.url as
    // well keeps custom fetch implementations and future browser behavior from
    // silently following something we did not validate first.
    if (response.url) {
      const finalUrl = parseImageSource(response.url);
      if (finalUrl.protocol === 'http:' || finalUrl.protocol === 'https:') {
        assertRemoteSourceAllowed(finalUrl, pageUrl);
      }
      if (withoutFragment(finalUrl) !== withoutFragment(currentUrl)) {
        throw unavailable('UNVALIDATED_REDIRECT_BLOCKED', 'The image request followed an unvalidated redirect.');
      }
    }

    const declaredLength = parseContentLength(response.headers?.get?.('content-length'));
    if (declaredLength > maxBytes) {
      throw unavailable('IMAGE_TOO_LARGE', `The encoded image exceeds the ${formatBytes(maxBytes)} safety limit.`);
    }
    const mimeType = normalizeMime(response.headers?.get?.('content-type'));
    const bytes = await readBoundedStream(response.body, { maxBytes, signal: controller.signal });
    return { bytes, mimeType };
  } finally {
    clearTimeout(timeout);
  }
}

export function inspectImageEnvelope(bytesLike, declaredMimeType = '') {
  const bytes = bytesLike instanceof Uint8Array ? bytesLike : new Uint8Array(bytesLike);
  const declared = normalizeMime(declaredMimeType);
  const declaredFormat = declared ? MIME_ALIASES.get(declared) : undefined;
  if (declared && !declaredFormat) {
    throw unavailable('UNSUPPORTED_MIME', `Images declared as ${declared} are unavailable.`);
  }

  const detected = detectImage(bytes);
  if (!detected) {
    throw unavailable('UNSUPPORTED_IMAGE', 'The image signature or dimensions are unsupported.');
  }
  if (declaredFormat && declaredFormat !== detected.format) {
    throw unavailable('MIME_MISMATCH', 'The image MIME type does not match its encoded signature.');
  }

  validateDecodedDimensions(detected.width, detected.height);
  return {
    ...detected,
    mimeType: canonicalMime(detected.format)
  };
}

export async function readBoundedStream(stream, { maxBytes = MAX_ENCODED_BYTES, signal } = {}) {
  if (!stream?.getReader) throw unavailable('EMPTY_BODY', 'The image response did not contain a readable body.');
  const reader = stream.getReader();
  const chunks = [];
  let total = 0;
  let rejectAbort;
  const aborted = new Promise((_, reject) => { rejectAbort = reject; });
  const onAbort = () => rejectAbort(unavailable('FETCH_TIMEOUT', 'The image stream timed out.', { retryable: true }));
  if (signal) {
    if (signal.aborted) onAbort();
    else signal.addEventListener('abort', onAbort, { once: true });
  }

  try {
    while (true) {
      const { done, value } = await Promise.race([reader.read(), aborted]);
      if (done) break;
      const chunk = value instanceof Uint8Array ? value : new Uint8Array(value);
      total += chunk.byteLength;
      if (total > maxBytes) {
        await reader.cancel().catch(() => {});
        throw unavailable('IMAGE_TOO_LARGE', `The encoded image exceeds the ${formatBytes(maxBytes)} safety limit.`);
      }
      chunks.push(chunk);
    }
  } finally {
    if (signal) signal.removeEventListener('abort', onAbort);
    reader.releaseLock?.();
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function detectImage(bytes) {
  return detectPng(bytes) || detectJpeg(bytes) || detectWebp(bytes) || detectAvif(bytes);
}

function detectPng(bytes) {
  if (bytes.length < 24) return null;
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (!signature.every((value, index) => bytes[index] === value) || ascii(bytes, 12, 16) !== 'IHDR') return null;
  const view = dataView(bytes);
  return { format: 'png', width: view.getUint32(16), height: view.getUint32(20) };
}

function detectJpeg(bytes) {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  const sofMarkers = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
  const view = dataView(bytes);
  let offset = 2;
  while (offset + 3 < bytes.length) {
    while (offset < bytes.length && bytes[offset] !== 0xff) offset += 1;
    while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
    if (offset >= bytes.length) break;
    const marker = bytes[offset++];
    if (marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01) continue;
    if (offset + 2 > bytes.length) break;
    const length = view.getUint16(offset);
    if (length < 2 || offset + length > bytes.length) break;
    if (sofMarkers.has(marker) && length >= 7) {
      return { format: 'jpeg', height: view.getUint16(offset + 3), width: view.getUint16(offset + 5) };
    }
    offset += length;
  }
  return null;
}

function detectWebp(bytes) {
  if (bytes.length < 30 || ascii(bytes, 0, 4) !== 'RIFF' || ascii(bytes, 8, 12) !== 'WEBP') return null;
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const type = ascii(bytes, offset, offset + 4);
    const length = readUint32LE(bytes, offset + 4);
    const dataOffset = offset + 8;
    if (dataOffset + length > bytes.length) return null;
    if (type === 'VP8X' && length >= 10) {
      return {
        format: 'webp',
        width: 1 + readUint24LE(bytes, dataOffset + 4),
        height: 1 + readUint24LE(bytes, dataOffset + 7)
      };
    }
    if (type === 'VP8L' && length >= 5 && bytes[dataOffset] === 0x2f) {
      const b1 = bytes[dataOffset + 1];
      const b2 = bytes[dataOffset + 2];
      const b3 = bytes[dataOffset + 3];
      const b4 = bytes[dataOffset + 4];
      return {
        format: 'webp',
        width: 1 + b1 + ((b2 & 0x3f) << 8),
        height: 1 + (b2 >> 6) + (b3 << 2) + ((b4 & 0x0f) << 10)
      };
    }
    if (
      type === 'VP8 ' && length >= 10
      && bytes[dataOffset + 3] === 0x9d && bytes[dataOffset + 4] === 0x01 && bytes[dataOffset + 5] === 0x2a
    ) {
      return {
        format: 'webp',
        width: readUint16LE(bytes, dataOffset + 6) & 0x3fff,
        height: readUint16LE(bytes, dataOffset + 8) & 0x3fff
      };
    }
    offset = dataOffset + length + (length % 2);
  }
  return null;
}

function detectAvif(bytes) {
  if (bytes.length < 20 || ascii(bytes, 4, 8) !== 'ftyp') return null;
  const boxLength = dataView(bytes).getUint32(0);
  if (boxLength < 16 || boxLength > bytes.length) return null;
  const brands = ascii(bytes, 8, Math.min(boxLength, 64));
  if (!brands.includes('avif') && !brands.includes('avis')) return null;

  let largest = null;
  for (let offset = 4; offset + 16 <= bytes.length; offset += 1) {
    if (bytes[offset] !== 0x69 || bytes[offset + 1] !== 0x73 || bytes[offset + 2] !== 0x70 || bytes[offset + 3] !== 0x65) continue;
    const start = offset - 4;
    const length = dataView(bytes).getUint32(start);
    if (length < 20 || start + length > bytes.length) continue;
    const width = dataView(bytes).getUint32(offset + 8);
    const height = dataView(bytes).getUint32(offset + 12);
    validateDecodedDimensions(width, height);
    if (!largest || width * height > largest.width * largest.height) largest = { width, height };
  }
  return largest ? { format: 'avif', ...largest } : null;
}

function parseContentLength(value) {
  if (value === null || value === undefined || value === '') return 0;
  const number = Number(value);
  return Number.isSafeInteger(number) && number >= 0 ? number : 0;
}

function normalizeMime(value) {
  return String(value || '').split(';', 1)[0].trim().toLowerCase();
}

function canonicalMime(format) {
  return format === 'jpeg' ? 'image/jpeg' : `image/${format}`;
}

function dataView(bytes) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function ascii(bytes, start, end) {
  return String.fromCharCode(...bytes.subarray(start, end));
}

function readUint16LE(bytes, offset) {
  return bytes[offset] | (bytes[offset + 1] << 8);
}

function readUint24LE(bytes, offset) {
  return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
}

function readUint32LE(bytes, offset) {
  return dataView(bytes).getUint32(offset, true);
}

function formatBytes(value) {
  return `${Math.floor(value / (1024 * 1024))} MiB`;
}

function withoutFragment(url) {
  const hash = url.href.indexOf('#');
  return hash === -1 ? url.href : url.href.slice(0, hash);
}

function parseRedirectTarget(location, currentUrl) {
  let next;
  try {
    next = new URL(location, currentUrl);
  } catch {
    throw unavailable('INVALID_REDIRECT', 'The image server returned an invalid redirect target.');
  }
  if (!['http:', 'https:'].includes(next.protocol)) {
    throw unavailable('UNSUPPORTED_REDIRECT', 'The image redirected to an unsupported URL protocol.');
  }
  return parseImageSource(next.href);
}

function unavailable(code, message, options) {
  const error = new ImageUnavailableError(code, message, options);
  if (options?.cause) error.cause = options.cause;
  return error;
}
