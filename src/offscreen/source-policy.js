export const MAX_ENCODED_BYTES = 20 * 1024 * 1024;
export const MAX_FALLBACK_BYTES = 2 * 1024 * 1024;
// This is a wire cap (including the data: prefix), not a decoded-byte estimate.
// Heavy-message admission bounds concurrency separately; the per-message limit
// prevents any one page-controlled source from monopolizing Stable Chrome's
// JSON serializer or the scheduler's hashing buffers.
export const MAX_DATA_URL_CHARS = 2 * 1024 * 1024;
export const MAX_REMOTE_SOURCE_CHARS = 32_768;
export const MAX_IMAGE_SIDE = 16_384;
export const MAX_IMAGE_PIXELS = 40_000_000;
export const ACQUISITION_TIMEOUT_MS = 12_000;

export const FALLBACK_DATA_URL_PREFIX = 'data:image/jpeg;base64,';
export const MAX_FALLBACK_DATA_URL_CHARS = FALLBACK_DATA_URL_PREFIX.length
  + 4 * Math.ceil(MAX_FALLBACK_BYTES / 3);

const LOCAL_HOST_SUFFIXES = ['.localhost'];
const NON_PUBLIC_HOST_SUFFIXES = ['.internal', '.local', '.lan', '.home', '.home.arpa'];

export class ImageUnavailableError extends Error {
  constructor(code, message, { retryable = false } = {}) {
    super(message);
    this.name = 'ImageUnavailableError';
    this.code = code;
    this.retryable = retryable;
  }
}

export function parseImageSource(source) {
  if (typeof source !== 'string' || source.length === 0) {
    throw unavailable('INVALID_SOURCE', 'The image source is missing.');
  }

  // Enforce the Chrome JSON-message wire bounds before URL parsing. Content
  // scripts call this before runtime.sendMessage(), and privileged contexts
  // repeat it rather than relying on the untrusted first-hop check.
  if (source.slice(0, 5).toLowerCase() === 'data:') {
    if (source.length > MAX_DATA_URL_CHARS) {
      throw unavailable('IMAGE_TOO_LARGE', 'The embedded image exceeds the safe input limit.');
    }
  } else if (source.length > MAX_REMOTE_SOURCE_CHARS) {
    throw unavailable('SOURCE_TOO_LONG', 'The image URL exceeds the safe request limit.');
  }

  let url;
  try {
    url = new URL(source);
  } catch {
    throw unavailable('INVALID_SOURCE', 'The image source is not a valid URL.');
  }

  // file: is accepted here (the wire-level source form) even though neither
  // fetchEncodedImage() nor a background fetch can ever retrieve a file: body.
  // A file: image can only ever reach analysis through the canvas-capture
  // fallback path, which supplies its bytes as fallbackDataUrl instead.
  if (!['http:', 'https:', 'data:', 'blob:', 'file:'].includes(url.protocol)) {
    throw unavailable('UNSUPPORTED_PROTOCOL', `Images using ${url.protocol || 'that'} URLs are unavailable.`);
  }
  if ((url.protocol === 'http:' || url.protocol === 'https:') && (url.username || url.password)) {
    throw unavailable('URL_CREDENTIALS_BLOCKED', 'Image URLs containing credentials are unavailable.');
  }
  return url;
}

/**
 * Validate the JSON-safe wire representation used only when a content script
 * must copy a page-local blob URL. The content script always renders that image
 * to JPEG, so accepting one exact canonical form keeps the message and MIME
 * surface deliberately narrow on Stable Chrome's JSON serializer.
 */
export function inspectFallbackDataUrl(value) {
  if (typeof value !== 'string' || value.length === 0) {
    throw unavailable('BLOB_UNAVAILABLE', 'The page-local image bytes are missing.');
  }
  if (value.length > MAX_FALLBACK_DATA_URL_CHARS) {
    throw unavailable('IMAGE_TOO_LARGE', 'The copied image exceeds the encoded-body safety limit.');
  }
  if (!value.startsWith(FALLBACK_DATA_URL_PREFIX)) {
    throw unavailable('UNSUPPORTED_MIME', 'Copied images must use the canonical JPEG data-URL format.');
  }

  const base64 = value.slice(FALLBACK_DATA_URL_PREFIX.length);
  const padding = canonicalBase64Padding(base64);
  if (padding === -1) {
    throw unavailable('INVALID_FALLBACK_ENCODING', 'The copied image has invalid base64 encoding.');
  }

  const byteLength = (base64.length / 4) * 3 - padding;
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength > MAX_FALLBACK_BYTES) {
    throw unavailable('IMAGE_TOO_LARGE', 'The copied image is empty or exceeds the encoded-body safety limit.');
  }
  return {
    dataUrl: value,
    mimeType: 'image/jpeg',
    byteLength
  };
}

function canonicalBase64Padding(value) {
  if (value.length === 0 || value.length % 4 !== 0) return -1;
  const padding = value.endsWith('==') ? 2 : value.endsWith('=') ? 1 : 0;
  const contentLength = value.length - padding;
  for (let index = 0; index < contentLength; index += 1) {
    if (base64Sextet(value.charCodeAt(index)) === -1) return -1;
  }
  for (let index = contentLength; index < value.length; index += 1) {
    if (value.charCodeAt(index) !== 61) return -1;
  }

  // Requiring zero unused bits prevents multiple strings from representing the
  // same bytes and makes the wire form genuinely canonical.
  const finalSextet = base64Sextet(value.charCodeAt(contentLength - 1));
  if ((padding === 2 && (finalSextet & 0x0f) !== 0) || (padding === 1 && (finalSextet & 0x03) !== 0)) {
    return -1;
  }
  return padding;
}

function base64Sextet(code) {
  if (code >= 65 && code <= 90) return code - 65;
  if (code >= 97 && code <= 122) return code - 71;
  if (code >= 48 && code <= 57) return code + 4;
  if (code === 43) return 62;
  if (code === 47) return 63;
  return -1;
}

export function assertRemoteSourceAllowed(url, trustedPageUrl) {
  if (!(url instanceof URL) || !['http:', 'https:'].includes(url.protocol)) {
    throw unavailable('UNSUPPORTED_PROTOCOL', 'Only HTTP and HTTPS remote images are supported.');
  }

  const targetKind = classifyHostname(url.hostname);
  if (targetKind === 'public') return;

  // A loopback or private-network/non-public target is reachable only from a
  // page that is itself local (loopback, a private-network/non-public host,
  // or a file: page). Same-kind local pages no longer need the exact same
  // origin as the target -- an http://localhost:8000 page may show an
  // http://127.0.0.1:9000 image, and a 192.168.1.50 page may show an image
  // hosted on a different private-network host/port. A genuinely PUBLIC page
  // can never reach either kind of target; that boundary does not move.
  if (isPageLocal(parseTrustedPageUrl(trustedPageUrl))) return;

  if (targetKind === 'loopback') {
    throw unavailable(
      'LOCAL_TARGET_BLOCKED',
      'Localhost images are available only from a local page (localhost, a private-network address, or a local file page).'
    );
  }

  throw unavailable(
    'PRIVATE_TARGET_BLOCKED',
    'Private-network and non-public image targets are available only from a local page.'
  );
}

export function validateDecodedDimensions(width, height) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw unavailable('INVALID_DIMENSIONS', 'The image dimensions could not be verified.');
  }
  if (width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE || width * height > MAX_IMAGE_PIXELS) {
    throw unavailable(
      'IMAGE_DIMENSIONS_EXCEEDED',
      `The decoded image exceeds the ${MAX_IMAGE_SIDE}px side or ${MAX_IMAGE_PIXELS.toLocaleString()}-pixel safety limit.`
    );
  }
  return { width, height };
}

export function classifyHostname(hostname) {
  const host = String(hostname || '').toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '');
  if (!host) return 'non-public';
  if (host === 'localhost' || LOCAL_HOST_SUFFIXES.some((suffix) => host.endsWith(suffix))) return 'loopback';

  const ipv4 = parseIpv4(host);
  if (ipv4) {
    if (ipv4[0] === 127) return 'loopback';
    return isPublicIpv4(ipv4) ? 'public' : 'non-public';
  }

  if (host.includes(':')) {
    if (host === '::1') return 'loopback';
    const mapped = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/.exec(host);
    if (mapped) return classifyHostname(mapped[1]);
    return isPublicIpv6(host) ? 'public' : 'non-public';
  }

  if (!host.includes('.') || NON_PUBLIC_HOST_SUFFIXES.some((suffix) => host.endsWith(suffix))) {
    return 'non-public';
  }
  return 'public';
}

function parseTrustedPageUrl(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:', 'file:'].includes(url.protocol) ? url : null;
  } catch {
    return null;
  }
}

// A file: page is always local by definition, regardless of its (often empty,
// occasionally UNC-style) hostname -- classifyHostname is not consulted for it.
function isPageLocal(page) {
  if (!page) return false;
  if (page.protocol === 'file:') return true;
  const kind = classifyHostname(page.hostname);
  return kind === 'loopback' || kind === 'non-public';
}

function parseIpv4(host) {
  if (!/^\d{1,3}(?:\.\d{1,3}){3}$/.test(host)) return null;
  const octets = host.split('.').map(Number);
  return octets.every((octet) => octet >= 0 && octet <= 255) ? octets : null;
}

function isPublicIpv4([a, b, c]) {
  if (a === 0 || a === 10 || a === 127 || a >= 224) return false;
  if (a === 100 && b >= 64 && b <= 127) return false;
  if (a === 169 && b === 254) return false;
  if (a === 172 && b >= 16 && b <= 31) return false;
  if (a === 192 && b === 168) return false;
  if (a === 192 && b === 0 && (c === 0 || c === 2)) return false;
  if (a === 192 && b === 88 && c === 99) return false;
  if (a === 198 && (b === 18 || b === 19)) return false;
  if (a === 198 && b === 51 && c === 100) return false;
  if (a === 203 && b === 0 && c === 113) return false;
  return true;
}

function isPublicIpv6(host) {
  const normalized = host.toLowerCase();
  if (normalized === '::' || normalized.startsWith('fc') || normalized.startsWith('fd')) return false;
  if (/^fe[89ab]/.test(normalized) || normalized.startsWith('ff')) return false;
  if (normalized.startsWith('2001:db8:')) return false;
  // Globally routable unicast space is currently 2000::/3. Reject everything
  // else instead of treating an unfamiliar local/reserved address as public.
  return normalized.startsWith('2') || normalized.startsWith('3');
}

function unavailable(code, message) {
  return new ImageUnavailableError(code, message);
}
