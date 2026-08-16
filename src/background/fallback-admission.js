import { ImageUnavailableError } from '../offscreen/source-policy.js';

export const HEAVY_IMAGE_ADMISSION_LIMIT = 2;
export const HEAVY_IMAGE_ADMISSION_UNUSED_TTL_MS = 15_000;
export const HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS = 30_000;

const TOKEN_PATTERN = /^[a-f0-9]{32}$/;
const ID_PATTERN = /^[A-Za-z0-9._:-]+$/;
const MAX_OWNER_ID_CHARS = 128;

/**
 * Build the identity Chrome, rather than the page payload, vouches for. A blob
 * reservation belongs to one content-script document and one request only;
 * navigation, another frame, or another extension cannot reuse it.
 */
export function createHeavyImageAdmissionOwner(sender, extensionId, requestId) {
  if (!extensionId || sender?.id !== extensionId) {
    throw unavailable('UNTRUSTED_SENDER', 'The fallback request did not come from this extension.');
  }
  const tabId = sender.tab?.id;
  const frameId = sender.frameId;
  const documentId = sender.documentId;
  if (!Number.isInteger(tabId) || tabId < 0 || !Number.isInteger(frameId) || frameId < 0) {
    throw unavailable('INVALID_SENDER_CONTEXT', 'The fallback request is missing its tab or frame identity.');
  }
  if (!boundedId(documentId) || !boundedId(requestId)) {
    throw unavailable('INVALID_SENDER_CONTEXT', 'The fallback request is missing its document or request identity.');
  }
  return Object.freeze({ extensionId, tabId, frameId, documentId, requestId });
}

export function requiresHeavyImageAdmission(payload) {
  if (!payload || typeof payload !== 'object') return false;
  if (typeof payload.fallbackDataUrl === 'string' && payload.fallbackDataUrl.length > 0) return true;
  return typeof payload.source === 'string' && payload.source.slice(0, 5).toLowerCase() === 'data:';
}

/**
 * Claim a heavyweight token before validation inspects a multi-megabyte data
 * URL/fallback body. This ordering is security-significant: an unauthenticated
 * heavy request must fail before canonical-base64 scans or scheduler hashing.
 */
export function authorizeAndSanitizeHeavyImagePayload({
  controller,
  wirePayload,
  sender,
  extensionId,
  sanitize
}) {
  if (!controller || typeof controller.beginAttempt !== 'function' || typeof controller.finishAttempt !== 'function') {
    throw new TypeError('controller must support heavyweight attempts.');
  }
  if (typeof sanitize !== 'function') throw new TypeError('sanitize must be a function.');

  let attempt;
  if (requiresHeavyImageAdmission(wirePayload)) {
    const owner = createHeavyImageAdmissionOwner(sender, extensionId, wirePayload?.requestId);
    attempt = controller.beginAttempt(wirePayload?.heavyAdmissionToken, owner);
  } else if (wirePayload?.heavyAdmissionToken !== undefined) {
    throw unavailable(
      'UNEXPECTED_HEAVY_IMAGE_ADMISSION',
      'A heavy-image admission token is allowed only with a data URL or page-local blob copy.'
    );
  }

  try {
    return { payload: sanitize(wirePayload, sender, extensionId), attempt };
  } catch (error) {
    if (attempt) controller.finishAttempt(attempt, { terminal: true });
    throw error;
  }
}

/**
 * A worker-lifetime, extension-global admission controller for live heavy
 * preparations and IPC. It intentionally does not persist tokens: after an
 * MV3 worker restart every old token is rejected before application-level
 * validation/hashing. A suspended renderer may retain its own page/body bytes
 * after a lease expires, but its thawed sender must re-activate before every
 * IPC attempt and therefore cannot send that body through this protocol.
 */
export class HeavyImageAdmissionController {
  constructor({
    limit = HEAVY_IMAGE_ADMISSION_LIMIT,
    unusedTtlMs = HEAVY_IMAGE_ADMISSION_UNUSED_TTL_MS,
    activeTtlMs = HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS,
    now = Date.now,
    createToken = randomToken
  } = {}) {
    if (!Number.isInteger(limit) || limit < 1) throw new TypeError('limit must be a positive integer.');
    if (!Number.isFinite(unusedTtlMs) || unusedTtlMs <= 0) throw new TypeError('unusedTtlMs must be positive.');
    if (!Number.isFinite(activeTtlMs) || activeTtlMs <= unusedTtlMs) {
      throw new TypeError('activeTtlMs must be greater than unusedTtlMs.');
    }
    if (typeof now !== 'function' || typeof createToken !== 'function') {
      throw new TypeError('now and createToken must be functions.');
    }
    this.limit = limit;
    this.unusedTtlMs = unusedTtlMs;
    this.activeTtlMs = activeTtlMs;
    this.now = now;
    this.createToken = createToken;
    this.entries = new Map();
    this.tokensByOwner = new Map();
    this.attemptSequence = 0;
  }

  get size() {
    this.sweep();
    return this.entries.size;
  }

  reserve(owner) {
    const normalizedOwner = normalizeOwner(owner);
    this.sweep();
    const ownerKey = serializeOwner(normalizedOwner);
    const existingToken = this.tokensByOwner.get(ownerKey);
    const existing = existingToken && this.entries.get(existingToken);
    // Reservation messages are idempotent. A lost response cannot consume a
    // second slot for the same document/request or extend its original lease.
    if (existing) return existing.token;

    if (this.entries.size >= this.limit) {
      throw unavailable(
        'HEAVY_IMAGE_CAPACITY',
        'The local heavy-image slots are busy. Retry the small reservation request.',
        { retryable: true }
      );
    }

    const token = this.uniqueToken();
    const entry = {
      token,
      owner: normalizedOwner,
      ownerKey,
      expiresAt: this.now() + this.unusedTtlMs,
      activated: false,
      inFlightAttempt: undefined,
      terminal: false,
      releaseRequested: false
    };
    this.entries.set(token, entry);
    this.tokensByOwner.set(ownerKey, token);
    return token;
  }

  activate(token, owner) {
    const normalizedOwner = normalizeOwner(owner);
    this.sweep();
    const entry = this.requireOwnedEntry(token, normalizedOwner);
    if (entry.terminal) {
      throw unavailable('HEAVY_IMAGE_REPLAY', 'The heavy-image admission token cannot be activated again.');
    }
    // Activation is idempotent so content can preflight both before preparing
    // a blob body and immediately before the large message. Activated entries
    // are renewable. In-flight attempts never expire; preparation and retry
    // gaps expire only after multiple missed heartbeats, so dead documents
    // cannot hold both global slots forever.
    entry.activated = true;
    entry.expiresAt = entry.inFlightAttempt === undefined
      ? this.now() + this.activeTtlMs
      : Number.POSITIVE_INFINITY;
    return true;
  }

  beginAttempt(token, owner) {
    const normalizedOwner = normalizeOwner(owner);
    this.sweep();
    const entry = this.requireOwnedEntry(token, normalizedOwner);
    if (!entry.activated) {
      throw unavailable('HEAVY_IMAGE_NOT_ACTIVATED', 'The heavy-image admission token was not activated before the large message.');
    }
    if (entry.terminal) {
      throw unavailable('HEAVY_IMAGE_REPLAY', 'The heavy-image admission token has already reached a terminal result.');
    }
    if (entry.inFlightAttempt !== undefined) {
      throw unavailable('HEAVY_IMAGE_REPLAY', 'The heavy-image admission token is already in use.');
    }
    const attempt = Object.freeze({ token, sequence: ++this.attemptSequence });
    entry.inFlightAttempt = attempt.sequence;
    // Never expire a token while its large message/inference is active. This
    // is what keeps the two-slot bound true even during a long cold start.
    entry.expiresAt = Number.POSITIVE_INFINITY;
    return attempt;
  }

  finishAttempt(attempt, { terminal = false } = {}) {
    if (!attempt || typeof attempt !== 'object') return false;
    const entry = this.entries.get(attempt.token);
    if (!entry || entry.inFlightAttempt !== attempt.sequence) return false;
    entry.inFlightAttempt = undefined;
    if (entry.releaseRequested) {
      this.deleteEntry(entry);
    } else {
      // The content script owns the encoded string until it receives the
      // response and sends RELEASE. Keep counting this slot during that final
      // cross-context hop; otherwise a third capture could overlap briefly.
      // OFFSCREEN_QUEUE_FULL remains retryable with this token, while terminal
      // attempts are sealed against replay. The heartbeat renews the bounded
      // idle lease while the content still owns a prepared body/retry loop.
      entry.terminal = Boolean(terminal);
      entry.expiresAt = this.now() + this.activeTtlMs;
    }
    return true;
  }

  release(token, owner) {
    const normalizedOwner = normalizeOwner(owner);
    this.sweep();
    const entry = this.entries.get(validateToken(token));
    if (!entry) return false; // Idempotent after a terminal background result.
    assertSameOwner(entry.owner, normalizedOwner);
    if (entry.inFlightAttempt !== undefined) {
      // A cancellation can race an already-delivered analysis message. Defer
      // deletion until that attempt settles so an active body stays counted.
      entry.releaseRequested = true;
    } else {
      this.deleteEntry(entry);
    }
    return true;
  }

  revokeTab(tabId) {
    if (!Number.isInteger(tabId) || tabId < 0) return 0;
    let revoked = 0;
    for (const entry of [...this.entries.values()]) {
      if (entry.owner.tabId !== tabId) continue;
      revoked += 1;
      if (entry.inFlightAttempt !== undefined) entry.releaseRequested = true;
      else this.deleteEntry(entry);
    }
    return revoked;
  }

  sweep() {
    const currentTime = this.now();
    for (const entry of this.entries.values()) {
      if (entry.inFlightAttempt === undefined && entry.expiresAt <= currentTime) {
        this.deleteEntry(entry);
      }
    }
  }

  requireOwnedEntry(token, owner) {
    const entry = this.entries.get(validateToken(token));
    if (!entry) {
      throw unavailable('HEAVY_IMAGE_TOKEN_INVALID', 'The heavy-image admission token is missing, expired, or from an earlier worker.');
    }
    assertSameOwner(entry.owner, owner);
    return entry;
  }

  uniqueToken() {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const token = validateToken(this.createToken());
      if (!this.entries.has(token)) return token;
    }
    throw unavailable('HEAVY_IMAGE_TOKEN_FAILURE', 'A unique heavy-image admission token could not be created.');
  }

  deleteEntry(entry) {
    if (this.entries.get(entry.token) !== entry) return;
    this.entries.delete(entry.token);
    if (this.tokensByOwner.get(entry.ownerKey) === entry.token) this.tokensByOwner.delete(entry.ownerKey);
  }
}

function randomToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return [...bytes].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function normalizeOwner(owner) {
  if (!owner || typeof owner !== 'object') throw unavailable('INVALID_SENDER_CONTEXT', 'Fallback ownership is missing.');
  const { extensionId, tabId, frameId, documentId, requestId } = owner;
  if (
    !boundedId(extensionId)
    || !Number.isInteger(tabId)
    || tabId < 0
    || !Number.isInteger(frameId)
    || frameId < 0
    || !boundedId(documentId)
    || !boundedId(requestId)
  ) {
    throw unavailable('INVALID_SENDER_CONTEXT', 'Fallback ownership is malformed.');
  }
  return Object.freeze({ extensionId, tabId, frameId, documentId, requestId });
}

function boundedId(value) {
  return typeof value === 'string'
    && value.length >= 1
    && value.length <= MAX_OWNER_ID_CHARS
    && ID_PATTERN.test(value);
}

function validateToken(token) {
  if (typeof token !== 'string' || !TOKEN_PATTERN.test(token)) {
    throw unavailable('HEAVY_IMAGE_TOKEN_INVALID', 'The heavy-image admission token is invalid.');
  }
  return token;
}

function serializeOwner(owner) {
  return JSON.stringify([owner.extensionId, owner.tabId, owner.frameId, owner.documentId, owner.requestId]);
}

function assertSameOwner(expected, actual) {
  if (serializeOwner(expected) !== serializeOwner(actual)) {
    throw unavailable('HEAVY_IMAGE_TOKEN_OWNER_MISMATCH', 'The heavy-image admission token belongs to another document or request.');
  }
}

function unavailable(code, message, options) {
  return new ImageUnavailableError(code, message, options);
}
