const DEFAULT_CACHE_ENTRIES = 150;

export class LruCache {
  constructor(maxEntries = DEFAULT_CACHE_ENTRIES) {
    if (!Number.isInteger(maxEntries) || maxEntries < 1) {
      throw new TypeError('maxEntries must be a positive integer.');
    }
    this.maxEntries = maxEntries;
    this.entries = new Map();
  }

  get size() {
    return this.entries.size;
  }

  get(key) {
    if (!this.entries.has(key)) return undefined;
    const value = this.entries.get(key);
    this.entries.delete(key);
    this.entries.set(key, value);
    return value;
  }

  set(key, value) {
    this.entries.delete(key);
    this.entries.set(key, value);
    while (this.entries.size > this.maxEntries) {
      this.entries.delete(this.entries.keys().next().value);
    }
  }
}

/**
 * This scheduler deliberately keeps only worker-lifetime state. MV3 may stop the
 * worker at any time, so cache hits and in-flight coalescing are optimizations,
 * never correctness requirements or claims of cross-restart persistence.
 */
export class InferenceScheduler {
  constructor({ maxCacheEntries = DEFAULT_CACHE_ENTRIES, keyFactory = createCacheKey } = {}) {
    this.cache = new LruCache(maxCacheEntries);
    this.inFlight = new Map();
    this.keyFactory = keyFactory;
  }

  async run(payload, infer) {
    const key = await this.keyFactory(payload);
    const requestId = payload?.requestId;
    const persistResult = isContentAddressedInput(payload);
    const cached = persistResult ? this.cache.get(key) : undefined;
    if (cached !== undefined) return forCaller(cached, requestId, { cached: true });

    let shared = this.inFlight.get(key);
    const coalesced = shared !== undefined;
    if (!shared) {
      shared = Promise.resolve().then(() => infer(payload)).then((rawResult) => {
        const result = withoutCallerFields(rawResult);
        if (result?.ok && persistResult) this.cache.set(key, result);
        return result;
      });
      this.inFlight.set(key, shared);
      const clearInFlight = () => {
        if (this.inFlight.get(key) === shared) this.inFlight.delete(key);
      };
      // Both handlers consume the cleanup branch, while callers still receive
      // the original promise and its original fulfilled/rejected outcome.
      void shared.then(clearInFlight, clearInFlight);
    }

    const result = await shared;
    return forCaller(result, requestId, { cached: false, coalesced });
  }
}

/**
 * Retain completed results only when the input bytes are part of the request.
 * Remote HTTP(S) resources can change without changing their URL, so they may
 * share concurrent work but must be fetched and inferred again later.
 */
export function isContentAddressedInput(payload) {
  let protocol;
  try {
    protocol = new URL(String(payload?.source || '')).protocol;
  } catch {
    return false;
  }

  if (protocol === 'data:') return true;
  return protocol === 'blob:' && typeof payload?.fallbackDataUrl === 'string';
}

/**
 * Aggregates tab results so completion order cannot erase an AI verdict. The
 * caller supplies Chrome's current badge text when a tab is first seen; this
 * lets a restarted worker recover a browser-held AI flag without pretending
 * its previous pending-request or cache maps survived.
 */
export class TabBadgeTracker {
  constructor({ threshold = 0.65 } = {}) {
    this.threshold = threshold;
    this.states = new Map();
    this.initializers = new Map();
    this.generations = new Map();
  }

  async begin(tabId, readCurrentBadge = async () => '') {
    if (!Number.isInteger(tabId)) return undefined;
    const state = await this.getOrCreateState(tabId, readCurrentBadge);
    state.pending += 1;
    return { tabId, state };
  }

  settle(token, result) {
    if (!token || this.states.get(token.tabId) !== token.state) return undefined;
    const state = token.state;
    state.pending = Math.max(0, state.pending - 1);
    if (result?.ok && Number.isFinite(result.score) && result.score >= this.threshold) {
      state.hasAi = true;
    }

    if (state.hasAi) return { text: 'AI', color: '#D84E42' };
    // Preserve the existing badge until every result for this tab is known.
    if (state.pending > 0) return undefined;
    return { text: '', color: '#0F8A70' };
  }

  reset(tabId) {
    if (!Number.isInteger(tabId)) return;
    this.generations.set(tabId, (this.generations.get(tabId) || 0) + 1);
    this.states.delete(tabId);
    this.initializers.delete(tabId);
  }

  async getOrCreateState(tabId, readCurrentBadge) {
    const existing = this.states.get(tabId);
    if (existing) return existing;

    const pendingInitializer = this.initializers.get(tabId);
    if (pendingInitializer) return pendingInitializer;

    const generation = this.generations.get(tabId) || 0;
    let initializer;
    initializer = (async () => {
      let badgeText = '';
      try {
        badgeText = await readCurrentBadge(tabId);
      } catch {
        // A missing/stale tab is equivalent to a clean badge.
      }
      const state = { pending: 0, hasAi: badgeText === 'AI', generation };
      if ((this.generations.get(tabId) || 0) === generation) this.states.set(tabId, state);
      return state;
    })().finally(() => {
      if (this.initializers.get(tabId) === initializer) this.initializers.delete(tabId);
    });
    this.initializers.set(tabId, initializer);
    return initializer;
  }
}

export async function createCacheKey(payload, digest = sha256Hex) {
  const source = String(payload?.source || '');
  const width = boundedDimension(payload?.naturalWidth);
  const height = boundedDimension(payload?.naturalHeight);
  const fallbackDataUrl = typeof payload?.fallbackDataUrl === 'string' ? payload.fallbackDataUrl : '';
  // Digest fallback content separately so the cache/in-flight key contains no
  // raw image data and building the small identity string does not duplicate a
  // potentially multi-megabyte data URL.
  const fallbackDigest = fallbackDataUrl ? await digest(fallbackDataUrl) : 'none';
  const fingerprint = await digest(`${source}\n${width}x${height}\n${fallbackDigest}`);
  return `image-v3:${fingerprint}`;
}

function boundedDimension(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return 0;
  return Math.min(Math.floor(number), 1_000_000_000);
}

async function sha256Hex(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

function withoutCallerFields(result) {
  if (!result || typeof result !== 'object') return result;
  const { requestId: _requestId, cached: _cached, coalesced: _coalesced, ...shared } = result;
  return shared;
}

function forCaller(result, requestId, flags) {
  if (!result || typeof result !== 'object') return result;
  return { ...result, ...flags, requestId };
}
