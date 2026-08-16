import { describe, expect, it, vi } from 'vitest';
import {
  HEAVY_IMAGE_ADMISSION_LIMIT,
  HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS,
  HEAVY_IMAGE_ADMISSION_UNUSED_TTL_MS,
  HeavyImageAdmissionController,
  authorizeAndSanitizeHeavyImagePayload,
  createHeavyImageAdmissionOwner,
  requiresHeavyImageAdmission
} from './fallback-admission.js';

const extensionId = 'abcdefghijklmnopabcdefghijklmnop';

describe('global heavy-image admission', () => {
  it('admits exactly two extension-global owners and keeps reservation messages idempotent', () => {
    const controller = controllerFixture();
    const first = owner('one');
    const second = owner('two', { tabId: 8 });
    const firstToken = controller.reserve(first);

    expect(controller.reserve(first)).toBe(firstToken);
    expect(controller.reserve(second)).not.toBe(firstToken);
    expect(controller.size).toBe(HEAVY_IMAGE_ADMISSION_LIMIT);
    expect(() => controller.reserve(owner('three', { frameId: 4 }))).toThrowError(
      expect.objectContaining({ code: 'HEAVY_IMAGE_CAPACITY', retryable: true })
    );
  });

  it('binds tokens to extension, tab, frame, document, and request and blocks replay', () => {
    const controller = controllerFixture();
    const expected = owner('bound');
    const token = controller.reserve(expected);

    for (const forged of [
      { ...expected, extensionId: 'ponmlkjihgfedcbaponmlkjihgfedcba' },
      { ...expected, tabId: expected.tabId + 1 },
      { ...expected, frameId: expected.frameId + 1 },
      { ...expected, documentId: 'new-document' },
      { ...expected, requestId: 'wrong-request' }
    ]) {
      expect(() => controller.beginAttempt(token, forged)).toThrowError(
        expect.objectContaining({ code: 'HEAVY_IMAGE_TOKEN_OWNER_MISMATCH' })
      );
    }

    expect(controller.activate(token, expected)).toBe(true);
    expect(controller.activate(token, expected)).toBe(true);
    const attempt = controller.beginAttempt(token, expected);
    expect(() => controller.beginAttempt(token, expected)).toThrowError(
      expect.objectContaining({ code: 'HEAVY_IMAGE_REPLAY' })
    );
    expect(controller.finishAttempt(attempt, { terminal: true })).toBe(true);
    expect(() => controller.beginAttempt(token, expected)).toThrowError(
      expect.objectContaining({ code: 'HEAVY_IMAGE_REPLAY' })
    );
    expect(controller.release(token, expected)).toBe(true);
    expect(controller.size).toBe(0);
  });

  it('expires an unused reservation at exactly 15 seconds', () => {
    let clock = 1_000;
    const controller = controllerFixture({ now: () => clock });
    const expected = owner('expires');
    const token = controller.reserve(expected);

    clock += HEAVY_IMAGE_ADMISSION_UNUSED_TTL_MS - 1;
    expect(controller.size).toBe(1);
    clock += 1;
    expect(controller.size).toBe(0);
    expect(() => controller.beginAttempt(token, expected)).toThrowError(
      expect.objectContaining({ code: 'HEAVY_IMAGE_TOKEN_INVALID' })
    );
  });

  it('never expires an in-flight body and renews the idle lease across queue retry', () => {
    let clock = 10;
    const controller = controllerFixture({ now: () => clock });
    const expected = owner('active');
    const token = controller.reserve(expected);
    controller.activate(token, expected);
    const attempt = controller.beginAttempt(token, expected);

    clock += 10 * HEAVY_IMAGE_ADMISSION_UNUSED_TTL_MS;
    expect(controller.size).toBe(1);
    expect(controller.finishAttempt(attempt, { terminal: false })).toBe(true);
    for (let heartbeat = 0; heartbeat < 5; heartbeat += 1) {
      clock += HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS - 1;
      expect(controller.activate(token, expected)).toBe(true);
      expect(controller.size).toBe(1);
    }
    expect(controller.release(token, expected)).toBe(true);
    expect(controller.size).toBe(0);
  });

  it('seals a terminal token and recovers it after missed active heartbeats', () => {
    let clock = 20;
    const controller = controllerFixture({ now: () => clock });
    const expected = owner('terminal-expiry');
    const token = controller.reserve(expected);
    controller.activate(token, expected);
    const attempt = controller.beginAttempt(token, expected);
    controller.finishAttempt(attempt, { terminal: true });

    expect(controller.size).toBe(1);
    expect(() => controller.beginAttempt(token, expected)).toThrowError(
      expect.objectContaining({ code: 'HEAVY_IMAGE_REPLAY' })
    );
    clock += HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS;
    expect(controller.size).toBe(0);
    expect(controller.release(token, expected)).toBe(false);
  });

  it('defers cancellation until an active message settles', () => {
    const controller = controllerFixture();
    const expected = owner('cancel');
    const token = controller.reserve(expected);
    controller.activate(token, expected);
    const attempt = controller.beginAttempt(token, expected);

    expect(controller.release(token, expected)).toBe(true);
    expect(controller.size).toBe(1);
    expect(controller.finishAttempt(attempt, { terminal: false })).toBe(true);
    expect(controller.size).toBe(0);
  });

  it('rejects old tokens after a service-worker restart', () => {
    const expected = owner('restart');
    const oldWorker = controllerFixture();
    const token = oldWorker.reserve(expected);
    const freshWorker = controllerFixture();

    expect(() => freshWorker.beginAttempt(token, expected)).toThrowError(
      expect.objectContaining({ code: 'HEAVY_IMAGE_TOKEN_INVALID' })
    );
  });

  it('recovers two abandoned activated owners despite continuing reservation traffic', () => {
    let clock = 0;
    const controller = controllerFixture({ now: () => clock });
    for (const requestId of ['abandoned-one', 'abandoned-two']) {
      const expected = owner(requestId);
      const token = controller.reserve(expected);
      controller.activate(token, expected);
    }

    for (clock = 5_000; clock < HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS; clock += 5_000) {
      expect(() => controller.reserve(owner('waiting'))).toThrowError(
        expect.objectContaining({ code: 'HEAVY_IMAGE_CAPACITY' })
      );
    }
    clock = HEAVY_IMAGE_ADMISSION_ACTIVE_TTL_MS;
    expect(controller.reserve(owner('waiting'))).toMatch(/^[a-f0-9]{32}$/);
    expect(controller.size).toBe(1);
  });

  it('revokes a navigating tab immediately but defers an in-flight revocation until settlement', () => {
    const controller = controllerFixture();
    const idleOwner = owner('idle-tab', { tabId: 20 });
    const activeOwner = owner('active-tab', { tabId: 21 });
    const idleToken = controller.reserve(idleOwner);
    controller.activate(idleToken, idleOwner);
    const activeToken = controller.reserve(activeOwner);
    controller.activate(activeToken, activeOwner);
    const attempt = controller.beginAttempt(activeToken, activeOwner);

    expect(controller.revokeTab(20)).toBe(1);
    expect(controller.size).toBe(1);
    expect(controller.revokeTab(21)).toBe(1);
    expect(controller.size).toBe(1);
    controller.finishAttempt(attempt, { terminal: false });
    expect(controller.size).toBe(0);
  });
});

describe('Chrome sender binding', () => {
  it('derives every owner field from Chrome sender context plus the request ID', () => {
    const sender = {
      id: extensionId,
      tab: { id: 17 },
      frameId: 3,
      documentId: '4cd52b14-4d56-4fe8-89f1-2eca9443ad64'
    };
    expect(createHeavyImageAdmissionOwner(sender, extensionId, 'pm-7')).toEqual({
      extensionId,
      tabId: 17,
      frameId: 3,
      documentId: sender.documentId,
      requestId: 'pm-7'
    });
    expect(() => createHeavyImageAdmissionOwner({ ...sender, documentId: undefined }, extensionId, 'pm-7')).toThrowError(
      expect.objectContaining({ code: 'INVALID_SENDER_CONTEXT' })
    );
  });

  it('requires the shared global gate for data sources and blob fallback bodies only', () => {
    expect(requiresHeavyImageAdmission({ source: 'data:image/jpeg;base64,/9j/' })).toBe(true);
    expect(requiresHeavyImageAdmission({
      source: 'blob:https://example.test/id',
      fallbackDataUrl: 'data:image/jpeg;base64,/9j/'
    })).toBe(true);
    expect(requiresHeavyImageAdmission({ source: 'https://images.example.test/a.jpg' })).toBe(false);
  });

  it('rejects an unclaimed heavy body before calling the expensive sanitizer', () => {
    const controller = controllerFixture();
    const sanitize = vi.fn();
    const wirePayload = {
      requestId: 'raw-heavy',
      source: 'blob:https://example.test/id',
      fallbackDataUrl: `data:image/jpeg;base64,${'A'.repeat(2_000_000)}`
    };
    expect(() => authorizeAndSanitizeHeavyImagePayload({
      controller,
      wirePayload,
      sender: chromeSender(),
      extensionId,
      sanitize
    })).toThrowError(expect.objectContaining({ code: 'HEAVY_IMAGE_TOKEN_INVALID' }));
    expect(sanitize).not.toHaveBeenCalled();
  });

  it('claims before sanitizing and terminal-seals a token when heavy validation fails', () => {
    const controller = controllerFixture();
    const requestId = 'validation-fails';
    const expected = owner(requestId);
    const token = controller.reserve(expected);
    controller.activate(token, expected);
    const validationError = Object.assign(new Error('invalid fallback'), { code: 'INVALID_FALLBACK_ENCODING' });
    const sanitize = vi.fn(() => { throw validationError; });

    expect(() => authorizeAndSanitizeHeavyImagePayload({
      controller,
      wirePayload: {
        requestId,
        source: 'data:image/jpeg;base64,invalid',
        heavyAdmissionToken: token
      },
      sender: chromeSender(),
      extensionId,
      sanitize
    })).toThrow(validationError);
    expect(sanitize).toHaveBeenCalledTimes(1);
    expect(() => controller.beginAttempt(token, expected)).toThrowError(
      expect.objectContaining({ code: 'HEAVY_IMAGE_REPLAY' })
    );
    expect(controller.release(token, expected)).toBe(true);
  });
});

function controllerFixture(overrides = {}) {
  let sequence = 0;
  return new HeavyImageAdmissionController({
    createToken: () => (++sequence).toString(16).padStart(32, '0'),
    ...overrides
  });
}

function owner(requestId, overrides = {}) {
  return {
    extensionId,
    tabId: 7,
    frameId: 0,
    documentId: 'document-one',
    requestId,
    ...overrides
  };
}

function chromeSender(overrides = {}) {
  return {
    id: extensionId,
    tab: { id: 7 },
    frameId: 0,
    documentId: 'document-one',
    ...overrides
  };
}
