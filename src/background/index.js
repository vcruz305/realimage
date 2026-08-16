import { MESSAGE } from '../shared/constants.js';
import { restrictCalibrationDebugRelay } from '../shared/calibration-debug-relay.js';
import { InferenceScheduler, TabBadgeTracker } from './scheduler.js';
import { startupError, waitForOffscreenReceiver } from './offscreen-readiness.js';
import { isTrustedExtensionSender, sanitizeAnalyzePayload, toUnavailableResponse } from './validation.js';
import {
  HEAVY_IMAGE_ADMISSION_UNUSED_TTL_MS,
  HeavyImageAdmissionController,
  authorizeAndSanitizeHeavyImagePayload,
  createHeavyImageAdmissionOwner,
} from './fallback-admission.js';

const OFFSCREEN_PATH = 'src/offscreen/offscreen.html';
// These maps are intentionally worker-lifetime optimizations. Chrome may stop an
// MV3 service worker between events; correctness must not depend on them surviving.
const inferenceScheduler = new InferenceScheduler({ maxCacheEntries: 150 });
const badgeTracker = new TabBadgeTracker({ threshold: 0.65 });
const heavyImageAdmissions = new HeavyImageAdmissionController();
let creatingOffscreen;

chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === 'install') chrome.runtime.openOptionsPage();
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const type = message?.type;
  if (type === MESSAGE.RESERVE_HEAVY_IMAGE) {
    try {
      const owner = createHeavyImageAdmissionOwner(sender, chrome.runtime.id, message?.requestId);
      const admissionToken = heavyImageAdmissions.reserve(owner);
      sendResponse({ ok: true, admissionToken, expiresInMs: HEAVY_IMAGE_ADMISSION_UNUSED_TTL_MS });
    } catch (error) {
      sendResponse(toUnavailableResponse(error, message?.requestId));
    }
    return false;
  }
  if (type === MESSAGE.ACTIVATE_HEAVY_IMAGE) {
    try {
      const owner = createHeavyImageAdmissionOwner(sender, chrome.runtime.id, message?.requestId);
      heavyImageAdmissions.activate(message?.admissionToken, owner);
      sendResponse({ ok: true, activated: true });
    } catch (error) {
      sendResponse(toUnavailableResponse(error, message?.requestId));
    }
    return false;
  }
  if (type === MESSAGE.RELEASE_HEAVY_IMAGE) {
    try {
      const owner = createHeavyImageAdmissionOwner(sender, chrome.runtime.id, message?.requestId);
      const released = heavyImageAdmissions.release(message?.admissionToken, owner);
      sendResponse({ ok: true, released });
    } catch (error) {
      sendResponse(toUnavailableResponse(error, message?.requestId));
    }
    return false;
  }
  if (type === MESSAGE.ANALYZE_IMAGE) {
    let authorized;
    try {
      authorized = authorizeAndSanitizeHeavyImagePayload({
        controller: heavyImageAdmissions,
        wirePayload: message.payload,
        sender,
        extensionId: chrome.runtime.id,
        sanitize: sanitizeAnalyzePayload
      });
    } catch (error) {
      sendResponse(toUnavailableResponse(error, message?.payload?.requestId));
      return false;
    }
    handleAnalyzeWithHeavyAdmission(authorized.payload, sender, authorized.attempt).then(sendResponse).catch((error) => {
      sendResponse(toUnavailableResponse(error, authorized.payload.requestId));
    });
    return true;
  }
  if ([MESSAGE.WARM_MODEL, MESSAGE.MODEL_STATUS].includes(type)) {
    if (!isTrustedExtensionSender(sender, chrome.runtime.id)) {
      sendResponse({ ...toUnavailableResponse(new Error('The request did not come from this extension.')), state: 'error' });
      return false;
    }
    forwardToOffscreen({ type }).then(sendResponse).catch((error) => {
      sendResponse({ ...toUnavailableResponse(error), state: 'error' });
    });
    return true;
  }
});

async function handleAnalyzeWithHeavyAdmission(payload, sender, attempt) {
  try {
    const result = await handleAnalyze(payload, sender);
    if (attempt) heavyImageAdmissions.finishAttempt(attempt, { terminal: !isOffscreenQueueFull(result) });
    return result;
  } catch (error) {
    if (attempt) heavyImageAdmissions.finishAttempt(attempt, { terminal: true });
    throw error;
  }
}

function isOffscreenQueueFull(result) {
  return result?.ok === false
    && result.retryable === true
    && result.code === 'OFFSCREEN_QUEUE_FULL';
}

chrome.tabs?.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status !== 'loading' && !changeInfo.url) return;
  if (changeInfo.status === 'loading') heavyImageAdmissions.revokeTab(tabId);
  badgeTracker.reset(tabId);
  chrome.action.setBadgeText({ tabId, text: '' }).catch(() => {});
});

chrome.tabs?.onRemoved.addListener((tabId) => {
  heavyImageAdmissions.revokeTab(tabId);
  badgeTracker.reset(tabId);
});

async function handleAnalyze(payload, sender) {
  const tabId = sender.tab?.id;
  const badgeToken = await badgeTracker.begin(
    tabId,
    (currentTabId) => chrome.action.getBadgeText({ tabId: currentTabId })
  );
  try {
    const scheduledResult = await inferenceScheduler.run(payload, (ownerPayload) => (
      forwardToOffscreen({ type: MESSAGE.OFFSCREEN_ANALYZE, payload: ownerPayload })
    ));
    const result = restrictCalibrationDebugRelay(scheduledResult, {
      pageUrl: payload.pageUrl,
      source: payload.source
    });
    applyBadgeDecision(tabId, badgeTracker.settle(badgeToken, result));
    return result;
  } catch (error) {
    applyBadgeDecision(tabId, badgeTracker.settle(badgeToken, { ok: false }));
    throw error;
  }
}

async function forwardToOffscreen(message) {
  await ensureOffscreenDocument();
  return chrome.runtime.sendMessage(message);
}

async function ensureOffscreenDocument() {
  if (!creatingOffscreen) {
    creatingOffscreen = createAndVerifyOffscreenDocument().finally(() => {
      creatingOffscreen = undefined;
    });
  }
  await creatingOffscreen;
}

async function createAndVerifyOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_PATH);
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [offscreenUrl]
  });

  if (!contexts.length) {
    try {
      await chrome.offscreen.createDocument({
        url: OFFSCREEN_PATH,
        reasons: ['WORKERS'],
        justification: 'Run the bundled local image classifier inside an isolated extension document.'
      });
    } catch (error) {
      throw startupError(
        'OFFSCREEN_CREATE_FAILED',
        'Chrome could not create the local inference document.',
        error
      );
    }
  }

  await waitForOffscreenReceiver({ sendMessage: (message) => chrome.runtime.sendMessage(message) });
}

function applyBadgeDecision(tabId, decision) {
  if (!Number.isInteger(tabId) || !decision) return;
  chrome.action.setBadgeText({ tabId, text: decision.text }).catch(() => {});
  chrome.action.setBadgeBackgroundColor({ tabId, color: decision.color }).catch(() => {});
}
