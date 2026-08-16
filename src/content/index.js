import { MESSAGE } from '../shared/constants.js';
import { takeCalibrationDebugRelay } from '../shared/calibration-debug-relay.js';
import { formatScorePercent } from '../shared/score-display.js';
import { getSettings } from '../shared/settings.js';
import { MAX_FALLBACK_BYTES, inspectFallbackDataUrl, parseImageSource } from '../offscreen/source-policy.js';
import { sendAnalysisWithQueueRetry } from './analysis-retry.js';
import { isCanvasCaptureRetryable } from './canvas-capture-retry.js';
import {
  applyBadgeImageLink,
  applyBadgeResultContract,
  clearBadgeImageLink
} from './badge-result-contract.js';
import { summarizeSemanticDeclarations } from './declaration-ui.js';
import { findBadgePlacement } from './badge-position.js';
import {
  beginDetailDialogRender,
  closeDetailDialog,
  commitDetailDialogRender,
  DETAIL_ACTION_SELECTOR,
  DETAIL_CLOSE_SELECTOR,
  focusDetailControl,
  isEscapeDialogKey
} from './detail-dialog.js';
import { withHeavyImageAdmission } from './fallback-admission.js';
import {
  cancelInvisiblePendingRecords,
  collectOpenTree,
  findTerminalEvictionCandidate,
  isCurrentRecord,
  isVisibleBadgeAnchor,
  isVisiblyRenderedImage,
  releaseRecord,
  reserveRecordWithTerminalEviction
} from './lifecycle.js';
import { createVisibilityRefreshController } from './visibility-refresh.js';
import './overlay.css';

const OBSERVATION_MARGIN_PX = 180;
const MAX_PENDING_ADMISSIONS = 200;
const MAX_VISIBILITY_MUTATION_ROOTS = 16;
const MAX_VISIBILITY_SCAN_NODES = 2048;
const MAX_VISIBILITY_RETRY_IMAGES = 200;

const pageState = {
  status: 'idle',
  scanned: 0,
  aiCount: 0,
  realCount: 0,
  errors: 0,
  results: []
};

const records = new Map();
const observed = new WeakSet();
const shadowObservers = new WeakMap();
const shadowTreatmentStyles = new WeakMap();
const pendingAdmissions = new Set();
let settings = await getSettings();
let requestSequence = 0;
let calibrationRuntimeSessionId;
let updateFrame;
let observationRefreshQueued = false;
let detailReturnFocus;

const overlayLayer = document.createElement('div');
overlayLayer.id = 'proofmark-overlay-layer';
overlayLayer.setAttribute('aria-live', 'polite');
document.documentElement.append(overlayLayer);

const detailPanel = createDetailPanel();
overlayLayer.append(detailPanel);

const intersectionObserver = new IntersectionObserver(handleIntersections, {
  rootMargin: `${OBSERVATION_MARGIN_PX}px 0px`,
  threshold: 0.01
});

const visibilityRefresh = createVisibilityRefreshController({
  belongsToDocument: belongsToThisDocument,
  hasRecord: (image) => records.has(image),
  observeImage,
  observeShadowRoot,
  shouldRefresh: () => settings.enabled,
  maxRoots: MAX_VISIBILITY_MUTATION_ROOTS,
  maxNodes: MAX_VISIBILITY_SCAN_NODES,
  maxImages: MAX_VISIBILITY_RETRY_IMAGES
});

discoverImages(document);
const mutationObserver = new MutationObserver(handleMutations);
mutationObserver.observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeFilter: ['src', 'srcset', 'sizes', 'hidden', 'class', 'style']
});

function handleMutations(mutations) {
  let visibilityMayHaveChanged = false;
  for (const mutation of mutations) {
    if (mutation.target === overlayLayer || overlayLayer.contains(mutation.target)) continue;
    if (mutation.type === 'attributes') {
      const sourceChanged = ['src', 'srcset', 'sizes'].includes(mutation.attributeName);
      if (sourceChanged && mutation.target instanceof HTMLImageElement) {
        resetImage(mutation.target);
      } else {
        visibilityMayHaveChanged = true;
        schedulePositions();
        visibilityRefresh.schedule(mutation.target);
      }
    }
    for (const node of mutation.removedNodes) {
      if (node instanceof Element && !belongsToThisDocument(node)) forgetRemovedImages(node);
    }
    for (const node of mutation.addedNodes) {
      if (node instanceof Element && belongsToThisDocument(node)) discoverImages(node);
    }
    if (mutation.type === 'childList') visibilityMayHaveChanged = true;
  }
  if (visibilityMayHaveChanged) {
    cancelInvisiblePendingRecords(records, {
      onCancel: (image, record) => {
        if (records.get(image) === record) resetImage(image);
      }
    });
  }
}

window.addEventListener('scroll', handleViewportChange, { passive: true });
window.addEventListener('resize', handleViewportChange, { passive: true });
document.addEventListener('keydown', handleDetailEscape, true);
document.addEventListener('visibilitychange', () => {
  schedulePositions();
  if (!document.hidden) discoverImages(document, { refresh: true });
});

chrome.storage.onChanged.addListener(async (changes, area) => {
  if (area !== 'local' || !changes.settings) return;
  settings = await getSettings();
  if (!settings.enabled) {
    pendingAdmissions.clear();
    visibilityRefresh.clear();
    hideAllOverlays();
  } else {
    enforceRecordLimit();
    renderAllResults();
    discoverImages(document, { refresh: true });
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === MESSAGE.GET_PAGE_STATE) {
    sendResponse({ ...pageState, results: pageState.results.slice(-12) });
  }
  if (message.type === MESSAGE.RESCAN_PAGE) {
    rescanPage();
    sendResponse({ ok: true });
  }
});

function discoverImages(root, { refresh = false } = {}) {
  if (!settings.enabled) return;
  const { images, shadowRoots } = collectOpenTree(root);
  for (const shadowRoot of shadowRoots) observeShadowRoot(shadowRoot);
  for (const image of images) {
    if (!belongsToThisDocument(image)) continue;
    observeImage(image, refresh);
  }
}

function observeShadowRoot(root) {
  if (shadowObservers.has(root) || !belongsToThisDocument(root.host)) return;
  const observer = new MutationObserver(handleMutations);
  observer.observe(root, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'srcset', 'sizes', 'hidden', 'class', 'style']
  });
  shadowObservers.set(root, observer);
}

function observeImage(image, refresh = false) {
  if (!belongsToThisDocument(image)) return;
  if (!observed.has(image)) {
    observed.add(image);
    image.addEventListener('load', handleImageLoad);
  }
  if (refresh) intersectionObserver.unobserve(image);
  intersectionObserver.observe(image);
}

function handleImageLoad(event) {
  const image = event.currentTarget;
  if (!(image instanceof HTMLImageElement)) return;
  if (!belongsToThisDocument(image)) {
    stopObservingImage(image);
    return;
  }
  const record = records.get(image);
  if (record && image.currentSrc !== record.source) releaseImageRecord(image, record);
  observeImage(image, true);
}

function handleIntersections(entries) {
  let viewportSlotMayHaveOpened = false;
  for (const entry of entries) {
    if (!(entry.target instanceof HTMLImageElement)) continue;
    const image = entry.target;
    if (!belongsToThisDocument(image)) {
      stopObservingImage(image);
      releaseImageRecord(image);
      continue;
    }
    if (!entry.isIntersecting) {
      pendingAdmissions.delete(image);
      visibilityRefresh.forget(image);
      viewportSlotMayHaveOpened = true;
      continue;
    }
    if (isAnalyzable(image)) {
      visibilityRefresh.forget(image);
      analyze(image);
    } else if (isVisibilityDeferredCandidate(image)) {
      visibilityRefresh.remember(image);
    } else {
      visibilityRefresh.forget(image);
    }
  }
  if (viewportSlotMayHaveOpened) scheduleObservationRefresh();
}

function isAnalyzable(image) {
  if (!settings.enabled || !belongsToThisDocument(image) || records.has(image)) return false;
  if (!image.complete || !image.currentSrc) return false;
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (Math.min(width, height) < settings.minimumDimension) return false;
  if (width * height < settings.minimumDimension ** 2) return false;
  if (!isVisiblyRenderedImage(image)) return false;
  const rect = image.getBoundingClientRect();
  return rect.width >= 48 && rect.height >= 48;
}

function isVisibilityDeferredCandidate(image) {
  if (!settings.enabled || !belongsToThisDocument(image) || records.has(image)) return false;
  if (!image.complete || !image.currentSrc) return false;
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (Math.min(width, height) < settings.minimumDimension) return false;
  if (width * height < settings.minimumDimension ** 2) return false;
  const rect = image.getBoundingClientRect();
  return rect.width >= 48 && rect.height >= 48 && !isVisiblyRenderedImage(image);
}

async function analyze(image) {
  if (!isAnalyzable(image)) return;
  const id = `pm-${Date.now().toString(36)}-${(++requestSequence).toString(36)}`;
  const record = {
    id,
    image,
    source: image.currentSrc,
    status: 'analyzing',
    result: null,
    badge: createBadge(id)
  };
  const admitted = reserveRecordWithTerminalEviction(records, image, record, settings.maxImagesPerPage, {
    isOffViewport: isOutsideObservationWindow,
    isProtected: (candidate) => !detailPanel.hidden && detailPanel.dataset.anchor === candidate.id,
    isVisible: isVisiblyRenderedImage,
    onEvict: (candidateImage, candidateRecord) => {
      releaseImageRecord(candidateImage, candidateRecord, { scheduleRefresh: false });
    }
  });
  if (!admitted) {
    queuePendingAdmission(image);
    return;
  }
  pendingAdmissions.delete(image);
  applyBadgeImageLink(record.image, record.badge);
  overlayLayer.append(record.badge);
  pageState.status = 'scanning';
  schedulePositions();

  try {
    // Fail before Chrome's JSON serializer sees a page-controlled URL. The
    // privileged background repeats the same bounds; this first-hop check is
    // what prevents a hostile >64 MiB data URL from reaching runtime messaging.
    const sourceUrl = parseImageSource(record.source);
    const shouldContinue = () => settings.enabled
      && isCurrentRecord(records, image, record, document)
      && isVisiblyRenderedImage(image);
    const sendAnalysis = (
      extraPayload = {},
      activateBeforeSend = async () => {}
    ) => sendAnalysisWithQueueRetry(async () => {
      await activateBeforeSend();
      if (!shouldContinue()) return undefined;
      return chrome.runtime.sendMessage({
        type: MESSAGE.ANALYZE_IMAGE,
        payload: {
          requestId: id,
          source: record.source,
          naturalWidth: image.naturalWidth,
          naturalHeight: image.naturalHeight,
          ...extraPayload
        }
      });
    }, { shouldContinue });
    // blob: and file: sources can never be fetched by the background itself
    // -- blob: URLs are page-local, and a background context cannot reach
    // file: at all -- so both always go straight to a canvas capture of the
    // already-rendered <img>. data: sources already carry their own bytes and
    // only need the heavy-admission slot, not a capture. http(s) tries the
    // ordinary direct fetch first, and retries once via the same capture path
    // only if the background could not reach the target itself (policy block
    // or a real fetch failure) -- see isCanvasCaptureRetryable(). A capture
    // that fails (e.g. a tainted canvas from a cross-origin image without
    // CORS headers) throws and surfaces as a normal analysis error, exactly
    // as an unrecoverable blob: capture failure already does today.
    const runCanvasCaptureAndSend = () => runWithHeavyImageAdmission(record, {
      requestId: id,
      sendControlMessage: (message) => chrome.runtime.sendMessage(message),
      prepare: async () => {
        const fallbackDataUrl = await captureImageDataUrl(image);
        if (!fallbackDataUrl) {
          throw new Error('The page-local image could not be copied within the safe byte limit.');
        }
        return fallbackDataUrl;
      },
      sendWithToken: ({ admissionToken, prepared, activateBeforeSend }) => sendAnalysis({
        fallbackDataUrl: prepared,
        heavyAdmissionToken: admissionToken
      }, activateBeforeSend),
      shouldContinue
    });

    let result;
    if (sourceUrl.protocol === 'data:') {
      result = await runWithHeavyImageAdmission(record, {
        requestId: id,
        sendControlMessage: (message) => chrome.runtime.sendMessage(message),
        sendWithToken: ({ admissionToken, activateBeforeSend }) => sendAnalysis({
          heavyAdmissionToken: admissionToken
        }, activateBeforeSend),
        shouldContinue
      });
    } else if (sourceUrl.protocol === 'blob:' || sourceUrl.protocol === 'file:') {
      result = await runCanvasCaptureAndSend();
    } else {
      result = await sendAnalysis();
      if (isCanvasCaptureRetryable(result)) {
        result = await runCanvasCaptureAndSend();
      }
    }
    if (!shouldContinue()) {
      if (records.get(image) === record) resetImage(image);
      return;
    }
    if (!result?.ok) throw new Error(result?.error || 'The image could not be analyzed.');
    const { publicResult, relay } = takeCalibrationDebugRelay(result, {
      pageUrl: location.href,
      source: record.source
    });
    if (relay) {
      if (calibrationRuntimeSessionId !== relay.runtimeReady.sessionId) {
        console.info('realimage:calibration-runtime-ready', JSON.stringify(relay.runtimeReady));
        calibrationRuntimeSessionId = relay.runtimeReady.sessionId;
      }
      console.info('realimage:calibration-analysis', JSON.stringify(relay.analysis));
    }
    record.status = 'complete';
    record.result = publicResult;
    pageState.scanned += 1;
    if (publicResult.score >= settings.threshold) pageState.aiCount += 1;
    else pageState.realCount += 1;
    pageState.results.push(toPublicResult(record));
    pageState.results = pageState.results.slice(-50);
    renderRecord(record);
  } catch (error) {
    if (!isCurrentRecord(records, image, record, document) || !isVisiblyRenderedImage(image)) {
      if (records.get(image) === record) resetImage(image);
      return;
    }
    record.status = 'error';
    record.error = error instanceof Error ? error.message : String(error);
    pageState.errors += 1;
    record.badge.className = 'proofmark-badge proofmark-badge--error';
    record.badge.innerHTML = '<span class="proofmark-badge__mark">!</span><span>Unreadable</span>';
    record.badge.title = record.error;
    record.badge.setAttribute('aria-label', `RealImage could not read this image: ${record.error}`);
    schedulePositions();
  } finally {
    updatePageStatus();
    // A request that finishes after it left the viewport may now be the safe
    // terminal eviction candidate a pending visible image was waiting for.
    scheduleObservationRefresh();
  }
}

async function runWithHeavyImageAdmission(record, options) {
  try {
    return await withHeavyImageAdmission({
      ...options,
      onAdmissionReserved(releaseAdmission) {
        record.releaseHeavyAdmission = releaseAdmission;
        if (record.cancelled) void releaseAdmission();
      }
    });
  } finally {
    delete record.releaseHeavyAdmission;
  }
}

function createBadge(id) {
  const badge = document.createElement('button');
  badge.className = 'proofmark-badge proofmark-badge--loading';
  badge.type = 'button';
  badge.dataset.proofmarkId = id;
  badge.setAttribute('aria-label', 'RealImage is analyzing this image');
  badge.innerHTML = '<span class="proofmark-badge__pulse"></span><span>Checking</span>';
  badge.addEventListener('click', () => {
    const record = [...records.values()].find((candidate) => candidate.id === id);
    if (record?.result) {
      showDetails(record, {
        focusSelector: DETAIL_CLOSE_SELECTOR,
        returnFocus: badge
      });
    }
  });
  return badge;
}

function renderRecord(record) {
  const score = record.result.score;
  const isAi = score >= settings.threshold;
  const displayScore = formatScorePercent(score, settings.threshold);
  const declaration = summarizeSemanticDeclarations(record.result.evidence);
  const showDeclarationBadge = !isAi && Boolean(declaration);
  applyBadgeImageLink(record.image, record.badge);
  record.badge.hidden = !isAi && !showDeclarationBadge && !settings.showRealScores;
  record.badge.className = `proofmark-badge proofmark-badge--${showDeclarationBadge ? 'declared' : isAi ? 'ai' : 'real'}`;
  applyBadgeResultContract(record.badge, {
    isAi,
    displayScore,
    decisionThreshold: settings.threshold,
    declaration
  });
  if (showDeclarationBadge) {
    record.badge.innerHTML = `
      <span class="proofmark-badge__mark">M</span>
      <span class="proofmark-badge__score">${escapeHtml(declaration.badgeText)}</span>
    `;
    record.badge.setAttribute(
      'aria-label',
      `${declaration.badgeText}. Detector model score ${displayScore} percent, below the decision threshold. Open details.`
    );
  } else {
    record.badge.innerHTML = `
      <span class="proofmark-badge__mark">${isAi ? 'AI' : 'R'}</span>
      <span class="proofmark-badge__score">${displayScore}% AI</span>
    `;
    record.badge.setAttribute('aria-label', `${displayScore} percent AI-likelihood score. Open details.`);
  }
  applyMediaTreatment(record);
  schedulePositions();
}

function renderAllResults() {
  for (const record of records.values()) {
    record.badge.hidden = false;
    if (record.result) renderRecord(record);
  }
  schedulePositions();
}

function schedulePositions() {
  if (updateFrame) return;
  updateFrame = requestAnimationFrame(() => {
    updateFrame = undefined;
    for (const record of records.values()) positionBadge(record);
    if (!detailPanel.hidden) positionDetailPanel();
  });
}

function positionBadge(record) {
  if (!record.image.isConnected || record.badge.hidden || !settings.enabled || !isVisibleBadgeAnchor(record.image)) {
    record.badge.style.display = 'none';
    return;
  }
  const rect = record.image.getBoundingClientRect();
  if (rect.bottom < 0 || rect.top > innerHeight || rect.right < 0 || rect.left > innerWidth || rect.width < 36 || rect.height < 36) {
    record.badge.style.display = 'none';
    return;
  }
  // Hide the badge from painted hit-testing while retaining layout so its
  // dimensions can be measured. Other RealImage UI is also ignored by the
  // blocker reader, preventing badges from displacing one another.
  record.badge.style.display = 'flex';
  record.badge.style.visibility = 'hidden';
  const placement = withTemporarilyVisibleHideAnchor(record.image, () => findBadgePlacement({
    imageRect: rect,
    badgeWidth: record.badge.offsetWidth,
    badgeHeight: record.badge.offsetHeight,
    viewportWidth: innerWidth,
    viewportHeight: innerHeight,
    readBlocker: (candidate) => readPaintedBlocker(candidate, record.image)
  }));
  if (!placement) {
    record.badge.style.display = 'none';
    record.badge.style.visibility = '';
    return;
  }
  record.badge.style.top = `${placement.top}px`;
  record.badge.style.left = `${placement.left}px`;
  record.badge.style.visibility = '';
}

function readPaintedBlocker(candidate, image) {
  const points = candidateSamplePoints(candidate);
  for (const [x, y] of points) {
    const stack = typeof document.elementsFromPoint === 'function' ? document.elementsFromPoint(x, y) : [];
    const topPageElement = stack.find((element) => !overlayLayer.contains(element));
    if (!topPageElement || isComposedAncestor(topPageElement, image) || isComposedAncestor(image, topPageElement)) continue;
    const blocker = nearestViewportChrome(topPageElement, candidate) || topPageElement;
    const blockerRect = blocker.getBoundingClientRect?.();
    if (blockerRect && blockerRect.width > 0 && blockerRect.height > 0) return blockerRect;
    // The painted stack proved the candidate is covered even if the covering
    // node did not expose useful geometry. Remove this candidate fail-closed.
    return candidate;
  }
  return undefined;
}

function candidateSamplePoints(rect) {
  const inset = 2;
  const centerX = (rect.left + rect.right) / 2;
  const centerY = (rect.top + rect.bottom) / 2;
  return [
    [rect.left + inset, rect.top + inset],
    [centerX, rect.top + inset],
    [rect.right - inset, rect.top + inset],
    [rect.left + inset, centerY],
    [centerX, centerY],
    [rect.right - inset, centerY],
    [rect.left + inset, rect.bottom - inset],
    [centerX, rect.bottom - inset],
    [rect.right - inset, rect.bottom - inset]
  ];
}

function nearestViewportChrome(element, candidate) {
  let current = element;
  for (let depth = 0; current && depth < 16; depth += 1) {
    let style;
    try {
      style = getComputedStyle(current);
    } catch {
      return undefined;
    }
    if (style?.position === 'fixed' || style?.position === 'sticky') {
      const rect = current.getBoundingClientRect?.();
      if (rectanglesOverlap(rect, candidate)) return current;
    }
    current = composedParent(current);
  }
  return undefined;
}

function rectanglesOverlap(first, second) {
  return first
    && first.left < second.right
    && first.right > second.left
    && first.top < second.bottom
    && first.bottom > second.top;
}

function isComposedAncestor(ancestor, node) {
  const seen = new Set();
  let current = node;
  for (let depth = 0; current && depth < 64; depth += 1) {
    if (current === ancestor) return true;
    if (seen.has(current)) return false;
    seen.add(current);
    current = composedParent(current);
  }
  return false;
}

function composedParent(node) {
  if (node?.assignedSlot) return node.assignedSlot;
  if (node?.parentElement) return node.parentElement;
  const root = typeof node?.getRootNode === 'function' ? node.getRootNode() : undefined;
  return root?.host;
}

function withTemporarilyVisibleHideAnchor(image, task) {
  if (image?.dataset?.proofmarkTreatment !== 'hide') return task();
  const snapshot = shadowTreatmentStyles.get(image);
  delete image.dataset.proofmarkTreatment;
  if (snapshot) restoreStyleProperty(image.style, 'visibility', snapshot.visibility);
  try {
    // Synchronous hit-testing forces style resolution before the treatment is
    // restored, so the hidden image can still identify its painted occluders.
    return task();
  } finally {
    image.dataset.proofmarkTreatment = 'hide';
    if (snapshot) image.style.setProperty('visibility', 'hidden', 'important');
  }
}

function createDetailPanel() {
  const panel = document.createElement('section');
  panel.className = 'proofmark-detail';
  panel.hidden = true;
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'Image authenticity details');
  panel.addEventListener('click', (event) => {
    if (event.target.closest('[data-proofmark-reveal]')) {
      const record = [...records.values()].find((candidate) => candidate.id === panel.dataset.anchor);
      if (record) {
        record.revealed = !record.revealed;
        applyMediaTreatment(record);
        // innerHTML replacement destroys the activated control, so restore
        // keyboard focus to its newly rendered counterpart.
        showDetails(record, { focusSelector: DETAIL_ACTION_SELECTOR });
      }
      return;
    }
    if (event.target.closest('[data-proofmark-close]')) closeDetails();
  });
  return panel;
}

function handleDetailEscape(event) {
  if (detailPanel.hidden || !isEscapeDialogKey(event)) return;
  event.preventDefault();
  event.stopPropagation();
  closeDetails();
}

function showDetails(record, { focusSelector, returnFocus } = {}) {
  if (returnFocus) detailReturnFocus = returnFocus;
  beginDetailDialogRender(detailPanel);
  const { result } = record;
  const isAi = result.score >= settings.threshold;
  const displayScore = formatScorePercent(result.score, settings.threshold);
  const declaration = summarizeSemanticDeclarations(result.evidence);
  const evidence = result.evidence.length
    ? result.evidence.map((item) => `<li><span>${escapeHtml(item.label)}</span><small>${escapeHtml(item.kind)}</small></li>`).join('')
    : '<li><span>No embedded generator or provenance markers found</span><small>model only</small></li>';
  const declarationDetails = declaration
    ? `<section class="proofmark-detail__declaration proofmark-detail__declaration--${escapeHtml(declaration.claim)}" data-proofmark-declaration-details>
        <strong data-proofmark-declaration-title>${escapeHtml(declaration.detailTitle)}</strong>
        <p data-proofmark-declaration-body>${escapeHtml(declaration.detailBody)}</p>
        <small data-proofmark-declaration-sources>${escapeHtml(declaration.sources.map((source) => source.label).join(' · '))}</small>
      </section>`
    : '';
  const treatmentButton = isAi && settings.aiImageAction !== 'label'
    ? `<button class="proofmark-detail__action" type="button" data-proofmark-reveal>${record.revealed ? `Reapply ${settings.aiImageAction}` : 'Reveal this image'}</button>`
    : '';
  const inferenceTiming = result.timings?.inferenceMs == null ? '' : ` · inference ${result.timings.inferenceMs} ms`;
  const activeThreads = result.runtime?.activeThreads;
  const threadTiming = activeThreads ? ` · ${activeThreads} thread${activeThreads === 1 ? '' : 's'}` : '';
  detailPanel.innerHTML = `
    <button class="proofmark-detail__close" type="button" data-proofmark-close aria-label="Close details">×</button>
    <p class="proofmark-detail__eyebrow">Local image check</p>
    <div class="proofmark-detail__score proofmark-detail__score--${isAi ? 'ai' : 'real'}">
      <strong>${displayScore}%</strong>
      <span>AI likelihood score</span>
    </div>
    <p class="proofmark-detail__verdict">Model result: ${isAi ? 'Likely AI-generated' : 'Likely real'}</p>
    <div class="proofmark-detail__meter"><i style="width:${Math.round(result.score * 100)}%"></i><b style="left:${Math.round(settings.threshold * 100)}%"></b></div>
    <div class="proofmark-detail__legend"><span>Real</span><span>${Math.round(settings.threshold * 100)}% threshold</span><span>AI</span></div>
    ${declarationDetails}
    <ul class="proofmark-detail__evidence">${evidence}</ul>
    ${treatmentButton}
    <p class="proofmark-detail__meta">${result.dimensions.width}×${result.dimensions.height} · ${escapeHtml(result.format.toUpperCase())} · total ${result.elapsedMs} ms${inferenceTiming} · ${escapeHtml(result.backend)}${threadTiming}</p>
    <p class="proofmark-detail__privacy">Pixels stayed in this browser. Scores are signals, not proof.</p>
  `;
  if (!commitDetailDialogRender(detailPanel, record.id)) return;
  positionDetailPanel();
  if (focusSelector) focusDetailControl(detailPanel, focusSelector);
}

function closeDetails({ restoreFocus = true } = {}) {
  const returnFocus = detailReturnFocus;
  detailReturnFocus = undefined;
  closeDetailDialog(detailPanel, returnFocus, { restoreFocus });
}

function positionDetailPanel() {
  const anchorId = detailPanel.dataset.anchor;
  const record = [...records.values()].find((candidate) => candidate.id === anchorId);
  if (!record) return;
  const rect = record.image.getBoundingClientRect();
  const width = Math.min(340, innerWidth - 20);
  detailPanel.style.width = `${width}px`;
  const left = Math.min(innerWidth - width - 10, Math.max(10, rect.right - width));
  const estimatedHeight = 390;
  const top = rect.bottom + 10 + estimatedHeight < innerHeight ? rect.bottom + 10 : Math.max(10, rect.top - estimatedHeight - 10);
  detailPanel.style.left = `${left}px`;
  detailPanel.style.top = `${top}px`;
}

async function captureImageDataUrl(image) {
  try {
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 1024 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    canvas.getContext('2d').drawImage(image, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.94));
    if (!(blob instanceof Blob) || blob.size <= 0 || blob.size > MAX_FALLBACK_BYTES) return undefined;
    const dataUrl = await readBlobAsDataUrl(blob);
    return inspectFallbackDataUrl(dataUrl).dataUrl;
  } catch {
    return undefined;
  }
}

function readBlobAsDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener('load', () => resolve(reader.result), { once: true });
    reader.addEventListener('error', () => reject(reader.error || new Error('The copied image could not be encoded.')), { once: true });
    reader.addEventListener('abort', () => reject(new Error('The copied image encoding was aborted.')), { once: true });
    reader.readAsDataURL(blob);
  });
}

function resetImage(image) {
  visibilityRefresh.forget(image);
  releaseImageRecord(image);
  if (belongsToThisDocument(image)) observeImage(image, true);
  else stopObservingImage(image);
}

function forgetRemovedImages(root) {
  if (belongsToThisDocument(root)) return;
  const { images, shadowRoots } = collectOpenTree(root);
  for (const image of images) {
    releaseImageRecord(image);
    stopObservingImage(image);
  }
  for (const shadowRoot of shadowRoots) {
    shadowObservers.get(shadowRoot)?.disconnect();
    shadowObservers.delete(shadowRoot);
  }
}

function rescanPage() {
  pendingAdmissions.clear();
  visibilityRefresh.clear();
  for (const image of [...records.keys()]) releaseImageRecord(image);
  Object.assign(pageState, { status: 'idle', scanned: 0, aiCount: 0, realCount: 0, errors: 0, results: [] });
  discoverImages(document, { refresh: true });
}

function releaseImageRecord(image, expectedRecord, { scheduleRefresh = true } = {}) {
  const record = releaseRecord(records, image, expectedRecord);
  if (!record) return undefined;
  record.cancelled = true;
  if (record.releaseHeavyAdmission) void record.releaseHeavyAdmission();
  clearBadgeImageLink(record.image, record.id);
  clearMediaTreatment(record.image);
  record.badge.remove();
  if (detailPanel.dataset.anchor === record.id) {
    closeDetails({ restoreFocus: false });
  }
  updatePageStatus();
  if (scheduleRefresh) scheduleObservationRefresh();
  return record;
}

function stopObservingImage(image) {
  pendingAdmissions.delete(image);
  visibilityRefresh.forget(image);
  intersectionObserver.unobserve(image);
  image.removeEventListener('load', handleImageLoad);
  observed.delete(image);
}

function enforceRecordLimit() {
  while (records.size > settings.maxImagesPerPage) {
    const candidate = findTerminalEvictionCandidate(records, {
      isOffViewport: isOutsideObservationWindow,
      isProtected: (record) => !detailPanel.hidden && detailPanel.dataset.anchor === record.id
    });
    if (!candidate) break;
    releaseImageRecord(candidate.image, candidate.record, { scheduleRefresh: false });
  }
  scheduleObservationRefresh();
}

function scheduleObservationRefresh() {
  if (observationRefreshQueued || !settings.enabled || pendingAdmissions.size === 0) return;
  observationRefreshQueued = true;
  queueMicrotask(() => {
    observationRefreshQueued = false;
    drainPendingAdmissions();
  });
}

function queuePendingAdmission(image) {
  if (
    pendingAdmissions.has(image)
    || pendingAdmissions.size >= MAX_PENDING_ADMISSIONS
    || !belongsToThisDocument(image)
  ) return;
  pendingAdmissions.add(image);
}

function drainPendingAdmissions() {
  if (!settings.enabled || pendingAdmissions.size === 0) return;
  const budget = Math.min(MAX_PENDING_ADMISSIONS, Math.max(1, settings.maxImagesPerPage));
  const candidates = [...pendingAdmissions].slice(0, budget);
  for (const image of candidates) {
    if (records.has(image) || !belongsToThisDocument(image) || !isInsideObservationWindow(image)) {
      pendingAdmissions.delete(image);
      continue;
    }
    pendingAdmissions.delete(image);
    if (isAnalyzable(image)) void analyze(image);
  }
}

function handleViewportChange() {
  schedulePositions();
  scheduleObservationRefresh();
}

function isOutsideObservationWindow(image) {
  return !isInsideObservationWindow(image);
}

function isInsideObservationWindow(image) {
  if (!belongsToThisDocument(image)) return false;
  const rect = image.getBoundingClientRect();
  return rect.bottom > -OBSERVATION_MARGIN_PX
    && rect.top < innerHeight + OBSERVATION_MARGIN_PX
    && rect.right > 0
    && rect.left < innerWidth;
}

function belongsToThisDocument(node) {
  return Boolean(node?.isConnected && node.ownerDocument === document);
}

function updatePageStatus() {
  if ([...records.values()].some((item) => item.status === 'analyzing')) {
    pageState.status = 'scanning';
  } else {
    pageState.status = pageState.scanned || pageState.errors ? 'ready' : 'idle';
  }
}

function hideAllOverlays() {
  for (const record of records.values()) {
    record.badge.style.display = 'none';
    clearMediaTreatment(record.image);
  }
  closeDetails({ restoreFocus: false });
}

function applyMediaTreatment(record) {
  const isAi = record.result?.score >= settings.threshold;
  if (!settings.enabled || !isAi || record.revealed || settings.aiImageAction === 'label') {
    clearMediaTreatment(record.image);
    return;
  }
  record.image.dataset.proofmarkTreatment = settings.aiImageAction;
  applyShadowTreatment(record.image, settings.aiImageAction);
}

function clearMediaTreatment(image) {
  delete image.dataset.proofmarkTreatment;
  const snapshot = shadowTreatmentStyles.get(image);
  if (!snapshot) return;
  restoreStyleProperty(image.style, 'filter', snapshot.filter);
  restoreStyleProperty(image.style, 'transition', snapshot.transition);
  restoreStyleProperty(image.style, 'visibility', snapshot.visibility);
  shadowTreatmentStyles.delete(image);
}

function applyShadowTreatment(image, action) {
  if (!(image.getRootNode() instanceof ShadowRoot)) return;
  let snapshot = shadowTreatmentStyles.get(image);
  if (!snapshot) {
    snapshot = {
      filter: readStyleProperty(image.style, 'filter'),
      transition: readStyleProperty(image.style, 'transition'),
      visibility: readStyleProperty(image.style, 'visibility')
    };
    shadowTreatmentStyles.set(image, snapshot);
  }
  restoreStyleProperty(image.style, 'filter', snapshot.filter);
  restoreStyleProperty(image.style, 'transition', snapshot.transition);
  restoreStyleProperty(image.style, 'visibility', snapshot.visibility);
  if (action === 'blur') {
    image.style.setProperty('filter', 'blur(18px) saturate(0.65)', 'important');
    image.style.setProperty('transition', 'filter 160ms ease', 'important');
  } else if (action === 'hide') {
    image.style.setProperty('visibility', 'hidden', 'important');
  }
}

function readStyleProperty(style, property) {
  return { value: style.getPropertyValue(property), priority: style.getPropertyPriority(property) };
}

function restoreStyleProperty(style, property, snapshot) {
  if (snapshot.value) style.setProperty(property, snapshot.value, snapshot.priority);
  else style.removeProperty(property);
}

function toPublicResult(record) {
  return {
    id: record.id,
    score: record.result.score,
    dimensions: record.result.dimensions,
    format: record.result.format,
    evidenceCount: record.result.evidence.length,
    treatment: record.result.score >= settings.threshold && !record.revealed ? settings.aiImageAction : 'none',
    source: redactSource(record.source)
  };
}

function redactSource(source) {
  try {
    const url = new URL(source);
    return `${url.hostname}${url.pathname.split('/').pop() ? `/${url.pathname.split('/').pop().slice(0, 42)}` : ''}`;
  } catch {
    return 'Embedded image';
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[character]);
}
