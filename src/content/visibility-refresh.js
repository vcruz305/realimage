import { collectOpenTreeBounded } from './lifecycle.js';

/**
 * Coalesce visibility-affecting ancestor mutations into one bounded recovery
 * pass. Known intersecting images that were deferred only because page CSS hid
 * them are retried first, so a large ancestor subtree cannot starve the exact
 * hidden-to-visible image that prompted recovery.
 */
export function createVisibilityRefreshController({
  belongsToDocument,
  hasRecord,
  observeImage,
  observeShadowRoot,
  shouldRefresh = () => true,
  scheduleTask = queueMicrotask,
  collectTree = collectOpenTreeBounded,
  maxRoots = 16,
  maxNodes = 2048,
  maxImages = 200
}) {
  const rootLimit = normalizeLimit(maxRoots);
  const nodeLimit = normalizeLimit(maxNodes);
  const imageLimit = normalizeLimit(maxImages);
  const roots = new Set();
  const retryCandidates = new Set();
  let queued = false;

  function schedule(root) {
    if (!shouldRefresh()) return;
    if (root && roots.size < rootLimit) roots.add(root);
    if (queued) return;
    queued = true;
    scheduleTask(flush);
  }

  function remember(image) {
    if (!image || retryCandidates.has(image)) return true;
    if (retryCandidates.size >= imageLimit) return false;
    retryCandidates.add(image);
    return true;
  }

  function forget(image) {
    retryCandidates.delete(image);
  }

  function clear() {
    roots.clear();
    retryCandidates.clear();
  }

  function flush() {
    queued = false;
    const batch = [...roots];
    roots.clear();
    if (!shouldRefresh()) return;

    const images = new Set();
    const shadowRoots = new Set();
    for (const image of retryCandidates) {
      if (images.size >= imageLimit) break;
      images.add(image);
    }

    let remainingNodes = nodeLimit;
    for (const root of batch) {
      if (remainingNodes <= 0 || images.size >= imageLimit) break;
      const result = collectTree(root, {
        maxNodes: remainingNodes,
        maxImages: imageLimit - images.size
      });
      remainingNodes = Math.max(0, remainingNodes - result.visitedNodes);
      for (const shadowRoot of result.shadowRoots) shadowRoots.add(shadowRoot);
      for (const image of result.images) images.add(image);
    }

    for (const shadowRoot of shadowRoots) observeShadowRoot(shadowRoot);
    for (const image of images) {
      if (!belongsToDocument(image)) {
        retryCandidates.delete(image);
        continue;
      }
      if (hasRecord(image)) {
        retryCandidates.delete(image);
        continue;
      }
      observeImage(image, true);
    }
  }

  return { schedule, remember, forget, clear };
}

function normalizeLimit(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.floor(value));
}
