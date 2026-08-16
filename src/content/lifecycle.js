/**
 * Return every image reachable through the supplied light-DOM subtree and any
 * nested *open* shadow roots. Closed roots are intentionally inaccessible to a
 * content script and are therefore not represented here.
 */
export function collectOpenTree(root) {
  const images = [];
  const shadowRoots = [];
  const seenImages = new Set();
  const seenScopes = new Set();
  const queue = [root];

  while (queue.length) {
    const scope = queue.shift();
    if (!scope || seenScopes.has(scope)) continue;
    seenScopes.add(scope);

    if (isImage(scope)) addOnce(images, seenImages, scope);
    for (const image of query(scope, 'img')) addOnce(images, seenImages, image);

    const hosts = isElement(scope) ? [scope, ...query(scope, '*')] : query(scope, '*');
    for (const host of hosts) {
      const shadowRoot = host?.shadowRoot;
      if (!shadowRoot || seenScopes.has(shadowRoot)) continue;
      shadowRoots.push(shadowRoot);
      queue.push(shadowRoot);
    }
  }

  return { images, shadowRoots };
}

/**
 * Walk a light-DOM subtree plus nested open shadow roots without first
 * materializing an unbounded querySelectorAll result. This is used for
 * visibility-only mutation recovery, where a hostile or very large page could
 * otherwise turn one class/style change into an unbounded scan.
 */
export function collectOpenTreeBounded(
  root,
  { maxNodes = 2048, maxImages = 200 } = {}
) {
  const nodeLimit = normalizePositiveLimit(maxNodes);
  const imageLimit = normalizePositiveLimit(maxImages);
  const images = [];
  const shadowRoots = [];
  const seenNodes = new Set();
  const seenImages = new Set();
  const stack = root ? [{ node: root, includeSibling: false }] : [];
  let visitedNodes = 0;
  let truncated = false;

  while (stack.length) {
    if (visitedNodes >= nodeLimit) {
      truncated = true;
      break;
    }

    const { node, includeSibling } = stack.pop();
    if (!node || seenNodes.has(node)) continue;
    seenNodes.add(node);
    visitedNodes += 1;

    if (includeSibling && node.nextElementSibling) {
      stack.push({ node: node.nextElementSibling, includeSibling: true });
    }

    const shadowRoot = isElement(node) ? node.shadowRoot : undefined;
    if (shadowRoot && !seenNodes.has(shadowRoot)) {
      shadowRoots.push(shadowRoot);
      stack.push({ node: shadowRoot, includeSibling: false });
    }

    const firstChild = node.firstElementChild;
    if (firstChild) stack.push({ node: firstChild, includeSibling: true });

    if (isImage(node) && !seenImages.has(node)) {
      if (images.length >= imageLimit) {
        truncated = true;
        break;
      }
      seenImages.add(node);
      images.push(node);
    }
  }

  if (stack.length) truncated = true;
  return { images, shadowRoots, visitedNodes, truncated };
}

/**
 * `isVisible` defaults to an always-pass check so every existing caller (and
 * every test that exercises capacity/eviction without a real DOM) keeps its
 * current behavior. Callers that enqueue real page images — see
 * reserveRecordWithTerminalEviction's caller in content/index.js — pass
 * `isVisiblyRenderedImage` so a node that only *discovery* time saw as
 * visible cannot be admitted here after it has gone visibility:hidden,
 * display:none, content-visibility:hidden, or opacity:0.
 */
export function reserveRecord(records, image, record, limit, { isVisible = () => true } = {}) {
  if (records.has(image) || records.size >= limit) return false;
  if (!isVisible(image)) return false;
  records.set(image, record);
  return true;
}

/**
 * Admit a record into a bounded live window. When the window is full, only the
 * oldest terminal record that is safely outside the observation viewport may
 * be released. The caller owns DOM cleanup through `onEvict` so the same
 * release path removes badges and media treatment before the slot is reused.
 *
 * The visibility check happens here, immediately before enqueue, rather than
 * relying solely on whatever gate ran when the image was first discovered.
 * A duplicate/hidden thumbnail (visibility:hidden with a nonzero layout rect,
 * as Google Images keeps for de-duplication) can pass an earlier discovery
 * check and still reach this call; without this re-check it would be
 * admitted and produce a ghost badge.
 */
export function reserveRecordWithTerminalEviction(
  records,
  image,
  record,
  limit,
  {
    isOffViewport,
    isProtected = () => false,
    isVisible = () => true,
    onEvict
  } = {}
) {
  if (records.has(image)) return false;
  if (!isVisible(image)) return false;
  if (records.size < limit) return reserveRecord(records, image, record, limit);
  if (typeof isOffViewport !== 'function' || typeof onEvict !== 'function') return false;

  const candidate = findTerminalEvictionCandidate(records, { isOffViewport, isProtected });
  if (!candidate) return false;
  onEvict(candidate.image, candidate.record);

  // Fail closed if cleanup did not actually free exactly one bounded slot.
  if (records.has(candidate.image) || records.size >= limit) return false;
  return reserveRecord(records, image, record, limit);
}

export function findTerminalEvictionCandidate(
  records,
  { isOffViewport, isProtected = () => false } = {}
) {
  if (typeof isOffViewport !== 'function') return undefined;
  for (const [image, record] of records) {
    if (!record || !['complete', 'error'].includes(record.status)) continue;
    if (isProtected(record, image) || !isOffViewport(image, record)) continue;
    return { image, record };
  }
  return undefined;
}

export function isCurrentRecord(records, image, record, ownerDocument) {
  return records.get(image) === record
    && image.isConnected
    && image.ownerDocument === ownerDocument
    && image.currentSrc === record.source;
}

/**
 * Cancel only in-flight records whose painted image disappeared. The records
 * map is already bounded by maxImagesPerPage, so checking it once per mutation
 * batch cannot turn an ancestor class change into an unbounded DOM traversal.
 */
export function cancelInvisiblePendingRecords(
  records,
  {
    isVisible = isVisiblyRenderedImage,
    onCancel
  } = {}
) {
  if (typeof onCancel !== 'function') throw new TypeError('onCancel must be a function.');
  let cancelled = 0;
  for (const [image, record] of records) {
    if (record?.status !== 'analyzing' || isVisible(image)) continue;
    onCancel(image, record);
    cancelled += 1;
  }
  return cancelled;
}

/**
 * Reject images that occupy layout space but are not actually painted. In
 * particular, Google Images keeps a visibility:hidden duplicate with a large
 * bounding box; treating that node as visible creates a second, ghost badge.
 */
export function isVisiblyRenderedImage(image, readComputedStyle = globalThis.getComputedStyle) {
  return readVisibility(image, readComputedStyle, false);
}

/**
 * Badge positioning must ignore only the visibility:hidden rule that RealImage
 * itself applies for the explicit "hide" treatment. Other page/ancestor hiding
 * signals still remove the overlay.
 */
export function isVisibleBadgeAnchor(image, readComputedStyle = globalThis.getComputedStyle) {
  return readVisibility(image, readComputedStyle, image?.dataset?.proofmarkTreatment === 'hide');
}

function readVisibility(image, readComputedStyle, allowRealImageVisibilityHide) {
  if (!image || image.hidden || typeof readComputedStyle !== 'function') return false;
  let style;
  try {
    style = readComputedStyle(image);
  } catch {
    return false;
  }
  if (!style) return false;
  if (style.display === 'none') return false;
  if ((style.visibility === 'hidden' || style.visibility === 'collapse') && !allowRealImageVisibilityHide) return false;
  if (style.contentVisibility === 'hidden') return false;
  if (Number.parseFloat(style.opacity) === 0) return false;
  if (typeof image.checkVisibility === 'function') {
    try {
      const visible = image.checkVisibility({ opacityProperty: true, visibilityProperty: !allowRealImageVisibilityHide });
      if (!visible) return false;
    } catch {
      return false;
    }
  }
  if (hasNonVisibleComposedAncestor(image, readComputedStyle)) return false;
  return true;
}

function hasNonVisibleComposedAncestor(image, readComputedStyle) {
  const seen = new Set();
  let ancestor = parentAcrossOpenShadowBoundary(image);
  for (let depth = 0; ancestor && depth < 64; depth += 1) {
    if (seen.has(ancestor)) return true;
    seen.add(ancestor);
    if (ancestor.hidden) return true;
    let style;
    try {
      style = readComputedStyle(ancestor);
    } catch {
      return true;
    }
    if (
      !style
      || style.display === 'none'
      || style.visibility === 'hidden'
      || style.visibility === 'collapse'
      || style.contentVisibility === 'hidden'
      || Number.parseFloat(style.opacity) === 0
    ) return true;
    ancestor = parentAcrossOpenShadowBoundary(ancestor);
  }
  // A composed tree deeper than the fixed inspection budget cannot be proven
  // visible. Fail closed instead of leaving a ghost reveal badge whose hidden
  // ancestor sits just beyond the bound.
  return Boolean(ancestor);
}

function parentAcrossOpenShadowBoundary(node) {
  // Slot assignment is the first composed parent. Reading parentElement first
  // skips CSS visibility inherited from a shadow-DOM slot.
  if (node?.assignedSlot) return node.assignedSlot;
  if (node?.parentElement) return node.parentElement;
  const root = typeof node?.getRootNode === 'function' ? node.getRootNode() : undefined;
  return root?.host || undefined;
}

export function releaseRecord(records, image, expectedRecord) {
  const record = records.get(image);
  if (!record || (expectedRecord && record !== expectedRecord)) return undefined;
  records.delete(image);
  return record;
}

function query(scope, selector) {
  return scope?.querySelectorAll ? [...scope.querySelectorAll(selector)] : [];
}

function isImage(node) {
  return String(node?.localName || node?.tagName || '').toLowerCase() === 'img';
}

function isElement(node) {
  return node?.nodeType === 1;
}

function addOnce(target, seen, value) {
  if (seen.has(value)) return;
  seen.add(value);
  target.push(value);
}

function normalizePositiveLimit(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.floor(value));
}
