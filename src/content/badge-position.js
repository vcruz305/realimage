const DEFAULT_EDGE_MARGIN = 6;
const DEFAULT_IMAGE_INSET = 8;
const DEFAULT_MAX_ATTEMPTS = 8;
const DEFAULT_MAX_REGIONS = 24;

/**
 * Find a badge rectangle inside the visible, unoccluded part of an image.
 *
 * `readBlocker` is deliberately injected: the browser integration can use the
 * painted hit-test stack, while unit tests can exercise the bounded geometry
 * without a DOM. A returned blocker is subtracted from the available image
 * regions and the preferred top-right placement is retried.
 */
export function findBadgePlacement({
  imageRect,
  badgeWidth,
  badgeHeight,
  viewportWidth,
  viewportHeight,
  readBlocker = () => undefined,
  edgeMargin = DEFAULT_EDGE_MARGIN,
  imageInset = DEFAULT_IMAGE_INSET,
  maxAttempts = DEFAULT_MAX_ATTEMPTS,
  maxRegions = DEFAULT_MAX_REGIONS
}) {
  const width = finitePositive(badgeWidth);
  const height = finitePositive(badgeHeight);
  const viewport = normalizeRect({ left: 0, top: 0, right: viewportWidth, bottom: viewportHeight });
  const image = normalizeRect(imageRect);
  if (!width || !height || !viewport || !image) return undefined;

  const viewportContent = insetRect(viewport, edgeMargin);
  const visibleImage = intersectRects(image, viewportContent);
  if (!visibleImage) return undefined;

  let regions = [visibleImage];
  const attempts = clampInteger(maxAttempts, 1, 16, DEFAULT_MAX_ATTEMPTS);
  const regionLimit = clampInteger(maxRegions, 1, 48, DEFAULT_MAX_REGIONS);

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const candidate = choosePreferredCandidate(regions, width, height, imageInset, viewportContent, visibleImage);
    if (!candidate) return undefined;
    const blocker = normalizeRect(readBlocker(candidate));
    if (!blocker || !intersectRects(candidate, blocker)) return candidate;

    const next = subtractBlocker(regions, blocker, regionLimit)
      .filter((region) => canFit(region, width, height, imageInset, visibleImage));
    if (!next.length || sameRegions(regions, next)) return undefined;
    regions = next;
  }
  return undefined;
}

function choosePreferredCandidate(regions, width, height, inset, viewport, visibleImage) {
  let best;
  for (const region of regions) {
    if (!canFit(region, width, height, inset, visibleImage)) continue;
    const left = Math.max(viewport.left, Math.min(viewport.right - width, region.right - width - inset));
    const candidate = {
      left,
      top: region.top + inset,
      right: left + width,
      bottom: region.top + inset + height,
      width,
      height
    };
    // Keep the established top-right behavior. If an occluder fragments the
    // image, choose the highest available region and then the rightmost one.
    const rank = [candidate.top, -candidate.right, -(region.width * region.height)];
    if (!best || compareRank(rank, best.rank) < 0) best = { candidate, rank };
  }
  return best?.candidate;
}

function subtractBlocker(regions, blocker, maxRegions) {
  const output = [];
  for (const region of regions) {
    const overlap = intersectRects(region, blocker);
    if (!overlap) {
      output.push(region);
      continue;
    }
    pushRect(output, region.left, region.top, region.right, overlap.top);
    pushRect(output, region.left, overlap.bottom, region.right, region.bottom);
    pushRect(output, region.left, overlap.top, overlap.left, overlap.bottom);
    pushRect(output, overlap.right, overlap.top, region.right, overlap.bottom);
  }
  return output
    .sort((a, b) => a.top - b.top || b.right - a.right || (b.width * b.height) - (a.width * a.height))
    .slice(0, maxRegions);
}

function pushRect(output, left, top, right, bottom) {
  const rect = normalizeRect({ left, top, right, bottom });
  if (rect) output.push(rect);
}

function canFit(rect, width, height, inset, visibleImage) {
  // Keep the historic right-edge behavior: a badge may overhang a narrow
  // image horizontally while its anchor stays in visible image area. Only a
  // region spanning the image's full visible width gets that exception; a
  // narrow fragment created by an occluder must fit the complete badge.
  const spansVisibleWidth = rect.left === visibleImage.left && rect.right === visibleImage.right;
  const fitsHorizontally = rect.width >= width + (2 * inset) || (spansVisibleWidth && rect.width >= 2 * inset);
  return fitsHorizontally && rect.height >= height + (2 * inset);
}

function insetRect(rect, amount) {
  return normalizeRect({
    left: rect.left + amount,
    top: rect.top + amount,
    right: rect.right - amount,
    bottom: rect.bottom - amount
  });
}

function intersectRects(first, second) {
  return normalizeRect({
    left: Math.max(first.left, second.left),
    top: Math.max(first.top, second.top),
    right: Math.min(first.right, second.right),
    bottom: Math.min(first.bottom, second.bottom)
  });
}

function normalizeRect(value) {
  if (!value) return undefined;
  const left = Number(value.left ?? value.x);
  const top = Number(value.top ?? value.y);
  const right = Number(value.right ?? (left + Number(value.width)));
  const bottom = Number(value.bottom ?? (top + Number(value.height)));
  if (![left, top, right, bottom].every(Number.isFinite) || right <= left || bottom <= top) return undefined;
  return { left, top, right, bottom, width: right - left, height: bottom - top };
}

function finitePositive(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : undefined;
}

function compareRank(first, second) {
  for (let index = 0; index < first.length; index += 1) {
    if (first[index] !== second[index]) return first[index] - second[index];
  }
  return 0;
}

function sameRegions(first, second) {
  if (first.length !== second.length) return false;
  return first.every((rect, index) => ['left', 'top', 'right', 'bottom']
    .every((key) => rect[key] === second[index][key]));
}

function clampInteger(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(maximum, Math.max(minimum, Math.trunc(number))) : fallback;
}
