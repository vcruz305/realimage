/**
 * Expose the displayed model result and the separate declaration summary to the
 * local browser harness without leaking the raw full-precision model score.
 */
export function applyBadgeResultContract(
  badge,
  { isAi, displayScore, decisionThreshold, declaration }
) {
  const dataset = badge?.dataset;
  if (!dataset) return;
  dataset.proofmarkDecisionThreshold = String(decisionThreshold);
  dataset.proofmarkModelVerdict = isAi ? 'ai' : 'real';
  dataset.proofmarkModelScorePercent = String(displayScore);
  if (declaration) {
    dataset.proofmarkDeclarationType = declaration.claim;
    dataset.proofmarkDeclarationSummary = declaration.badgeText;
  } else {
    delete dataset.proofmarkDeclarationType;
    delete dataset.proofmarkDeclarationSummary;
  }
}

/**
 * Give the page-local validation harness an opaque one-to-one association
 * between an image and its overlay badge. The identifier contains no source or
 * benchmark identity and is already used as the extension request ID.
 */
export function applyBadgeImageLink(image, badge) {
  const id = badge?.dataset?.proofmarkId;
  if (!image?.dataset || typeof id !== 'string' || !id) return false;
  image.dataset.proofmarkBadgeId = id;
  return true;
}

/** Remove only the association owned by the record being released. */
export function clearBadgeImageLink(image, expectedId) {
  if (!image?.dataset || image.dataset.proofmarkBadgeId !== expectedId) return false;
  delete image.dataset.proofmarkBadgeId;
  return true;
}
