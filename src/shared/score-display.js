const DISPLAY_DECIMALS = 1;
const DISPLAY_SCALE = 10 ** DISPLAY_DECIMALS;

/**
 * Formats a detector score without ever rounding a below-threshold value up to the
 * threshold shown in the UI. The decision still uses the full-precision score.
 */
export function formatScorePercent(score, threshold) {
  if (!Number.isFinite(score) || score < 0 || score > 1) throw new RangeError('score must be finite and in [0, 1]');
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError('threshold must be finite and in [0, 1]');

  // Work in integer tenths-of-a-percent for the public display. Comparing a
  // separately multiplied threshold percentage can put mathematically equal
  // values on opposite sides of the boundary (notably 55% and 56%).
  const roundedUnits = Math.round(score * 100 * DISPLAY_SCALE);
  const roundedScore = roundedUnits / (100 * DISPLAY_SCALE);
  const safeUnits = score < threshold && roundedScore >= threshold
    ? roundedUnits - 1
    : roundedUnits;
  return (safeUnits / DISPLAY_SCALE).toFixed(DISPLAY_DECIMALS);
}
