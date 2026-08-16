/**
 * Error codes that mean "the background could not fetch this http(s) image
 * itself" (blocked by policy or a real network failure) but say nothing about
 * whether the image is actually unavailable -- the page's own <img> element
 * may already have the pixels rendered. These, and only these, are worth one
 * canvas-capture retry. Codes like IMAGE_TOO_LARGE or UNSUPPORTED_MIME are not
 * included here because a canvas capture cannot fix them.
 */
export const CANVAS_CAPTURE_RETRY_CODES = Object.freeze([
  'PRIVATE_TARGET_BLOCKED',
  'LOCAL_TARGET_BLOCKED',
  'FETCH_BLOCKED',
  'HTTP_UNAVAILABLE'
]);

/**
 * Decide whether a failed direct-fetch analysis result is worth retrying via
 * canvas capture. `result` is the response from a plain sendAnalysis() call
 * (or `undefined`/`null` when the image went stale mid-flight, which is never
 * retryable).
 */
export function isCanvasCaptureRetryable(result) {
  return Boolean(result) && result.ok === false && CANVAS_CAPTURE_RETRY_CODES.includes(result.code);
}
