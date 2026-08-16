import { ImageUnavailableError } from './source-policy.js';

export function extractModelLogit(output) {
  const logit = output?.logits?.data?.[0];
  if (typeof logit !== 'number' || !Number.isFinite(logit)) {
    throw new ImageUnavailableError(
      'MODEL_OUTPUT_INVALID',
      'The local model did not return a valid detector score.'
    );
  }
  return logit;
}
