export function createUnavailableResponse(error, requestId, humanize = defaultHumanize) {
  return {
    ok: false,
    unavailable: true,
    code: typeof error?.code === 'string' ? error.code : 'ANALYSIS_UNAVAILABLE',
    retryable: Boolean(error?.retryable),
    requestId,
    error: humanize(error)
  };
}

function defaultHumanize(error) {
  return error instanceof Error ? error.message : String(error || 'The image could not be analyzed.');
}
