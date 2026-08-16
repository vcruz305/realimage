export const MAX_WASM_THREADS = 12;

/**
 * Use more than ONNX Runtime Web's conservative four-thread default for this
 * serial, compute-bound ViT workload. The cap leaves headroom on high-core
 * systems and preserves the existing behavior on four-core (or smaller)
 * machines.
 */
export function selectWasmThreadCount(hardwareConcurrency) {
  const logicalCores = Number.isFinite(hardwareConcurrency)
    ? Math.max(1, Math.floor(hardwareConcurrency))
    : 1;
  return Math.min(MAX_WASM_THREADS, logicalCores);
}

// A WebGPU startup self-test is allowed this long (model load plus one
// inference pass) before it is treated as a failure and the offscreen
// document falls back to the WASM path.
export const WEBGPU_SELF_TEST_TIMEOUT_MS = 6000;

// The self-test logit and the stored reference (SELF_TEST_REFERENCE_LOGIT in
// self-test-fixture.js) come from two different execution providers running
// the same frozen FP32 graph, so they are expected to differ only by
// ordinary floating-point rounding noise, not by this much. An empirical
// WASM-vs-WebGPU comparison on the current candidate measured an actual
// difference near 8e-6; this tolerance stays far above that noise floor
// while still rejecting a WebGPU backend that loaded but is silently
// producing wrong output (wrong dtype, an unimplemented op falling back to
// zeros, and similar failures).
export const WEBGPU_SELF_TEST_TOLERANCE = 1e-3;

/**
 * Pure pass/fail check for the WebGPU startup self-test. Kept separate from
 * the backend-selection decision below so both can be unit tested without
 * touching a real WebGPU or WASM runtime.
 */
export function isSelfTestWithinTolerance(logit, referenceLogit, tolerance = WEBGPU_SELF_TEST_TOLERANCE) {
  return typeof logit === 'number'
    && Number.isFinite(logit)
    && typeof referenceLogit === 'number'
    && Number.isFinite(referenceLogit)
    && Math.abs(logit - referenceLogit) <= tolerance;
}

/**
 * Decide which inference backend the offscreen document should use and how
 * many WASM threads that choice implies, from the outcome of at most one
 * WebGPU self-test attempt. This function makes no runtime calls itself —
 * the caller runs (or skips) the actual WebGPU load/inference and passes in
 * the outcome — so the decision itself stays a pure, easily tested function.
 *
 * @param {{
 *   hardwareConcurrency: number,
 *   crossOriginIsolated: boolean,
 *   referenceLogit: number,
 *   webgpuSelfTest?: { ok: true, logit: number } | { ok: false }
 * }} params
 */
export function selectBackendPlan({ hardwareConcurrency, crossOriginIsolated, referenceLogit, webgpuSelfTest }) {
  const requestedThreads = selectWasmThreadCount(hardwareConcurrency);
  const wasmActiveThreads = crossOriginIsolated ? requestedThreads : 1;
  const webgpuPassed = webgpuSelfTest?.ok === true && isSelfTestWithinTolerance(webgpuSelfTest.logit, referenceLogit);
  if (webgpuPassed) {
    return { backend: 'webgpu', requestedThreads, activeThreads: 1 };
  }
  return { backend: 'wasm', requestedThreads, activeThreads: wasmActiveThreads };
}
