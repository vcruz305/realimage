export const SELF_TEST_IMAGE_SIZE = 64;

/**
 * Deterministic RGB pixel data for the local backend self-test. There is no
 * entropy source here (no Math.random, no Date.now, no I/O), so every probe
 * — the WebGPU startup check, the WASM fallback, and the empirical reference
 * measurement recorded as SELF_TEST_REFERENCE_LOGIT in runtime-config.js —
 * encodes and infers the exact same input bytes.
 */
export function createSelfTestPixelData(size = SELF_TEST_IMAGE_SIZE) {
  const channels = 3;
  const data = new Uint8Array(size * size * channels);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const offset = (y * size + x) * channels;
      data[offset] = (x * 5 + y * 3) % 256;
      data[offset + 1] = (x * 3 + y * 7 + 41) % 256;
      data[offset + 2] = (x * 11 + y * 2 + 83) % 256;
    }
  }
  return data;
}

/**
 * Empirically measured (not guessed) for the current candidate,
 * RealImage/broad-v1-modern-v1-1471e3ef, by loading it through the real WASM
 * execution provider (onnxruntime-web, single thread, the same
 * ort-wasm-simd-threaded.jsep.wasm binary this extension ships) in an actual
 * Chromium tab, running the processor + model over the deterministic 64x64
 * image above, and reading output.logits.data[0]. Measured with
 * scripts/perf-probe/self-test-reference.html (serve
 * scripts/perf-probe/vite.config.js, e.g. `npx vite --config
 * scripts/perf-probe/vite.config.js`, then open that page and read
 * `runs.wasm.logit`). A same-tab WebGPU run of the identical input measured
 * -4.577672481536865, ~2.4e-6 from this value -- consistent with the
 * ordinary floating-point noise floor between execution providers and well
 * inside WEBGPU_SELF_TEST_TOLERANCE (1e-3) in runtime-config.js.
 *
 * Whenever MODEL in src/shared/constants.js changes (a new model swap), this
 * constant goes stale and must be re-measured the same way, or the WebGPU
 * self-test will fail closed on every load and force a WASM-only fallback
 * (safe, but a real performance regression rather than a correctness bug).
 */
export const SELF_TEST_REFERENCE_LOGIT = -4.577674865722656;
