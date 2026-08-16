// Empirically measures SELF_TEST_REFERENCE_LOGIT for the current MODEL by
// running the exact same self-test image (src/offscreen/self-test-fixture.js)
// through the exact same processor+model call shape offscreen.js's WebGPU
// self-test uses (src/offscreen/offscreen.js probeWebgpuBackend/
// runSelfTestInference), but on the WASM execution provider, in a real
// Chromium tab. Re-run this (npx vite --config scripts/perf-probe/vite.config.js,
// then open /self-test-reference.html) whenever MODEL in src/shared/constants.js
// changes, and copy the resulting wasm.logit into SELF_TEST_REFERENCE_LOGIT.
import {
  AutoImageProcessor,
  AutoModelForImageClassification,
  RawImage,
  env
} from '@huggingface/transformers';
import { MODEL } from '../../src/shared/constants.js';
import { createSelfTestPixelData, SELF_TEST_IMAGE_SIZE } from '../../src/offscreen/self-test-fixture.js';
import { extractModelLogit } from '../../src/offscreen/model-output.js';

const output = document.querySelector('#output');
const report = { environment: {}, model: MODEL.id, runs: {} };

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;
env.localModelPath = `${location.origin}/models/`;
env.backends.onnx.wasm.wasmPaths = `${location.origin}/wasm/`;
env.backends.onnx.wasm.numThreads = 1;

report.environment = {
  crossOriginIsolated,
  hardwareConcurrency: navigator.hardwareConcurrency,
  webgpu: Boolean(navigator.gpu)
};
publish();

try {
  const processor = await AutoImageProcessor.from_pretrained(MODEL.id);
  const selfTestImage = new RawImage(createSelfTestPixelData(SELF_TEST_IMAGE_SIZE), SELF_TEST_IMAGE_SIZE, SELF_TEST_IMAGE_SIZE, 3);

  for (const device of ['wasm', 'webgpu']) {
    try {
      const model = await AutoModelForImageClassification.from_pretrained(MODEL.id, { dtype: MODEL.dtype, device });
      const inputs = await processor(selfTestImage);
      const modelOutput = await model(inputs);
      const logit = extractModelLogit(modelOutput);
      report.runs[device] = { ok: true, logit };
    } catch (error) {
      report.runs[device] = { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
    publish();
  }
} catch (error) {
  report.error = error instanceof Error ? error.stack || error.message : String(error);
  publish();
}

function publish() {
  output.textContent = JSON.stringify(report, null, 2);
  window.__realimageSelfTestReference = structuredClone(report);
}
