import {
  AutoImageProcessor,
  AutoModelForImageClassification,
  RawImage,
  env
} from '@huggingface/transformers';
import { MODEL } from '../../src/shared/constants.js';

const MODEL_ID = MODEL.id;
const output = document.querySelector('#output');
const report = { environment: {}, runs: [] };
const search = new URLSearchParams(location.search);
const threadOverride = Number(search.get('threads'));
const devices = (search.get('devices') || 'wasm,webgpu').split(',').filter(Boolean);

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;
env.localModelPath = `${location.origin}/models/`;
env.backends.onnx.wasm.wasmPaths = `${location.origin}/wasm/`;
env.backends.onnx.wasm.numThreads = Number.isInteger(threadOverride) && threadOverride > 0
  ? threadOverride
  : Math.min(4, navigator.hardwareConcurrency || 1);

report.environment = {
  crossOriginIsolated,
  hardwareConcurrency: navigator.hardwareConcurrency,
  webgpu: Boolean(navigator.gpu),
  wasmThreads: env.backends.onnx.wasm.numThreads
};
publish();

try {
  const rawImage = await makeFixture();
  const processor = await AutoImageProcessor.from_pretrained(MODEL_ID);
  const inputs = await processor(rawImage);

  for (const device of devices) {
    const run = { device };
    report.runs.push(run);
    publish();
    try {
      const loadStarted = performance.now();
      const model = await AutoModelForImageClassification.from_pretrained(MODEL_ID, {
        dtype: 'fp32',
        device
      });
      run.loadMs = Math.round(performance.now() - loadStarted);
      run.inferenceMs = [];
      for (let index = 0; index < 7; index += 1) {
        const started = performance.now();
        const result = await model(inputs);
        run.inferenceMs.push(Math.round(performance.now() - started));
        run.logit = Number(result.logits.data[0]);
        publish();
      }
    } catch (error) {
      run.error = error instanceof Error ? error.message : String(error);
    }
    publish();
  }
} catch (error) {
  report.error = error instanceof Error ? error.stack || error.message : String(error);
  publish();
}

function publish() {
  output.textContent = JSON.stringify(report, null, 2);
  window.__realimagePerformanceProbe = structuredClone(report);
}

async function makeFixture() {
  const canvas = document.createElement('canvas');
  canvas.width = 386;
  canvas.height = 518;
  const context = canvas.getContext('2d', { alpha: false });
  const gradient = context.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, '#15324b');
  gradient.addColorStop(0.5, '#c98b57');
  gradient.addColorStop(1, '#ece4d2');
  context.fillStyle = gradient;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#1d1d1d';
  context.fillRect(47, 61, 173, 89);
  context.fillStyle = '#f1ca64';
  context.beginPath();
  context.arc(286, 173, 64, 0, Math.PI * 2);
  context.fill();
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  return RawImage.fromBlob(blob);
}
