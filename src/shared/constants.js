export const MESSAGE = Object.freeze({
  ANALYZE_IMAGE: 'proofmark/analyze-image',
  RESERVE_HEAVY_IMAGE: 'proofmark/reserve-heavy-image',
  ACTIVATE_HEAVY_IMAGE: 'proofmark/activate-heavy-image',
  RELEASE_HEAVY_IMAGE: 'proofmark/release-heavy-image',
  OFFSCREEN_ANALYZE: 'proofmark/offscreen-analyze',
  OFFSCREEN_PING: 'proofmark/offscreen-ping',
  MODEL_STATUS: 'proofmark/model-status',
  WARM_MODEL: 'proofmark/warm-model',
  PAGE_RESULT: 'proofmark/page-result',
  GET_PAGE_STATE: 'proofmark/get-page-state',
  RESCAN_PAGE: 'proofmark/rescan-page',
  SETTINGS_CHANGED: 'proofmark/settings-changed'
});

export const DEFAULT_SETTINGS = Object.freeze({
  enabled: true,
  threshold: 0.65,
  minimumDimension: 64,
  maxImagesPerPage: 40,
  showRealScores: true,
  aiImageAction: 'blur'
});

export const RELEASE = Object.freeze({
  name: 'RealImage',
  shortName: 'RealImage',
  version: '0.2.4',
  status: 'shipping RealImage/broad-v1-modern-v1-1471e3ef'
});

export const MODEL = Object.freeze({
  id: 'RealImage/broad-v1-modern-v1-1471e3ef',
  candidateId: 'realimage-broad-v1-mlp-modern-v1-fp32-1471e3ef',
  upstreamId: 'OwensLab/commfor-model-384',
  revision: '6076002bf0d9dd37537f965ee2f06f826c333b61',
  dtype: 'fp32',
  inputSize: 384,
  resizeSize: 440,
  outputActivation: 'sigmoid',
  decisionPolicy: 'model-only',
  license: 'MIT backbone (OwensLab/commfor-model-384); head trained on a broader multi-generator corpus with mixed provenance (CC-BY-4.0, Apache-2.0, and public-domain rows, plus rows with an open provenance gap or non-redistributable dev-only origin) -- see MODEL_CARD.md for the exact per-source breakdown',
  weightFile: 'onnx/model.onnx',
  sourceWeightSha256: 'b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387',
  sourceModelSha256: '8c7762fe3b7f407a15b8cc7796e3b286fc4a05ae5c2e580a936730ca5f9a4a33',
  weightSha256: '1471e3eff3a05d5ef8c068abfdef2f3f43d41060b27306f19763968cf8d38098',
  configSha256: '91c4edb9ee494e7f6afd510546850f30409a19a6817c6741552610a2cf087d81',
  preprocessorConfigSha256: '14c65215bdb2d8041bacf31d8b68953830c3081e7e7dd4ac25a4d3af58ceffaa',
  calibration: Object.freeze({
    rawThreshold: 0.646794855594635,
    displayThreshold: 0.65
  })
});

// These release identities are reported by the internal installed-Chrome
// parity channel. They are also verified before every build, so a parity report
// cannot silently describe a different local runtime than the packaged files.
export const LOCAL_RUNTIME = Object.freeze({
  transformersJsVersion: '3.8.1',
  onnxRuntimeWebVersion: '1.22.0-dev.20250409-89f8206ba4',
  wasmModule: 'ort-wasm-simd-threaded.jsep.wasm',
  wasmSha256: 'c46655e8a94afc45338d4cb2b840475f88e5012d524509916e505079c00bfa39',
  wasmLoader: 'ort-wasm-simd-threaded.jsep.mjs',
  wasmLoaderSha256: '08fb86ec433c78bfb032c5d84a68b8e8e5a8d81268fa39e24314179a5767a5b9'
});

export const MIN_ANALYZABLE_AREA = 96 * 96;
