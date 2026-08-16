import { LOCAL_RUNTIME, MODEL } from '../src/shared/constants.js';

export const MODEL_ID = MODEL.id;
export const MODEL_UPSTREAM_ID = MODEL.upstreamId;
export const MODEL_REVISION = MODEL.revision;
export const MODEL_WEIGHT_FILE = MODEL.weightFile;
export const MODEL_FILES = [
  'config.json',
  'preprocessor_config.json',
  MODEL_WEIGHT_FILE
];
export const MODEL_WEIGHT_SHA256 = MODEL.weightSha256;
export const MODEL_CONFIG_SHA256 = MODEL.configSha256;
export const MODEL_PREPROCESSOR_SHA256 = MODEL.preprocessorConfigSha256;

export const MODEL_ROOT = new URL('../public/models/', import.meta.url);
export const RUNTIME_FILES = [
  LOCAL_RUNTIME.wasmLoader,
  LOCAL_RUNTIME.wasmModule
];
export const RUNTIME_SHA256 = Object.freeze({
  [LOCAL_RUNTIME.wasmLoader]: LOCAL_RUNTIME.wasmLoaderSha256,
  [LOCAL_RUNTIME.wasmModule]: LOCAL_RUNTIME.wasmSha256
});
export const RUNTIME_SOURCE_ROOT = new URL('../node_modules/@huggingface/transformers/dist/', import.meta.url);
export const RUNTIME_DESTINATION_ROOT = new URL('../public/wasm/', import.meta.url);
