import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MESSAGE } from '../shared/constants.js';

vi.mock('@huggingface/transformers', () => ({
  AutoImageProcessor: { from_pretrained: vi.fn() },
  AutoModelForImageClassification: { from_pretrained: vi.fn() },
  RawImage: { fromBlob: vi.fn() },
  env: {
    allowLocalModels: false,
    allowRemoteModels: true,
    useBrowserCache: true,
    localModelPath: '',
    backends: { onnx: { wasm: {} } }
  }
}));

describe('offscreen entry bootstrap', () => {
  let addMessageListener;

  beforeEach(() => {
    vi.resetModules();
    addMessageListener = vi.fn();
    vi.stubGlobal('navigator', {
      hardwareConcurrency: 20,
      platform: 'test-platform',
      userAgent: 'RealImage offscreen bootstrap test'
    });
    vi.stubGlobal('crossOriginIsolated', true);
    vi.stubGlobal('chrome', {
      runtime: {
        // This deliberately mirrors the restricted offscreen runtime surface:
        // getManifest is absent and must never be required during bootstrap.
        getURL: (path) => `chrome-extension://abcdefghijklmnopabcdefghijklmnop/${path}`,
        id: 'abcdefghijklmnopabcdefghijklmnop',
        onMessage: { addListener: addMessageListener }
      }
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('registers its receiver without chrome.runtime.getManifest', async () => {
    expect(chrome.runtime.getManifest).toBeUndefined();

    await import('./offscreen.js');

    expect(addMessageListener).toHaveBeenCalledOnce();
    expect(addMessageListener).toHaveBeenCalledWith(expect.any(Function));
  });

  it('answers a background ping but ignores internal messages sent from a tab', async () => {
    await import('./offscreen.js');
    const listener = addMessageListener.mock.calls[0][0];
    const sendResponse = vi.fn();
    const background = { id: chrome.runtime.id };
    const contentScript = { id: chrome.runtime.id, tab: { id: 7 } };

    expect(listener({ type: MESSAGE.OFFSCREEN_PING, nonce: 'ready' }, background, sendResponse)).toBe(false);
    expect(sendResponse).toHaveBeenCalledWith({ ok: true, receiver: 'offscreen', nonce: 'ready' });

    sendResponse.mockClear();
    expect(listener({ type: MESSAGE.OFFSCREEN_PING, nonce: 'forged' }, contentScript, sendResponse)).toBe(false);
    expect(listener({ type: MESSAGE.OFFSCREEN_ANALYZE, payload: {} }, contentScript, sendResponse)).toBe(false);
    expect(sendResponse).not.toHaveBeenCalled();
  });

  it('disables Transformers.js browser caching before loading packaged extension models', async () => {
    const runtime = await import('@huggingface/transformers');
    runtime.AutoImageProcessor.from_pretrained.mockImplementationOnce(() => new Promise(() => {}));

    await import('./offscreen.js');
    const listener = addMessageListener.mock.calls[0][0];

    expect(listener(
      { type: MESSAGE.WARM_MODEL },
      { id: chrome.runtime.id },
      vi.fn()
    )).toBe(true);

    await vi.waitFor(() => {
      expect(runtime.AutoImageProcessor.from_pretrained).toHaveBeenCalledOnce();
    });
    expect(runtime.env).toMatchObject({
      allowLocalModels: true,
      allowRemoteModels: false,
      useBrowserCache: false,
      localModelPath: `chrome-extension://${chrome.runtime.id}/models/`,
      backends: { onnx: { wasm: { numThreads: 12 } } }
    });
  });

  // Regression test for a real bug found via manual end-to-end testing:
  // the bundled @huggingface/transformers caches the FIRST
  // InferenceSession.create() call's promise (successful or not) to avoid
  // double-initializing the shared ORT-web runtime. If a WebGPU session
  // creation attempt is allowed to fail, every later session creation --
  // including an explicit WASM fallback -- immediately re-throws that same
  // stale error instead of trying WASM at all. offscreen.js must never
  // attempt a WebGPU model load (device:'webgpu') when no usable adapter is
  // present, so that failure mode can never be triggered.
  describe('WebGPU adapter preflight', () => {
    it('never attempts a WebGPU model load when navigator.gpu is absent, and loads WASM directly', async () => {
      const runtime = await import('@huggingface/transformers');
      runtime.AutoImageProcessor.from_pretrained.mockResolvedValueOnce({});
      runtime.AutoModelForImageClassification.from_pretrained.mockResolvedValueOnce({});

      await import('./offscreen.js');
      const listener = addMessageListener.mock.calls[0][0];
      const sendResponse = vi.fn();
      listener({ type: MESSAGE.WARM_MODEL }, { id: chrome.runtime.id }, sendResponse);

      await vi.waitFor(() => {
        expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalled();
      });
      expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalledTimes(1);
      expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ device: 'wasm' })
      );
    });

    it('never attempts a WebGPU model load when requestAdapter() rejects', async () => {
      vi.stubGlobal('navigator', {
        hardwareConcurrency: 20,
        platform: 'test-platform',
        userAgent: 'RealImage offscreen bootstrap test',
        gpu: { requestAdapter: vi.fn().mockRejectedValue(new Error('Failed to get GPU adapter.')) }
      });
      const runtime = await import('@huggingface/transformers');
      runtime.AutoImageProcessor.from_pretrained.mockResolvedValueOnce({});
      runtime.AutoModelForImageClassification.from_pretrained.mockResolvedValueOnce({});

      await import('./offscreen.js');
      const listener = addMessageListener.mock.calls[0][0];
      listener({ type: MESSAGE.WARM_MODEL }, { id: chrome.runtime.id }, vi.fn());

      await vi.waitFor(() => {
        expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalled();
      });
      expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalledTimes(1);
      expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ device: 'wasm' })
      );
    });

    it('never attempts a WebGPU model load when requestAdapter() resolves null', async () => {
      vi.stubGlobal('navigator', {
        hardwareConcurrency: 20,
        platform: 'test-platform',
        userAgent: 'RealImage offscreen bootstrap test',
        gpu: { requestAdapter: vi.fn().mockResolvedValue(null) }
      });
      const runtime = await import('@huggingface/transformers');
      runtime.AutoImageProcessor.from_pretrained.mockResolvedValueOnce({});
      runtime.AutoModelForImageClassification.from_pretrained.mockResolvedValueOnce({});

      await import('./offscreen.js');
      const listener = addMessageListener.mock.calls[0][0];
      listener({ type: MESSAGE.WARM_MODEL }, { id: chrome.runtime.id }, vi.fn());

      await vi.waitFor(() => {
        expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalled();
      });
      expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ device: 'wasm' })
      );
    });

    it('does attempt a WebGPU model load when a usable adapter is present', async () => {
      vi.stubGlobal('navigator', {
        hardwareConcurrency: 20,
        platform: 'test-platform',
        userAgent: 'RealImage offscreen bootstrap test',
        gpu: { requestAdapter: vi.fn().mockResolvedValue({ features: [] }) }
      });
      const runtime = await import('@huggingface/transformers');
      runtime.AutoImageProcessor.from_pretrained.mockResolvedValueOnce({});
      // First call is the WebGPU self-test probe; let it hang so the test can
      // observe that it was attempted without needing a full self-test image
      // pipeline (RawImage.fromBlob isn't wired up for this mock).
      runtime.AutoModelForImageClassification.from_pretrained.mockImplementationOnce(() => new Promise(() => {}));

      await import('./offscreen.js');
      const listener = addMessageListener.mock.calls[0][0];
      listener({ type: MESSAGE.WARM_MODEL }, { id: chrome.runtime.id }, vi.fn());

      await vi.waitFor(() => {
        expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalled();
      });
      expect(runtime.AutoModelForImageClassification.from_pretrained).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ device: 'webgpu' })
      );
    });
  });
});
