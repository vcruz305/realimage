import { describe, expect, it } from 'vitest';
import {
  MAX_WASM_THREADS,
  WEBGPU_SELF_TEST_TOLERANCE,
  isSelfTestWithinTolerance,
  selectBackendPlan,
  selectWasmThreadCount
} from './runtime-config.js';

describe('WASM runtime configuration', () => {
  it('keeps the existing thread count on four-core and smaller systems', () => {
    expect(selectWasmThreadCount(1)).toBe(1);
    expect(selectWasmThreadCount(2)).toBe(2);
    expect(selectWasmThreadCount(4)).toBe(4);
  });

  it('scales the serial ViT workload without exceeding the measured cap', () => {
    expect(selectWasmThreadCount(8)).toBe(8);
    expect(selectWasmThreadCount(20)).toBe(MAX_WASM_THREADS);
    expect(selectWasmThreadCount(128)).toBe(MAX_WASM_THREADS);
  });

  it('fails closed to one thread for missing or invalid reports', () => {
    expect(selectWasmThreadCount(undefined)).toBe(1);
    expect(selectWasmThreadCount(Number.NaN)).toBe(1);
    expect(selectWasmThreadCount(0)).toBe(1);
    expect(selectWasmThreadCount(-4)).toBe(1);
  });
});

describe('WebGPU self-test tolerance check', () => {
  it('passes a logit within the fixed tolerance of the reference', () => {
    expect(isSelfTestWithinTolerance(-4.325838, -4.325838)).toBe(true);
    expect(isSelfTestWithinTolerance(-4.325, -4.325838, WEBGPU_SELF_TEST_TOLERANCE)).toBe(true);
    expect(isSelfTestWithinTolerance(-4.3259, -4.325838)).toBe(true);
  });

  it('fails a logit outside the tolerance, or that is missing/non-finite', () => {
    expect(isSelfTestWithinTolerance(-4.3, -4.325838)).toBe(false);
    expect(isSelfTestWithinTolerance(0, -4.325838)).toBe(false);
    expect(isSelfTestWithinTolerance(Number.NaN, -4.325838)).toBe(false);
    expect(isSelfTestWithinTolerance(undefined, -4.325838)).toBe(false);
    expect(isSelfTestWithinTolerance(-4.325838, undefined)).toBe(false);
  });
});

describe('backend selection plan', () => {
  const referenceLogit = -4.325838088989258;

  it('selects WebGPU with a single active thread when the self-test passes', () => {
    const plan = selectBackendPlan({
      hardwareConcurrency: 8,
      crossOriginIsolated: true,
      referenceLogit,
      webgpuSelfTest: { ok: true, logit: -4.325845718383789 }
    });
    expect(plan).toEqual({ backend: 'webgpu', requestedThreads: 8, activeThreads: 1 });
  });

  it('falls back to WASM when the WebGPU probe never ran or threw', () => {
    const plan = selectBackendPlan({
      hardwareConcurrency: 8,
      crossOriginIsolated: true,
      referenceLogit,
      webgpuSelfTest: { ok: false }
    });
    expect(plan).toEqual({ backend: 'wasm', requestedThreads: 8, activeThreads: 8 });
  });

  it('falls back to WASM when the WebGPU probe was never attempted', () => {
    const plan = selectBackendPlan({ hardwareConcurrency: 4, crossOriginIsolated: true, referenceLogit });
    expect(plan).toEqual({ backend: 'wasm', requestedThreads: 4, activeThreads: 4 });
  });

  it('falls back to WASM when the self-test logit mismatches beyond tolerance', () => {
    const plan = selectBackendPlan({
      hardwareConcurrency: 8,
      crossOriginIsolated: true,
      referenceLogit,
      webgpuSelfTest: { ok: true, logit: referenceLogit + 1 }
    });
    expect(plan).toEqual({ backend: 'wasm', requestedThreads: 8, activeThreads: 8 });
  });

  it('falls back to WASM single-threaded when the tab is not cross-origin isolated', () => {
    const plan = selectBackendPlan({
      hardwareConcurrency: 8,
      crossOriginIsolated: false,
      referenceLogit,
      webgpuSelfTest: { ok: false }
    });
    expect(plan).toEqual({ backend: 'wasm', requestedThreads: 8, activeThreads: 1 });
  });
});
