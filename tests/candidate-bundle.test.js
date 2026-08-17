import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { calibrateDecisionScore } from '../src/analysis/forensics.js';
import manifest from '../src/manifest.js';
import { DEFAULT_SETTINGS, MODEL, RELEASE } from '../src/shared/constants.js';

const projectRoot = resolve(import.meta.dirname, '..');
const modelRoot = resolve(projectRoot, 'public/models', MODEL.id);

describe('frozen FP32 Chrome-calibration candidate', () => {
  it('uses Stable Chrome JSON messaging without a channel-gated manifest opt-in', () => {
    expect(manifest).not.toHaveProperty('message_serialization');
  });

  it('declares only the minimal extension permissions, with no network-request tricks', () => {
    expect(manifest.permissions).toEqual([
      'storage',
      'offscreen',
      'activeTab'
    ]);
  });

  it('declares the generated RealImage icon family for Chrome and the toolbar', () => {
    expect(manifest.icons).toEqual({
      16: 'icons/icon-16.png',
      32: 'icons/icon-32.png',
      48: 'icons/icon-48.png',
      128: 'icons/icon-128.png'
    });
    expect(manifest.action.default_icon).toEqual({
      16: 'icons/icon-16.png',
      32: 'icons/icon-32.png',
      48: 'icons/icon-48.png'
    });
  });

  it('pins one unmistakable candidate identity and model-only operating point', () => {
    // NOTE: MODEL previously carried finalistLockSha256/protocolSha256/
    // toolingSha256/qualificationRead fields belonging to an abandoned
    // installed-Chrome hash-lock/qualification-ceremony system (see
    // docs/INSTALLED_CHROME_PARITY_GATE.md for what remains of that
    // protocol's own document contract). That ceremony never ran for any
    // shipped candidate and was retired; the fields were removed from
    // MODEL rather than carried forward unused. RELEASE.name is the
    // user-facing extension name shown in chrome://extensions, so it was
    // cleaned up to drop the internal candidate-id jargon.
    expect(RELEASE).toEqual(expect.objectContaining({
      name: 'RealImage',
      version: '1.1.0'
    }));
    expect(MODEL).toEqual(expect.objectContaining({
      id: 'RealImage/broad-v1-modern-v1-1471e3ef',
      candidateId: 'realimage-broad-v1-mlp-modern-v1-fp32-1471e3ef',
      dtype: 'fp32',
      weightFile: 'onnx/model.onnx',
      decisionPolicy: 'model-only',
      weightSha256: '1471e3eff3a05d5ef8c068abfdef2f3f43d41060b27306f19763968cf8d38098'
    }));
    expect(MODEL).not.toHaveProperty('finalistLockSha256');
    expect(MODEL).not.toHaveProperty('protocolSha256');
    expect(MODEL).not.toHaveProperty('toolingSha256');
    expect(MODEL).not.toHaveProperty('qualificationRead');
    expect(MODEL.calibration).toEqual({
      rawThreshold: 0.646794855594635,
      displayThreshold: 0.65
    });
    expect(DEFAULT_SETTINGS.threshold).toBe(MODEL.calibration.displayThreshold);
    expect(calibrateDecisionScore(
      MODEL.calibration.rawThreshold,
      MODEL.calibration.rawThreshold,
      MODEL.calibration.displayThreshold
    )).toBe(MODEL.calibration.displayThreshold);
  });

  it('bundles exactly the frozen config, preprocessor, and FP32 graph bytes', async () => {
    for (const [relativePath, bytes, sha256] of [
      ['config.json', 510, MODEL.configSha256],
      ['preprocessor_config.json', 458, MODEL.preprocessorConfigSha256],
      [MODEL.weightFile, 87_437_806, MODEL.weightSha256]
    ]) {
      const path = resolve(modelRoot, relativePath);
      expect((await stat(path)).size).toBe(bytes);
      expect(await fileSha256(path)).toBe(sha256);
    }
  });
});

async function fileSha256(path) {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest('hex');
}
