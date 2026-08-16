import { describe, expect, it } from 'vitest';
import { formatScorePercent } from './score-display.js';

describe('formatScorePercent', () => {
  it('does not display a below-threshold score as meeting the threshold', () => {
    expect(formatScorePercent(0.649999, 0.65)).toBe('64.9');
  });

  it('keeps every supported integer-percent threshold fail-safe at rounding boundaries', () => {
    for (let thresholdPercent = 50; thresholdPercent <= 95; thresholdPercent += 1) {
      const threshold = thresholdPercent / 100;
      for (const distance of [Number.EPSILON, 0.000001, 0.000499]) {
        const score = threshold - distance;
        const displayedScore = Number(formatScorePercent(score, threshold)) / 100;
        expect(displayedScore, `threshold=${threshold} score=${score}`).toBeLessThan(threshold);
      }
      expect(Number(formatScorePercent(threshold, threshold)) / 100).toBeGreaterThanOrEqual(threshold);
    }
  });

  it('displays the exact threshold and above-threshold values consistently', () => {
    expect(formatScorePercent(0.65, 0.65)).toBe('65.0');
    expect(formatScorePercent(0.65001, 0.65)).toBe('65.0');
  });

  it('uses ordinary one-decimal rounding away from the boundary', () => {
    expect(formatScorePercent(0.1236, 0.65)).toBe('12.4');
    expect(formatScorePercent(1, 0.65)).toBe('100.0');
  });

  it('rejects invalid score inputs', () => {
    expect(() => formatScorePercent(Number.NaN, 0.65)).toThrow(RangeError);
    expect(() => formatScorePercent(1.01, 0.65)).toThrow(RangeError);
  });
});
