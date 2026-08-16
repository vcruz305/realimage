import { describe, expect, it } from 'vitest';
import {
  applyBadgeImageLink,
  applyBadgeResultContract,
  clearBadgeImageLink
} from './badge-result-contract.js';

describe('badge browser-validation result contract', () => {
  it('keeps a below-threshold model verdict separate from a declaration', () => {
    const badge = { dataset: {} };
    applyBadgeResultContract(badge, {
      isAi: false,
      displayScore: '0.8',
      decisionThreshold: 0.65,
      declaration: { claim: 'edited', badgeText: 'Declared AI-edited' }
    });

    expect(badge.dataset).toEqual({
      proofmarkDecisionThreshold: '0.65',
      proofmarkModelVerdict: 'real',
      proofmarkModelScorePercent: '0.8',
      proofmarkDeclarationType: 'edited',
      proofmarkDeclarationSummary: 'Declared AI-edited'
    });
  });

  it('reports an AI model verdict independently when declaration metadata also exists', () => {
    const badge = { dataset: {} };
    applyBadgeResultContract(badge, {
      isAi: true,
      displayScore: '93.6',
      decisionThreshold: 0.65,
      declaration: { claim: 'composite', badgeText: 'Declared AI composite' }
    });

    expect(badge.dataset.proofmarkModelVerdict).toBe('ai');
    expect(badge.dataset.proofmarkModelScorePercent).toBe('93.6');
    expect(badge.dataset.proofmarkDeclarationType).toBe('composite');
  });

  it('deletes stale declaration attributes when a rerender has no declaration', () => {
    const badge = {
      dataset: {
        proofmarkDeclarationType: 'generated',
        proofmarkDeclarationSummary: 'Declared AI-generated'
      }
    };
    applyBadgeResultContract(badge, {
      isAi: false,
      displayScore: '12.4',
      decisionThreshold: 0.65,
      declaration: null
    });

    expect(badge.dataset).toEqual({
      proofmarkDecisionThreshold: '0.65',
      proofmarkModelVerdict: 'real',
      proofmarkModelScorePercent: '12.4'
    });
  });

  it('links an image to one opaque badge ID and clears only the matching record link', () => {
    const image = { dataset: {} };
    const badge = { dataset: { proofmarkId: 'pm-opaque-1' } };

    expect(applyBadgeImageLink(image, badge)).toBe(true);
    expect(image.dataset).toEqual({ proofmarkBadgeId: 'pm-opaque-1' });
    expect(clearBadgeImageLink(image, 'pm-other')).toBe(false);
    expect(image.dataset.proofmarkBadgeId).toBe('pm-opaque-1');
    expect(clearBadgeImageLink(image, 'pm-opaque-1')).toBe(true);
    expect(image.dataset).toEqual({});

    expect(applyBadgeImageLink({}, badge)).toBe(false);
    expect(applyBadgeImageLink(image, { dataset: {} })).toBe(false);
  });
});
