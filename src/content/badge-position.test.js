import { describe, expect, it } from 'vitest';
import { findBadgePlacement } from './badge-position.js';

const base = {
  imageRect: { left: 100, top: 80, right: 500, bottom: 480 },
  badgeWidth: 120,
  badgeHeight: 32,
  viewportWidth: 800,
  viewportHeight: 600
};

describe('occlusion-aware badge positioning', () => {
  it('preserves the normal top-right image anchor', () => {
    expect(findBadgePlacement(base)).toMatchObject({ left: 372, top: 88, right: 492, bottom: 120 });
  });

  it('moves below a sticky top header that covers the top of the image', () => {
    const stickyHeader = { left: 0, top: 0, right: 800, bottom: 170 };
    const placement = findBadgePlacement({
      ...base,
      readBlocker: (candidate) => overlaps(candidate, stickyHeader) ? stickyHeader : undefined
    });
    expect(placement).toMatchObject({ left: 372, top: 178, right: 492, bottom: 210 });
  });

  it('moves left of a fixed side rail while retaining the highest usable anchor', () => {
    const sideRail = { left: 420, top: 0, right: 800, bottom: 600 };
    const placement = findBadgePlacement({
      ...base,
      readBlocker: (candidate) => overlaps(candidate, sideRail) ? sideRail : undefined
    });
    expect(placement).toMatchObject({ left: 292, top: 88, right: 412, bottom: 120 });
  });

  it('hides the badge when page chrome fully occludes the visible image', () => {
    const fullCover = { left: 0, top: 0, right: 800, bottom: 600 };
    expect(findBadgePlacement({ ...base, readBlocker: () => fullCover })).toBeUndefined();
  });

  it('keeps the badge inside viewport margins at the right edge', () => {
    const placement = findBadgePlacement({
      ...base,
      imageRect: { left: 690, top: 40, right: 850, bottom: 300 }
    });
    expect(placement).toMatchObject({ left: 666, right: 786, top: 48 });
  });
});

function overlaps(first, second) {
  return first.left < second.right
    && first.right > second.left
    && first.top < second.bottom
    && first.bottom > second.top;
}
