import { describe, expect, it } from 'vitest';
import {
  cancelInvisiblePendingRecords,
  collectOpenTree,
  collectOpenTreeBounded,
  findTerminalEvictionCandidate,
  isCurrentRecord,
  isVisibleBadgeAnchor,
  isVisiblyRenderedImage,
  releaseRecord,
  reserveRecord,
  reserveRecordWithTerminalEviction
} from './lifecycle.js';

describe('open-tree discovery', () => {
  it('finds light-DOM images and images in nested open shadow roots exactly once', () => {
    const lightImage = fakeImage('light');
    const firstShadowImage = fakeImage('first-shadow');
    const nestedShadowImage = fakeImage('nested-shadow');
    const closedImage = fakeImage('closed');

    const nestedHost = fakeElement();
    nestedHost.shadowRoot = fakeScope({ images: [nestedShadowImage], host: nestedHost });
    const openHost = fakeElement();
    openHost.shadowRoot = fakeScope({ images: [firstShadowImage], elements: [nestedHost], host: openHost });
    const closedHost = fakeElement();
    closedHost.closedRootForTest = fakeScope({ images: [closedImage], host: closedHost });
    const document = fakeScope({ images: [lightImage], elements: [openHost, closedHost] });

    const result = collectOpenTree(document);

    expect(result.images).toEqual([lightImage, firstShadowImage, nestedShadowImage]);
    expect(result.shadowRoots).toEqual([openHost.shadowRoot, nestedHost.shadowRoot]);
  });

  it('bounds visibility-recovery traversal while crossing nested open shadow roots', () => {
    const lightImage = fakeImage('bounded-light');
    const shadowImage = fakeImage('bounded-shadow');
    const shadowRoot = fakeElement();
    linkChildren(shadowRoot, [shadowImage]);
    const host = fakeElement();
    host.shadowRoot = shadowRoot;
    const root = fakeElement();
    linkChildren(root, [lightImage, host]);

    const complete = collectOpenTreeBounded(root, { maxNodes: 8, maxImages: 4 });
    expect(complete.images).toEqual([lightImage, shadowImage]);
    expect(complete.shadowRoots).toEqual([shadowRoot]);
    expect(complete.visitedNodes).toBe(5);
    expect(complete.truncated).toBe(false);

    const bounded = collectOpenTreeBounded(root, { maxNodes: 2, maxImages: 4 });
    expect(bounded.visitedNodes).toBe(2);
    expect(bounded.images).toEqual([lightImage]);
    expect(bounded.truncated).toBe(true);
  });
});

describe('image record lifecycle', () => {
  it('cancels pending records hidden at the image, ancestor, slot, or shadow host', () => {
    const direct = fakeImage('direct-hidden');
    direct.hidden = true;

    const ancestorHidden = fakeImage('ancestor-hidden');
    const ancestor = fakeElement();
    ancestorHidden.parentElement = ancestor;

    const slotHidden = fakeImage('slot-hidden');
    const slot = fakeElement();
    slotHidden.assignedSlot = slot;

    const hostHidden = fakeImage('host-hidden');
    const host = fakeElement();
    hostHidden.getRootNode = () => ({ host });

    const visible = fakeImage('visible-pending');
    const terminalHidden = fakeImage('terminal-hidden');
    terminalHidden.hidden = true;
    const records = new Map([
      [direct, { id: 'direct', status: 'analyzing' }],
      [ancestorHidden, { id: 'ancestor', status: 'analyzing' }],
      [slotHidden, { id: 'slot', status: 'analyzing' }],
      [hostHidden, { id: 'host', status: 'analyzing' }],
      [visible, { id: 'visible', status: 'analyzing' }],
      [terminalHidden, { id: 'terminal', status: 'complete' }]
    ]);
    const hiddenStyles = new Map([
      [ancestor, { display: 'none' }],
      [slot, { visibility: 'hidden' }],
      [host, { opacity: '0' }]
    ]);
    const readStyle = (node) => style(hiddenStyles.get(node));
    const cancelled = [];

    expect(cancelInvisiblePendingRecords(records, {
      isVisible: (image) => isVisiblyRenderedImage(image, readStyle),
      onCancel(image, record) {
        cancelled.push(record.id);
        releaseRecord(records, image, record);
      }
    })).toBe(4);

    expect(cancelled).toEqual(['direct', 'ancestor', 'slot', 'host']);
    expect([...records.values()].map((record) => record.id)).toEqual(['visible', 'terminal']);
  });

  it('rejects layout-sized images that CSS keeps hidden', () => {
    const image = fakeImage('google-hidden-duplicate');
    const style = (overrides = {}) => ({
      contentVisibility: 'visible',
      display: 'block',
      opacity: '1',
      visibility: 'visible',
      ...overrides
    });

    expect(isVisiblyRenderedImage(image, () => style())).toBe(true);
    expect(isVisiblyRenderedImage(image, () => style({ visibility: 'hidden' }))).toBe(false);
    expect(isVisiblyRenderedImage(image, () => style({ visibility: 'collapse' }))).toBe(false);
    expect(isVisiblyRenderedImage(image, () => style({ display: 'none' }))).toBe(false);
    expect(isVisiblyRenderedImage(image, () => style({ contentVisibility: 'hidden' }))).toBe(false);
    expect(isVisiblyRenderedImage(image, () => style({ opacity: '0' }))).toBe(false);
    image.hidden = true;
    expect(isVisiblyRenderedImage(image, () => style())).toBe(false);
  });

  it('keeps the reveal badge for RealImage hide treatment while rejecting page hiding', () => {
    const image = fakeImage('realimage-hidden-ai');
    const style = (overrides = {}) => ({
      contentVisibility: 'visible',
      display: 'block',
      opacity: '1',
      visibility: 'visible',
      ...overrides
    });

    image.dataset.proofmarkTreatment = 'hide';
    expect(isVisibleBadgeAnchor(image, () => style({ visibility: 'hidden' }))).toBe(true);
    expect(isVisibleBadgeAnchor(image, () => style({ display: 'none', visibility: 'hidden' }))).toBe(false);
    expect(isVisibleBadgeAnchor(image, () => style({ opacity: '0', visibility: 'hidden' }))).toBe(false);
    image.checkVisibility = () => false;
    expect(isVisibleBadgeAnchor(image, () => style({ visibility: 'hidden' }))).toBe(false);

    delete image.dataset.proofmarkTreatment;
    delete image.checkVisibility;
    expect(isVisibleBadgeAnchor(image, () => style({ visibility: 'hidden' }))).toBe(false);
  });

  it('rejects a RealImage-hidden shadow image when an ancestor host is page-hidden', () => {
    const image = fakeImage('shadow-hidden-ai');
    const host = fakeElement();
    const shadowRoot = { host };
    image.dataset.proofmarkTreatment = 'hide';
    image.getRootNode = () => shadowRoot;
    const readStyle = (node) => ({
      contentVisibility: 'visible',
      display: 'block',
      opacity: '1',
      visibility: node === host ? 'hidden' : 'visible'
    });

    expect(isVisibleBadgeAnchor(image, readStyle)).toBe(false);
    expect(isVisibleBadgeAnchor(image, () => ({
      contentVisibility: 'visible',
      display: 'block',
      opacity: '1',
      visibility: 'visible'
    }))).toBe(true);
  });

  it('uses the assigned slot as the first composed ancestor for badge visibility', () => {
    const image = fakeImage('slotted-hidden-ai');
    const lightParent = fakeElement();
    const slot = fakeElement();
    image.dataset.proofmarkTreatment = 'hide';
    image.parentElement = lightParent;
    image.assignedSlot = slot;

    const readStyle = (node) => ({
      contentVisibility: 'visible',
      display: 'block',
      opacity: '1',
      visibility: node === slot ? 'hidden' : 'visible'
    });

    expect(isVisibleBadgeAnchor(image, readStyle)).toBe(false);
  });

  it('fails closed when the composed ancestor chain exceeds the 64-node bound', () => {
    const image = fakeImage('deep-hidden-ai');
    image.dataset.proofmarkTreatment = 'hide';
    let child = image;
    const ancestors = [];
    for (let index = 0; index < 65; index += 1) {
      const ancestor = fakeElement();
      ancestors.push(ancestor);
      child.parentElement = ancestor;
      child = ancestor;
    }

    const readStyle = (node) => ({
      contentVisibility: 'visible',
      display: 'block',
      opacity: '1',
      visibility: node === ancestors[64] ? 'hidden' : 'visible'
    });
    expect(isVisibleBadgeAnchor(image, readStyle)).toBe(false);

    ancestors[63].parentElement = undefined;
    expect(isVisibleBadgeAnchor(image, () => ({
      contentVisibility: 'visible',
      display: 'block',
      opacity: '1',
      visibility: 'visible'
    }))).toBe(true);
  });

  it('refuses to enqueue a visibility:hidden image even with a nonzero layout rect', () => {
    // Reproduces the Google Images duplicate-thumbnail ghost badge: the node
    // is visibility:hidden but still occupies real layout space, so a check
    // that only inspected getBoundingClientRect would have admitted it.
    const image = fakeImage('google-hidden-duplicate-enqueue');
    image.getBoundingClientRect = () => ({ width: 200, height: 150, top: 0, left: 0, right: 200, bottom: 150 });
    const hiddenStyle = () => ({ contentVisibility: 'visible', display: 'block', opacity: '1', visibility: 'hidden' });
    const isVisible = (candidate) => isVisiblyRenderedImage(candidate, hiddenStyle);
    const records = new Map();

    expect(image.getBoundingClientRect().width).toBeGreaterThan(0);
    expect(reserveRecord(records, image, { source: image.currentSrc }, 5, { isVisible })).toBe(false);
    expect(records.has(image)).toBe(false);

    expect(reserveRecordWithTerminalEviction(records, image, { source: image.currentSrc }, 5, {
      isOffViewport: () => true,
      isVisible,
      onEvict: () => { throw new Error('must not evict for a rejected admission'); }
    })).toBe(false);
    expect(records.has(image)).toBe(false);

    const visibleStyle = () => ({ contentVisibility: 'visible', display: 'block', opacity: '1', visibility: 'visible' });
    expect(reserveRecord(records, image, { source: image.currentSrc }, 5, {
      isVisible: (candidate) => isVisiblyRenderedImage(candidate, visibleStyle)
    })).toBe(true);
  });

  it('enforces the cap atomically and reuses a slot after release', () => {
    const records = new Map();
    const first = fakeImage('one');
    const second = fakeImage('two');

    expect(reserveRecord(records, first, { source: first.currentSrc }, 1)).toBe(true);
    expect(reserveRecord(records, second, { source: second.currentSrc }, 1)).toBe(false);
    expect(releaseRecord(records, first)).toBeDefined();
    expect(reserveRecord(records, second, { source: second.currentSrc }, 1)).toBe(true);
    expect(records.size).toBe(1);
  });

  it('rejects replaced, disconnected, adopted, and source-stale async records', () => {
    const ownerDocument = {};
    const image = fakeImage('original', ownerDocument);
    const original = { source: image.currentSrc };
    const records = new Map([[image, original]]);

    expect(isCurrentRecord(records, image, original, ownerDocument)).toBe(true);
    image.currentSrc = 'https://example.test/recycled.jpg';
    expect(isCurrentRecord(records, image, original, ownerDocument)).toBe(false);
    image.currentSrc = original.source;
    image.isConnected = false;
    expect(isCurrentRecord(records, image, original, ownerDocument)).toBe(false);
    image.isConnected = true;
    image.ownerDocument = {};
    expect(isCurrentRecord(records, image, original, ownerDocument)).toBe(false);
    image.ownerDocument = ownerDocument;
    records.set(image, { source: image.currentSrc });
    expect(isCurrentRecord(records, image, original, ownerDocument)).toBe(false);
    expect(releaseRecord(records, image, original)).toBeUndefined();
  });

  it('advances a static gallery beyond the live-record cap as the viewport moves', () => {
    const images = Array.from({ length: 6 }, (_, index) => fakeImage(`gallery-${index}`));
    const records = new Map();
    const observed = new Set(images);
    const outside = new Set();
    const evicted = [];
    const cleanup = (image, record) => {
      evicted.push(record.id);
      delete image.dataset.proofmarkTreatment;
      record.badge.remove();
      releaseRecord(records, image, record);
    };

    for (const image of images.slice(0, 2)) {
      image.dataset.proofmarkTreatment = 'blur';
      expect(reserveRecord(records, image, terminalRecord(image.name, image), 2)).toBe(true);
    }

    for (let index = 2; index < images.length; index += 1) {
      outside.add(images[index - 2]);
      const image = images[index];
      image.dataset.proofmarkTreatment = 'blur';
      const admitted = reserveRecordWithTerminalEviction(records, image, terminalRecord(image.name, image), 2, {
        isOffViewport: (candidate) => outside.has(candidate),
        onEvict: cleanup
      });
      expect(admitted).toBe(true);
    }

    expect([...records.keys()]).toEqual(images.slice(-2));
    expect(evicted).toEqual(images.slice(0, -2).map((image) => image.name));
    expect(images.slice(0, -2).every((image) => image.dataset.proofmarkTreatment === undefined)).toBe(true);
    expect(images.slice(0, -2).every((image) => image.badgeRemoved)).toBe(true);
    expect(observed.size).toBe(images.length);
  });

  it('never evicts analyzing or protected-detail records', () => {
    const activeImage = fakeImage('active');
    const detailImage = fakeImage('detail');
    const safeImage = fakeImage('safe');
    const incomingImage = fakeImage('incoming');
    const active = { id: 'active', status: 'analyzing' };
    const detail = terminalRecord('detail');
    const safe = terminalRecord('safe');
    const records = new Map([
      [activeImage, active],
      [detailImage, detail],
      [safeImage, safe]
    ]);
    let evicted;

    expect(reserveRecordWithTerminalEviction(records, incomingImage, terminalRecord('incoming'), 3, {
      isOffViewport: () => true,
      isProtected: (record) => record === detail,
      onEvict: (image, record) => {
        evicted = record;
        releaseRecord(records, image, record);
      }
    })).toBe(true);

    expect(evicted).toBe(safe);
    expect(records.get(activeImage)).toBe(active);
    expect(records.get(detailImage)).toBe(detail);
    expect(records.has(incomingImage)).toBe(true);
  });

  it('fails closed when no terminal off-viewport record can be released', () => {
    const activeImage = fakeImage('active-only');
    const visibleImage = fakeImage('visible-only');
    const incomingImage = fakeImage('blocked');
    const records = new Map([
      [activeImage, { id: 'active-only', status: 'analyzing' }],
      [visibleImage, terminalRecord('visible-only')]
    ]);
    const onEvict = () => { throw new Error('must not evict'); };

    expect(findTerminalEvictionCandidate(records, { isOffViewport: () => false })).toBeUndefined();
    expect(reserveRecordWithTerminalEviction(records, incomingImage, terminalRecord('blocked'), 2, {
      isOffViewport: () => false,
      onEvict
    })).toBe(false);
    expect(records.has(incomingImage)).toBe(false);
    expect(records.size).toBe(2);
  });
});

function fakeImage(name, ownerDocument = {}) {
  return {
    localName: 'img',
    name,
    currentSrc: `https://example.test/${name}.jpg`,
    isConnected: true,
    ownerDocument,
    dataset: {},
    badgeRemoved: false,
    nodeType: 1,
    querySelectorAll: () => []
  };
}

function terminalRecord(id, image) {
  return {
    id,
    status: 'complete',
    badge: {
      remove() {
        if (image) image.badgeRemoved = true;
      }
    }
  };
}

function fakeElement() {
  return { localName: 'div', nodeType: 1, querySelectorAll: () => [] };
}

function style(overrides = {}) {
  return {
    contentVisibility: 'visible',
    display: 'block',
    opacity: '1',
    visibility: 'visible',
    ...overrides
  };
}

function linkChildren(parent, children) {
  parent.firstElementChild = children[0];
  for (let index = 0; index < children.length; index += 1) {
    children[index].nextElementSibling = children[index + 1];
    children[index].parentElement = parent.nodeType === 1 ? parent : undefined;
  }
}

function fakeScope({ images = [], elements = [], host } = {}) {
  return {
    host,
    nodeType: host ? 11 : 9,
    querySelectorAll(selector) {
      if (selector === 'img') return images;
      if (selector === '*') return [...images, ...elements];
      return [];
    }
  };
}
