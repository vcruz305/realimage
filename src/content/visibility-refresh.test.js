import { describe, expect, it } from 'vitest';
import { isVisiblyRenderedImage } from './lifecycle.js';
import { createVisibilityRefreshController } from './visibility-refresh.js';

describe('coalesced visibility refresh', () => {
  it('re-observes an intersecting image revealed only by an ancestor style change', () => {
    const tasks = [];
    const ancestor = fakeElement();
    const image = fakeImage();
    linkChildren(ancestor, [image]);
    let visible = false;
    let analyzed = 0;
    const observerCalls = [];
    const controller = createVisibilityRefreshController({
      belongsToDocument: () => true,
      hasRecord: () => analyzed > 0,
      observeImage(candidate, refresh) {
        observerCalls.push({ candidate, refresh });
        if (isVisiblyRenderedImage(candidate, () => style({
          visibility: visible ? 'visible' : 'hidden'
        }))) analyzed += 1;
      },
      observeShadowRoot: () => {},
      scheduleTask: (task) => tasks.push(task)
    });

    expect(isVisiblyRenderedImage(image, () => style({ visibility: 'hidden' }))).toBe(false);
    controller.remember(image);
    controller.schedule(ancestor);
    controller.schedule(ancestor);
    expect(tasks).toHaveLength(1);

    visible = true;
    tasks.shift()();

    expect(observerCalls).toEqual([{ candidate: image, refresh: true }]);
    expect(analyzed).toBe(1);
  });

  it('finds descendant images in open shadow trees within one global work budget', () => {
    const tasks = [];
    const image = fakeImage();
    const shadowRoot = fakeElement();
    shadowRoot.nodeType = 11;
    linkChildren(shadowRoot, [image]);
    const host = fakeElement();
    host.shadowRoot = shadowRoot;
    const ancestor = fakeElement();
    linkChildren(ancestor, [host]);
    const observedImages = [];
    const observedRoots = [];
    const controller = createVisibilityRefreshController({
      belongsToDocument: () => true,
      hasRecord: () => false,
      observeImage: (candidate, refresh) => observedImages.push([candidate, refresh]),
      observeShadowRoot: (root) => observedRoots.push(root),
      scheduleTask: (task) => tasks.push(task),
      maxNodes: 8,
      maxImages: 2
    });

    controller.schedule(ancestor);
    tasks.shift()();

    expect(observedRoots).toEqual([shadowRoot]);
    expect(observedImages).toEqual([[image, true]]);
  });

  it('caps remembered hidden-image retries', () => {
    const tasks = [];
    const calls = [];
    const remembered = [fakeImage(), fakeImage(), fakeImage()];
    const controller = createVisibilityRefreshController({
      belongsToDocument: () => true,
      hasRecord: () => false,
      observeImage: (image) => calls.push(image),
      observeShadowRoot: () => {},
      scheduleTask: (task) => tasks.push(task),
      maxRoots: 2,
      maxNodes: 7,
      maxImages: 2
    });

    expect(controller.remember(remembered[0])).toBe(true);
    expect(controller.remember(remembered[1])).toBe(true);
    expect(controller.remember(remembered[2])).toBe(false);
    controller.schedule(fakeElement());
    controller.schedule(fakeElement());
    controller.schedule(fakeElement());
    expect(tasks).toHaveLength(1);
    tasks.shift()();

    // Remembered hidden candidates have priority and consume the image budget,
    // so no descendant scan can expand work past that cap.
    expect(calls).toEqual(remembered.slice(0, 2));
  });

  it('caps coalesced mutation roots and shares one node/image budget across them', () => {
    const tasks = [];
    const roots = [fakeElement(), fakeElement(), fakeElement()];
    const images = [fakeImage(), fakeImage()];
    const scans = [];
    const observed = [];
    const controller = createVisibilityRefreshController({
      belongsToDocument: () => true,
      hasRecord: () => false,
      observeImage: (image) => observed.push(image),
      observeShadowRoot: () => {},
      scheduleTask: (task) => tasks.push(task),
      collectTree(root, limits) {
        scans.push({ root, limits });
        const index = scans.length - 1;
        return {
          images: [images[index]],
          shadowRoots: [],
          visitedNodes: index === 0 ? 4 : 3,
          truncated: false
        };
      },
      maxRoots: 2,
      maxNodes: 7,
      maxImages: 2
    });

    for (const root of roots) controller.schedule(root);
    expect(tasks).toHaveLength(1);
    tasks.shift()();

    expect(scans).toEqual([
      { root: roots[0], limits: { maxNodes: 7, maxImages: 2 } },
      { root: roots[1], limits: { maxNodes: 3, maxImages: 1 } }
    ]);
    expect(observed).toEqual(images);
  });
});

function fakeElement() {
  return { localName: 'div', nodeType: 1 };
}

function fakeImage() {
  return {
    localName: 'img',
    nodeType: 1,
    hidden: false,
    dataset: {}
  };
}

function linkChildren(parent, children) {
  parent.firstElementChild = children[0];
  for (let index = 0; index < children.length; index += 1) {
    children[index].nextElementSibling = children[index + 1];
    children[index].parentElement = parent.nodeType === 1 ? parent : undefined;
  }
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
