import { describe, expect, it, vi } from 'vitest';
import {
  beginDetailDialogRender,
  closeDetailDialog,
  commitDetailDialogRender,
  DETAIL_ACTION_SELECTOR,
  DETAIL_CLOSE_SELECTOR,
  focusDetailControl,
  isEscapeDialogKey
} from './detail-dialog.js';

describe('detail dialog keyboard focus', () => {
  it('clears and hides stale content until a completed render commits its new anchor', () => {
    const panel = { hidden: false, dataset: { anchor: 'pm-stale' } };

    expect(beginDetailDialogRender(panel)).toBe(true);
    expect(panel).toEqual({ hidden: true, dataset: {} });

    // A render that throws before commit remains hidden and unanchored.
    expect(panel.hidden).toBe(true);
    expect(panel.dataset.anchor).toBeUndefined();

    expect(commitDetailDialogRender(panel, 'pm-current')).toBe(true);
    expect(panel).toEqual({ hidden: false, dataset: { anchor: 'pm-current' } });
    expect(commitDetailDialogRender(panel, '')).toBe(false);
  });

  it('moves focus to close on open and to the replacement action after rerender', () => {
    const close = { focus: vi.fn() };
    const firstAction = { focus: vi.fn() };
    const replacementAction = { focus: vi.fn() };
    let action = firstAction;
    const panel = {
      querySelector(selector) {
        if (selector === DETAIL_CLOSE_SELECTOR) return close;
        if (selector === DETAIL_ACTION_SELECTOR) return action;
        return undefined;
      }
    };

    expect(focusDetailControl(panel)).toBe(true);
    expect(close.focus).toHaveBeenCalledWith({ preventScroll: true });

    action = replacementAction;
    expect(focusDetailControl(panel, DETAIL_ACTION_SELECTOR)).toBe(true);
    expect(firstAction.focus).not.toHaveBeenCalled();
    expect(replacementAction.focus).toHaveBeenCalledWith({ preventScroll: true });
  });

  it('recognizes an unhandled, non-composing Escape only', () => {
    expect(isEscapeDialogKey({ key: 'Escape' })).toBe(true);
    expect(isEscapeDialogKey({ key: 'Enter' })).toBe(false);
    expect(isEscapeDialogKey({ key: 'Escape', defaultPrevented: true })).toBe(false);
    expect(isEscapeDialogKey({ key: 'Escape', isComposing: true })).toBe(false);
  });

  it('closes, clears the anchor, and returns focus to the invoking badge', () => {
    const panel = { hidden: false, dataset: { anchor: 'pm-one' } };
    const trigger = {
      focus: vi.fn(),
      hidden: false,
      isConnected: true,
      style: { display: 'flex' }
    };

    expect(closeDetailDialog(panel, trigger)).toBe(true);
    expect(panel.hidden).toBe(true);
    expect(panel.dataset.anchor).toBeUndefined();
    expect(trigger.focus).toHaveBeenCalledWith({ preventScroll: true });
  });

  it('does not focus a removed/hidden trigger or restore focus during teardown', () => {
    const removed = { focus: vi.fn(), hidden: false, isConnected: false, style: {} };
    const hidden = { focus: vi.fn(), hidden: true, isConnected: true, style: {} };

    expect(closeDetailDialog({ hidden: false, dataset: {} }, removed)).toBe(false);
    expect(closeDetailDialog({ hidden: false, dataset: {} }, hidden)).toBe(false);
    expect(closeDetailDialog(
      { hidden: false, dataset: {} },
      { focus: vi.fn(), hidden: false, isConnected: true, style: {} },
      { restoreFocus: false }
    )).toBe(false);
    expect(removed.focus).not.toHaveBeenCalled();
    expect(hidden.focus).not.toHaveBeenCalled();
  });
});
