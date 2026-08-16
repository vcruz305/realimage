export const DETAIL_CLOSE_SELECTOR = '[data-proofmark-close]';
export const DETAIL_ACTION_SELECTOR = '[data-proofmark-reveal]';

export function beginDetailDialogRender(panel) {
  if (!panel) return false;
  panel.hidden = true;
  if (panel.dataset) delete panel.dataset.anchor;
  return true;
}

export function commitDetailDialogRender(panel, anchor) {
  if (!panel?.dataset || typeof anchor !== 'string' || !anchor) return false;
  panel.dataset.anchor = anchor;
  panel.hidden = false;
  return true;
}

export function focusDetailControl(panel, selector = DETAIL_CLOSE_SELECTOR) {
  const target = panel?.querySelector?.(selector);
  if (!target || typeof target.focus !== 'function') return false;
  try {
    target.focus({ preventScroll: true });
    return true;
  } catch {
    return false;
  }
}

export function isEscapeDialogKey(event) {
  return Boolean(event && event.key === 'Escape' && !event.defaultPrevented && !event.isComposing);
}

export function closeDetailDialog(panel, returnFocus, { restoreFocus = true } = {}) {
  if (panel) {
    panel.hidden = true;
    if (panel.dataset) delete panel.dataset.anchor;
  }
  if (!restoreFocus || !isFocusableTrigger(returnFocus)) return false;
  try {
    returnFocus.focus({ preventScroll: true });
    return true;
  } catch {
    return false;
  }
}

function isFocusableTrigger(trigger) {
  return Boolean(
    trigger
    && trigger.isConnected
    && !trigger.hidden
    && trigger.style?.display !== 'none'
    && typeof trigger.focus === 'function'
  );
}
