/**
 * OFFSCREEN_* messages are private service-worker-to-offscreen control traffic.
 * A same-extension content script also has our extension ID, but Chrome adds a
 * sender.tab for that path; it must not be able to bypass background validation.
 */
export function isTrustedOffscreenInternalSender(sender, extensionId) {
  return sender?.id === extensionId && sender.tab === undefined;
}
