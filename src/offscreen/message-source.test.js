import { describe, expect, it } from 'vitest';
import { isTrustedOffscreenInternalSender } from './message-source.js';

describe('offscreen internal message source', () => {
  const extensionId = 'abcdefghijklmnopabcdefghijklmnop';

  it('accepts background messages that have the extension ID and no tab', () => {
    expect(isTrustedOffscreenInternalSender({ id: extensionId }, extensionId)).toBe(true);
  });

  it('rejects same-extension tab senders and foreign senders', () => {
    expect(isTrustedOffscreenInternalSender({ id: extensionId, tab: { id: 7 } }, extensionId)).toBe(false);
    expect(isTrustedOffscreenInternalSender({ id: 'foreign-extension' }, extensionId)).toBe(false);
    expect(isTrustedOffscreenInternalSender(undefined, extensionId)).toBe(false);
  });
});
