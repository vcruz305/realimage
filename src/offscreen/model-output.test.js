import { describe, expect, it } from 'vitest';
import { extractModelLogit } from './model-output.js';
import { createUnavailableResponse } from './unavailable-response.js';

describe('model output contract', () => {
  it('accepts a numeric zero without replacing it', () => {
    expect(extractModelLogit({ logits: { data: new Float32Array([0]) } })).toBe(0);
  });

  it.each([NaN, Infinity, -Infinity, '0', null, undefined])(
    'turns malformed logit %s into an explicit unavailable response without a score',
    (value) => {
      let error;
      try {
        extractModelLogit({ logits: { data: [value] } });
      } catch (caught) {
        error = caught;
      }

      const response = createUnavailableResponse(error, 'request-1');
      expect(response).toMatchObject({
        ok: false,
        unavailable: true,
        code: 'MODEL_OUTPUT_INVALID',
        requestId: 'request-1'
      });
      expect(response).not.toHaveProperty('score');
    }
  );

  it('rejects a missing logits container', () => {
    expect(() => extractModelLogit({})).toThrowError(expect.objectContaining({ code: 'MODEL_OUTPUT_INVALID' }));
  });
});
