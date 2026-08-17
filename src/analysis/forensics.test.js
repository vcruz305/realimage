import { describe, expect, it } from 'vitest';
import { calibrateDecisionScore, collectPageContextEvidence, computePixelSignals, fuseEvidence, inspectEncodedImage } from './forensics.js';

describe('encoded image forensics', () => {
  it('finds Stable Diffusion generation parameters in PNG text chunks', () => {
    const bytes = makePngWithText('parameters\0a lighthouse\nSteps: 30, Sampler: Euler, Seed: 42, Model hash: abc123, stable diffusion');
    const result = inspectEncodedImage(bytes.buffer, 'image/png');
    expect(result.format).toBe('png');
    expect(result.evidence.map((item) => item.label)).toContain('Stable Diffusion generator metadata');
    expect(result.evidence.map((item) => item.label)).toContain('Embedded generation parameters');
    expect(result.evidence.filter((item) => item.claim)).toEqual([]);
  });

  it('surfaces provenance and watermark labels separately', () => {
    const bytes = makeWebpWithXmp('Content Credentials c2pa synthid');
    const result = inspectEncodedImage(bytes.buffer, 'image/webp');
    expect(result.provenance.length).toBeGreaterThan(0);
    expect(result.watermarks[0].label).toMatch(/SynthID/);
  });

  it('does not treat ordinary EXIF camera text as AI evidence', () => {
    const bytes = new TextEncoder().encode('Exif\0\0Make=Fujifilm;Model=X-T5;Software=Capture One');
    const result = inspectEncodedImage(bytes.buffer, 'image/jpeg');
    expect(result.evidence).toEqual([]);
    expect(result.watermarks).toEqual([]);
  });

  it('scans late PNG metadata without treating compressed pixel bytes as metadata', () => {
    const pixelPayload = new Uint8Array(200_000);
    pixelPayload.set(new TextEncoder().encode('stable diffusion, steps: 30, seed: 42'), 100_000);
    const withoutMetadata = makePngWithChunks([
      makePngChunk('IDAT', pixelPayload),
      makePngChunk('IEND', new Uint8Array())
    ]);
    expect(inspectEncodedImage(withoutMetadata.buffer, 'image/png').evidence).toEqual([]);

    const withMetadata = makePngWithChunks([
      makePngChunk('IDAT', new Uint8Array(200_000)),
      makePngChunk('tEXt', new TextEncoder().encode('parameters\0stable diffusion\nSteps: 30, Seed: 42')),
      makePngChunk('IEND', new Uint8Array())
    ]);
    expect(inspectEncodedImage(withMetadata.buffer, 'image/png').evidence.length).toBeGreaterThan(0);
  });

  it('distinguishes exact IPTC XMP attribute/element declarations and a standalone APP13 edit field', () => {
    const cases = [
      ['trainedAlgorithmicMedia', 'generated', 'IPTC: created using Generative AI'],
      ['compositeWithTrainedAlgorithmicMedia', 'edited', 'IPTC: edited using Generative AI'],
      ['compositeSynthetic', 'composite', 'IPTC: composite includes Generative AI elements']
    ];
    for (const [identifier, claim, label] of cases) {
      const uri = `http://cv.iptc.org/newscodes/digitalsourcetype/${identifier}`;
      for (const property of [
        `<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${uri}"/>`,
        `<rdf:Description rdf:about=""><Iptc4xmpExt:DigitalSourceType>${uri}</Iptc4xmpExt:DigitalSourceType></rdf:Description>`
      ]) {
        const result = inspectEncodedImage(makeJpegWithMetadataSegments([
          [0xe1, standardXmp(property)]
        ]).buffer, 'image/jpeg');
        expect(result.evidence).toContainEqual(expect.objectContaining({ authority: 'iptc', claim, label }));
      }
    }

    const edited = inspectEncodedImage(makeJpegWithMetadataSegments([
      [0xe1, standardXmp('<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"/>')],
      [0xed, makePhotoshopIptcResource()]
    ]).buffer, 'image/jpeg');
    expect(edited.evidence).toEqual(expect.arrayContaining([
      expect.objectContaining({ authority: 'iptc', claim: 'edited' }),
      expect.objectContaining({ authority: 'self-declared', claim: 'edited', label: 'Self-declared Google AI edit metadata' })
    ]));
  });

  it('fails closed for free text, wrong properties, qcodes, suffixes, negation, extensions, and conflicting IPTC values', () => {
    const generated = 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia';
    const edited = 'http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia';
    const probes = [
      [0xe1, standardXmp(`<dc:description>${generated}</dc:description>`) ],
      [0xe1, standardXmp(`<rdf:Description rdf:about=""><dc:description><![CDATA[<Iptc4xmpExt:DigitalSourceType>${generated}</Iptc4xmpExt:DigitalSourceType>]]></dc:description></rdf:Description>`) ],
      [0xe1, standardXmp(`<rdf:Description rdf:about="" Wrong:DigitalSourceType="${generated}"/>`)],
      [0xe1, standardXmp(`<rdf:Description rdf:about="" wrong:Iptc4xmpExt:DigitalSourceType="${generated}"/>`)],
      [0xe1, standardXmp('<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="digsrctype:compositeSynthetic"/>')],
      [0xe1, standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}Fake"/>`)],
      [0xe1, standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="prefix${generated}"/>`)],
      [0xe1, standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}/"/>`)],
      [0xe1, standardXmp('<dc:description>Edited with Google AI</dc:description>')],
      [0xed, 'This photo was not Edited with Google AI'],
      [0xed, '\0Edited with Google AI\0'],
      [0xed, '\x1cEdited with Google AI tools\x1c'],
      [0xed, makePhotoshopIptcResource({ resourceId: 0x040c })]
    ];
    for (const probe of probes) {
      const result = inspectEncodedImage(makeJpegWithMetadataSegments([probe]).buffer, 'image/jpeg');
      expect(result.evidence.filter((item) => item.claim)).toEqual([]);
    }

    const conflicting = inspectEncodedImage(makeJpegWithMetadataSegments([
      [0xe1, standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`)],
      [0xe1, standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${edited}"/>`)]
    ]).buffer, 'image/jpeg');
    expect(conflicting.evidence.filter((item) => item.authority === 'iptc')).toEqual([]);

    const overflow = inspectEncodedImage(makeJpegWithMetadataSegments([
      ...Array.from({ length: 64 }, () => [
        0xe1,
        standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`)
      ]),
      [0xe1, standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${edited}"/>`)]
    ]).buffer, 'image/jpeg');
    expect(overflow.metadataTruncated).toBe(true);
    expect(overflow.evidence.filter((item) => item.authority === 'iptc')).toEqual([]);
  });

  it('requires the image RDF subject, top-level hierarchy, a single root, and a valid IPTC Photoshop resource', () => {
    const generated = 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia';
    const xmpProbes = [
      standardXmp(`<rdf:Description Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`<rdf:Description rdf:about="https://example.invalid/other" Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`<dc:description><rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/></dc:description>`),
      `${standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`)}GARBAGE`,
      `${standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`)}${standardXmp('')}`,
      standardXmp(`<evil/><rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`GARBAGE<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`).replace(
        '</x:xmpmeta>',
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"></rdf:RDF></x:xmpmeta>'
      ),
      standardXmp(`<rdf:Description rdf:about="" rdf:ID="other" Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`<rdf:Description rdf:about="" rdf:nodeID="other" Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`<rdf:Description rdf:about=""><Iptc4xmpExt:DigitalSourceType rdf:resource="${generated}">${generated}</Iptc4xmpExt:DigitalSourceType></rdf:Description>`),
      standardXmp(`<rdf:Description rdf:about="" rdf:resource="other" Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`<![CDATA[GARBAGE]]><rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"/>`),
      standardXmp(`<rdf:Description rdf:about="" Iptc4xmpExt:DigitalSourceType="${generated}"><![CDATA[GARBAGE]]></rdf:Description>`)
    ];
    for (const [index, xmp] of xmpProbes.entries()) {
      const result = inspectEncodedImage(makeJpegWithMetadataSegments([[0xe1, xmp]]).buffer, 'image/jpeg');
      if ([3, 4, 5, 6, 7, 12, 13].includes(index)) {
        expect(result.metadataTruncated, `XMP structural probe ${index}`).toBe(true);
      }
      expect(result.evidence.filter((item) => item.claim)).toEqual([]);
    }

    const malformedIrb = makePhotoshopIptcResource();
    malformedIrb[20] = 0xff;
    for (const payload of [
      makeIimField(2, 110, GOOGLE_AI_EDIT_TEST_TEXT),
      makePhotoshopIptcResource({ resourceId: 0x040c }),
      malformedIrb
    ]) {
      const result = inspectEncodedImage(makeJpegWithMetadataSegments([[0xed, payload]]).buffer, 'image/jpeg');
      expect(result.evidence.filter((item) => item.claim)).toEqual([]);
    }

    for (const payload of [
      makePhotoshopIptcResource({ creditValues: [GOOGLE_AI_EDIT_TEST_TEXT, 'Photographer Credit'] }),
      [
        makePhotoshopIptcResource(),
        makePhotoshopIptcResource({ creditValues: ['Photographer Credit'] })
      ]
    ]) {
      const segments = Array.isArray(payload) ? payload.map((value) => [0xed, value]) : [[0xed, payload]];
      const result = inspectEncodedImage(makeJpegWithMetadataSegments(segments).buffer, 'image/jpeg');
      expect(result.evidence.filter((item) => item.claim)).toEqual([]);
    }
  });

  it('bounds JPEG, PNG, and WebP container floods before descriptor growth', () => {
    const jpeg = inspectEncodedImage(makeJpegSegmentFlood(5_000).buffer, 'image/jpeg');
    const png = inspectEncodedImage(makePngChunkFlood(5_000).buffer, 'image/png');
    const webp = inspectEncodedImage(makeWebpChunkFlood(5_000).buffer, 'image/webp');

    expect(jpeg.metadataTruncated).toBe(true);
    expect(png.metadataTruncated).toBe(true);
    expect(webp.metadataTruncated).toBe(true);
    expect(png.chunks.length).toBeLessThanOrEqual(64);
    expect(webp.chunks.length).toBeLessThanOrEqual(64);
  });

  it('marks malformed and structurally truncated containers incomplete', () => {
    const jpeg = Uint8Array.from([0xff, 0xd8, 0xff, 0xe1, 0x00, 0x20, 0x41]);
    const png = Uint8Array.from([
      137, 80, 78, 71, 13, 10, 26, 10,
      0, 0, 0, 20, 116, 69, 88, 116, 1, 2, 3
    ]);
    const webp = new Uint8Array(12);
    webp.set(new TextEncoder().encode('RIFF'), 0);
    new DataView(webp.buffer).setUint32(4, 100, true);
    webp.set(new TextEncoder().encode('WEBP'), 8);

    expect(inspectEncodedImage(jpeg.buffer, 'image/jpeg').metadataTruncated).toBe(true);
    expect(inspectEncodedImage(png.buffer, 'image/png').metadataTruncated).toBe(true);
    expect(inspectEncodedImage(webp.buffer, 'image/webp').metadataTruncated).toBe(true);
  });
});

describe('hybrid score fusion', () => {
  it('maps the locked operating point to the visible 65% decision line', () => {
    expect(calibrateDecisionScore(0.646794855594635)).toBe(0.65);
    expect(calibrateDecisionScore(0.1)).toBeLessThan(0.65);
    expect(calibrateDecisionScore(0.8)).toBeGreaterThan(0.65);
  });

  it('preserves a genuine zero instead of fabricating an uncertain score', () => {
    const encoded = { evidence: [], watermarks: [] };
    expect(fuseEvidence(0, encoded, { adjustment: 0 })).toBe(0);
    expect(calibrateDecisionScore(0)).toBe(0);
  });

  it.each([NaN, Infinity, -Infinity, '0', null, undefined])(
    'rejects malformed detector probability %s',
    (value) => {
      const encoded = { evidence: [], watermarks: [] };
      expect(() => fuseEvidence(value, encoded)).toThrowError(expect.objectContaining({ code: 'MODEL_OUTPUT_INVALID' }));
      expect(() => calibrateDecisionScore(value)).toThrowError(expect.objectContaining({ code: 'MODEL_OUTPUT_INVALID' }));
    }
  );

  it('lets explicit generator metadata override an uncertain model', () => {
    const encoded = {
      evidence: [{ kind: 'metadata', label: 'ComfyUI workflow metadata', strength: 0.98 }],
      watermarks: []
    };
    expect(fuseEvidence(0.4, encoded, { adjustment: 0 })).toBeGreaterThanOrEqual(0.985);
  });

  it('keeps weak pixel calibration bounded', () => {
    const encoded = { evidence: [], watermarks: [] };
    expect(fuseEvidence(0.5, encoded, { adjustment: 0.06 })).toBeLessThan(0.57);
    expect(fuseEvidence(0.5, encoded, { adjustment: -0.04 })).toBeGreaterThan(0.45);
  });

  it('returns finite signals for a simple image', () => {
    const data = new Uint8ClampedArray(16 * 16 * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = (i / 4) % 255;
      data[i + 1] = 90;
      data[i + 2] = 150;
      data[i + 3] = 255;
    }
    const result = computePixelSignals({ data, width: 16, height: 16 });
    expect(Number.isFinite(result.adjustment)).toBe(true);
    expect(Number.isFinite(result.metrics.luminanceEntropy)).toBe(true);
  });
});

describe('page-context evidence', () => {
  it('returns no evidence for non-string, empty, or whitespace-only input', () => {
    expect(collectPageContextEvidence(undefined)).toEqual([]);
    expect(collectPageContextEvidence(null)).toEqual([]);
    expect(collectPageContextEvidence(42)).toEqual([]);
    expect(collectPageContextEvidence({})).toEqual([]);
    expect(collectPageContextEvidence('')).toEqual([]);
    expect(collectPageContextEvidence('   \n\t  ')).toEqual([]);
  });

  it.each([
    ['ai-generated (hyphen)', 'This AI-generated artwork went viral overnight.'],
    ['ai generated (space)', 'An AI generated image of a lighthouse at dusk.'],
    ['ai-created', 'This AI-created illustration was made in seconds.'],
    ['generated by ai', 'This picture was generated by AI for the article.'],
    ['generated with artificial intelligence', 'A portrait generated with artificial intelligence.'],
    ['midjourney-generated', 'A midjourney-generated landscape of the coast.'],
    ['dall-e generated', 'A dall-e generated portrait of a cat astronaut.'],
    ['stable diffusion generated', 'A stable diffusion generated cityscape at night.'],
    ['generated by midjourney', 'This artwork was generated by Midjourney in 2024.'],
    ['generated with dall-e', 'This image was generated with DALL-E from a prompt.'],
    ['generated using stable diffusion', 'This scene was generated using Stable Diffusion.'],
    ['generated by adobe firefly', 'The banner was generated by Adobe Firefly.']
  ])('matches the declarative phrase: %s', (_label, caption) => {
    const evidence = collectPageContextEvidence(caption);
    expect(evidence.length).toBeGreaterThan(0);
    for (const item of evidence) {
      expect(item).toMatchObject({ kind: 'page-context', strength: 0.9 });
      expect(item.label).toEqual(expect.stringContaining('Page caption declares:'));
    }
  });

  it('does not match a stock-photo category listing that merely mentions AI', () => {
    expect(collectPageContextEvidence('590,100+ Ai Stock Photos')).toEqual([]);
  });

  it('does not match generic AI-generator marketing copy with no generation verb', () => {
    expect(collectPageContextEvidence('AI Image Generator - 100% Free')).toEqual([]);
  });

  it('does not match a bare generator name with no declarative phrasing', () => {
    expect(collectPageContextEvidence('Photos discussing Midjourney and Stable Diffusion tools')).toEqual([]);
  });

  it('is case-insensitive', () => {
    expect(collectPageContextEvidence('AI-GENERATED ARTWORK').length).toBeGreaterThan(0);
    expect(collectPageContextEvidence('Generated By MIDJOURNEY').length).toBeGreaterThan(0);
  });

  it('handles a very long input without crashing, truncating to the bound and still matching', () => {
    const longPrefix = 'x'.repeat(600);
    const caption = `${longPrefix} AI-generated artwork`;
    expect(() => collectPageContextEvidence(caption)).not.toThrow();
    // The declarative phrase falls past the 500-character bound once padded
    // with a 600-character prefix, so it is truncated away and produces no
    // evidence -- this documents the truncate-don't-scan-forever behavior
    // rather than asserting a specific match outcome.
    expect(collectPageContextEvidence(caption)).toEqual([]);
    const shortEnoughCaption = `${'x'.repeat(50)} AI-generated artwork`;
    expect(collectPageContextEvidence(shortEnoughCaption).length).toBeGreaterThan(0);
  });

  it('deduplicates identical matches and caps the number of returned entries', () => {
    const caption = 'AI-generated, ai created, generated by ai, generated with artificial intelligence, midjourney-generated, generated by midjourney';
    const evidence = collectPageContextEvidence(caption);
    expect(evidence.length).toBeLessThanOrEqual(5);
    const labels = evidence.map((item) => item.label);
    expect(new Set(labels).size).toBe(labels.length);
  });
});

function makePngWithText(text) {
  return makePngWithChunks([makePngChunk('tEXt', new TextEncoder().encode(text))]);
}

function makePngWithChunks(chunks) {
  const signature = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);
  return Uint8Array.from([...signature, ...chunks.flatMap((chunk) => [...chunk])]);
}

function makePngChunk(type, payload) {
  const chunk = new Uint8Array(12 + payload.length);
  new DataView(chunk.buffer).setUint32(0, payload.length, false);
  chunk.set(new TextEncoder().encode(type), 4);
  chunk.set(payload, 8);
  return chunk;
}

function makeWebpWithXmp(text) {
  const payload = new TextEncoder().encode(text);
  const paddedLength = payload.length + (payload.length & 1);
  const bytes = new Uint8Array(12 + 8 + paddedLength);
  bytes.set(new TextEncoder().encode('RIFF'), 0);
  new DataView(bytes.buffer).setUint32(4, bytes.length - 8, true);
  bytes.set(new TextEncoder().encode('WEBP'), 8);
  bytes.set(new TextEncoder().encode('XMP '), 12);
  new DataView(bytes.buffer).setUint32(16, payload.length, true);
  bytes.set(payload, 20);
  return bytes;
}

function makeJpegWithMetadataSegments(segments) {
  const parts = [Uint8Array.from([0xff, 0xd8])];
  for (const [marker, value] of segments) {
    const payload = typeof value === 'string' ? new TextEncoder().encode(value) : value;
    const segment = new Uint8Array(payload.length + 4);
    segment.set([0xff, marker], 0);
    new DataView(segment.buffer).setUint16(2, payload.length + 2, false);
    segment.set(payload, 4);
    parts.push(segment);
  }
  parts.push(Uint8Array.from([0xff, 0xd9]));
  return Uint8Array.from(parts.flatMap((part) => [...part]));
}

function standardXmp(xml) {
  return `http://ns.adobe.com/xap/1.0/\0<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">${xml}</rdf:RDF></x:xmpmeta>`;
}

const GOOGLE_AI_EDIT_TEST_TEXT = 'Edited with Google AI';

function makeIimField(record, dataset, value) {
  const encoded = new TextEncoder().encode(value);
  const field = new Uint8Array(5 + encoded.length);
  field.set([0x1c, record, dataset], 0);
  new DataView(field.buffer).setUint16(3, encoded.length, false);
  field.set(encoded, 5);
  return field;
}

function makePhotoshopIptcResource({ resourceId = 0x0404, creditValues = [GOOGLE_AI_EDIT_TEST_TEXT] } = {}) {
  const iim = Uint8Array.from(creditValues.flatMap((value) => [...makeIimField(2, 110, value)]));
  const header = new TextEncoder().encode('Photoshop 3.0\0');
  const resource = new Uint8Array(12 + iim.length + (iim.length & 1));
  resource.set(new TextEncoder().encode('8BIM'), 0);
  new DataView(resource.buffer).setUint16(4, resourceId, false);
  resource[6] = 0;
  resource[7] = 0;
  new DataView(resource.buffer).setUint32(8, iim.length, false);
  resource.set(iim, 12);
  return Uint8Array.from([...header, ...resource]);
}

function makeJpegSegmentFlood(count) {
  const bytes = new Uint8Array(2 + count * 4 + 2);
  bytes.set([0xff, 0xd8], 0);
  for (let index = 0, offset = 2; index < count; index += 1, offset += 4) {
    bytes.set([0xff, 0xe1, 0x00, 0x02], offset);
  }
  bytes.set([0xff, 0xd9], bytes.length - 2);
  return bytes;
}

function makePngChunkFlood(count) {
  const signature = Uint8Array.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const chunk = makePngChunk('aaAa', new Uint8Array());
  const bytes = new Uint8Array(signature.length + count * chunk.length);
  bytes.set(signature, 0);
  for (let index = 0, offset = signature.length; index < count; index += 1, offset += chunk.length) {
    bytes.set(chunk, offset);
  }
  return bytes;
}

function makeWebpChunkFlood(count) {
  const bytes = new Uint8Array(12 + count * 8);
  bytes.set(new TextEncoder().encode('RIFF'), 0);
  new DataView(bytes.buffer).setUint32(4, bytes.length - 8, true);
  bytes.set(new TextEncoder().encode('WEBP'), 8);
  for (let index = 0, offset = 12; index < count; index += 1, offset += 8) {
    bytes.set(new TextEncoder().encode('META'), offset);
  }
  return bytes;
}
