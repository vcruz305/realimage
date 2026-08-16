import { describe, expect, it } from 'vitest';
import { summarizeSemanticDeclarations } from './declaration-ui.js';

const exactDeclarations = {
  generated: {
    kind: 'metadata',
    authority: 'iptc',
    claim: 'generated',
    label: 'IPTC: created using Generative AI'
  },
  edited: {
    kind: 'metadata',
    authority: 'iptc',
    claim: 'edited',
    label: 'IPTC: edited using Generative AI'
  },
  composite: {
    kind: 'metadata',
    authority: 'iptc',
    claim: 'composite',
    label: 'IPTC: composite includes Generative AI elements'
  },
  googleEdited: {
    kind: 'metadata',
    authority: 'self-declared',
    claim: 'edited',
    label: 'Self-declared Google AI edit metadata'
  }
};

describe('semantic declaration UI summary', () => {
  it.each([
    ['generated', exactDeclarations.generated, 'Declared AI-generated', 'Metadata declares AI generation'],
    ['edited', exactDeclarations.edited, 'Declared AI-edited', 'Metadata declares AI editing'],
    ['composite', exactDeclarations.composite, 'Declared AI composite', 'Metadata declares an AI composite']
  ])('distinguishes an exact %s declaration', (claim, declaration, badgeText, detailTitle) => {
    const summary = summarizeSemanticDeclarations([declaration]);

    expect(summary).toMatchObject({
      claim,
      claims: [claim],
      badgeText,
      detailTitle,
      sourceCount: 1
    });
    expect(summary.detailBody).toContain('separate from the detector model result');
    expect(summary.detailBody).toContain('not proof that the metadata is authentic');
  });

  it('accepts the exact Google AI edit declaration as edited, not generated', () => {
    const summary = summarizeSemanticDeclarations([exactDeclarations.googleEdited]);

    expect(summary).toMatchObject({
      claim: 'edited',
      claims: ['edited'],
      badgeText: 'Declared AI-edited',
      sources: [{
        authority: 'self-declared',
        claim: 'edited',
        label: 'Google AI edit declaration'
      }]
    });
  });

  it('deduplicates repeated declarations while retaining distinct corroborating sources', () => {
    const summary = summarizeSemanticDeclarations([
      exactDeclarations.edited,
      { ...exactDeclarations.edited },
      exactDeclarations.googleEdited,
      { ...exactDeclarations.googleEdited }
    ]);

    expect(summary).toMatchObject({
      claim: 'edited',
      claims: ['edited'],
      sourceCount: 2,
      sources: [
        { authority: 'iptc', claim: 'edited', label: 'IPTC digital-source declaration', context: 'direct' },
        { authority: 'self-declared', claim: 'edited', label: 'Google AI edit declaration', context: 'direct' }
      ]
    });
  });

  it('uses cautious generic copy when exact declarations disagree about the AI use', () => {
    const summary = summarizeSemanticDeclarations([
      exactDeclarations.composite,
      exactDeclarations.generated,
      exactDeclarations.edited
    ]);

    expect(summary).toMatchObject({
      claim: 'mixed',
      claims: ['generated', 'edited', 'composite'],
      badgeText: 'Declared AI involvement',
      detailTitle: 'Metadata contains multiple AI-use declarations',
      sourceCount: 3
    });
    expect(summary.detailBody).toContain('AI generation, AI editing, and AI-composite content');
  });

  it.each([
    ['missing input', undefined],
    ['non-array input', exactDeclarations.generated],
    ['legacy claimed marker', [{ kind: 'metadata', claim: 'generated', label: 'Stable Diffusion generator metadata' }]],
    ['unknown authority', [{ ...exactDeclarations.generated, authority: 'unknown' }]],
    ['unknown claim', [{ ...exactDeclarations.generated, claim: 'enhanced' }]],
    ['wrong kind', [{ ...exactDeclarations.generated, kind: 'provenance' }]],
    ['fuzzy label', [{ ...exactDeclarations.generated, label: `${exactDeclarations.generated.label} maybe` }]],
    ['case variant', [{ ...exactDeclarations.generated, claim: 'Generated' }]],
    ['self-declared generation', [{ ...exactDeclarations.googleEdited, claim: 'generated' }]],
    ['arbitrary object', [{}]]
  ])('fails closed for %s', (_description, evidence) => {
    expect(summarizeSemanticDeclarations(evidence)).toBeNull();
  });

  it('ignores unknown evidence while preserving an exact declaration', () => {
    const summary = summarizeSemanticDeclarations([
      { kind: 'metadata', claim: 'generated', label: 'Embedded generation parameters' },
      { kind: 'metadata', authority: 'future-standard', claim: 'generated', label: 'Future declaration' },
      exactDeclarations.composite
    ]);

    expect(summary).toMatchObject({ claim: 'composite', sourceCount: 1 });
  });

  it('never presents declaration copy as a probability or model verdict', () => {
    const summaries = [
      summarizeSemanticDeclarations([exactDeclarations.generated]),
      summarizeSemanticDeclarations([exactDeclarations.edited]),
      summarizeSemanticDeclarations([exactDeclarations.composite]),
      summarizeSemanticDeclarations([exactDeclarations.generated, exactDeclarations.edited])
    ];
    const allUiCopy = summaries
      .flatMap(({ badgeText, detailTitle, detailBody }) => [badgeText, detailTitle, detailBody])
      .join(' ');

    expect(allUiCopy).not.toMatch(/probab|likelihood|confidence|score|percent|likely/i);
  });
});
