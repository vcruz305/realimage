const DECLARATION_RULES = Object.freeze([
  Object.freeze({
    authority: 'iptc',
    claim: 'generated',
    label: 'IPTC: created using Generative AI',
    sourceLabel: 'IPTC digital-source declaration'
  }),
  Object.freeze({
    authority: 'iptc',
    claim: 'edited',
    label: 'IPTC: edited using Generative AI',
    sourceLabel: 'IPTC digital-source declaration'
  }),
  Object.freeze({
    authority: 'iptc',
    claim: 'composite',
    label: 'IPTC: composite includes Generative AI elements',
    sourceLabel: 'IPTC digital-source declaration'
  }),
  Object.freeze({
    authority: 'self-declared',
    claim: 'edited',
    label: 'Self-declared Google AI edit metadata',
    sourceLabel: 'Google AI edit declaration'
  })
]);

const SINGLE_CLAIM_COPY = Object.freeze({
  generated: Object.freeze({
    badgeText: 'Declared AI-generated',
    detailTitle: 'Metadata declares AI generation',
    detailLead: 'Recognized metadata says this image was created using generative AI.'
  }),
  edited: Object.freeze({
    badgeText: 'Declared AI-edited',
    detailTitle: 'Metadata declares AI editing',
    detailLead: 'Recognized metadata says generative AI was used to edit this image.'
  }),
  composite: Object.freeze({
    badgeText: 'Declared AI composite',
    detailTitle: 'Metadata declares an AI composite',
    detailLead: 'Recognized metadata says this composite includes generative AI elements.'
  })
});

const CLAIM_ORDER = Object.freeze(['generated', 'edited', 'composite']);
const MIXED_CLAIM_LABELS = Object.freeze({
  generated: 'AI generation',
  edited: 'AI editing',
  composite: 'AI-composite content'
});
const DECLARATION_CAVEAT = 'This declaration is separate from the detector model result and is not proof that the metadata is authentic.';

/**
 * Converts exact, allowlisted semantic metadata declarations into cautious UI
 * copy. This function deliberately ignores legacy/fuzzy evidence and does not
 * change or interpret a detector score.
 *
 * @param {unknown} evidence
 * @returns {null | {
 *   claim: 'generated' | 'edited' | 'composite' | 'mixed',
 *   claims: Array<'generated' | 'edited' | 'composite'>,
 *   badgeText: string,
 *   detailTitle: string,
 *   detailBody: string,
 *   sources: Array<{authority: string, claim: string, label: string}>,
 *   sourceCount: number
 * }}
 */
export function summarizeSemanticDeclarations(evidence) {
  if (!Array.isArray(evidence)) return null;

  const matchedRuleIndexes = new Set();
  for (const item of evidence) {
    if (!item || typeof item !== 'object' || item.kind !== 'metadata') continue;
    const ruleIndex = DECLARATION_RULES.findIndex((rule) => (
      item.authority === rule.authority
      && item.claim === rule.claim
      && item.label === rule.label
    ));
    if (ruleIndex !== -1) matchedRuleIndexes.add(ruleIndex);
  }

  if (matchedRuleIndexes.size === 0) return null;

  const sources = [...matchedRuleIndexes]
    .sort((left, right) => left - right)
    .map((ruleIndex) => {
      const rule = DECLARATION_RULES[ruleIndex];
      return Object.freeze({
        authority: rule.authority,
        claim: rule.claim,
        label: rule.sourceLabel,
        context: 'direct'
      });
    });
  const presentClaims = new Set(sources.map((source) => source.claim));
  const claims = CLAIM_ORDER.filter((claim) => presentClaims.has(claim));

  if (claims.length === 1) {
    const claim = claims[0];
    const copy = SINGLE_CLAIM_COPY[claim];
    return freezeSummary({
      claim,
      claims,
      badgeText: copy.badgeText,
      detailTitle: copy.detailTitle,
      detailBody: `${copy.detailLead} ${DECLARATION_CAVEAT}`,
      sources
    });
  }

  const declarationList = formatList(claims.map((claim) => MIXED_CLAIM_LABELS[claim]));
  return freezeSummary({
    claim: 'mixed',
    claims,
    badgeText: 'Declared AI involvement',
    detailTitle: 'Metadata contains multiple AI-use declarations',
    detailBody: `Recognized metadata separately declares ${declarationList}. ${DECLARATION_CAVEAT}`,
    sources
  });
}

function freezeSummary({ claim, claims, badgeText, detailTitle, detailBody, sources }) {
  return Object.freeze({
    claim,
    claims: Object.freeze([...claims]),
    badgeText,
    detailTitle,
    detailBody,
    sources: Object.freeze(sources),
    sourceCount: sources.length
  });
}

function formatList(items) {
  if (items.length < 2) return items[0] ?? '';
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(', ')}, and ${items.at(-1)}`;
}
