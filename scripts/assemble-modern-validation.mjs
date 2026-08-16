import { createHash } from 'node:crypto';
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve } from 'node:path';

const PLAN_SCHEMA = 'realimage-validation-plan-v1';
const INVENTORY_SCHEMA = 'realimage-validation-inventory-v1';
const SELECTION_SCHEMA = 'realimage-validation-selection-v1';
const RESERVED_ROOT = resolve('.bench-data/modern-browser-validation-v1');
const SHA256 = /^[a-f0-9]{64}$/;

await main().catch((error) => {
  console.error(`Modern-validation assembly failed: ${error.message}`);
  process.exitCode = 1;
});

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const planPath = resolve(options.plan || 'benchmark/modern-browser-validation-v1.plan.json');
  const planBytes = await readFile(planPath);
  const plan = parseJson(planBytes, planPath);
  const report = validatePlan(plan);
  const planSha256 = digest(planBytes);

  if (options.requireFrozen) assertFrozen(plan);

  if (options.writeInventoryTemplate) {
    const outputPath = containedReservedPath(options.writeInventoryTemplate, '--write-inventory-template');
    await assertMissing(outputPath);
    const template = {
      schema: INVENTORY_SCHEMA,
      benchmarkId: plan.benchmarkId,
      planSha256,
      createdAt: null,
      note: 'Metadata only. Do not place image bytes, model scores, qualification rows, or private diagnostic rows here.',
      requiredRowFields: [
        'sourceUnitId',
        'sourceRow',
        'sourceUrl or localCommitPath',
        'immutableRevision',
        'upstreamAssetKey',
        'width',
        'height',
        'contentLength',
        'rightsProfileId',
        'licenseIdentifier',
        'rightsEvidenceUrl (remote rows) or localCommitPath (local rows)',
        'licenseVerifiedAt',
        'attribution',
        'rightsStatus=approved',
        'provenanceEvidenceUrl (remote rows) or localCommitPath (local rows)',
        'creator',
        'labelMethod',
        'provenanceVerifiedAt',
        'provenanceStatus=verified',
        'unit-specific requiredInventoryFields'
      ],
      rows: []
    };
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(template, null, 2)}\n`, { flag: 'wx' });
    console.log(`Wrote metadata-only inventory template: ${outputPath}`);
  }

  if (options.inventory || options.selectionLock) {
    if (!options.inventory || !options.selectionLock) {
      throw new Error('--inventory and --selection-lock must be supplied together.');
    }
    assertSelectionReady(plan);
    const inventoryPath = containedReservedPath(options.inventory, '--inventory');
    const outputPath = containedReservedPath(options.selectionLock, '--selection-lock');
    await assertMissing(outputPath);
    const inventoryBytes = await readFile(inventoryPath);
    const inventory = parseJson(inventoryBytes, inventoryPath);
    const selected = selectRows(plan, inventory, planSha256);
    const selection = {
      schema: SELECTION_SCHEMA,
      benchmarkId: plan.benchmarkId,
      seed: plan.seed,
      plan: { path: relative(process.cwd(), planPath).replaceAll('\\', '/'), sha256: planSha256 },
      inventory: { path: relative(process.cwd(), inventoryPath).replaceAll('\\', '/'), sha256: digest(inventoryBytes) },
      algorithm: plan.cohort.selectionOrder,
      counts: summarize(selected),
      rows: selected
    };
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(outputPath, `${JSON.stringify(selection, null, 2)}\n`, { flag: 'wx' });
    console.log(`Wrote deterministic selection lock: ${outputPath}`);
  }

  console.log(JSON.stringify({
    plan: relative(process.cwd(), planPath).replaceAll('\\', '/'),
    planSha256,
    status: plan.status,
    counts: report.counts,
    sourceUnits: report.sourceUnits,
    freezeBlockers: freezeBlockers(plan)
  }, null, 2));
}

function parseArgs(argv) {
  const options = {};
  for (const value of argv) {
    if (value === '--help' || value === '-h') options.help = true;
    else if (value === '--require-frozen') options.requireFrozen = true;
    else if (value.startsWith('--plan=')) options.plan = value.slice('--plan='.length);
    else if (value.startsWith('--inventory=')) options.inventory = value.slice('--inventory='.length);
    else if (value.startsWith('--selection-lock=')) options.selectionLock = value.slice('--selection-lock='.length);
    else if (value.startsWith('--write-inventory-template=')) options.writeInventoryTemplate = value.slice('--write-inventory-template='.length);
    else throw new Error(`Unknown option: ${value}`);
  }
  return options;
}

function printUsage() {
  console.log('Usage: node scripts/assemble-modern-validation.mjs [options]');
  console.log('  --plan=PATH                       Validate the tracked plan (default plan path shown in the protocol).');
  console.log('  --write-inventory-template=PATH   Write an empty metadata-only inventory beneath the reserved .bench-data directory.');
  console.log('  --inventory=PATH                  Read a completed local metadata inventory beneath the reserved directory.');
  console.log('  --selection-lock=PATH             Write a deterministic selection lock; must accompany --inventory.');
  console.log('  --require-frozen                  Fail unless all final plan locks are populated.');
  console.log('This tool has no network, image decode, model, benchmark, or Chrome capability.');
}

function validatePlan(plan) {
  if (!plan || plan.schema !== PLAN_SCHEMA) throw new Error(`Plan schema must be ${PLAN_SCHEMA}.`);
  requiredString(plan.benchmarkId, 'benchmarkId');
  if (!['draft-unfetched', 'selection-ready', 'frozen'].includes(plan.status)) {
    throw new Error('status must be draft-unfetched, selection-ready, or frozen.');
  }
  if (!Number.isSafeInteger(plan.seed)) throw new Error('seed must be an integer.');
  if (plan.seed !== 323) throw new Error('This preregistered campaign requires seed 323.');
  if (!plan.freeze || plan.freeze.displayDecisionThreshold !== 0.65) {
    throw new Error('The frozen display decision threshold must be exactly 0.65.');
  }
  if (!plan.separation || !Array.isArray(plan.separation.excludedSourceFamilies)) {
    throw new Error('separation.excludedSourceFamilies is required.');
  }

  const rights = new Map();
  for (const [index, profile] of requiredArray(plan.rightsProfiles, 'rightsProfiles').entries()) {
    const id = requiredString(profile.id, `rightsProfiles[${index}].id`);
    if (rights.has(normalize(id))) throw new Error(`Duplicate rights profile: ${id}`);
    for (const field of ['commercialUse', 'evaluationUse', 'derivatives', 'localCopy']) {
      if (profile[field] !== true) throw new Error(`rightsProfiles[${index}].${field} must be explicitly true.`);
    }
    httpsUrl(profile.sourceEvidenceUrl, `rightsProfiles[${index}].sourceEvidenceUrl`);
    httpsUrl(profile.licenseUrl, `rightsProfiles[${index}].licenseUrl`);
    const allowed = requiredArray(profile.allowedLicenseIdentifiers, `rightsProfiles[${index}].allowedLicenseIdentifiers`);
    for (const [allowedIndex, identifier] of allowed.entries()) {
      requiredString(identifier, `rightsProfiles[${index}].allowedLicenseIdentifiers[${allowedIndex}]`);
    }
    if (profile.rowVerificationRequired !== true) {
      throw new Error(`rightsProfiles[${index}].rowVerificationRequired must be true.`);
    }
    rights.set(normalize(id), profile);
  }

  const cohort = plan.cohort;
  if (!cohort || cohort.split !== 'single-final-holdout') throw new Error('cohort.split must be single-final-holdout.');
  if (cohort.trainingOriginals !== 0 || cohort.developmentOriginals !== 0) {
    throw new Error('Final cohort may not contain training or development rows.');
  }
  const units = requiredArray(cohort.sourceUnits, 'cohort.sourceUnits');
  if (units.length !== 19) throw new Error('Cohort must contain exactly 19 source units (9 AI domains and 10 real slices).');
  const unitIds = new Set();
  const counts = { ai: 0, real: 0, total: 0 };
  const aiDomains = new Map();
  const realSlices = new Map();
  for (const [index, unit] of units.entries()) {
    const prefix = `cohort.sourceUnits[${index}]`;
    const id = requiredString(unit.id, `${prefix}.id`);
    if (unitIds.has(normalize(id))) throw new Error(`Duplicate source unit: ${id}`);
    unitIds.add(normalize(id));
    if (unit.label !== 'ai' && unit.label !== 'real') throw new Error(`${prefix}.label must be ai or real.`);
    const domain = requiredString(unit.domain, `${prefix}.domain`);
    const slice = requiredString(unit.slice, `${prefix}.slice`);
    requiredString(unit.sourceDataset, `${prefix}.sourceDataset`);
    requiredString(unit.sourceGroup, `${prefix}.sourceGroup`);
    const unitIdentity = canonical([unit.id, domain, slice, unit.sourceDataset, unit.sourceGroup, unit.selector].join(' '));
    const forbidden = plan.separation.excludedSourceFamilies.find((family) => unitIdentity.includes(canonical(family)));
    if (forbidden) throw new Error(`${prefix} violates excluded source family: ${forbidden}.`);
    if (unit.sourceUrl !== null) httpsUrl(unit.sourceUrl, `${prefix}.sourceUrl`);
    if (!Number.isSafeInteger(unit.count) || unit.count <= 0) throw new Error(`${prefix}.count must be positive.`);
    if (!Number.isSafeInteger(unit.maxBytesPerFile) || unit.maxBytesPerFile <= 0) {
      throw new Error(`${prefix}.maxBytesPerFile must be positive.`);
    }
    if (!rights.has(normalize(unit.rightsProfileId))) throw new Error(`${prefix} refers to unknown rights profile.`);
    requiredArray(unit.requiredInventoryFields, `${prefix}.requiredInventoryFields`);
    for (const [ruleIndex, rule] of requiredArray(unit.maxPerKey, `${prefix}.maxPerKey`).entries()) {
      requiredString(rule.field, `${prefix}.maxPerKey[${ruleIndex}].field`);
      if (!Number.isSafeInteger(rule.max) || rule.max <= 0) throw new Error(`${prefix}.maxPerKey[${ruleIndex}].max must be positive.`);
    }
    counts[unit.label] += unit.count;
    counts.total += unit.count;
    const strata = unit.label === 'ai' ? aiDomains : realSlices;
    const stratum = unit.label === 'ai' ? domain : slice;
    strata.set(stratum, (strata.get(stratum) || 0) + unit.count);
  }

  if (counts.ai !== 216 || counts.real !== 216 || counts.total !== 432) {
    throw new Error(`Cohort must total 216 AI + 216 real = 432; found ${JSON.stringify(counts)}.`);
  }
  if (cohort.aiOriginals !== 216 || cohort.realOriginals !== 216 || cohort.finalOriginals !== 432) {
    throw new Error('Declared cohort counts do not match the preregistration.');
  }
  if (aiDomains.size !== 9 || [...aiDomains.values()].some((count) => count !== 24)) {
    throw new Error('AI cohort must have exactly 9 domains with 24 originals each.');
  }
  if (realSlices.size !== 10 || [...realSlices.values()].some((count) => count < 20)) {
    throw new Error('Real cohort must have exactly 10 slices with at least 20 originals each.');
  }
  if (plan.transforms?.cleanViews !== 432 || plan.transforms?.stressViews !== 1728 || plan.transforms?.allViews !== 2160) {
    throw new Error('Transform counts must be 432 clean, 1728 stress, and 2160 total.');
  }
  validateDevelopmentScreen(plan.developmentScreen, units);
  const regression = plan.scoreProtocol?.regressionGates;
  if (!regression
    || regression.modelGraphAndWeightsBytesMax !== 50000000
    || regression.unpackedExtensionBytesMax !== 65000000
    || regression.installedChromeWarmP50InferenceMsMax !== 140
    || regression.installedChromeWarmP95InferenceMsMax !== 220
    || regression.pairedP50InferenceRatioToV4Max !== 0.40
    || regression.extensionTaskPeakPrivateMemoryBytesMax !== 402653184) {
    throw new Error('Architecture-neutral package, installed-Chrome latency, and memory gates changed from preregistration.');
  }
  if (plan.browserProtocol?.fullDirect?.totalPages !== 37 || plan.browserProtocol?.deliveryAudit?.cases !== 304) {
    throw new Error('Browser protocol must retain the preregistered 37 direct pages and 304 delivery-audit cases.');
  }
  return { counts, sourceUnits: units.length };
}

function validateDevelopmentScreen(screen, finalUnits) {
  if (!screen || screen.benchmarkId !== 'rapid-development-v1' || screen.seed !== 1323) {
    throw new Error('The bounded rapid-development-v1 screen is required.');
  }
  const counts = screen.counts || {};
  if (counts.originals !== 96 || counts.fitOriginals !== 64 || counts.checkOriginals !== 32
    || counts.aiOriginals !== 48 || counts.realOriginals !== 48) {
    throw new Error('Development screen must be 96 originals: 64 fit, 32 check, balanced 48/48.');
  }
  const devUnits = requiredArray(screen.sourceUnits, 'developmentScreen.sourceUnits');
  if (devUnits.length !== 4) throw new Error('Development screen must contain exactly four source units.');
  let ai = 0;
  let real = 0;
  const devGroups = new Set();
  for (const [index, unit] of devUnits.entries()) {
    const prefix = `developmentScreen.sourceUnits[${index}]`;
    if (unit.label !== 'ai' && unit.label !== 'real') throw new Error(`${prefix}.label must be ai or real.`);
    if (unit.originals !== 24 || unit.fit !== 16 || unit.check !== 8) {
      throw new Error(`${prefix} must contain 24 originals partitioned 16 fit/8 check.`);
    }
    const group = normalize(requiredString(unit.sourceGroup, `${prefix}.sourceGroup`));
    if (devGroups.has(group)) throw new Error(`Duplicate development source group: ${group}`);
    devGroups.add(group);
    if (unit.label === 'ai') ai += unit.originals;
    else real += unit.originals;
  }
  if (ai !== 48 || real !== 48) throw new Error('Development source-unit labels must total 48 AI and 48 real.');
  const finalGroups = new Set(finalUnits.map((unit) => normalize(unit.sourceGroup)));
  for (const group of devGroups) {
    if (finalGroups.has(group)) throw new Error(`Development/final source-group overlap: ${group}`);
  }
  if (screen.strictFirewall?.finalSourceDatasetsAllowed !== false
    || screen.strictFirewall?.finalSourceGroupsAllowed !== false
    || screen.strictFirewall?.finalGeneratorFamiliesAllowed !== false
    || screen.strictFirewall?.finalRowsOrMetadataAllowed !== false
    || screen.strictFirewall?.freepikIncidentMayBeFitOrCheck !== false) {
    throw new Error('Development-to-final firewall must fail closed, including the known incident.');
  }
  if (screen.budget?.hardStopNetworkBytes > 1000000000 || screen.budget?.minimumFreeDiskBytes > 4000000000) {
    throw new Error('Development screen exceeds the 1 GB network or 4 GB disk ceiling.');
  }
  if (screen.budget?.paidGenerationRequired !== false || screen.budget?.abortIfReservedRowsUnavailable !== true) {
    throw new Error('Development screen must require no paid generation and abort if reserved rows are unavailable.');
  }
  if (screen.candidateBudget?.maximumCandidateFamiliesOnCheckSplit !== 3
    || screen.candidateBudget?.configurationFrozenBeforeCheck !== true) {
    throw new Error('Development screen candidate budget must be at most three preregistered families.');
  }
}

function assertSelectionReady(plan) {
  if (plan.status !== 'selection-ready') {
    throw new Error(`Selection requires status=selection-ready; current status is ${plan.status}.`);
  }
  const blockers = selectionBlockers(plan);
  if (blockers.length) throw new Error(`Plan is not selection-ready: ${blockers.join('; ')}`);
}

function assertFrozen(plan) {
  if (plan.status !== 'frozen') throw new Error(`Plan status is ${plan.status}, not frozen.`);
  const blockers = freezeBlockers(plan);
  if (blockers.length) throw new Error(`Frozen plan is incomplete: ${blockers.join('; ')}`);
}

function selectionBlockers(plan) {
  const blockers = [];
  if (plan.freeze?.upstreamRevisionsPinned !== true) blockers.push('upstreamRevisionsPinned is not true');
  for (const unit of plan.cohort.sourceUnits) {
    if (!immutableRevision(unit.immutableRevision)) blockers.push(`${unit.id} lacks an immutable revision`);
  }
  for (const profile of plan.rightsProfiles) {
    if (profile.status !== 'approved-for-row-review') blockers.push(`${profile.id} rights profile is not approved-for-row-review`);
  }
  return blockers;
}

function freezeBlockers(plan) {
  const blockers = selectionBlockers(plan);
  const freeze = plan.freeze || {};
  for (const field of ['candidateModelSha256', 'extensionPackageSha256', 'sourceSelectionLockSha256', 'sourceManifestSha256', 'transformManifestSha256']) {
    if (!SHA256.test(freeze[field] || '')) blockers.push(`${field} is not a SHA-256 digest`);
  }
  for (const field of ['selectedRowsPinned', 'candidateLockedBeforeImageDecode']) {
    if (freeze[field] !== true) blockers.push(`${field} is not true`);
  }
  return blockers;
}

function selectRows(plan, inventory, planSha256) {
  if (!inventory || inventory.schema !== INVENTORY_SCHEMA) throw new Error(`Inventory schema must be ${INVENTORY_SCHEMA}.`);
  if (inventory.benchmarkId !== plan.benchmarkId) throw new Error('Inventory benchmarkId does not match plan.');
  if (inventory.planSha256 !== planSha256) throw new Error('Inventory planSha256 does not match current plan bytes.');
  const rows = requiredArray(inventory.rows, 'inventory.rows');
  const units = new Map(plan.cohort.sourceUnits.map((unit) => [unit.id, unit]));
  const candidates = new Map(plan.cohort.sourceUnits.map((unit) => [unit.id, []]));
  const seenSourceRows = new Set();

  for (const [index, row] of rows.entries()) {
    const prefix = `inventory.rows[${index}]`;
    const unit = units.get(requiredString(row.sourceUnitId, `${prefix}.sourceUnitId`));
    if (!unit) throw new Error(`${prefix} has unknown sourceUnitId.`);
    const sourceRow = requiredString(String(row.sourceRow ?? ''), `${prefix}.sourceRow`);
    const sourceIdentity = `${normalize(unit.sourceDataset)}\u0000${normalize(sourceRow)}`;
    if (seenSourceRows.has(sourceIdentity)) throw new Error(`Duplicate inventory source row: ${unit.sourceDataset}/${sourceRow}`);
    seenSourceRows.add(sourceIdentity);
    if (row.immutableRevision !== unit.immutableRevision) throw new Error(`${prefix}.immutableRevision does not match the plan pin.`);
    if (row.rightsProfileId !== unit.rightsProfileId || row.rightsStatus !== 'approved') {
      throw new Error(`${prefix} lacks approved rights under the planned profile.`);
    }
    const profile = plan.rightsProfiles.find((item) => item.id === unit.rightsProfileId);
    const licenseIdentifier = requiredString(row.licenseIdentifier, `${prefix}.licenseIdentifier`);
    if (!profile.allowedLicenseIdentifiers.includes(licenseIdentifier)) {
      throw new Error(`${prefix}.licenseIdentifier is not allowed by ${profile.id}.`);
    }
    requiredIsoDate(row.licenseVerifiedAt, `${prefix}.licenseVerifiedAt`);
    requiredString(row.attribution, `${prefix}.attribution`);
    if (row.provenanceStatus !== 'verified') throw new Error(`${prefix}.provenanceStatus must be verified.`);
    requiredString(row.creator, `${prefix}.creator`);
    requiredString(row.labelMethod, `${prefix}.labelMethod`);
    requiredIsoDate(row.provenanceVerifiedAt, `${prefix}.provenanceVerifiedAt`);
    if (!Number.isSafeInteger(row.width) || !Number.isSafeInteger(row.height)
      || row.width < plan.cohort.minimumWidth || row.height < plan.cohort.minimumHeight) {
      throw new Error(`${prefix} is below minimum dimensions.`);
    }
    if (!Number.isSafeInteger(row.contentLength) || row.contentLength <= 0 || row.contentLength > unit.maxBytesPerFile) {
      throw new Error(`${prefix}.contentLength exceeds the unit cap or is invalid.`);
    }
    if (unit.sourceUrl === null) {
      requiredString(row.localCommitPath, `${prefix}.localCommitPath`);
      if (!immutableRevision(row.immutableRevision)) throw new Error(`${prefix} lacks a pinned local commit.`);
    } else {
      httpsUrl(row.sourceUrl, `${prefix}.sourceUrl`);
      httpsUrl(row.rightsEvidenceUrl, `${prefix}.rightsEvidenceUrl`);
      httpsUrl(row.provenanceEvidenceUrl, `${prefix}.provenanceEvidenceUrl`);
    }
    for (const field of unit.requiredInventoryFields) requiredString(String(row[field] ?? ''), `${prefix}.${field}`);
    candidates.get(unit.id).push({ ...row, sourceRow });
  }

  const globalRules = requiredArray(plan.cohort.globalMaxPerKey, 'cohort.globalMaxPerKey');
  const globalCounts = new Map();
  const selected = [];
  for (const unit of plan.cohort.sourceUnits) {
    const ordered = candidates.get(unit.id).toSorted((left, right) => {
      const leftKey = selectionKey(plan, unit, left);
      const rightKey = selectionKey(plan, unit, right);
      return leftKey.localeCompare(rightKey) || left.sourceRow.localeCompare(right.sourceRow);
    });
    const localCounts = new Map();
    const unitSelected = [];
    for (const row of ordered) {
      if (violatesRules(row, unit.maxPerKey, localCounts, unit.id)) continue;
      const relevantGlobal = globalRules.filter((rule) => rule.sourceDataset === unit.sourceDataset);
      if (violatesRules(row, relevantGlobal, globalCounts, unit.sourceDataset)) continue;
      incrementRules(row, unit.maxPerKey, localCounts, unit.id);
      incrementRules(row, relevantGlobal, globalCounts, unit.sourceDataset);
      unitSelected.push({
        sourceUnitId: unit.id,
        familyId: `${unit.id}-${String(unitSelected.length + 1).padStart(3, '0')}`,
        label: unit.label,
        domain: unit.domain,
        slice: unit.slice,
        sourceDataset: unit.sourceDataset,
        sourceGroup: unit.sourceGroup,
        sourceRow: row.sourceRow,
        immutableRevision: row.immutableRevision,
        sourceUrl: row.sourceUrl || null,
        localCommitPath: row.localCommitPath || null,
        upstreamAssetKey: row.upstreamAssetKey,
        license: {
          status: row.rightsStatus,
          profileId: row.rightsProfileId,
          identifier: row.licenseIdentifier,
          evidenceUrl: row.rightsEvidenceUrl || null,
          verifiedAt: row.licenseVerifiedAt,
          attribution: row.attribution
        },
        provenance: {
          status: row.provenanceStatus,
          evidenceUrl: row.provenanceEvidenceUrl || null,
          creator: row.creator,
          labelMethod: row.labelMethod,
          verifiedAt: row.provenanceVerifiedAt
        },
        contentLength: row.contentLength,
        width: row.width,
        height: row.height,
        selectionKey: selectionKey(plan, unit, row)
      });
      if (unitSelected.length === unit.count) break;
    }
    if (unitSelected.length !== unit.count) {
      throw new Error(`${unit.id} has only ${unitSelected.length}/${unit.count} eligible diverse inventory rows.`);
    }
    selected.push(...unitSelected);
  }
  if (selected.length !== plan.cohort.finalOriginals) throw new Error('Internal selection count mismatch.');
  return selected;
}

function selectionKey(plan, unit, row) {
  return digest(`${plan.seed}\u0000${plan.benchmarkId}\u0000${unit.id}\u0000${row.sourceRow}`);
}

function violatesRules(row, rules, counts, scope) {
  return rules.some((rule) => {
    const value = requiredString(String(row[rule.field] ?? ''), `${scope}.${rule.field}`);
    return (counts.get(`${scope}\u0000${rule.field}\u0000${normalize(value)}`) || 0) >= rule.max;
  });
}

function incrementRules(row, rules, counts, scope) {
  for (const rule of rules) {
    const key = `${scope}\u0000${rule.field}\u0000${normalize(row[rule.field])}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
}

function summarize(rows) {
  return {
    originals: rows.length,
    ai: rows.filter((row) => row.label === 'ai').length,
    real: rows.filter((row) => row.label === 'real').length,
    sourceUnits: new Set(rows.map((row) => row.sourceUnitId)).size
  };
}

function containedReservedPath(value, field) {
  const result = resolve(requiredString(value, field));
  const rel = relative(RESERVED_ROOT, result);
  if (!rel || rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error(`${field} must be a file below ${RESERVED_ROOT}.`);
  }
  const unsafe = rel.split(/[\\/]/).some((part) => /^(?:private|sealed|qualification|webwild)(?:[-_.]|$)/i.test(part));
  if (unsafe) throw new Error(`${field} may not reference a private, sealed, qualification, or WebWild path.`);
  return result;
}

async function assertMissing(path) {
  await access(path).then(
    () => { throw new Error(`Refusing to overwrite existing path: ${path}`); },
    (error) => { if (error.code !== 'ENOENT') throw error; }
  );
}

function requiredArray(value, field) {
  if (!Array.isArray(value) || value.length === 0) throw new Error(`${field} must be a non-empty array.`);
  return value;
}

function requiredString(value, field) {
  if (typeof value !== 'string' || value.trim() === '') throw new Error(`${field} must be a non-empty string.`);
  return value.trim();
}

function httpsUrl(value, field) {
  requiredString(value, field);
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${field} must be a valid URL.`);
  }
  if (url.protocol !== 'https:') throw new Error(`${field} must use HTTPS.`);
}

function requiredIsoDate(value, field) {
  requiredString(value, field);
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(value) || Number.isNaN(Date.parse(value))) {
    throw new Error(`${field} must be an ISO-8601 UTC timestamp.`);
  }
}

function immutableRevision(value) {
  return typeof value === 'string' && /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/i.test(value);
}

function parseJson(bytes, path) {
  try {
    return JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`Invalid JSON in ${path}: ${error.message}`);
  }
}

function digest(value) {
  return createHash('sha256').update(value).digest('hex');
}

function normalize(value) {
  return String(value).trim().toLowerCase();
}

function canonical(value) {
  return normalize(value).replace(/[^a-z0-9]+/g, '');
}
