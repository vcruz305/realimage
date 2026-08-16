// Splits .bench-data/realimage-broad-v1/_staging/<ai|real>/<domain>/ into
// train/calibration/holdout image-root directories plus a manifest.json,
// which the prior agent's report claimed already existed but did not
// (report was empty/null and only _staging was present on disk).
//
// Split policy:
//  - "nano-banana", "flux-krea", "imagen4", "recraft-v3" (all ai) are
//    FULLY-HOLDOUT-ONLY generators: zero images go to train or calibration,
//    100% go to holdout, so the acceptance criteria's "fully-holdout-only
//    generator" check has real out-of-distribution generators to test
//    against. nano-banana/flux-krea are small non-redistributable dev
//    fixtures kept for eval signal only; imagen4/recraft-v3 (added
//    2026-08-16) are large, licensed, modern-generator sources deliberately
//    withheld from training to test generalization to unseen generators.
//  - "usgs" (real) has exactly 1 staged image; too small to split
//    meaningfully, so it is assigned entirely to train (documented, not
//    used for any per-domain recall claim).
//  - Every other domain is split 70/15/15 (train/calibration/holdout) by a
//    seeded deterministic shuffle of its sha256-named files, so the exact
//    same split is reproducible from the same _staging tree.
//  - No image appears in more than one split (image-level disjointness).
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const ROOT = resolve('.bench-data/realimage-broad-v1');
const STAGING = resolve(ROOT, '_staging');
const SEED = 323;
// nano-banana and flux-krea: locally-authored, non-redistributable dev
// fixtures, eval-signal-only by design. imagen4 and recraft-v3: two of the
// 2026-08-16 "modern generator" additions deliberately withheld from
// train/calibration entirely, to serve as true unseen-generator
// generalization checks distinct from nano-banana/flux-krea (see
// manifest.json provenanceNotes for the rationale).
const FULLY_HOLDOUT_ONLY = new Set(['nano-banana', 'flux-krea', 'imagen4', 'recraft-v3']);
const TOO_SMALL_TO_SPLIT = new Set(['usgs']);
const RATIOS = { train: 0.70, calibration: 0.15, holdout: 0.15 };

// Per-domain provenance, kept in the split script itself (not hand-patched
// into manifest.json after the fact) so every regeneration from _staging
// carries accurate sourceDataset/sourceLicense automatically. Verified via
// scripts/map-hf-corpus-sources.mjs (Zitacron aggregator boundary scan),
// direct HF datasets-server / duckdb+httpfs cross-checks against upstream
// metadata (see MODEL_CARD.md's "Training data" section for the
// diffusiondb/sd15-stock investigation writeup), and the pull manifests
// already on disk (_staging/zitacron-pull-manifest.json,
// _staging/local-reuse-summary.json, _staging/tiny-genimage-pull-manifest.json,
// produced by scripts/prepare-genimage-replacement.mjs).
const PROVENANCE = {
  'ai/diffusiondb': { sourceDataset: 'poloclub/diffusiondb', sourceLicense: 'cc0-1.0', note: 'Confirmed 2026-08-16: NOT a Zitacron/real-vs-ai-corpus source_dataset tag (that aggregator only carries 4 tags -- AbstractPhil/synthetic-characters, LucasFang/FLUX-Reason-6M, skylenage/DeepVision-103K, laion/laion2B-en-aesthetic -- confirmed by a full boundary scan of its 12,336,207-row train split, no residue). Instead these are genuine poloclub/diffusiondb (CC0-1.0) images pulled directly: a 12-file random sample had every filename (UUID.png) and every width/height match an exact row in diffusiondb\'s own metadata.parquet (part_id=1 of the 2m subset), 12/12.' },
  'ai/flux-krea': { sourceDataset: 'reused-local', sourceLicense: 'proprietary-generation, locally-authored dev sample (unpublished, not redistributed)', note: 'FLUX.1 Krea generations already in repo dev fixtures. Moved to fully-holdout-only 2026-08-16 (previously had train/calibration images) -- non-redistributable, small, kept for evaluation signal only.' },
  'ai/flux-reason': { sourceDataset: 'LucasFang/FLUX-Reason-6M', sourceLicense: 'apache-2.0', note: 'Zitacron/real-vs-ai-corpus source_dataset tag; see _staging/zitacron-pull-manifest.json' },
  'ai/genimage-midjourney': { sourceDataset: 'TheKernel01/Tiny-GenImage', sourceLicense: 'cc-by-nc-sa-4.0', note: 'Added 2026-08-16 to replace the retired ai/sd15-stock folder (see below). validation split, generator=Midjourney, label=fake. Pulled via scripts/prepare-genimage-replacement.mjs; see _staging/tiny-genimage-pull-manifest.json. NOTE: this is Midjourney v5-era (GenImage\'s Midjourney generator); see ai/midjourney-v6 below for the genuinely newer v6 addition.' },
  'ai/genimage-sd15': { sourceDataset: 'TheKernel01/Tiny-GenImage', sourceLicense: 'cc-by-nc-sa-4.0', note: 'Added 2026-08-16 to replace the retired ai/sd15-stock folder (see below). validation split, generator=SD15, label=fake. Pulled via scripts/prepare-genimage-replacement.mjs; see _staging/tiny-genimage-pull-manifest.json.' },
  'ai/gpt-image': { sourceDataset: 'Rapidata/OpenAI-4o_t2i_human_preference', sourceLicense: 'cdla-permissive-2.0', note: 'Added 2026-08-16 (modern-generator expansion). GPT-4o native image generation ("4o-26-3-25" snapshot, i.e. ChatGPT/GPT-Image-style generations), extracted from Rapidata\'s pairwise human-preference comparisons -- only the image on the "4o-26-3-25" side of each pair is kept. Pulled via scripts/prepare-rapidata-modern-generators.mjs; see _staging/rapidata-modern-generators-pull-manifest.json.' },
  'ai/imagen4': { sourceDataset: 'Rapidata/Imagen4_t2i_human_preference', sourceLicense: 'cdla-permissive-2.0', note: 'Added 2026-08-16 (modern-generator expansion). Google Imagen 4 Ultra (experimental, "imagen-4.0-ultra-generate-exp-05-20" snapshot), extracted from Rapidata pairwise comparisons. Deliberately FULLY HOLDOUT (0% train/calibration) as one of this expansion\'s >=2 unseen-generator generalization checks. Pulled via scripts/prepare-rapidata-modern-generators.mjs; see _staging/rapidata-modern-generators-pull-manifest.json.' },
  'ai/midjourney-v6': { sourceDataset: 'Photoroom/midjourney-v6-recap', sourceLicense: 'mit', note: 'Added 2026-08-16 (modern-generator expansion). Genuinely v6-era Midjourney (distinct from the v5-era ai/genimage-midjourney already in this corpus), COCO-prompt recaptioning dataset published by Photoroom. Pulled via scripts/prepare-midjourney-v6.mjs; see _staging/midjourney-v6-pull-manifest.json.' },
  'ai/nano-banana': { sourceDataset: 'reused-local', sourceLicense: 'proprietary-generation, locally-authored dev sample (unpublished, not redistributed)', note: 'Nano Banana (Gemini image) generations already in repo dev fixtures; fully-holdout-only' },
  'ai/recraft-v3': { sourceDataset: 'Rapidata/Recraft-v3-24-7-25_t2i_human_preference', sourceLicense: 'cdla-permissive-2.0', note: 'Added 2026-08-16 (modern-generator expansion). Recraft v3 ("recraft-v3-24-7-25" snapshot), extracted from Rapidata pairwise comparisons. Deliberately FULLY HOLDOUT (0% train/calibration) as one of this expansion\'s >=2 unseen-generator generalization checks. Pulled via scripts/prepare-rapidata-modern-generators.mjs (spec target is 220, matching ai/imagen4); this build\'s snapshot has only 47 images -- this specific 30GB HF dataset repo confirmed 100% row-level model match but only a ~2% image-download success rate even at conservative concurrency=3 with exponential backoff (up to 45s/attempt), apparently because its cached-assets are rarely-enough-accessed to need fresh server-side decode on nearly every request. Time-boxed at 47 images (still clears the >=30-40-in-holdout bar) rather than waiting the multiple hours the full 220 would have required; the puller is checkpointed/resumable (see _staging/_checkpoints/recraft-v3.json) so re-running scripts/prepare-rapidata-modern-generators.mjs later will continue from where this run stopped, not restart from zero. See _staging/rapidata-modern-generators-pull-manifest.json.' },
  'ai/sd35': { sourceDataset: 'data-is-better-together/open-image-preferences-v1', sourceLicense: 'apache-2.0', note: 'Added 2026-08-16 (modern-generator expansion). The dataset\'s "_sd" image columns (paired against a "_dev"/FLUX.1-dev column per prompt, which we do NOT pull from, to avoid overlap with ai/flux-krea/ai/flux-reason). Exact SD checkpoint version is not machine-readable from the parquet/dataset-card (only inferable from the "_sd" column-naming convention and the dataset\'s "flux"/"stable-diffusion" tags) -- recorded honestly as SD3.x-class rather than a guessed exact version. Pulled via scripts/prepare-open-image-preferences-sd35.mjs; see _staging/open-image-preferences-sd35-pull-manifest.json.' },
  'ai/synthetic-characters': { sourceDataset: 'AbstractPhil/synthetic-characters', sourceLicense: 'cc-by-4.0', note: 'Zitacron/real-vs-ai-corpus source_dataset tag; see _staging/zitacron-pull-manifest.json. Capped at its current size 2026-08-16 -- not grown further, to keep this source <=10% of the total AI pool as the pool grows.' },
  'real/coco-val2017': { sourceDataset: 'rafaelpadilla/coco2017', sourceLicense: 'cc-by-4.0 (annotations); per-image photo license inherited from the original Flickr upload, not independently re-verified per COCO\'s own terms of use', note: 'Added 2026-08-16 (real-photo diversity expansion). COCO 2017 validation split -- everyday real photographs (people, objects, street/indoor/outdoor scenes across 80 object categories), for false-positive control against ordinary real photography. Same "dataset-level tag confirmed, per-image copyright not independently verified" caveat already applied to real/laion-aesthetic in this manifest. Pulled via scripts/prepare-real-diversity.mjs; see _staging/real-diversity-pull-manifest.json.' },
  'real/docci': { sourceDataset: 'reused-local', sourceLicense: 'cc-by-4.0 (Google DOCCI dataset)', note: 'DOCCI real photographs already in repo dev fixtures (see _staging/local-reuse-summary.json); no HF dataset id recorded in repo provenance' },
  'real/japan-cc0': { sourceDataset: 'ThePioneer/japanese-photos', sourceLicense: 'unverified', note: '"cc0" is this repo\'s own directory-naming assumption, not a confirmed dataset-card license' },
  'real/laion-aesthetic': { sourceDataset: 'laion/laion2B-en-aesthetic', sourceLicense: 'cc-by-4.0', note: 'CC-BY-4.0 tag covers the metadata/URL index only; per-image copyright not independently verified. Zitacron/real-vs-ai-corpus source_dataset tag; see _staging/zitacron-pull-manifest.json' },
  'real/memes-imgflip': { sourceDataset: 'SassyRong/meme-imgflip-small-test-dataset', sourceLicense: 'cc0-1.0', note: 'Added 2026-08-16 (real-photo diversity expansion). Real (human-made, not AI-generated) imgflip meme template captures -- non-photo real content, for false-positive control against memes/graphics rather than photographs. Pulled via scripts/prepare-real-diversity.mjs; see _staging/real-diversity-pull-manifest.json.' },
  'real/nasa': { sourceDataset: 'reused-local', sourceLicense: 'public-domain (NASA image and media usage guidelines)', note: 'NASA photography already in repo dev fixtures' },
  'real/usgs': { sourceDataset: 'reused-local', sourceLicense: 'public-domain (USGS visual identity / U.S. Government work)', note: 'USGS photography already in repo dev fixtures; too small to split' },
  'real/visual-logic': { sourceDataset: 'skylenage/DeepVision-103K', sourceLicense: 'cc-by-4.0', note: 'Zitacron/real-vs-ai-corpus source_dataset tag; see _staging/zitacron-pull-manifest.json' },
  'real/web-screenshots': { sourceDataset: 'Zexanima/website_screenshots_image_dataset', sourceLicense: 'mit', note: 'Added 2026-08-16 (real-photo diversity expansion). Real website screenshots (Roboflow-sourced UI-element-annotated captures) -- non-photo real content, for false-positive control against ordinary web-page screenshots rather than photographs. Pulled via scripts/prepare-real-diversity.mjs; see _staging/real-diversity-pull-manifest.json.' },
};

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle(items, seed) {
  const random = mulberry32(seed);
  const copy = items.slice();
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function domainSeed(label, domain) {
  const digest = createHash('sha256').update(`${SEED}:${label}:${domain}`).digest();
  return digest.readUInt32BE(0);
}

async function main() {
  const manifestRows = [];
  const sourceSummary = [];

  for (const label of ['ai', 'real']) {
    const labelDir = resolve(STAGING, label);
    const domains = (await readdir(labelDir, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .sort();

    for (const domain of domains) {
      const domainDir = resolve(labelDir, domain);
      const files = (await readdir(domainDir)).sort();
      let assignment;

      if (FULLY_HOLDOUT_ONLY.has(domain)) {
        assignment = files.map((file) => ({ file, split: 'holdout' }));
      } else if (TOO_SMALL_TO_SPLIT.has(domain)) {
        assignment = files.map((file) => ({ file, split: 'train' }));
      } else {
        const shuffled = seededShuffle(files, domainSeed(label, domain));
        const trainCount = Math.round(shuffled.length * RATIOS.train);
        const calibrationCount = Math.round(shuffled.length * RATIOS.calibration);
        const holdoutCount = shuffled.length - trainCount - calibrationCount;
        assignment = [
          ...shuffled.slice(0, trainCount).map((file) => ({ file, split: 'train' })),
          ...shuffled.slice(trainCount, trainCount + calibrationCount).map((file) => ({ file, split: 'calibration' })),
          ...shuffled.slice(trainCount + calibrationCount).map((file) => ({ file, split: 'holdout' })),
        ];
        void holdoutCount;
      }

      const counts = { train: 0, calibration: 0, holdout: 0 };
      for (const { file, split } of assignment) {
        const from = resolve(domainDir, file);
        const toDir = resolve(ROOT, split, label, domain);
        await mkdir(toDir, { recursive: true });
        await copyFile(from, resolve(toDir, file));
        counts[split] += 1;
        manifestRows.push({ label, domain, split, fileName: file });
      }

      const provenance = PROVENANCE[`${label}/${domain}`];
      if (!provenance) throw new Error(`No PROVENANCE entry for ${label}/${domain} -- add one before regenerating manifest.json`);
      sourceSummary.push({
        label,
        domain,
        totalStaged: files.length,
        fullyHoldoutOnly: FULLY_HOLDOUT_ONLY.has(domain),
        tooSmallToSplit: TOO_SMALL_TO_SPLIT.has(domain),
        counts,
        sourceDataset: provenance.sourceDataset,
        sourceLicense: provenance.sourceLicense,
        provenanceNote: provenance.note,
      });
      console.log(`${label}/${domain}: staged=${files.length} -> train=${counts.train} calibration=${counts.calibration} holdout=${counts.holdout}`);
    }
  }

  const manifest = {
    schema: 'realimage-broad-v1-split-manifest-v1',
    generatedAt: new Date().toISOString(),
    generatedBy: 'scripts/split-realimage-broad-v1.mjs',
    seed: SEED,
    splitRatios: RATIOS,
    fullyHoldoutOnlyGenerators: [...FULLY_HOLDOUT_ONLY],
    tooSmallToSplitDomains: [...TOO_SMALL_TO_SPLIT],
    provenanceNotes: [
      'This manifest was built from .bench-data/realimage-broad-v1/_staging by scripts/split-realimage-broad-v1.mjs. Per-source provenance (sourceDataset/sourceLicense/provenanceNote) is recorded on each entry in "sources" below, sourced from the PROVENANCE table in this script.',
      'ai/nano-banana, ai/flux-krea, ai/imagen4, and ai/recraft-v3 are deliberately 100% holdout (never seen in train or calibration) to serve as the acceptance criteria\'s "fully-holdout-only generator" generalization checks.',
      'real/usgs has only 1 staged image and was assigned entirely to train; it is too small to serve as an evaluable holdout domain on its own.',
      '2026-08-16 provenance resolution: the previously-unverified ai/diffusiondb (340 images) is now CONFIRMED as poloclub/diffusiondb (CC0-1.0) -- see the "sources" entry for detail on the verification method (filename+dimension cross-check against the dataset\'s own metadata.parquet, 12/12 sampled matches). The previously-unverified ai/sd15-stock (340 images, no matching source_dataset anywhere in Zitacron/real-vs-ai-corpus, no build script, no citable licensed origin found) was RETIRED and deleted from _staging/train/calibration/holdout, replaced 1:1 in count by ai/genimage-sd15 (170) + ai/genimage-midjourney (170), both freshly pulled from TheKernel01/Tiny-GenImage (CC-BY-NC-SA-4.0) via scripts/prepare-genimage-replacement.mjs -- see _staging/tiny-genimage-pull-manifest.json. Total AI pool size is unchanged (1748 images before and after).',
      'ai/nano-banana is locally-authored, an unpublished proprietary-model generation reused from a prior dev fixture (see _staging/local-reuse-summary.json) -- not redistributable, dev/eval use only.',
      '2026-08-16 corpus expansion (this pass): ai/cifake-sd14 (340 images, 32x32 CIFAR-derived, unverified license, uninformative under every stress transform) was DROPPED entirely -- deleted from _staging/train/calibration/holdout and removed from PROVENANCE below. ai/synthetic-characters was capped at its existing 300-image size (not grown further) to stay <=10% of the total AI pool as the pool grows. Five modern photoreal generators were added -- ai/gpt-image (Rapidata/OpenAI-4o_t2i_human_preference, CDLA-Permissive-2.0), ai/midjourney-v6 (Photoroom/midjourney-v6-recap, MIT, genuinely v6-era unlike the v5-era ai/genimage-midjourney), ai/sd35 (data-is-better-together/open-image-preferences-v1, Apache-2.0), ai/imagen4 and ai/recraft-v3 (both Rapidata, CDLA-Permissive-2.0, both FULLY HOLDOUT as unseen-generator checks) -- all non-NC licensed. ai/flux-krea was moved from a 70/15/15 split to fully-holdout-only (its train/calibration images were relocated into holdout, no images deleted or added). Three real-diversity sources were added for false-positive control -- real/coco-val2017 (rafaelpadilla/coco2017, ordinary everyday photographs), real/web-screenshots (Zexanima/website_screenshots_image_dataset, MIT), real/memes-imgflip (SassyRong/meme-imgflip-small-test-dataset, CC0-1.0) -- the latter two being non-photo real content (screenshots/memes) rather than photographs. See COMPETITION_EXECUTION_PLAN.md / the corpus-expansion report for full before/after counts and the resulting real:ai ratio.',
      'real/japan-cc0 remains an unverified-license source (unrelated to the above; out of scope for that fix) -- see MODEL_CARD.md for detail.',
      'ai/recraft-v3 fell short of its 220-image pull target (47 staged instead) due to severe, dataset-specific rate-limiting on the Rapidata/Recraft-v3-24-7-25_t2i_human_preference HF repo -- 100% row-level model match but ~2% image-download success rate, time-boxed rather than waited out. 47 still clears the ">=30-40 images in holdout" acceptance bar for a fully-held-out generator; see the ai/recraft-v3 provenanceNote above for detail and PROVENANCE table for the resumable-checkpoint mechanism to top it up later.',
    ],
    sources: sourceSummary,
    rows: manifestRows,
  };

  await writeFile(resolve(ROOT, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`\nWrote manifest.json with ${manifestRows.length} rows to ${resolve(ROOT, 'manifest.json')}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
