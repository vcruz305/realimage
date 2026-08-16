# Attribution notice

The extension packages `RealImage/broad-v1-modern-v1-1471e3ef`, a one-head
derivative of the MIT-licensed `OwensLab/commfor-model-384` checkpoint. Its
FP32 ONNX SHA-256 is
`1471e3eff3a05d5ef8c068abfdef2f3f43d41060b27306f19763968cf8d38098`.
The replacement 384→32→1 MLP head was trained on a further-expanded,
public-only multi-generator, multi-domain corpus (21 sources, 4,650 staged
images; see `MODEL_CARD.md` for the full per-source dataset/license
breakdown) that excludes the two locally-authored, non-redistributable
dev-only sources (`flux-krea`, `nano-banana`) from training and calibration
entirely, keeping them eval-only. Training images are not shipped. See
`MODEL_CARD.md` for the leakage-corrected holdout results this candidate was
selected on.

This supersedes the prior `RealImage/broad-v1-02195715` candidate (FP32 ONNX
SHA-256 `02195715c69b2eade82df6359079f43f7312d04a5e541707216f1167164fe5d9`,
384→64→1 MLP head on a 14-source corpus that included the two
non-redistributable sources in its train/calibration split), which has been
removed from `public/models/`. That candidate itself superseded the earlier
`RealImage/license-clean-v4-fp32-a6297c3a` candidate (FP32 ONNX SHA-256
`7cedb42842b2e12359971acf0d86c935873d8a94fd069899851140b33263c7a2`), trained
on 100 originals from `bitmind/nano-banana` (MIT), `lioooox/T2I-CoReBench-Images`
FLUX.1-Krea-dev rows (Apache-2.0), and Google DOCCI photographs (CC BY 4.0),
also already removed from `public/models/`.

RealImage began from Proofmark commit
`ef986acb51c9ed6768d512bfc76174070940458b`, published by the Proofmark
contributors under the MIT License. The original copyright and permission
notice remain in `LICENSE`.

The Proofmark repository labels the bundled `proofmark-webwild-v3` adaptation
as MIT and identifies `OwensLab/commfor-model-384` revision
`6076002bf0d9dd37537f965ee2f06f826c333b61`. Its checked-in Q8 ONNX weight
has SHA-256
`ed17ceb332bef84d0adcc2fa537eef85ed3ac6fb32c30393c326321fbbe54683`.
RealImage treats that adapted head as a retained evaluation baseline only; its
weights are not included in the isolated FP32 candidate build. The exact
head-training manifest is missing, so its training-data provenance and
redistribution basis cannot be verified. Tiny-GenImage's CC-BY-NC-SA rows are
present in the inherited evaluation workflow; the available artifacts do not
establish that they trained the shipped head.

Competition-fork changes are tracked separately and must not be attributed to
the upstream Proofmark authors. The first change removes a redundant pixel
heuristic/decode pass after proving that it changes zero decisions across the
3,498 checked-in sealed clean and web-stress results. Scores are not identical:
mean absolute displayed-score movement is about 3.5 points and the observed
maximum is about 8.8 points. It also adds phase-level runtime telemetry and
explicit decision-parity/audit gates.
