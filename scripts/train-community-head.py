"""Train a small browser-safe head on frozen Community Forensics features."""

import argparse
import hashlib
import io
import json
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torchvision.transforms as T
from onnx import TensorProto, helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image, ImageOps
from safetensors.torch import load_file


SOURCE_SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
FEATURE_OUTPUT = "/vit/Gather_1_output_0"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-extra", type=Path, action="append", default=[])
    parser.add_argument("--train-robust", type=Path, action="append", default=[])
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--development", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=323)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def samples(root):
    output = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".avif"}:
            continue
        relative = path.relative_to(root).parts
        label_index = next((index for index, part in enumerate(relative) if part in {"ai", "real"}), None)
        if label_index is None:
            continue
        label = relative[label_index]
        generator = re.search(r"generator-([a-z0-9]+)", path.name.lower())
        domain = generator.group(1) if generator else (relative[label_index + 1] if len(relative) > label_index + 2 else "unknown")
        output.append((path, 1.0 if label == "ai" else 0.0, domain))
    return output


exact_transform = T.Compose([
    T.Resize(440, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(384),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

class RandomThumbnailRoundTrip:
    """Simulates a destructive small-thumbnail round trip: downscale to a
    randomized short side, re-encode as WebP or JPEG at a randomized
    low-moderate quality, then upscale back to the original resolution.

    This mirrors `thumbnail192-webp-q50-upscale` in
    scripts/build-holdout-stress-tiers.mjs -- the single hard-tier transform
    isolated (see .bench-data/realimage-broad-v1/hard-tier-breakdown.log) as
    responsible for the shipped model's hard-tier collapse: pooled AI recall
    fell to ~28% under that one transform vs. 85-92% for the other two hard
    transforms, because nothing in training previously exposed the head to
    this degradation. The short side, codec, and quality are all randomized
    per call (rather than fixed at exactly 192px/WebP/q50) so the head learns
    general robustness to small-thumbnail degradation instead of memorizing
    one exact configuration.
    """

    def __init__(self, short_side_range=(160, 224), quality_range=(30, 65)):
        self.short_side_range = short_side_range
        self.quality_range = quality_range

    def __call__(self, image):
        width, height = image.size
        short_side = max(1, min(width, height))
        target_short = int(torch.randint(self.short_side_range[0], self.short_side_range[1] + 1, (1,)).item())
        scale = target_short / short_side
        thumb_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        thumbnail = image.resize(thumb_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        quality = int(torch.randint(self.quality_range[0], self.quality_range[1] + 1, (1,)).item())
        if torch.rand(1).item() < 0.5:
            thumbnail.save(buffer, "WEBP", quality=quality)
        else:
            thumbnail.save(buffer, "JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as degraded:
            return degraded.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


# THUMBNAIL_AUG_MODE lets a run temporarily swap out the
# RandomThumbnailRoundTrip step of augmented_transform without touching the
# default behavior any other invocation relies on (unset / "default"
# reproduces the exact p=0.4, 160-224px, quality 30-65 config used for the
# shipped candidate-mlp-modern-v1 (C2) run -- this env var did not exist
# when that run happened, so it is a strict no-op for reproducing it):
#   - "default" (or unset): p=0.4, short side 160-224px, quality 30-65
#     (unchanged from before this env var existed)
#   - "none": the thumbnail round-trip step is omitted entirely (C3)
#   - "mild": p=0.2, short side 224-320px, quality 55-80 (C4)
# This is read once at import time, so each training process (one per
# candidate) picks a single fixed mode for its whole run; it never mutates
# shared state that another concurrently-running process could observe.
THUMBNAIL_AUG_MODE = os.environ.get("THUMBNAIL_AUG_MODE", "default")
if THUMBNAIL_AUG_MODE not in ("default", "none", "mild"):
    raise ValueError(f"Unknown THUMBNAIL_AUG_MODE={THUMBNAIL_AUG_MODE!r}; expected 'default', 'none', or 'mild'")


def _thumbnail_round_trip_steps():
    if THUMBNAIL_AUG_MODE == "none":
        return []
    if THUMBNAIL_AUG_MODE == "mild":
        return [T.RandomApply([RandomThumbnailRoundTrip(short_side_range=(224, 320), quality_range=(55, 80))], p=0.2)]
    return [T.RandomApply([RandomThumbnailRoundTrip()], p=0.4)]


augmented_transform = T.Compose([
    # Applied first (before the existing crop/normalize steps) so the
    # thumbnail degradation -- when it fires -- affects the whole image,
    # the same way the hard-tier stress transform degrades the whole
    # original before any cropping happens downstream in this pipeline.
    # (Zero or one step here depending on THUMBNAIL_AUG_MODE -- see above.)
    *_thumbnail_round_trip_steps(),
    T.RandomResizedCrop(384, scale=(0.72, 1.0), ratio=(0.88, 1.14), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(brightness=0.14, contrast=0.14, saturation=0.12, hue=0.025)], p=0.8),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.3, 1.0))], p=0.25),
    T.RandomRotation(3, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.18, scale=(0.01, 0.08), ratio=(0.4, 2.5), value="random"),
])


def load_image(path, augmented, rng):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if augmented:
            buffer = io.BytesIO()
            quality = rng.randint(58, 90)
            image.save(buffer, "JPEG", quality=quality, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as recompressed:
                return augmented_transform(recompressed.convert("RGB")).numpy()
        return exact_transform(image).numpy()


def feature_model(source, destination):
    model = onnx.load(source)
    if not any(output.name == FEATURE_OUTPUT for output in model.graph.output):
        model.graph.output.append(helper.make_tensor_value_info(FEATURE_OUTPUT, TensorProto.FLOAT, ["batch", 384]))
    onnx.save(model, destination)


def extract(session, records, cache_path, augmented_views, seed):
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["x"], cached["y"], cached["domains"]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    features, labels, domains = [], [], []
    pending, pending_meta = [], []
    total = len(records) * (1 + augmented_views)
    completed = 0
    for record in records:
        for view in range(1 + augmented_views):
            pending.append(load_image(record[0], view > 0, rng))
            pending_meta.append((record[1], record[2]))
            if len(pending) == 8:
                completed += run_feature_batch(session, pending, pending_meta, features, labels, domains)
                pending, pending_meta = [], []
                print(f"\rfeatures {cache_path.stem}: {completed}/{total}", end="", flush=True)
    if pending:
        completed += run_feature_batch(session, pending, pending_meta, features, labels, domains)
    print(f"\rfeatures {cache_path.stem}: {completed}/{total}")
    x = np.concatenate(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    domain_array = np.asarray(domains)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y, domains=domain_array)
    return x, y, domain_array


def run_feature_batch(session, images, metadata, features, labels, domains):
    outputs = session.run([FEATURE_OUTPUT], {"pixel_values": np.stack(images).astype(np.float32)})[0]
    features.append(outputs)
    labels.extend(item[0] for item in metadata)
    domains.extend(item[1] for item in metadata)
    return len(images)


def metrics(logits, y, domains, threshold):
    scores = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    predicted = scores >= threshold
    ai = y == 1
    real = ~ai
    ai_recall = float(predicted[ai].mean())
    real_recall = float((~predicted[real]).mean())
    per_domain = []
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        correct = predicted[mask] == y[mask].astype(bool)
        per_domain.append({"domain": domain, "label": "ai" if y[mask][0] else "real", "recall": float(correct.mean()), "total": int(mask.sum())})
    return {
        "threshold": float(threshold),
        "balancedAccuracy": (ai_recall + real_recall) / 2,
        "aiRecall": ai_recall,
        "realRecall": real_recall,
        "minimumDomainRecall": min(item["recall"] for item in per_domain),
        "domains": per_domain,
    }


def choose_threshold(logits, y, domains):
    scores = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    ordered = np.unique(scores)
    thresholds = np.concatenate(([0.001], (ordered[:-1] + ordered[1:]) / 2, [0.999]))
    candidates = [metrics(logits, y, domains, value) for value in thresholds]
    eligible = [item for item in candidates if item["realRecall"] >= 0.9]
    return max(eligible or candidates, key=lambda item: (item["minimumDomainRecall"], item["balancedAccuracy"], item["aiRecall"]))


def fit_candidates(train_x, train_y, train_domains, cal_x, cal_y, cal_domains, dev, original_weight, original_bias, seed):
    torch.manual_seed(seed)
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y).reshape(-1, 1)
    sample_weights = torch.tensor([
        4.0 if domain == "adm" else 2.0 if domain == "biggan" else 1.5 if domain == "midjourney" else 1.0
        for domain in train_domains
    ], dtype=torch.float32).reshape(-1, 1)
    results = []
    best = None
    for decay in (0.1, 0.03, 0.01, 0.003, 0.001):
        head = nn.Linear(384, 1)
        with torch.no_grad():
            head.weight.copy_(original_weight)
            head.bias.copy_(original_bias)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.015, weight_decay=decay)
        for step in range(800):
            generator = torch.Generator().manual_seed(seed + step)
            indices = torch.randperm(len(x), generator=generator)[: min(768, len(x))]
            losses = nn.functional.binary_cross_entropy_with_logits(head(x[indices]), y[indices], reduction="none")
            loss = (losses * sample_weights[indices]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        learned_weight = head.weight.detach().clone()
        learned_bias = head.bias.detach().clone()
        for alpha in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
            weight = original_weight * (1 - alpha) + learned_weight * alpha
            bias = original_bias * (1 - alpha) + learned_bias * alpha
            cal_logits = cal_x @ weight.numpy().reshape(-1) + float(bias)
            calibration = choose_threshold(cal_logits, cal_y, cal_domains)
            development = None
            if dev:
                dev_x, dev_y, dev_domains = dev
                dev_logits = dev_x @ weight.numpy().reshape(-1) + float(bias)
                development = metrics(dev_logits, dev_y, dev_domains, calibration["threshold"])
            objective = min(calibration["balancedAccuracy"], development["balancedAccuracy"] if development else 1.0)
            if calibration["realRecall"] < 0.9:
                objective -= 1
            if development:
                named_domains = {item["domain"]: item["recall"] for item in development["domains"]}
                major_generators = ["adm", "biggan", "flux-reason", "glide", "sd15", "synthetic-characters", "vqdm", "wukong"]
                if development["realRecall"] < 0.9 or any(named_domains.get(name, 1.0) < 0.75 for name in major_generators):
                    objective -= 1
                if named_domains.get("midjourney", 1.0) < 0.55:
                    objective -= 1
            record = {"weightDecay": decay, "alpha": alpha, "objective": objective, "calibration": calibration, "development": development}
            results.append(record)
            if best is None or (objective, calibration["minimumDomainRecall"], calibration["balancedAccuracy"]) > best[0]:
                best = ((objective, calibration["minimumDomainRecall"], calibration["balancedAccuracy"]), weight, bias, record)
    return best, results


def fit_mlp_candidates(train_x, train_y, train_domains, cal_x, cal_y, cal_domains, dev, seed):
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y).reshape(-1, 1)
    sample_weights = torch.tensor([
        3.0 if domain == "adm" else 1.7 if domain == "biggan" else 1.35 if domain == "midjourney" else 1.0
        for domain in train_domains
    ], dtype=torch.float32).reshape(-1, 1)
    results = []
    best = None
    for hidden in (16, 32, 64):
        for decay in (0.01, 0.003, 0.001):
            torch.manual_seed(seed + hidden)
            head = nn.Sequential(nn.Linear(384, hidden), nn.GELU(), nn.Linear(hidden, 1))
            optimizer = torch.optim.AdamW(head.parameters(), lr=0.0015, weight_decay=decay)
            for step in range(1800):
                generator = torch.Generator().manual_seed(seed + hidden * 10000 + step)
                indices = torch.randperm(len(x), generator=generator)[: min(1024, len(x))]
                losses = nn.functional.binary_cross_entropy_with_logits(head(x[indices]), y[indices], reduction="none")
                loss = (losses * sample_weights[indices]).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                cal_logits = head(torch.from_numpy(cal_x)).numpy().reshape(-1)
                calibration = choose_threshold(cal_logits, cal_y, cal_domains)
                development = None
                if dev:
                    dev_x, dev_y, dev_domains = dev
                    dev_logits = head(torch.from_numpy(dev_x)).numpy().reshape(-1)
                    development = metrics(dev_logits, dev_y, dev_domains, calibration["threshold"])
            objective = min(calibration["balancedAccuracy"], development["balancedAccuracy"] if development else 1.0)
            eligible = calibration["realRecall"] >= 0.9 and calibration["minimumDomainRecall"] >= 0.8
            if development:
                named = {item["domain"]: item["recall"] for item in development["domains"]}
                major = ["adm", "biggan", "flux-reason", "glide", "sd15", "synthetic-characters", "vqdm", "wukong"]
                eligible = eligible and development["realRecall"] >= 0.9 and all(named.get(name, 1.0) >= 0.75 for name in major) and named.get("midjourney", 1.0) >= 0.55
            if not eligible:
                objective -= 1
            record = {"hidden": hidden, "weightDecay": decay, "objective": objective, "eligible": eligible, "calibration": calibration, "development": development}
            results.append(record)
            rank = (objective, calibration["minimumDomainRecall"], calibration["balancedAccuracy"])
            if best is None or rank > best[0]:
                best = (rank, head, record)
    return best, results


def config_source_dir(onnx_path):
    """Locate config.json/preprocessor_config.json for the ViT-S/16 backbone.

    Historically this script assumed --onnx lived two directories below a
    model root that also held these two files (e.g.
    .../CommunityForensics/official-q8/onnx/model.onnx). When --onnx is
    instead passed directly as .../commfor-direct-export/model_fp32.onnx
    (one level, not two), onnx_path.parent.parent lands on .bench-data/models
    which has neither file, and the original unconditional shutil.copy2
    crashed here -- after all feature extraction and candidate fitting had
    already finished. This checks the historical location first and falls
    back to the known-good CommunityForensics/official-q8 config (verified
    to describe the same architecture/preprocessing: ViT-S/16, 384px,
    shortest-edge-440 + center-crop-384, ImageNet mean/std -- identical to
    this script's own exact_transform)."""
    primary = onnx_path.parent.parent
    if (primary / "config.json").exists() and (primary / "preprocessor_config.json").exists():
        return primary
    fallback = Path(".bench-data/models/CommunityForensics/official-q8")
    if (fallback / "config.json").exists() and (fallback / "preprocessor_config.json").exists():
        return fallback
    raise RuntimeError(f"Could not locate config.json/preprocessor_config.json near {onnx_path} or at {fallback}")


def export_model(checkpoint, head, output):
    import timm

    class ViTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
            self.vit.head = nn.Linear(384, 1)

        def forward(self, pixel_values):
            return self.vit(pixel_values)

    model = ViTClassifier().eval()
    state = load_file(checkpoint)
    model.load_state_dict(state, strict=True)
    model.vit.head = head.eval()
    output.mkdir(parents=True, exist_ok=True)
    onnx_dir = output / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    fp32 = onnx_dir / "model_fp32.onnx"
    quantized = onnx_dir / "model_quantized.onnx"
    torch.onnx.export(model, (torch.zeros(1, 3, 384, 384),), fp32, input_names=["pixel_values"], output_names=["logits"], dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18, dynamo=False)
    quantize_dynamic(fp32, quantized, op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8, per_channel=True)
    # NOTE: previously this unconditionally deleted the FP32 export
    # (`fp32.unlink()`) right after quantizing, so the FP32 ONNX never
    # persisted to disk -- only the quantized model survived. That made it
    # impossible to do a row-by-row FP32-vs-Q8 shipping comparison after the
    # fact. Keep the FP32 file; both precisions are cheap enough to retain
    # for the shipping decision and are not committed to git regardless
    # (.bench-data/ is a build-scratch directory).
    return fp32, quantized


def main():
    args = parse_args()
    if sha256(args.checkpoint) != SOURCE_SHA256:
        raise RuntimeError("Official checkpoint checksum mismatch")
    args.cache.mkdir(parents=True, exist_ok=True)
    feature_path = args.cache / "feature-model.onnx"
    feature_model(args.onnx, feature_path)
    session = ort.InferenceSession(feature_path, providers=["CPUExecutionProvider"])
    base_records = samples(args.train)
    train_parts = [extract(session, base_records, args.cache / "train.npz", 2, args.seed)]
    for index, root in enumerate(args.train_extra):
        records = samples(root)
        part = extract(session, records, args.cache / f"train-extra-{index}.npz", 2, args.seed + 10 + index)
        train_parts.append((part[0], part[1], np.repeat(np.asarray([record[2] for record in records]), 3)))
    for index, root in enumerate(args.train_robust):
        records = samples(root)
        part = extract(session, records, args.cache / f"train-robust-{index}.npz", 0, args.seed + 20 + index)
        train_parts.append((part[0], part[1], np.asarray([record[2] for record in records])))
    train = (
        np.concatenate([part[0] for part in train_parts]),
        np.concatenate([part[1] for part in train_parts]),
        np.concatenate([part[2] for part in train_parts]),
    )
    calibration_records = sum((samples(root) for root in args.calibration), [])
    calibration = extract(session, calibration_records, args.cache / "calibration.npz", 0, args.seed + 1)
    development_records = sum((samples(root) for root in args.development), [])
    development = extract(session, development_records, args.cache / "development.npz", 0, args.seed + 2) if development_records else None
    state = load_file(args.checkpoint)
    best, candidates = fit_mlp_candidates(*train, *calibration, development, args.seed)
    _, head, selected = best
    fp32, quantized = export_model(args.checkpoint, head, args.output)
    source_model = config_source_dir(args.onnx)
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy2(source_model / name, args.output / name)
    report = {
        "seed": args.seed,
        "thumbnailAugMode": THUMBNAIL_AUG_MODE,
        "trainingImages": len(samples(args.train)) + sum(len(samples(root)) for root in args.train_extra) + sum(len(samples(root)) for root in args.train_robust),
        "trainingFeatureRows": len(train[0]),
        "calibrationImages": len(calibration[0]),
        "developmentImages": len(development[0]) if development else 0,
        "selected": selected,
        "candidates": candidates,
        "modelSha256": sha256(quantized),
        "modelBytes": quantized.stat().st_size,
        "fp32Sha256": sha256(fp32),
        "fp32Bytes": fp32.stat().st_size,
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["selected"], indent=2))
    print(f"Model SHA-256: {report['modelSha256']}")


if __name__ == "__main__":
    main()
