"""Score a final ONNX head against clean/web/hard holdout tiers using
onnxruntime, with preprocessing matching exact_transform in
train-community-head.py (resize shorter side to 440 bicubic, center-crop
384, ImageNet mean/std normalize) -- the same preprocessing the real
extension applies.

Computes, per tier: overall balanced accuracy, real recall, ai recall, and
per-AI-generator recall (by source-domain folder name), at a fixed raw
sigmoid threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

exact_transform = T.Compose([
    T.Resize(440, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(384),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


def samples(root):
    output = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".avif"}:
            continue
        relative = path.relative_to(root).parts
        label_index = next((i for i, part in enumerate(relative) if part in {"ai", "real"}), None)
        if label_index is None:
            continue
        label = relative[label_index]
        domain = relative[label_index + 1] if len(relative) > label_index + 2 else "unknown"
        output.append((path, 1.0 if label == "ai" else 0.0, domain))
    return output


def load_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return exact_transform(image).numpy()


def run_tier(session, root, threshold, batch_size=16):
    records = samples(root)
    if not records:
        return None
    logits = []
    labels = []
    domains = []
    pending = []
    for record in records:
        pending.append(record)
        if len(pending) == batch_size:
            logits.append(run_batch(session, pending))
            labels.extend(r[1] for r in pending)
            domains.extend(r[2] for r in pending)
            pending = []
    if pending:
        logits.append(run_batch(session, pending))
        labels.extend(r[1] for r in pending)
        domains.extend(r[2] for r in pending)
    logits = np.concatenate(logits)
    labels = np.asarray(labels, dtype=np.float32)
    domains = np.asarray(domains)
    return compute_metrics(logits, labels, domains, threshold, root)


def run_batch(session, records):
    images = np.stack([load_image(r[0]) for r in records]).astype(np.float32)
    return session.run(["logits"], {"pixel_values": images})[0].reshape(-1)


def compute_metrics(logits, y, domains, threshold, root):
    scores = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    predicted = scores >= threshold
    ai_mask = y == 1
    real_mask = ~ai_mask
    ai_recall = float(predicted[ai_mask].mean()) if ai_mask.any() else None
    real_recall = float((~predicted[real_mask]).mean()) if real_mask.any() else None
    balanced_accuracy = (ai_recall + real_recall) / 2 if ai_recall is not None and real_recall is not None else None

    per_generator = []
    for domain in sorted(set(domains[ai_mask].tolist())):
        mask = ai_mask & (domains == domain)
        recall = float(predicted[mask].mean())
        per_generator.append({"domain": domain, "n": int(mask.sum()), "recall": recall})

    min_generator_recall = min((g["recall"] for g in per_generator), default=None)

    return {
        "root": str(root),
        "n": int(len(y)),
        "nAi": int(ai_mask.sum()),
        "nReal": int(real_mask.sum()),
        "threshold": float(threshold),
        "balancedAccuracy": balanced_accuracy,
        "aiRecall": ai_recall,
        "realRecall": real_recall,
        "perGeneratorRecall": per_generator,
        "minimumGeneratorRecall": min_generator_recall,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--web", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    report = {
        "onnx": str(args.onnx),
        "threshold": args.threshold,
        "tiers": {
            "clean": run_tier(session, args.clean, args.threshold),
            "web": run_tier(session, args.web, args.threshold),
            "hard": run_tier(session, args.hard, args.threshold),
        },
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
