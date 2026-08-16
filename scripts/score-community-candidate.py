"""Score local Community Forensics candidates without changing the extension.

This is a research diagnostic, not an accuracy claim or a threshold-selection
protocol. It performs no network requests and records every raw per-view logit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


CROP = 384
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
NORMALIZATIONS = {
    "imagenet": (
        np.asarray([0.485, 0.456, 0.406], dtype=np.float32),
        np.asarray([0.229, 0.224, 0.225], dtype=np.float32),
    ),
    "clip": (
        np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32),
        np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", action="append", choices=("official", "native", "squash"), required=True)
    parser.add_argument("--normalization", choices=tuple(NORMALIZATIONS), required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_for(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "ai" in parts or "fake" in parts:
        return "ai"
    if "real" in parts:
        return "real"
    return None


def collect_samples(root: Path) -> list[tuple[Path, str]]:
    output = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_for(path.relative_to(root))
        if label is not None:
            output.append((path, label))
    return output


def official_view(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = 440 / min(width, height)
    resized = image.resize((max(440, round(width * scale)), max(440, round(height * scale))), Image.Resampling.BICUBIC)
    width, height = resized.size
    left, top = (width - CROP) // 2, (height - CROP) // 2
    return resized.crop((left, top, left + CROP, top + CROP))


def native_view(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width < CROP or height < CROP:
        pixels = np.asarray(image, dtype=np.uint8)
        pixels = np.pad(
            pixels,
            ((0, max(0, CROP - height)), (0, max(0, CROP - width)), (0, 0)),
            mode="reflect",
        )
        image = Image.fromarray(pixels)
        width, height = image.size
    left, top = (width - CROP) // 2, (height - CROP) // 2
    return image.crop((left, top, left + CROP, top + CROP))


def squash_view(image: Image.Image) -> Image.Image:
    return image.resize((CROP, CROP), Image.Resampling.BICUBIC)


VIEW_FUNCTIONS = {"official": official_view, "native": native_view, "squash": squash_view}


def tensor_for(image: Image.Image, normalization: str) -> np.ndarray:
    mean, std = NORMALIZATIONS[normalization]
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - mean) / std
    return np.transpose(pixels, (2, 0, 1))[None].astype(np.float32, copy=False)


def sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    if args.threads < 1 or args.threads > 32:
        raise ValueError("--threads must be in [1, 32]")

    model_digest = sha256(args.model)
    if args.expected_sha256 and model_digest.lower() != args.expected_sha256.lower():
        raise RuntimeError(f"Model checksum mismatch: {model_digest}")

    samples = collect_samples(args.root)
    if not samples:
        raise RuntimeError(f"No labeled images below {args.root}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rows = []
    for path, label in samples:
        started = time.perf_counter()
        with Image.open(path) as source:
            image = source.convert("RGB")
        scores = {}
        for view in args.view:
            view_started = time.perf_counter()
            transformed = VIEW_FUNCTIONS[view](image)
            tensor = tensor_for(transformed, args.normalization)
            logit = float(np.asarray(session.run([output_name], {input_name: tensor})[0]).reshape(-1)[0])
            scores[view] = {
                "logit": logit,
                "probability": sigmoid(logit),
                "elapsedMs": round((time.perf_counter() - view_started) * 1000, 3),
            }
        mean_probability = float(np.mean([scores[view]["probability"] for view in args.view]))
        rows.append({
            "file": path.relative_to(args.root).as_posix(),
            "label": label,
            "views": scores,
            "meanProbability": mean_probability,
            "predicted": "ai" if mean_probability >= args.threshold else "real",
            "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
        })

    ai_rows = [row for row in rows if row["label"] == "ai"]
    real_rows = [row for row in rows if row["label"] == "real"]
    ai_correct = sum(row["predicted"] == "ai" for row in ai_rows)
    real_correct = sum(row["predicted"] == "real" for row in real_rows)
    recall = ai_correct / len(ai_rows) if ai_rows else 0.0
    specificity = real_correct / len(real_rows) if real_rows else 0.0
    balanced_accuracy = (recall + specificity) / 2
    report = {
        "schema": "realimage-community-candidate-diagnostic-v1",
        "model": {"path": str(args.model.resolve()), "sha256": model_digest},
        "root": str(args.root.resolve()),
        "runtime": "onnxruntime-cpu",
        "threads": args.threads,
        "normalization": args.normalization,
        "views": args.view,
        "aggregation": "probability-mean",
        "threshold": args.threshold,
        "metrics": {
            "aiRecall": recall,
            "realRecall": specificity,
            "balancedAccuracy": balanced_accuracy,
            "aiCorrect": ai_correct,
            "aiTotal": len(ai_rows),
            "realCorrect": real_correct,
            "realTotal": len(real_rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)

    print(f"Model SHA-256: {model_digest}")
    print(f"Views: {','.join(args.view)}; normalization: {args.normalization}; threshold: {args.threshold:.8f}")
    print(f"AI recall: {ai_correct}/{len(ai_rows)} ({recall * 100:.1f}%)")
    print(f"Real recall: {real_correct}/{len(real_rows)} ({specificity * 100:.1f}%)")
    print(f"Balanced accuracy: {balanced_accuracy * 100:.1f}%")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
