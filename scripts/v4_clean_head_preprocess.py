"""Frozen preprocessing helpers for the v4 clean linear-head challenger.

This module has no model-loading or selection code.  The protocol lock pins its
file hash before any TRAIN or CALIBRATION feature is extracted.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CROP_SIZE = 384
RESIZE_SHORTEST_EDGE = 440
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class MatchedRecord:
    path: Path
    relative_path: str
    transform: str
    role: str
    label: str
    source: str
    sample_id: str
    byte_count: int
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_not_reparse(path: Path) -> None:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    is_junction = getattr(os.path, "isjunction", lambda _path: False)
    if path.is_symlink() or bool(attributes & reparse_flag) or is_junction(path):
        raise RuntimeError(f"symlink/reparse paths are forbidden: {path}")


def official_rgb_view(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("decoded image has invalid dimensions")
    # Every registered v4 input is an already-materialized square transform.
    # Refusing any other geometry removes a silent Python-vs-browser rounding
    # degree of freedom from this small-sample experiment.
    if width != height:
        raise ValueError("v4 accepts only the registered square matched inputs")
    resized_width = RESIZE_SHORTEST_EDGE
    resized_height = RESIZE_SHORTEST_EDGE
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left = (resized_width - CROP_SIZE) // 2
    top = (resized_height - CROP_SIZE) // 2
    return resized.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))


def official_tensor(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        view = official_rgb_view(source)
    pixels = np.asarray(view, dtype=np.float32) / np.float32(255.0)
    normalized = (pixels - MEAN) / STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


def scan_allowed_records(
    root: Path,
    transforms: tuple[str, ...],
    roles: tuple[str, ...],
) -> list[MatchedRecord]:
    """Enumerate only explicitly allowed roots; never walk their common parent."""

    records: list[MatchedRecord] = []
    resolved_root = root.resolve(strict=True)
    assert_not_reparse(root)
    for transform in transforms:
        for role in roles:
            role_root = root / transform / role
            if not role_root.is_dir():
                raise FileNotFoundError(f"missing allowed role root: {role_root}")
            for component in (root / transform, role_root):
                assert_not_reparse(component)
                component.resolve(strict=True).relative_to(resolved_root)
            for path in sorted(role_root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
                assert_not_reparse(path)
                if not path.is_file():
                    continue
                path.resolve(strict=True).relative_to(resolved_root)
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise RuntimeError(f"unexpected regular file below allowed root: {path}")
                relative = path.relative_to(root)
                parts = relative.parts
                if len(parts) != 5:
                    raise RuntimeError(f"unexpected matched-input path layout: {relative.as_posix()}")
                observed_transform, observed_role, label, source, _ = parts
                if observed_transform != transform or observed_role != role:
                    raise RuntimeError(f"role-root escape detected: {relative.as_posix()}")
                if label not in {"ai", "real"}:
                    raise RuntimeError(f"unexpected label in {relative.as_posix()}")
                digest = file_sha256(path)
                records.append(
                    MatchedRecord(
                        path=path,
                        relative_path=relative.as_posix(),
                        transform=transform,
                        role=role,
                        label=label,
                        source=source,
                        sample_id=path.stem,
                        byte_count=path.stat().st_size,
                        sha256=digest,
                    )
                )
    return sorted(records, key=lambda item: item.relative_path)


def canonical_inventory(records: list[MatchedRecord]) -> tuple[bytes, str]:
    canonical = "".join(
        f"{record.relative_path}\t{record.byte_count}\t{record.sha256}\n"
        for record in sorted(records, key=lambda item: item.relative_path)
    ).encode("utf-8")
    return canonical, hashlib.sha256(canonical).hexdigest()
