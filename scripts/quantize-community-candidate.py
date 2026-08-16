"""Create a checked, local MatMul-Q8 research artifact from a pinned ONNX file."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    source_digest = sha256(args.input)
    if source_digest.lower() != args.expected_sha256.lower():
        raise RuntimeError(f"Input checksum mismatch: {source_digest}")
    if args.input.resolve() == args.output.resolve():
        raise RuntimeError("Refusing to overwrite the source model")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    quantize_dynamic(
        args.input,
        temporary,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    onnx.checker.check_model(onnx.load(temporary))
    temporary.replace(args.output)
    print(f"Source SHA-256: {source_digest}")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"Output SHA-256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
