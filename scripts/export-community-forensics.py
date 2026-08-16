"""Export a reproducible official-head Community Forensics browser baseline.

This does not reproduce Proofmark WebWild v3, whose 384→32→1 adapted head and
exact training artifact are not available in this repository.
"""

import argparse
import hashlib
from pathlib import Path

import onnx
import timm
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from safetensors.torch import load_file


SOURCE_SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"


class ViTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch16_384.augreg_in21k_ft_in1k",
            pretrained=False,
        )
        self.vit.head = nn.Linear(384, 1, bias=True)

    def forward(self, pixel_values):
        return self.vit(pixel_values)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path, help="Official model.safetensors file")
parser.add_argument("output", type=Path, help="Destination model_quantized.onnx")
parser.add_argument(
    "--fp32-output",
    type=Path,
    help="Optional destination for the checked FP32 ONNX graph used to create Q8",
)
args = parser.parse_args()

source_digest = sha256(args.checkpoint)
if source_digest != SOURCE_SHA256:
    raise RuntimeError(f"Checkpoint checksum mismatch: {source_digest}")

model = ViTClassifier().eval()
missing, unexpected = model.load_state_dict(load_file(args.checkpoint), strict=False)
if missing or unexpected:
    raise RuntimeError(f"Checkpoint mismatch. Missing={missing}; unexpected={unexpected}")

args.output.parent.mkdir(parents=True, exist_ok=True)
if args.fp32_output and args.fp32_output.resolve() == args.output.resolve():
    raise RuntimeError("FP32 and Q8 outputs must be different files")

full_precision = args.fp32_output or args.output.with_name("model_fp32.onnx")
full_precision.parent.mkdir(parents=True, exist_ok=True)
fp32_temporary = full_precision.with_suffix(full_precision.suffix + ".tmp")
q8_temporary = args.output.with_suffix(args.output.suffix + ".tmp")
fp32_temporary.unlink(missing_ok=True)
q8_temporary.unlink(missing_ok=True)
dummy = torch.zeros(1, 3, 384, 384, dtype=torch.float32)
torch.onnx.export(
    model,
    (dummy,),
    fp32_temporary,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=18,
    dynamo=False,
)
onnx.checker.check_model(onnx.load(fp32_temporary))
fp32_temporary.replace(full_precision)

# Keep Conv in fp32 because ONNX Runtime Web does not implement ConvInteger.
quantize_dynamic(
    full_precision,
    q8_temporary,
    op_types_to_quantize=["MatMul"],
    weight_type=QuantType.QInt8,
    per_channel=True,
)
onnx.checker.check_model(onnx.load(q8_temporary))
q8_temporary.replace(args.output)
if not args.fp32_output:
    full_precision.unlink()
else:
    print(f"Wrote {full_precision} ({full_precision.stat().st_size} bytes)")
    print(f"FP32 SHA-256 {sha256(full_precision)}")
print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
print(f"Q8 SHA-256 {sha256(args.output)}")
