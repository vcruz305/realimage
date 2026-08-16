"""Run the frozen v4 clean linear-head challenger without opening qualification.

The runner has no configurable data roots, roles, transforms, model family, or
hyperparameter grid.  It verifies an execution lock that pins the protocol and
tooling before it decodes any allowed TRAIN or CALIBRATION image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import PIL
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from v4_clean_head_preprocess import (
    MatchedRecord,
    assert_not_reparse,
    canonical_inventory,
    file_sha256,
    official_tensor,
    scan_allowed_records,
)


PROTOCOL_RELATIVE = Path(".bench-data/license-clean-challenger-v4/protocol-lock.json")
TOOLING_RELATIVE = Path(".bench-data/license-clean-challenger-v4/tooling-lock.json")
SOURCE_MODEL_RELATIVE = Path(".bench-data/models/commfor-direct-export/model_fp32.onnx")
MATCHED_ROOT_RELATIVE = Path(".bench-data/license-clean-calibration-v3/matched-inputs")
OUTPUT_ROOT_RELATIVE = Path(".bench-data/license-clean-challenger-v4")
EXPECTED_SOURCE_SHA256 = "8c7762fe3b7f407a15b8cc7796e3b286fc4a05ae5c2e580a936730ca5f9a4a33"
EXPECTED_SOURCE_BYTES = 87388774
SOURCE_MANIFEST_RELATIVE = Path(".bench-data/license-clean-calibration-v3/manifest.json")
MATCHED_MANIFEST_RELATIVE = Path(".bench-data/license-clean-calibration-v3/matched-inputs/manifest.json")
MODEL_CONFIG_ROOT_RELATIVE = Path("public/models/RealImage/license-clean-v4-fp32-a6297c3a")
FEATURE_OUTPUT = "/vit/Gather_1_output_0"
WEIGHT_NAME = "vit.head.weight"
BIAS_NAME = "vit.head.bias"
TRANSFORMS = (
    "matched-square-jpeg-q75-768",
    "matched-square-webp-q72-768",
    "matched-square-thumbnail-jpeg-q75-512",
)
ROLES = ("train", "calibration")
REGULARIZATION_GRID = (
    ("official", math.inf),
    ("proximal-30", 30.0),
    ("proximal-10", 10.0),
    ("proximal-3", 3.0),
    ("proximal-1", 1.0),
    ("proximal-0p3", 0.3),
    ("proximal-0p1", 0.1),
    ("proximal-0p03", 0.03),
)
EXPECTED_ROLE_COUNTS = {
    ("train", "ai", "nanoBanana"): 75,
    ("train", "ai", "fluxKrea"): 75,
    ("train", "real", "docci"): 150,
    ("calibration", "ai", "nanoBanana"): 45,
    ("calibration", "ai", "fluxKrea"): 45,
    ("calibration", "real", "docci"): 90,
}
SOURCE_TOTAL_WEIGHTS = {
    ("ai", "nanoBanana"): 0.25,
    ("ai", "fluxKrea"): 0.25,
    ("real", "docci"): 0.50,
}
SEED = 323
BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def relative_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def assert_file(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"byte-count mismatch for {path}: {path.stat().st_size}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed}")


def dependency_tree_fingerprint(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    assert_not_reparse(root)
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
            continue
        assert_not_reparse(path)
        if not path.is_file():
            continue
        path.resolve(strict=True).relative_to(resolved_root)
        digest = file_sha256(path)
        rows.append((path.relative_to(root).as_posix(), path.stat().st_size, digest))
    canonical = "".join(f"{name}\t{size}\t{digest}\n" for name, size, digest in rows).encode("utf-8")
    return {
        "files": len(rows),
        "bytes": sum(size for _, size, _ in rows),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def load_and_verify_locks(
    project_root: Path,
    execution_lock_path: Path,
    *,
    preflight_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution_path = execution_lock_path.resolve()
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("schema") != "realimage-clean-head-execution-lock-v4":
        raise RuntimeError("unexpected execution-lock schema")
    if preflight_only:
        if execution.get("releasedForZeroScorePreflight") is not True:
            raise RuntimeError("execution lock has not released the zero-score preflight")
    elif execution.get("releasedForTrainCalibration") is not True:
        raise RuntimeError("execution lock has not released TRAIN/CALIBRATION scoring")
    if execution.get("qualificationAuthorized") is not False:
        raise RuntimeError("execution lock must keep qualification unauthorized")
    cross_lane = execution.get("crossLane", {})
    if cross_lane.get("bothHypothesisClassesFrozenBeforeScoring") is not True:
        raise RuntimeError("cross-lane hypothesis classes were not frozen before scoring")
    if cross_lane.get("v4aResultsWithheldFromCleanHeadLane") is not True:
        raise RuntimeError("v4a results are not attested as withheld")
    if cross_lane.get("cleanHeadResultsWithheldFromV4aLane") is not True:
        raise RuntimeError("clean-head results are not attested as withheld")

    protocol_path = project_root / PROTOCOL_RELATIVE
    tooling_path = project_root / TOOLING_RELATIVE
    assert_file(protocol_path, execution["protocol"]["sha256"])
    assert_file(tooling_path, execution["tooling"]["sha256"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tooling = json.loads(tooling_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen-before-feature-extraction":
        raise RuntimeError("protocol is not frozen")
    if protocol.get("qualificationRead") is not False:
        raise RuntimeError("protocol does not attest a sealed qualification")

    for item in tooling["files"]:
        assert_file(project_root / item["path"], item["sha256"], item["bytes"])
    for item in tooling["dependencyTrees"]:
        observed = dependency_tree_fingerprint(project_root / item["path"])
        expected = {key: item[key] for key in ("files", "bytes", "sha256")}
        if observed != expected:
            raise RuntimeError(
                f"pinned dependency tree changed for {item['path']}: expected {expected}; observed {observed}"
            )
    for item in execution.get("artifacts", []):
        assert_file(project_root / item["path"], item["sha256"], item["bytes"])
    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
    }
    if observed_runtime != tooling["runtime"]:
        raise RuntimeError(
            f"pinned Python runtime changed: expected {tooling['runtime']}; observed {observed_runtime}"
        )
    executable = tooling["pythonExecutable"]
    observed_executable = Path(sys.executable).resolve()
    expected_executable = Path(executable["path"]).resolve()
    if observed_executable != expected_executable:
        raise RuntimeError(
            f"Python executable changed: expected {expected_executable}; observed {observed_executable}"
        )
    assert_file(expected_executable, executable["sha256"], executable["bytes"])
    return protocol, tooling


def verify_records(records: list[MatchedRecord]) -> None:
    observed: dict[tuple[str, str, str], int] = {}
    for record in records:
        key = (record.role, record.label, record.source)
        observed[key] = observed.get(key, 0) + 1
    if observed != EXPECTED_ROLE_COUNTS:
        raise RuntimeError(f"allowed data counts changed: {observed}")

    role_sample_ids: dict[str, set[str]] = {}
    for role, expected_unique in (("train", 100), ("calibration", 60)):
        role_rows = [record for record in records if record.role == role]
        grouped: dict[str, list[MatchedRecord]] = {}
        for record in role_rows:
            grouped.setdefault(record.sample_id, []).append(record)
        if len(grouped) != expected_unique:
            raise RuntimeError(f"{role} unique-original count changed: {len(grouped)}")
        role_sample_ids[role] = set(grouped)
        for sample_id, rows in grouped.items():
            if len(rows) != len(TRANSFORMS):
                raise RuntimeError(f"{role}/{sample_id} is not present in exactly three transforms")
            if {row.transform for row in rows} != set(TRANSFORMS):
                raise RuntimeError(f"{role}/{sample_id} transform set changed")
            if len({(row.label, row.source) for row in rows}) != 1:
                raise RuntimeError(f"{role}/{sample_id} label/source changed across transforms")
    overlap = role_sample_ids["train"] & role_sample_ids["calibration"]
    if overlap:
        raise RuntimeError(f"TRAIN/CALIBRATION sample-ID overlap detected: {sorted(overlap)[:3]}")


def build_feature_model(source_path: Path, destination: Path) -> dict[str, Any]:
    model = onnx.load(source_path, load_external_data=False)
    onnx.checker.check_model(model)
    outputs = [item.name for item in model.graph.output]
    if outputs != ["logits"]:
        raise RuntimeError(f"unexpected source outputs: {outputs}")
    producers = [node for node in model.graph.node if FEATURE_OUTPUT in node.output]
    if len(producers) != 1 or producers[0].op_type != "Gather":
        raise RuntimeError("penultimate feature tensor attachment point changed")
    model.graph.output.append(
        helper.make_tensor_value_info(FEATURE_OUTPUT, TensorProto.FLOAT, ["batch", 384])
    )
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination, save_as_external_data=False)
    return {
        "path": destination,
        "sha256": file_sha256(destination),
        "bytes": destination.stat().st_size,
    }


def session_for(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def initializer_array(model: onnx.ModelProto, name: str) -> np.ndarray:
    matches = [item for item in model.graph.initializer if item.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one initializer named {name}")
    return np.asarray(numpy_helper.to_array(matches[0]), dtype=np.float64)


def extract_features(
    session: ort.InferenceSession,
    records: list[MatchedRecord],
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    graph_logits: list[np.ndarray] = []
    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start : start + BATCH_SIZE]
        tensors = np.stack([official_tensor(record.path) for record in batch_records]).astype(np.float32)
        logits, batch_features = session.run(["logits", FEATURE_OUTPUT], {"pixel_values": tensors})
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        batch_features = np.asarray(batch_features, dtype=np.float32)
        if batch_features.shape != (len(batch_records), 384):
            raise RuntimeError(f"unexpected feature shape: {batch_features.shape}")
        if not np.isfinite(batch_features).all() or not np.isfinite(logits).all():
            raise RuntimeError("feature extraction produced a non-finite value")
        features.append(batch_features)
        graph_logits.append(logits)
        print(f"\rfeatures {start + len(batch_records)}/{len(records)}", end="", flush=True)
    print()
    return np.concatenate(features), np.concatenate(graph_logits)


def write_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def record_document(records: list[MatchedRecord]) -> dict[str, Any]:
    return {
        "schema": "realimage-clean-head-feature-rows-v4",
        "rows": [
            {
                "relativePath": record.relative_path,
                "transform": record.transform,
                "role": record.role,
                "label": record.label,
                "source": record.source,
                "sampleId": record.sample_id,
                "bytes": record.byte_count,
                "sha256": record.sha256,
            }
            for record in records
        ],
    }


def labels_for(records: list[MatchedRecord]) -> np.ndarray:
    return np.asarray([1.0 if record.label == "ai" else 0.0 for record in records], dtype=np.float64)


def loss_weights(records: list[MatchedRecord]) -> np.ndarray:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.label, record.source)
        counts[key] = counts.get(key, 0) + 1
    if set(counts) != set(SOURCE_TOTAL_WEIGHTS):
        raise RuntimeError(f"unexpected training sources: {counts}")
    weights = np.asarray(
        [SOURCE_TOTAL_WEIGHTS[(record.label, record.source)] / counts[(record.label, record.source)] for record in records],
        dtype=np.float64,
    )
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("training weights do not sum to one")
    return weights


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def calibrate_for_display(
    raw_probabilities: np.ndarray,
    raw_threshold: float,
    display_threshold: float = 0.65,
) -> np.ndarray:
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if (
        not 0 < raw_threshold < 1
        or not 0 < display_threshold < 1
        or np.any(probabilities <= 0)
        or np.any(probabilities >= 1)
    ):
        raise ValueError("display calibration expects open probabilities")
    logits = np.log(probabilities / (1.0 - probabilities))
    offset = math.log(display_threshold / (1.0 - display_threshold)) - math.log(
        raw_threshold / (1.0 - raw_threshold)
    )
    return stable_sigmoid(logits + offset)


def weighted_objective(
    delta: np.ndarray,
    z: np.ndarray,
    base_logits: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> float:
    logits = base_logits + z @ delta
    losses = np.maximum(logits, 0.0) - y * logits + np.log1p(np.exp(-np.abs(logits)))
    return float(weights @ losses + 0.5 * regularization * (delta @ delta))


def fit_proximal_delta(
    z: np.ndarray,
    base_logits: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    delta = np.zeros(z.shape[1], dtype=np.float64)
    objective = weighted_objective(delta, z, base_logits, y, weights, regularization)
    total_reductions = 0
    converged = False
    reason = "maximum-iterations"
    iterations = 0
    last_gradient_norm = math.inf
    last_relative_improvement = math.inf

    for iteration in range(1, 101):
        iterations = iteration
        logits = base_logits + z @ delta
        probabilities = stable_sigmoid(logits)
        residual = weights * (probabilities - y)
        gradient = z.T @ residual + regularization * delta
        last_gradient_norm = float(np.max(np.abs(gradient)))
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = z.T @ (z * curvature[:, None])
        hessian.flat[:: hessian.shape[0] + 1] += regularization
        if last_gradient_norm <= 1e-10:
            converged = True
            reason = "gradient-infinity-norm"
            break
        newton_step = np.linalg.solve(hessian, gradient)
        directional_derivative = float(gradient @ newton_step)
        accepted = False
        step_size = 1.0
        for reduction in range(31):
            proposal = delta - step_size * newton_step
            proposal_objective = weighted_objective(
                proposal, z, base_logits, y, weights, regularization
            )
            if proposal_objective <= objective - 1e-4 * step_size * directional_derivative:
                accepted = True
                total_reductions += reduction
                break
            step_size *= 0.5
        if not accepted:
            raise RuntimeError(f"Newton line search failed for lambda={regularization}")
        last_relative_improvement = (objective - proposal_objective) / max(1.0, abs(objective))
        delta = proposal
        objective = proposal_objective
        if last_relative_improvement <= 1e-12:
            converged = True
            reason = "relative-objective"
            break

    logits = base_logits + z @ delta
    probabilities = stable_sigmoid(logits)
    curvature = weights * probabilities * (1.0 - probabilities)
    hessian = z.T @ (z * curvature[:, None])
    hessian.flat[:: hessian.shape[0] + 1] += regularization
    eigenvalues = np.linalg.eigvalsh(hessian)
    diagnostics = {
        "converged": converged,
        "reason": reason,
        "iterations": iterations,
        "objective": objective,
        "gradientInfinityNorm": last_gradient_norm,
        "relativeObjectiveImprovement": last_relative_improvement,
        "lineSearchReductions": total_reductions,
        "hessianMinimumEigenvalue": float(eigenvalues[0]),
        "hessianMaximumEigenvalue": float(eigenvalues[-1]),
        "hessianConditionNumber": float(eigenvalues[-1] / eigenvalues[0]),
    }
    if not converged:
        raise RuntimeError(f"Newton optimizer did not converge for lambda={regularization}")
    return delta, diagnostics


def equivalent_raw_head(
    official_weight: np.ndarray,
    official_bias: float,
    delta: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, float]:
    adjustment = delta / std
    weight = official_weight + adjustment
    bias = official_bias - float(mean @ adjustment)
    return weight, bias


def metric_report(
    logits: np.ndarray,
    records: list[MatchedRecord],
    threshold: float,
) -> dict[str, Any]:
    probabilities = stable_sigmoid(logits)
    predicted = probabilities >= threshold
    per_transform = []
    for transform in TRANSFORMS:
        indices = np.asarray([index for index, record in enumerate(records) if record.transform == transform])
        rows = [records[index] for index in indices]
        transform_predictions = predicted[indices]
        ai_mask = np.asarray([row.label == "ai" for row in rows])
        real_mask = ~ai_mask
        ai_correct = int(transform_predictions[ai_mask].sum())
        real_correct = int((~transform_predictions[real_mask]).sum())
        ai_total = int(ai_mask.sum())
        real_total = int(real_mask.sum())
        source_metrics = []
        for source in ("nanoBanana", "fluxKrea", "docci"):
            source_indices = np.asarray([index for index, row in enumerate(rows) if row.source == source])
            if len(source_indices) == 0:
                continue
            expected_ai = rows[int(source_indices[0])].label == "ai"
            correct = (
                transform_predictions[source_indices]
                if expected_ai
                else ~transform_predictions[source_indices]
            )
            source_metrics.append(
                {
                    "source": source,
                    "label": "ai" if expected_ai else "real",
                    "correct": int(correct.sum()),
                    "total": int(len(source_indices)),
                    "recall": float(correct.mean()),
                }
            )
        ai_recall = ai_correct / ai_total
        real_recall = real_correct / real_total
        per_transform.append(
            {
                "transform": transform,
                "balancedAccuracy": 0.5 * (ai_recall + real_recall),
                "aiRecall": ai_recall,
                "realRecall": real_recall,
                "aiCorrect": ai_correct,
                "aiTotal": ai_total,
                "realCorrect": real_correct,
                "realTotal": real_total,
                "sources": source_metrics,
            }
        )
    generator_recalls = [
        item["recall"]
        for transform in per_transform
        for item in transform["sources"]
        if item["label"] == "ai"
    ]
    real_recalls = [item["realRecall"] for item in per_transform]
    balanced = [item["balancedAccuracy"] for item in per_transform]
    eligible = all(
        transform["realRecall"] >= 0.90
        and all(item["recall"] >= 0.70 for item in transform["sources"] if item["label"] == "ai")
        for transform in per_transform
    )
    return {
        "threshold": float(threshold),
        "eligible": eligible,
        "minimumBalancedAccuracy": float(min(balanced)),
        "meanBalancedAccuracy": float(np.mean(balanced)),
        "minimumGeneratorRecall": float(min(generator_recalls)),
        "minimumRealRecall": float(min(real_recalls)),
        "perTransform": per_transform,
    }


def attainable_thresholds(logits: np.ndarray) -> np.ndarray:
    scores = np.unique(stable_sigmoid(logits))
    if len(scores) == 0 or not np.isfinite(scores).all() or scores[0] <= 0 or scores[-1] >= 1:
        raise RuntimeError("calibration scores are not finite open probabilities")
    above_maximum = np.nextafter(scores[-1], 1.0)
    if not above_maximum > scores[-1]:
        raise RuntimeError("could not construct the all-real threshold")
    return np.concatenate((scores, np.asarray([above_maximum], dtype=np.float64)))


def report_rank(report: dict[str, Any], regularization_strength: float, candidate_index: int) -> tuple[float, ...]:
    return (
        1.0 if report["eligible"] else 0.0,
        report["minimumBalancedAccuracy"],
        report["meanBalancedAccuracy"],
        report["minimumGeneratorRecall"],
        report["minimumRealRecall"],
        regularization_strength,
        report["threshold"],
        float(-candidate_index),
    )


def choose_threshold(logits: np.ndarray, records: list[MatchedRecord]) -> dict[str, Any]:
    reports = [metric_report(logits, records, float(value)) for value in attainable_thresholds(logits)]
    return max(
        reports,
        key=lambda report: (
            1.0 if report["eligible"] else 0.0,
            report["minimumBalancedAccuracy"],
            report["meanBalancedAccuracy"],
            report["minimumGeneratorRecall"],
            report["minimumRealRecall"],
            report["threshold"],
        ),
    )


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            replacement = numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)
            model.graph.initializer[index].CopyFrom(replacement)
            return
    raise RuntimeError(f"initializer not found: {name}")


def protobuf_sequence_sha256(messages: list[Any]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        encoded = message.SerializeToString(deterministic=True)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def inspect_graph_surgery(source_path: Path, candidate_path: Path) -> dict[str, Any]:
    source = onnx.load(source_path, load_external_data=False)
    candidate = onnx.load(candidate_path, load_external_data=False)
    source_nodes = protobuf_sequence_sha256(list(source.graph.node))
    candidate_nodes = protobuf_sequence_sha256(list(candidate.graph.node))
    source_non_head = [
        item for item in source.graph.initializer if item.name not in {WEIGHT_NAME, BIAS_NAME}
    ]
    candidate_non_head = [
        item for item in candidate.graph.initializer if item.name not in {WEIGHT_NAME, BIAS_NAME}
    ]
    source_non_head_digest = protobuf_sequence_sha256(source_non_head)
    candidate_non_head_digest = protobuf_sequence_sha256(candidate_non_head)
    source_inputs = protobuf_sequence_sha256(list(source.graph.input))
    candidate_inputs = protobuf_sequence_sha256(list(candidate.graph.input))
    source_outputs = protobuf_sequence_sha256(list(source.graph.output))
    candidate_outputs = protobuf_sequence_sha256(list(candidate.graph.output))
    if source_nodes != candidate_nodes:
        raise RuntimeError("FP32 export changed backbone graph nodes")
    if source_non_head_digest != candidate_non_head_digest:
        raise RuntimeError("FP32 export changed a non-head initializer")
    if source_inputs != candidate_inputs or source_outputs != candidate_outputs:
        raise RuntimeError("FP32 export changed model input/output contracts")
    return {
        "nodeDigest": source_nodes,
        "nonHeadInitializerDigest": source_non_head_digest,
        "inputContractDigest": source_inputs,
        "outputContractDigest": source_outputs,
        "onlyHeadInitializersChanged": True,
    }


def save_fp32_candidate(
    source_path: Path,
    destination: Path,
    weight: np.ndarray,
    bias: float,
    protocol_sha256: str,
    candidate_id: str,
    threshold: float,
) -> None:
    model = onnx.load(source_path, load_external_data=False)
    replace_initializer(model, WEIGHT_NAME, weight.reshape(1, 384))
    replace_initializer(model, BIAS_NAME, np.asarray([bias], dtype=np.float32))
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update(
        {
            "realimage.v4.protocol_sha256": protocol_sha256,
            "realimage.v4.candidate": candidate_id,
            "realimage.v4.raw_threshold": format(threshold, ".17g"),
            "realimage.v4.training": "100 originals; 3 repeated matched transforms; frozen backbone; proximal linear head",
        }
    )
    helper.set_model_props(model, dict(sorted(metadata.items())))
    if [item.name for item in model.graph.output] != ["logits"]:
        raise RuntimeError("candidate output contract changed")
    onnx.checker.check_model(model)
    onnx.save_model(model, destination, save_as_external_data=False)


def deterministic_fp32_export(
    source_path: Path,
    destination: Path,
    weight: np.ndarray,
    bias: float,
    protocol_sha256: str,
    candidate_id: str,
    threshold: float,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    first = destination.with_suffix(".first.onnx")
    second = destination.with_suffix(".second.onnx")
    for path in (first, second):
        path.unlink(missing_ok=True)
        save_fp32_candidate(source_path, path, weight, bias, protocol_sha256, candidate_id, threshold)
    first_digest, second_digest = file_sha256(first), file_sha256(second)
    if first_digest != second_digest:
        raise RuntimeError("FP32 graph surgery was not deterministic")
    os.replace(first, destination)
    second.unlink()
    return {
        "path": destination,
        "sha256": first_digest,
        "bytes": destination.stat().st_size,
        "graphSurgery": inspect_graph_surgery(source_path, destination),
    }


def quantize_once(source: Path, destination: Path) -> None:
    quantize_dynamic(
        source,
        destination,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    onnx.checker.check_model(onnx.load(destination, load_external_data=False))


def deterministic_q8_export(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    first = destination.with_suffix(".first.onnx")
    second = destination.with_suffix(".second.onnx")
    for path in (first, second):
        path.unlink(missing_ok=True)
        quantize_once(source, path)
    first_digest, second_digest = file_sha256(first), file_sha256(second)
    if first_digest != second_digest:
        raise RuntimeError("Q8 export was not deterministic")
    os.replace(first, destination)
    second.unlink()
    return {"path": destination, "sha256": first_digest, "bytes": destination.stat().st_size}


def score_full_model(model_path: Path, records: list[MatchedRecord]) -> np.ndarray:
    session = session_for(model_path)
    output: list[np.ndarray] = []
    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start : start + BATCH_SIZE]
        tensors = np.stack([official_tensor(record.path) for record in batch_records]).astype(np.float32)
        logits = session.run(["logits"], {"pixel_values": tensors})[0]
        output.append(np.asarray(logits, dtype=np.float64).reshape(-1))
        print(f"\rvalidate {model_path.name} {start + len(batch_records)}/{len(records)}", end="", flush=True)
    print()
    return np.concatenate(output)


def artifact_entry(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path, project_root),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def write_chrome_calibration_manifests(
    output_root: Path,
    records: list[MatchedRecord],
    project_root: Path,
) -> list[dict[str, Any]]:
    artifacts = []
    for transform in TRANSFORMS:
        selected = [record for record in records if record.transform == transform]
        if len(selected) != 60:
            raise RuntimeError(f"Chrome calibration manifest count changed for {transform}")
        path = output_root / "chrome-gate" / f"{transform}.manifest.json"
        document = {
            "schema": "realimage-clean-head-chrome-calibration-manifest-v4",
            "benchmarkId": f"clean-head-v4-{transform}",
            "purpose": "Installed-Chrome calibration parity only; qualification remains sealed.",
            "rows": [
                {
                    "sampleId": f"{record.sample_id}:{record.transform}",
                    "familyId": record.sample_id,
                    "label": record.label,
                    "domain": record.source,
                    "slice": "v4-clean-head-calibration",
                    "transformId": record.transform,
                    "view": "single-official",
                    "relativePath": relative_path(record.path, project_root),
                    "sha256": record.sha256,
                }
                for record in selected
            ],
        }
        write_json_atomic(path, document)
        artifacts.append(artifact_entry(path, project_root))
    return artifacts


def copy_browser_configs(project_root: Path, bundle_root: Path) -> list[Path]:
    source_root = project_root / MODEL_CONFIG_ROOT_RELATIVE
    copied = []
    for name in ("config.json", "preprocessor_config.json"):
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = bundle_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        copied.append(destination)
    return copied


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = project_root / OUTPUT_ROOT_RELATIVE
    qualification_score_root = project_root / ".bench-data/license-clean-calibration-v3/scores/qualification"
    if qualification_score_root.exists():
        raise RuntimeError("qualification score path exists; refusing to run")
    finalist_path = output_root / "offline-finalists-lock.json"
    if finalist_path.exists():
        raise RuntimeError("offline-finalists-lock.json already exists; refusing to overwrite a frozen decision")

    protocol, tooling = load_and_verify_locks(
        project_root, args.execution_lock, preflight_only=args.preflight_only
    )
    execution = json.loads(args.execution_lock.resolve().read_text(encoding="utf-8"))
    protocol_sha256 = execution["protocol"]["sha256"]
    source_model = project_root / SOURCE_MODEL_RELATIVE
    assert_file(source_model, EXPECTED_SOURCE_SHA256, EXPECTED_SOURCE_BYTES)
    assert_file(
        project_root / SOURCE_MANIFEST_RELATIVE,
        protocol["data"]["sourceManifest"]["sha256"],
    )
    assert_file(
        project_root / MATCHED_MANIFEST_RELATIVE,
        protocol["data"]["matchedInputManifest"]["sha256"],
    )

    matched_root = project_root / MATCHED_ROOT_RELATIVE
    records = scan_allowed_records(matched_root, TRANSFORMS, ROLES)
    verify_records(records)
    _, inventory_sha256 = canonical_inventory(records)
    inventory = protocol["data"]["allowedInventory"]
    if inventory_sha256 != inventory["sha256"] or len(records) != inventory["files"]:
        raise RuntimeError("allowed TRAIN/CALIBRATION inventory changed")
    if sum(record.byte_count for record in records) != inventory["bytes"]:
        raise RuntimeError("allowed TRAIN/CALIBRATION byte count changed")

    preflight_path = output_root / "preflight-lock.json"
    preflight = {
        "schema": "realimage-clean-head-preflight-lock-v4",
        "protocol": {"path": PROTOCOL_RELATIVE.as_posix(), "sha256": protocol_sha256},
        "tooling": {"path": TOOLING_RELATIVE.as_posix(), "sha256": execution["tooling"]["sha256"]},
        "qualificationRead": False,
        "qualificationScorePathAbsent": not qualification_score_root.exists(),
        "sourceModel": artifact_entry(source_model, project_root),
        "sourceManifest": artifact_entry(project_root / SOURCE_MANIFEST_RELATIVE, project_root),
        "matchedInputManifest": artifact_entry(project_root / MATCHED_MANIFEST_RELATIVE, project_root),
        "allowedInventorySha256": inventory_sha256,
        "allowedFiles": len(records),
        "trainRows": sum(record.role == "train" for record in records),
        "calibrationRows": sum(record.role == "calibration" for record in records),
        "trainUniqueSampleIds": len({record.sample_id for record in records if record.role == "train"}),
        "calibrationUniqueSampleIds": len({record.sample_id for record in records if record.role == "calibration"}),
        "trainCalibrationSampleIdOverlap": 0,
        "decodedImages": 0,
        "featureExtractions": 0,
        "modelInferenceCalls": 0,
        "status": "PASS",
    }
    if args.preflight_only:
        if preflight_path.exists():
            raise RuntimeError("preflight-lock.json already exists; refusing to overwrite")
        write_json_atomic(preflight_path, preflight)
        print(canonical_json({"preflight": artifact_entry(preflight_path, project_root), **preflight}))
        return
    if not preflight_path.is_file():
        raise RuntimeError("run --preflight-only and freeze preflight-lock.json before feature extraction")
    frozen_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if frozen_preflight != preflight:
        raise RuntimeError("frozen preflight no longer matches the zero-score preflight")

    work_root = output_root / "work"
    feature_model_path = work_root / "feature-model.onnx"
    feature_model_artifact = build_feature_model(source_model, feature_model_path)
    feature_session = session_for(feature_model_path)
    source_graph = onnx.load(source_model, load_external_data=False)
    official_weight = initializer_array(source_graph, WEIGHT_NAME).reshape(-1)
    official_bias = float(initializer_array(source_graph, BIAS_NAME).reshape(-1)[0])
    if official_weight.shape != (384,):
        raise RuntimeError(f"unexpected official head shape: {official_weight.shape}")

    role_features: dict[str, np.ndarray] = {}
    role_logits: dict[str, np.ndarray] = {}
    role_records: dict[str, list[MatchedRecord]] = {}
    feature_artifacts: list[dict[str, Any]] = []
    for role in ROLES:
        selected_records = [record for record in records if record.role == role]
        features, graph_logits = extract_features(feature_session, selected_records)
        algebraic_logits = features.astype(np.float64) @ official_weight + official_bias
        max_logit_error = float(np.max(np.abs(algebraic_logits - graph_logits.astype(np.float64))))
        if max_logit_error > 2e-5:
            raise RuntimeError(f"feature attachment changed official logits by {max_logit_error}")
        role_features[role] = features.astype(np.float64)
        role_logits[role] = algebraic_logits
        role_records[role] = selected_records
        feature_path = work_root / f"{role}-features.npy"
        row_path = work_root / f"{role}-rows.json"
        write_npy_atomic(feature_path, features)
        write_json_atomic(row_path, record_document(selected_records))
        feature_artifacts.extend(
            [artifact_entry(feature_path, project_root), artifact_entry(row_path, project_root)]
        )

    feature_lock_path = output_root / "feature-lock.json"
    feature_lock = {
        "schema": "realimage-clean-head-feature-lock-v4",
        "protocol": {"path": PROTOCOL_RELATIVE.as_posix(), "sha256": protocol_sha256},
        "tooling": {"path": TOOLING_RELATIVE.as_posix(), "sha256": execution["tooling"]["sha256"]},
        "qualificationRead": False,
        "sourceModel": artifact_entry(source_model, project_root),
        "preflight": artifact_entry(preflight_path, project_root),
        "featureModel": {
            "path": relative_path(feature_model_artifact["path"], project_root),
            "sha256": feature_model_artifact["sha256"],
            "bytes": feature_model_artifact["bytes"],
            "output": FEATURE_OUTPUT,
        },
        "allowedInventorySha256": inventory_sha256,
        "artifacts": feature_artifacts,
    }
    write_json_atomic(feature_lock_path, feature_lock)

    train_x = role_features["train"]
    train_records = role_records["train"]
    train_y = labels_for(train_records)
    weights = loss_weights(train_records)
    feature_mean = weights @ train_x
    centered = train_x - feature_mean
    feature_variance = weights @ (centered * centered)
    feature_std = np.sqrt(np.maximum(feature_variance, 0.0))
    floored_dimensions = int((feature_std < 1e-6).sum())
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    train_z = centered / feature_std
    base_train_logits = role_logits["train"]

    calibration_x = role_features["calibration"]
    calibration_records = role_records["calibration"]
    calibration_z = (calibration_x - feature_mean) / feature_std
    base_calibration_logits = role_logits["calibration"]

    candidates: list[dict[str, Any]] = []
    fitted: dict[str, tuple[np.ndarray, float, np.ndarray]] = {}
    for candidate_index, (candidate_id, regularization) in enumerate(REGULARIZATION_GRID):
        if math.isinf(regularization):
            delta = np.zeros(384, dtype=np.float64)
            diagnostics = {
                "converged": True,
                "reason": "official-zero-delta",
                "iterations": 0,
                "objective": weighted_objective(
                    delta, train_z, base_train_logits, train_y, weights, 0.0
                ),
                "gradientInfinityNorm": None,
                "relativeObjectiveImprovement": None,
                "lineSearchReductions": 0,
                "hessianMinimumEigenvalue": None,
                "hessianMaximumEigenvalue": None,
                "hessianConditionNumber": None,
            }
        else:
            delta, diagnostics = fit_proximal_delta(
                train_z, base_train_logits, train_y, weights, regularization
            )
        head_weight, head_bias = equivalent_raw_head(
            official_weight, official_bias, delta, feature_mean, feature_std
        )
        calibration_logits = base_calibration_logits + calibration_z @ delta
        selected_threshold = choose_threshold(calibration_logits, calibration_records)
        raw_delta = head_weight - official_weight
        regularization_strength = 1e300 if math.isinf(regularization) else regularization
        record = {
            "id": candidate_id,
            "candidateIndex": candidate_index,
            "lambda": "infinity" if math.isinf(regularization) else regularization,
            "eligible": selected_threshold["eligible"],
            "selectedThreshold": selected_threshold,
            "standardizedDeltaL2": float(np.linalg.norm(delta)),
            "rawHeadDeltaL2": float(np.linalg.norm(raw_delta)),
            "officialWeightL2": float(np.linalg.norm(official_weight)),
            "relativeRawHeadDeltaL2": float(
                np.linalg.norm(raw_delta) / max(np.linalg.norm(official_weight), np.finfo(np.float64).tiny)
            ),
            "maximumAbsoluteRawWeightDelta": float(np.max(np.abs(raw_delta))),
            "optimization": diagnostics,
        }
        record["selectionRank"] = list(
            report_rank(selected_threshold, regularization_strength, candidate_index)
        )
        candidates.append(record)
        fitted[candidate_id] = (head_weight, head_bias, calibration_logits)

    eligible_candidates = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(eligible_candidates, key=lambda candidate: tuple(candidate["selectionRank"])) if eligible_candidates else None
    selection_report_path = output_root / "selection-report.json"
    selection_report = {
        "schema": "realimage-clean-head-selection-report-v4",
        "protocol": {"path": PROTOCOL_RELATIVE.as_posix(), "sha256": protocol_sha256},
        "featureLock": artifact_entry(feature_lock_path, project_root),
        "qualificationRead": False,
        "sampleUnit": "100 TRAIN originals and 60 CALIBRATION originals; three transforms are repeated measures",
        "featureStandardization": {
            "trainingOnly": True,
            "flooredDimensions": floored_dimensions,
            "meanL2": float(np.linalg.norm(feature_mean)),
            "stdMinimum": float(feature_std.min()),
            "stdMaximum": float(feature_std.max()),
        },
        "maximumStandardizedDeltaL2AcrossGrid": float(max(item["standardizedDeltaL2"] for item in candidates)),
        "maximumRawHeadDeltaL2AcrossGrid": float(max(item["rawHeadDeltaL2"] for item in candidates)),
        "candidates": candidates,
        "selectedCandidate": selected["id"] if selected else None,
    }
    write_json_atomic(selection_report_path, selection_report)

    models: list[dict[str, Any]] = []
    precision: dict[str, Any] | None = None
    score_reference_artifact: dict[str, Any] | None = None
    chrome_manifests = write_chrome_calibration_manifests(
        output_root, calibration_records, project_root
    )
    if selected:
        selected_id = selected["id"]
        selected_weight, selected_bias, selected_calibration_logits = fitted[selected_id]
        threshold = selected["selectedThreshold"]["threshold"]
        fp32_bundle = output_root / "browser-models/v4-fp32"
        q8_bundle = output_root / "browser-models/v4-q8"
        fp32_path = fp32_bundle / "onnx/model.onnx"
        q8_path = q8_bundle / "onnx/model_quantized.onnx"
        fp32 = deterministic_fp32_export(
            source_model,
            fp32_path,
            selected_weight,
            selected_bias,
            protocol_sha256,
            selected_id,
            threshold,
        )
        q8 = deterministic_q8_export(fp32_path, q8_path)
        fp32_configs = copy_browser_configs(project_root, fp32_bundle)
        q8_configs = copy_browser_configs(project_root, q8_bundle)
        fp32_logits = score_full_model(fp32_path, calibration_records)
        formula_logit_error = float(np.max(np.abs(fp32_logits - selected_calibration_logits)))
        fp32_probabilities = stable_sigmoid(fp32_logits)
        formula_probabilities = stable_sigmoid(selected_calibration_logits)
        formula_decision_disagreements = int(
            np.sum((fp32_probabilities >= threshold) != (formula_probabilities >= threshold))
        )
        fp32_metrics = metric_report(fp32_logits, calibration_records, threshold)
        fp32_eligible = (
            formula_logit_error <= 2e-4
            and formula_decision_disagreements == 0
            and fp32_metrics["eligible"]
        )
        q8_logits = score_full_model(q8_path, calibration_records)
        q8_probabilities = stable_sigmoid(q8_logits)
        decision_disagreements = int(
            np.sum((fp32_probabilities >= threshold) != (q8_probabilities >= threshold))
        )
        maximum_probability_error = float(np.max(np.abs(fp32_probabilities - q8_probabilities)))
        q8_metrics = metric_report(q8_logits, calibration_records, threshold)
        q8_eligible = (
            fp32_eligible
            and decision_disagreements == 0
            and maximum_probability_error <= 0.02
            and q8_metrics["eligible"]
        )
        fp32_display = calibrate_for_display(fp32_probabilities, threshold)
        q8_display = calibrate_for_display(q8_probabilities, threshold)
        score_reference_path = output_root / "chrome-gate/offline-calibration-reference.json"
        score_reference = {
            "schema": "realimage-clean-head-offline-calibration-reference-v4",
            "protocolSha256": protocol_sha256,
            "qualificationRead": False,
            "rawThreshold": threshold,
            "displayThreshold": 0.65,
            "rows": [
                {
                    "sampleId": f"{record.sample_id}:{record.transform}",
                    "familyId": record.sample_id,
                    "transform": record.transform,
                    "label": record.label,
                    "source": record.source,
                    "fp32RawScore": float(fp32_probabilities[index]),
                    "fp32DisplayScore": float(fp32_display[index]),
                    "fp32Predicted": "ai" if fp32_probabilities[index] >= threshold else "real",
                    "q8RawScore": float(q8_probabilities[index]),
                    "q8DisplayScore": float(q8_display[index]),
                    "q8Predicted": "ai" if q8_probabilities[index] >= threshold else "real",
                }
                for index, record in enumerate(calibration_records)
            ],
        }
        write_json_atomic(score_reference_path, score_reference)
        score_reference_artifact = artifact_entry(score_reference_path, project_root)
        models = [
            {
                "id": "fp32-reference",
                "path": relative_path(fp32["path"], project_root),
                "sha256": fp32["sha256"],
                "bytes": fp32["bytes"],
                "eligible": fp32_eligible,
                "graphSurgery": fp32["graphSurgery"],
                "bundleFiles": [artifact_entry(path, project_root) for path in fp32_configs],
            },
            {
                "id": "matmul-dynamic-q8",
                "path": relative_path(q8["path"], project_root),
                "sha256": q8["sha256"],
                "bytes": q8["bytes"],
                "eligible": q8_eligible,
                "bundleFiles": [artifact_entry(path, project_root) for path in q8_configs],
            },
        ]
        precision = {
            "rawThreshold": threshold,
            "fp32FormulaMaximumAbsoluteLogitError": formula_logit_error,
            "fp32FormulaDecisionDisagreements": formula_decision_disagreements,
            "fp32Metrics": fp32_metrics,
            "fp32Eligible": fp32_eligible,
            "q8MaximumAbsoluteProbabilityError": maximum_probability_error,
            "q8DecisionDisagreements": decision_disagreements,
            "q8Metrics": q8_metrics,
            "q8Eligible": q8_eligible,
            "preferredRoute": "matmul-dynamic-q8" if q8_eligible else "fp32-reference",
        }

    offline_eligible = selected is not None and any(model["eligible"] for model in models)
    finalist_lock = {
        "schema": "realimage-clean-head-offline-finalists-lock-v4",
        "frozenBeforeQualification": True,
        "qualificationRead": False,
        "protocol": {"path": PROTOCOL_RELATIVE.as_posix(), "sha256": protocol_sha256},
        "tooling": {"path": TOOLING_RELATIVE.as_posix(), "sha256": execution["tooling"]["sha256"]},
        "executionLock": artifact_entry(args.execution_lock.resolve(), project_root),
        "sourceModel": artifact_entry(source_model, project_root),
        "featureLock": artifact_entry(feature_lock_path, project_root),
        "selectionReport": artifact_entry(selection_report_path, project_root),
        "selectedHead": None
        if selected is None
        else {
            "id": selected["id"],
            "lambda": selected["lambda"],
            "rawThreshold": selected["selectedThreshold"]["threshold"],
            "calibration": selected["selectedThreshold"],
            "standardizedDeltaL2": selected["standardizedDeltaL2"],
            "rawHeadDeltaL2": selected["rawHeadDeltaL2"],
            "relativeRawHeadDeltaL2": selected["relativeRawHeadDeltaL2"],
            "maximumAbsoluteRawWeightDelta": selected["maximumAbsoluteRawWeightDelta"],
            "optimization": selected["optimization"],
        },
        "precision": precision,
        "models": models,
        "chromeCalibrationManifests": chrome_manifests,
        "offlineCalibrationReference": score_reference_artifact,
        "offlineCandidateEligible": offline_eligible,
        "installedChromeGate": {
            "required": offline_eligible,
            "observations": 180,
            "manifests": chrome_manifests,
            "route": precision["preferredRoute"] if precision else None,
            "requirements": {
                "terminalSuccesses": 180,
                "errorsOrTimeouts": 0,
                "offlineChromeDecisionDisagreements": 0,
                "maximumAbsoluteSigmoidScoreError": 0.02,
                "independentPerTransformConstraints": "DOCCI >=27/30, Nano Banana >=11/15, FLUX >=11/15 on each transform",
                "packagedRoute": "the exact candidate bytes, Transformers.js processor configuration, ORT-WASM asset, threshold mapping, and extension build later frozen in the joint lock",
            },
            "status": "pending" if offline_eligible else "not-applicable",
        },
        "qualificationAuthorized": False,
        "nextStep": "Run the frozen installed-Chrome calibration parity gate, then freeze a joint one-shot finalist lock before opening qualification."
        if offline_eligible
        else "Hard NO-GO. Keep qualification sealed.",
        "interpretation": "Small, same-source development gate only; never an accuracy or broad-generalization claim.",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
        },
    }
    write_json_atomic(finalist_path, finalist_lock)
    print(canonical_json({
        "selectedHead": finalist_lock["selectedHead"],
        "precision": precision,
        "offlineCandidateEligible": finalist_lock["offlineCandidateEligible"],
        "qualificationAuthorized": False,
        "offlineFinalistsLock": artifact_entry(finalist_path, project_root),
    }))


if __name__ == "__main__":
    main()
