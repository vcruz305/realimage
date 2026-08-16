"""Fit a SIMPLE L2-regularized logistic-regression (linear) head on the same
frozen Community Forensics features used by train-community-head.py, as a
baseline to compare against the MLP head.

Reuses:
  - the numerically-stable proximal-Newton/IRLS logistic fit
    (weighted_objective / fit_proximal_delta / stable_sigmoid) adapted from
    scripts/run_v4_clean_head.py (not imported -- that script is hard-locked
    to the old narrow protocol and refuses to run; the fitting math is
    self-contained and copied here instead).
  - samples() / metrics() / choose_threshold() / export_model() from
    scripts/train-community-head.py, loaded directly from that file so the
    threshold-selection and eligibility gates are identical between the two
    head types (an apples-to-apples comparison on the same calibration and
    development splits).

Expects --cache to already contain train.npz / calibration.npz /
development.npz written by a prior train-community-head.py run (features are
not re-extracted here).
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_train_community_head_module():
    path = Path(__file__).parent / "train-community-head.py"
    spec = importlib.util.spec_from_file_location("train_community_head", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def weighted_objective(delta, z, base_logits, y, weights, regularization):
    logits = base_logits + z @ delta
    losses = np.maximum(logits, 0.0) - y * logits + np.log1p(np.exp(-np.abs(logits)))
    return float(weights @ losses + 0.5 * regularization * (delta @ delta))


def fit_proximal_delta(z, base_logits, y, weights, regularization):
    """Numerically-stable proximal-Newton (IRLS) fit of an L2-regularized
    logistic delta on top of base_logits, adapted from
    scripts/run_v4_clean_head.py's fit_proximal_delta."""
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    delta = np.zeros(z.shape[1], dtype=np.float64)
    objective = weighted_objective(delta, z, base_logits, y, weights, regularization)
    converged = False
    reason = "maximum-iterations"
    iterations = 0

    for iteration in range(1, 101):
        iterations = iteration
        logits = base_logits + z @ delta
        probabilities = stable_sigmoid(logits)
        residual = weights * (probabilities - y)
        gradient = z.T @ residual + regularization * delta
        gradient_norm = float(np.max(np.abs(gradient)))
        curvature = weights * probabilities * (1.0 - probabilities)
        hessian = z.T @ (z * curvature[:, None])
        hessian.flat[:: hessian.shape[0] + 1] += regularization
        if gradient_norm <= 1e-10:
            converged = True
            reason = "gradient-infinity-norm"
            break
        newton_step = np.linalg.solve(hessian, gradient)
        directional_derivative = float(gradient @ newton_step)
        accepted = False
        step_size = 1.0
        for _ in range(31):
            proposal = delta - step_size * newton_step
            proposal_objective = weighted_objective(proposal, z, base_logits, y, weights, regularization)
            if proposal_objective <= objective - 1e-4 * step_size * directional_derivative:
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            raise RuntimeError(f"Newton line search failed for lambda={regularization}")
        relative_improvement = (objective - proposal_objective) / max(1.0, abs(objective))
        delta = proposal
        objective = proposal_objective
        if relative_improvement <= 1e-12:
            converged = True
            reason = "relative-objective"
            break

    return delta, {"converged": converged, "reason": reason, "iterations": iterations, "objective": objective}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True, help="Feature cache dir with train/calibration/development .npz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=323)
    return parser.parse_args()


def main():
    args = parse_args()
    tch = load_train_community_head_module()

    train = np.load(args.cache / "train.npz", allow_pickle=False)
    train_x, train_y, train_domains = train["x"].astype(np.float64), train["y"].astype(np.float64), train["domains"]
    calibration = np.load(args.cache / "calibration.npz", allow_pickle=False)
    cal_x, cal_y, cal_domains = calibration["x"].astype(np.float64), calibration["y"].astype(np.float64), calibration["domains"]
    development = np.load(args.cache / "development.npz", allow_pickle=False)
    dev_x, dev_y, dev_domains = development["x"].astype(np.float64), development["y"].astype(np.float64), development["domains"]

    weights = np.full(len(train_x), 1.0 / len(train_x), dtype=np.float64)
    feature_mean = weights @ train_x
    centered = train_x - feature_mean
    feature_variance = weights @ (centered * centered)
    feature_std = np.sqrt(np.maximum(feature_variance, 0.0))
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    train_z = centered / feature_std
    cal_z = (cal_x - feature_mean) / feature_std
    dev_z = (dev_x - feature_mean) / feature_std
    base_train_logits = np.zeros(len(train_x), dtype=np.float64)

    major_generators = ["adm", "biggan", "flux-reason", "glide", "sd15", "synthetic-characters", "vqdm", "wukong"]
    results = []
    best = None
    for regularization in (300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3):
        delta, diagnostics = fit_proximal_delta(train_z, base_train_logits, train_y, weights, regularization)
        weight = delta / feature_std
        bias = -float(feature_mean @ (delta / feature_std))

        cal_logits = cal_x @ weight + bias
        calibration_metrics = tch.choose_threshold(cal_logits, cal_y, cal_domains)
        dev_logits = dev_x @ weight + bias
        development_metrics = tch.metrics(dev_logits, dev_y, dev_domains, calibration_metrics["threshold"])

        objective = min(calibration_metrics["balancedAccuracy"], development_metrics["balancedAccuracy"])
        named = {item["domain"]: item["recall"] for item in development_metrics["domains"]}
        eligible = (
            calibration_metrics["realRecall"] >= 0.9
            and development_metrics["realRecall"] >= 0.9
            and all(named.get(name, 1.0) >= 0.75 for name in major_generators)
        )
        if not eligible:
            objective -= 1

        record = {
            "regularization": regularization,
            "objective": objective,
            "eligible": eligible,
            "diagnostics": diagnostics,
            "calibration": calibration_metrics,
            "development": development_metrics,
            "weightL2": float(np.linalg.norm(weight)),
            "bias": bias,
        }
        results.append(record)
        rank = (objective, calibration_metrics["minimumDomainRecall"], calibration_metrics["balancedAccuracy"])
        if best is None or rank > best[0]:
            best = (rank, weight, bias, record)

    _, weight, bias, selected = best

    head = nn.Linear(384, 1)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(weight.reshape(1, -1)).float())
        head.bias.copy_(torch.from_numpy(np.asarray([bias])).float())

    fp32, quantized = tch.export_model(args.checkpoint, head, args.output)
    source_model_dir = args.output  # placeholder; caller copies config separately
    report = {
        "headType": "logistic-regression",
        "seed": args.seed,
        "trainingFeatureRows": len(train_x),
        "calibrationImages": len(cal_x),
        "developmentImages": len(dev_x),
        "selected": selected,
        "candidates": results,
        "modelSha256": tch.sha256(quantized),
        "modelBytes": quantized.stat().st_size,
        "fp32Sha256": tch.sha256(fp32),
        "fp32Bytes": fp32.stat().st_size,
    }
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(selected["calibration"], indent=2, default=float))
    print(json.dumps(selected["development"], indent=2, default=float))
    print(f"Model SHA-256: {report['modelSha256']}")


if __name__ == "__main__":
    main()
