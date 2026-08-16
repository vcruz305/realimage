from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from PIL import Image
from onnx import TensorProto, helper, numpy_helper

import run_v4_clean_head as challenger
from v4_clean_head_preprocess import official_rgb_view, official_tensor


class PreprocessTests(unittest.TestCase):
    def test_registered_square_view_has_frozen_shape_and_tensor_layout(self) -> None:
        pixels = np.zeros((768, 768, 3), dtype=np.uint8)
        pixels[:, :, 0] = 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            Image.fromarray(pixels).save(path, quality=95)
            tensor = official_tensor(path)
        self.assertEqual(tensor.shape, (3, 384, 384))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertGreater(float(tensor[0].mean()), float(tensor[1].mean()))

    def test_non_square_input_is_rejected(self) -> None:
        image = Image.new("RGB", (768, 512))
        with self.assertRaisesRegex(ValueError, "square matched inputs"):
            official_rgb_view(image)


class OptimizationTests(unittest.TestCase):
    def test_proximal_newton_reduces_convex_objective_deterministically(self) -> None:
        z = np.asarray(
            [
                [-2.0, 0.1],
                [-1.0, -0.2],
                [1.0, 0.2],
                [2.0, -0.1],
            ],
            dtype=np.float64,
        )
        base = np.zeros(4, dtype=np.float64)
        y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        weights = np.full(4, 0.25, dtype=np.float64)
        initial = challenger.weighted_objective(np.zeros(2), z, base, y, weights, 1.0)
        first, diagnostics = challenger.fit_proximal_delta(z, base, y, weights, 1.0)
        second, repeated = challenger.fit_proximal_delta(z, base, y, weights, 1.0)
        self.assertTrue(diagnostics["converged"])
        self.assertLess(diagnostics["objective"], initial)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(diagnostics, repeated)

    def test_raw_head_is_algebraically_equivalent(self) -> None:
        rng = np.random.default_rng(323)
        x = rng.normal(size=(10, 3))
        mean = rng.normal(size=3)
        std = np.asarray([0.5, 1.5, 2.0])
        official_weight = rng.normal(size=3)
        official_bias = 0.25
        delta = rng.normal(size=3)
        weight, bias = challenger.equivalent_raw_head(
            official_weight, official_bias, delta, mean, std
        )
        expected = x @ official_weight + official_bias + ((x - mean) / std) @ delta
        np.testing.assert_allclose(x @ weight + bias, expected, rtol=1e-12, atol=1e-12)


class ExportTests(unittest.TestCase):
    def test_graph_surgery_changes_only_the_registered_head_initializers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.onnx"
            candidate_path = root / "candidate.onnx"
            input_value = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 384])
            output_value = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])
            weight = numpy_helper.from_array(np.zeros((1, 384), dtype=np.float32), challenger.WEIGHT_NAME)
            bias = numpy_helper.from_array(np.zeros(1, dtype=np.float32), challenger.BIAS_NAME)
            node = helper.make_node(
                "Gemm",
                ["pixel_values", challenger.WEIGHT_NAME, challenger.BIAS_NAME],
                ["logits"],
                name="/vit/head/Gemm",
                transB=1,
            )
            graph = helper.make_graph([node], "fixture", [input_value], [output_value], [weight, bias])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
            onnx.save_model(model, source_path)
            challenger.save_fp32_candidate(
                source_path,
                candidate_path,
                np.ones(384, dtype=np.float64),
                -0.25,
                "a" * 64,
                "fixture",
                0.5,
            )
            audit = challenger.inspect_graph_surgery(source_path, candidate_path)
            self.assertTrue(audit["onlyHeadInitializersChanged"])
            candidate = onnx.load(candidate_path)
            np.testing.assert_array_equal(
                challenger.initializer_array(candidate, challenger.WEIGHT_NAME),
                np.ones((1, 384), dtype=np.float64),
            )


class SelectionTests(unittest.TestCase):
    def make_records(self) -> list:
        records = []
        for transform in challenger.TRANSFORMS:
            for source, label, count in (
                ("nanoBanana", "ai", 15),
                ("fluxKrea", "ai", 15),
                ("docci", "real", 30),
            ):
                for index in range(count):
                    records.append(
                        type(
                            "Record",
                            (),
                            {"transform": transform, "source": source, "label": label},
                        )()
                    )
        return records

    def test_threshold_selector_enforces_every_transform_source_gate(self) -> None:
        records = self.make_records()
        logits = np.asarray([3.0 if record.label == "ai" else -3.0 for record in records])
        report = challenger.choose_threshold(logits, records)
        self.assertTrue(report["eligible"])
        self.assertEqual(report["minimumBalancedAccuracy"], 1.0)
        self.assertEqual(report["minimumGeneratorRecall"], 1.0)
        self.assertEqual(report["minimumRealRecall"], 1.0)

    def test_attainable_thresholds_include_all_real_partition(self) -> None:
        thresholds = challenger.attainable_thresholds(np.asarray([-1.0, 0.0, 1.0]))
        self.assertEqual(len(thresholds), 4)
        self.assertGreater(thresholds[-1], thresholds[-2])

    def test_display_mapping_preserves_the_frozen_decision_boundary(self) -> None:
        raw_threshold = 0.123456789
        raw = np.asarray([raw_threshold - 1e-8, raw_threshold, raw_threshold + 1e-8])
        displayed = challenger.calibrate_for_display(raw, raw_threshold)
        self.assertLess(displayed[0], 0.65)
        self.assertAlmostEqual(displayed[1], 0.65, places=14)
        self.assertGreater(displayed[2], 0.65)


if __name__ == "__main__":
    unittest.main()
