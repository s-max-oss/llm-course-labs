from __future__ import annotations

import sys
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_level_b_replay import (  # noqa: E402
    apply_transform,
    rotate,
    validate_input_manifest,
    validate_manifest_and_predictions,
)
from extract_level_b_bag import nearest_by_stamp  # noqa: E402
from run_level_b_rfdetr import validate_protocol_bindings  # noqa: E402
from validate_level_b_run import validate_run  # noqa: E402


class LevelBToolTests(unittest.TestCase):
    def load_v2_inputs(self):
        root = PROJECT_ROOT / "experiments" / "level_b_v2"
        manifest_path = root / "inputs" / "input_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = json.loads(
            (PROJECT_ROOT / "configs" / "level_b_replay_protocol_v2.json").read_text(encoding="utf-8")
        )
        predictions = [
            json.loads(line)
            for line in (root / "outputs" / "rfdetr_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return manifest_path.parent, manifest, predictions, protocol

    def test_nearest_stamp_uses_both_sides(self):
        sequence = [(10, "a"), (20, "b"), (30, "c")]
        self.assertEqual(nearest_by_stamp(sequence, 24), (20, "b", 4))
        self.assertEqual(nearest_by_stamp(sequence, 28), (30, "c", 2))

    def test_quaternion_identity_rotation(self):
        vector = np.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(rotate(np.asarray([0.0, 0.0, 0.0, 1.0]), vector), vector)

    def test_tf_translation_is_applied_in_parent_frame(self):
        result = apply_transform(
            np.asarray([1.0, 2.0, 3.0]),
            {
                "translation": [0.5, -1.0, 2.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )
        np.testing.assert_allclose(result, [1.5, 1.0, 5.0])

    def test_accepted_input_manifest_separates_target_truth_from_pose_truth(self):
        manifest_text = (
            PROJECT_ROOT / "experiments" / "level_b_v1" / "inputs" / "input_manifest.json"
        ).read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        self.assertNotIn('"danger_truth"', manifest_text)
        self.assertFalse(manifest["source_bag"]["ground_truth_topics_present"])
        self.assertIn("/Odometry_gazebo", manifest["source_bag"]["topics"])
        run = json.loads(
            (PROJECT_ROOT / "experiments" / "level_b_v1" / "level_b_run.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(run["input"]["danger_target_truth_available_to_inference"])
        self.assertTrue(run["input"]["pose_ground_truth_available_to_replay"])
        self.assertEqual(run["input"]["pose_ground_truth_topic"], "/Odometry_gazebo")
        self.assertEqual(
            run["gates"]["G2_truth_isolation"],
            "failed_pose_ground_truth_present_reclassified_debug_only",
        )

    def test_v2_protocol_explicitly_records_pose_ground_truth(self):
        _, manifest, _, protocol = self.load_v2_inputs()
        self.assertTrue(protocol["acquisition"]["runtime"]["referee_odometry"])
        self.assertIn("/Odometry_gazebo", protocol["acquisition"]["required_topics"])
        self.assertIn("/Odometry_gazebo", manifest["source_bag"]["topics"])

    def test_v2_protocol_binding_is_accepted(self):
        root, manifest, _, protocol = self.load_v2_inputs()
        validate_protocol_bindings(protocol, manifest, root)

    def test_protocol_binding_rejects_cross_version_manifest(self):
        root, manifest, _, protocol = self.load_v2_inputs()
        forged = copy.deepcopy(manifest)
        forged["protocol_id"] = "level_b_replay_v1"
        with self.assertRaisesRegex(ValueError, "manifest/protocol mismatch"):
            validate_protocol_bindings(protocol, forged, root)

    def test_sealed_v2_run_validates(self):
        run_path = PROJECT_ROOT / "experiments" / "level_b_v2" / "level_b_run.json"
        self.assertEqual(validate_run(run_path), [])

    def test_sealed_v2_run_rejects_artifact_hash_tampering(self):
        source = PROJECT_ROOT / "experiments" / "level_b_v2" / "level_b_run.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["inference"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "level_b_run.json"
            forged.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn(
                "artifact hash mismatch: experiments/level_b_v2/outputs/rfdetr_predictions.jsonl",
                validate_run(forged),
            )

    def test_sealed_v2_run_rejects_claim_overreach(self):
        source = PROJECT_ROOT / "experiments" / "level_b_v2" / "level_b_run.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["claim_boundary"]["formal_benchmark"] = True
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "level_b_run.json"
            forged.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn(
                "claim boundary is missing or overstates the development replay",
                validate_run(forged),
            )

    def test_sealed_v2_run_rejects_unknown_claim(self):
        source = PROJECT_ROOT / "experiments" / "level_b_v2" / "level_b_run.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["claim_boundary"]["complete_building_search"] = True
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "level_b_run.json"
            forged.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIn(
                "claim boundary is missing or overstates the development replay",
                validate_run(forged),
            )

    def test_sealed_v2_run_rejects_false_pose_truth_isolation(self):
        source = PROJECT_ROOT / "experiments" / "level_b_v2" / "level_b_run.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["input"]["pose_ground_truth_available_to_replay"] = False
        payload["gates"]["G2_truth_isolation"] = "passed"
        with tempfile.TemporaryDirectory() as directory:
            forged = Path(directory) / "level_b_run.json"
            forged.write_text(json.dumps(payload), encoding="utf-8")
            errors = validate_run(forged)
            self.assertIn(
                "Level-B gates do not preserve the failed truth-isolation finding",
                errors,
            )
            self.assertIn(
                "run does not declare both target-truth and pose-truth boundaries",
                errors,
            )

    def test_runner_rejects_recursive_truth_before_model_loading(self):
        root, manifest, _, protocol = self.load_v2_inputs()
        forged = copy.deepcopy(manifest)
        forged["records"][0]["camera_info"]["danger_truth"] = [7.7, 22.55, 0.17]
        with self.assertRaisesRegex(ValueError, "forbidden inference field"):
            validate_protocol_bindings(protocol, forged, root)

    def test_runner_rejects_rgb_path_escape_before_model_loading(self):
        root, manifest, _, protocol = self.load_v2_inputs()
        forged = copy.deepcopy(manifest)
        forged["records"][0]["rgb"]["path"] = "../../level_b_run.json"
        with self.assertRaisesRegex(ValueError, "escapes manifest root"):
            validate_protocol_bindings(protocol, forged, root)

    def test_v2_manifest_rejects_recursive_truth_field(self):
        root, manifest, predictions, protocol = self.load_v2_inputs()
        forged = copy.deepcopy(manifest)
        forged["records"][0]["camera_info"]["danger_truth"] = [7.7, 22.55, 0.17]
        with self.assertRaisesRegex(ValueError, "forbidden inference field"):
            validate_manifest_and_predictions(forged, predictions, protocol, root)

    def test_v2_predictions_reject_model_forgery(self):
        root, manifest, predictions, protocol = self.load_v2_inputs()
        forged = copy.deepcopy(predictions)
        forged[0]["source_model_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model binding mismatch"):
            validate_manifest_and_predictions(manifest, forged, protocol, root)

    def test_v2_predictions_reject_class_mapping_forgery(self):
        root, manifest, predictions, protocol = self.load_v2_inputs()
        forged = copy.deepcopy(predictions)
        forged[0]["predictions"][0]["canonical_label"] = "danger_red_sphere"
        with self.assertRaisesRegex(ValueError, "class mapping mismatch"):
            validate_manifest_and_predictions(manifest, forged, protocol, root)

    def test_v2_predictions_reject_invalid_latency_and_area(self):
        root, manifest, predictions, protocol = self.load_v2_inputs()
        forged_latency = copy.deepcopy(predictions)
        forged_latency[0]["latency_ms"] = -1
        with self.assertRaisesRegex(ValueError, "invalid prediction latency"):
            validate_manifest_and_predictions(manifest, forged_latency, protocol, root)
        forged_area = copy.deepcopy(predictions)
        forged_area[0]["predictions"][0]["bbox_area_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "bbox area fraction mismatch"):
            validate_manifest_and_predictions(manifest, forged_area, protocol, root)

    def test_v2_manifest_rejects_depth_hash_tampering(self):
        root, manifest, predictions, protocol = self.load_v2_inputs()
        forged = copy.deepcopy(manifest)
        forged["records"][0]["depth"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "depth frame hash mismatch"):
            validate_manifest_and_predictions(forged, predictions, protocol, root)


if __name__ == "__main__":
    unittest.main()
