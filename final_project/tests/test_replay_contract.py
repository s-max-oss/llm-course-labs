from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_replay.contracts import (  # noqa: E402
    ReplayContractError,
    load_replay_contract,
    validate_detection_frame,
    validate_replay_contract,
)


CONTRACT_PATH = PROJECT_ROOT / "configs" / "replay_event_contract_v1.json"


class ReplayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_replay_contract(CONTRACT_PATH)

    def valid_frame(self):
        return {
            "schema_version": 1,
            "contract_id": "robot_replay_detection_frame_v1",
            "record_id": "replay::seed101_001_R0309",
            "sample_id": "seed101_001_R0309",
            "sequence_index": 0,
            "sequence_semantics": "seed_grouped_storyboard_non_temporal",
            "stamp": None,
            "stamp_source": "unavailable_legacy_rgb",
            "frame_id": "replay_camera",
            "frame_id_source": "offline_namespace_not_ros_tf",
            "input_sha256": "f" * 64,
            "inference_record_sha256": "e" * 64,
            "source_model": "RF-DETR-Seg Medium",
            "source_model_sha256": "ec2de357f402901cf624835f11d5c54bb0eab8980d1268d7ee8a9122a446fa50",
            "inference_run_id": "rfdetr_pilot_cpu_full_v1",
            "processing_ms": 653.051,
            "detections": [
                {
                    "prediction_index": 0,
                    "track_id": None,
                    "raw_class_name": "Red_ball",
                    "class_name": "danger_red_sphere",
                    "confidence": 0.9,
                    "bbox_xyxy": [10.0, 20.0, 40.0, 60.0],
                    "temporal_hits": 0,
                    "temporal_confirmed": False,
                    "localized": False,
                    "position": None,
                    "position_frame": None,
                    "quality_accepted": False,
                    "quality_rejection_reasons": ["level_a_missing_rgbd_pose"],
                }
            ],
        }

    def test_frozen_contract_and_valid_frame(self):
        self.assertEqual(self.contract["replay_level"], "A")
        self.assertFalse(
            self.contract["g0_decision"]["world_coordinate_output_allowed"]
        )
        validate_detection_frame(self.valid_frame(), self.contract)

    def test_frozen_contract_rejects_boundary_weakening(self):
        mutations = []
        source = copy.deepcopy(self.contract)
        source["authoritative_robot_source"]["commit"] = "0" * 40
        mutations.append(source)
        interface = copy.deepcopy(self.contract)
        interface["interfaces"]["detection_output_equivalent"]["topic"] = "/other"
        mutations.append(interface)
        fields = copy.deepcopy(self.contract)
        fields["detection_required_fields"].remove("position_frame")
        mutations.append(fields)
        forbidden = copy.deepcopy(self.contract)
        forbidden["forbidden_runtime_fields"].remove("ground_truth")
        mutations.append(forbidden)
        gate = copy.deepcopy(self.contract)
        gate["g0_decision"]["missing_resources"] = []
        mutations.append(gate)
        for contract in mutations:
            with self.assertRaises(ReplayContractError):
                validate_replay_contract(contract)

    def test_empty_detection_array_is_valid_heartbeat(self):
        frame = self.valid_frame()
        frame["detections"] = []
        validate_detection_frame(frame, self.contract)

    def test_truth_fields_fail_closed_at_any_depth(self):
        frame = self.valid_frame()
        frame["ground_truth"] = {"target_label": "danger_red_sphere"}
        with self.assertRaises(ReplayContractError):
            validate_detection_frame(frame, self.contract)

        frame = self.valid_frame()
        frame["detections"][0]["scene_seed"] = 101
        with self.assertRaises(ReplayContractError):
            validate_detection_frame(frame, self.contract)

    def test_level_a_rejects_time_tracking_and_position_claims(self):
        corruptions = (
            ("stamp", 123.45),
            ("frame_id", "world"),
        )
        for key, value in corruptions:
            with self.subTest(key=key):
                frame = self.valid_frame()
                frame[key] = value
                with self.assertRaises(ReplayContractError):
                    validate_detection_frame(frame, self.contract)

        for key, value in (
            ("track_id", 1),
            ("temporal_hits", 1),
            ("temporal_confirmed", True),
            ("localized", True),
            ("position", [1.0, 2.0, 3.0]),
            ("position_frame", "world"),
            ("quality_accepted", True),
        ):
            with self.subTest(key=key):
                frame = self.valid_frame()
                frame["detections"][0][key] = value
                with self.assertRaises(ReplayContractError):
                    validate_detection_frame(frame, self.contract)

    def test_alias_bbox_hash_and_order_are_strict(self):
        mutations = []
        alias = self.valid_frame()
        alias["detections"][0]["class_name"] = "distractor_red_box"
        mutations.append(alias)
        bbox = self.valid_frame()
        bbox["detections"][0]["bbox_xyxy"] = [10.0, 20.0, 9.0, 60.0]
        mutations.append(bbox)
        digest = self.valid_frame()
        digest["input_sha256"] = "not-a-hash"
        mutations.append(digest)
        for frame in mutations:
            with self.assertRaises(ReplayContractError):
                validate_detection_frame(frame, self.contract)

        frame = self.valid_frame()
        second = copy.deepcopy(frame["detections"][0])
        second["prediction_index"] = 1
        second["confidence"] = 0.95
        frame["detections"].append(second)
        with self.assertRaises(ReplayContractError):
            validate_detection_frame(frame, self.contract)
