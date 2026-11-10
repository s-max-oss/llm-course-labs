from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from robot_replay.contracts import ReplayContractError  # noqa: E402
from robot_replay.replay import (  # noqa: E402
    build_replay_records,
    load_replay_protocol,
    render_outputs,
)
from validate_replay_run import validate_run_manifest  # noqa: E402


class ReplayPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol_path = PROJECT_ROOT / "configs" / "replay_protocol_v1.json"
        cls.protocol = load_replay_protocol(PROJECT_ROOT, cls.protocol_path)

    def test_build_is_label_free_and_level_a_only(self):
        manifest, events, traces, summary = build_replay_records(PROJECT_ROOT, self.protocol)
        self.assertEqual(len(events), 172)
        self.assertEqual(len(manifest), len(events))
        self.assertEqual(len(traces), len(events))
        self.assertEqual(summary["confirmed_count"], 0)
        self.assertEqual(summary["reported_count"], 0)
        self.assertEqual(summary["world_coordinate_count"], 0)
        payload = json.dumps([manifest, events, traces])
        for forbidden in ("target_label", "ground_truth", "scene_seed", "room_id", "world_layout"):
            self.assertNotIn(f'"{forbidden}"', payload)
        self.assertTrue(all(event["stamp"] is None for event in events))
        self.assertTrue(all(detection["position"] is None for event in events for detection in event["detections"]))
        self.assertTrue(all(trace["final_state"] not in {"CONFIRMED", "REPORTED"} for trace in traces))

    def test_outputs_are_byte_deterministic(self):
        self.assertEqual(render_outputs(PROJECT_ROOT, self.protocol), render_outputs(PROJECT_ROOT, self.protocol))

    def test_frozen_policy_rejects_weakening(self):
        mutations = []
        threshold = copy.deepcopy(self.protocol)
        threshold["decision_policy"]["detector_threshold"] = 0.0
        mutations.append(threshold)
        temporal = copy.deepcopy(self.protocol)
        temporal["ordering"]["trusted_time_available"] = True
        mutations.append(temporal)
        locked = copy.deepcopy(self.protocol)
        locked["development_inputs"][0]["fold_id"] = "fold_03"
        mutations.append(locked)
        relabel = copy.deepcopy(self.protocol)
        relabel["development_inputs"][0].update({
            "job_path": "experiments/pilot_v1/baseline_jobs/fold_03/rfdetr_images.jsonl",
            "job_sha256": "705705791432249a39d1f05c9556f7d0e5c1bd1c79d720d7a57e29a41cdc2625",
            "result_path": "experiments/pilot_v1/results/rfdetr_fold_03.jsonl",
            "result_sha256": "aa417af6020a817584f92af932a25c7b28c5d5e898f4cedad0d3b64d68df8ef6",
            "expected_records": 75,
        })
        mutations.append(relabel)
        redirect = copy.deepcopy(self.protocol)
        redirect["outputs"]["manifest"] = "deliverables/2024311281_孙驰_期末报告_RF-DETR_SmolVLM.docx"
        mutations.append(redirect)
        truth = copy.deepcopy(self.protocol)
        truth["truth_join_policy"] = "allowed"
        mutations.append(truth)
        storyboard = copy.deepcopy(self.protocol)
        storyboard["storyboard_policy"]["cases_per_state"] = 1
        mutations.append(storyboard)
        for protocol in mutations:
            temp = PROJECT_ROOT / "configs" / ".invalid_replay_protocol_test.json"
            try:
                temp.write_text(json.dumps(protocol), encoding="utf-8")
                with self.assertRaises(ReplayContractError):
                    load_replay_protocol(PROJECT_ROOT, temp)
            finally:
                temp.unlink(missing_ok=True)

    def test_state_policy_has_all_non_temporal_branches(self):
        _, events, traces, summary = build_replay_records(PROJECT_ROOT, self.protocol)
        self.assertGreater(summary["state_counts"]["SEARCHING"], 0)
        self.assertGreater(summary["state_counts"]["REJECTED"], 0)
        self.assertGreater(summary["state_counts"]["HELD_UNLOCALIZED"], 0)
        by_id = {event["record_id"]: event for event in events}
        for trace in traces:
            event = by_id[trace["event_record_id"]]
            danger = any(item["class_name"] == "danger_red_sphere" for item in event["detections"])
            if danger:
                self.assertEqual(trace["final_state"], "HELD_UNLOCALIZED")

    def test_run_manifest_rejects_missing_artifact(self):
        source = PROJECT_ROOT / "experiments" / "replay_v1" / "replay_run.json"
        mutated = json.loads(source.read_text(encoding="utf-8"))
        mutated["artifacts"] = mutated["artifacts"][:-1]
        temp = PROJECT_ROOT / "experiments" / "replay_v1" / ".invalid_replay_run_test.json"
        try:
            temp.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_run_manifest(temp)
        finally:
            temp.unlink(missing_ok=True)
