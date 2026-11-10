from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from prepare_cascade_jobs import build  # noqa: E402
from evaluate_cascade_feasibility import evaluate  # noqa: E402
from robot_vlm_baselines.cascade import (  # noqa: E402
    build_cascade_jobs,
    load_cascade_protocol,
    padded_region,
    validate_cascade_jobs,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
)
from validate_cascade_jobs import validate  # noqa: E402
import validate_cascade_run as run_seal  # noqa: E402
from robot_vlm_baselines.contracts import write_jsonl  # noqa: E402
from run_cascade_b2 import run  # noqa: E402
from run_cascade_b2 import reserve_output, write_reserved_jsonl  # noqa: E402


PROTOCOL_PATH = PROJECT_ROOT / "configs" / "cascade_feasibility_v1.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "baselines.json"


class CascadeToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_cascade_protocol(PROJECT_ROOT, PROTOCOL_PATH)

    def test_frozen_queue_is_label_free_and_exact(self):
        jobs = build_cascade_jobs(PROJECT_ROOT, self.protocol)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({item["sample_id"] for item in jobs}), 22)
        self.assertEqual({item["fold_id"] for item in jobs}, {"fold_01", "fold_02"})
        payload = json.dumps(jobs)
        for forbidden in ("target_label", "scene_seed", "oracle", "ground_truth"):
            self.assertNotIn(forbidden, payload)
        validate()

    def test_generation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            build(PROTOCOL_PATH, CONFIG_PATH, first_root)
            build(PROTOCOL_PATH, CONFIG_PATH, second_root)
            for name in ("development.jsonl", "protocol.json"):
                self.assertEqual(
                    (first_root / name).read_bytes(),
                    (second_root / name).read_bytes(),
                )

    def test_job_contract_rejects_truth_and_route_corruption(self):
        jobs = build_cascade_jobs(PROJECT_ROOT, self.protocol)
        truth_leak = copy.deepcopy(jobs)
        truth_leak[0]["target_label"] = "danger_red_sphere"
        with self.assertRaises(BaselineContractError):
            validate_cascade_jobs(truth_leak, self.protocol, {"target_label"})

        out_of_route = copy.deepcopy(jobs)
        out_of_route[0]["detector_confidence"] = 0.9
        with self.assertRaises(BaselineContractError):
            validate_cascade_jobs(out_of_route, self.protocol, set())

        oracle_source = copy.deepcopy(jobs)
        oracle_source[0]["proposal_source"] = "offline_annotation_oracle"
        with self.assertRaises(BaselineContractError):
            validate_cascade_jobs(oracle_source, self.protocol, set())

    def test_padding_is_clipped(self):
        self.assertEqual(padded_region([0.0, 0.0, 10.0, 20.0], 100, 100, 0.15), [0.0, 0.0, 11.5, 23.0])
        with self.assertRaises(BaselineContractError):
            padded_region([-1.0, 0.0, 10.0, 20.0], 100, 100, 0.15)

    def _identity_results(self):
        reverse = {
            "danger_red_sphere": "red ball",
            "distractor_red_box": "red box",
            "distractor_green_sphere": "green ball",
        }
        natural_order = ["red ball", "red box", "green ball", "none"]
        records = []
        for job in build_cascade_jobs(PROJECT_ROOT, self.protocol):
            top1 = reverse[job["detector_label"]]
            scores = {label: (-1.0 if label == top1 else -4.0) for label in natural_order}
            top2 = next(label for label in natural_order if label != top1)
            records.append(
                {
                    "schema_version": 1,
                    "protocol_id": self.protocol["protocol_id"],
                    "b2_protocol_id": "smolvlm_b2_exploratory_v1",
                    "variant": "b2_b_fixed_candidate_scoring",
                    "job_id": job["job_id"],
                    "fold_id": job["fold_id"],
                    "sample_id": job["sample_id"],
                    "prediction_index": job["prediction_index"],
                    "image_sha256": job["image_sha256"],
                    "input_variant": job["input_variant"],
                    "proposal_source": job["proposal_source"],
                    "detector_label": job["detector_label"],
                    "detector_confidence": job["detector_confidence"],
                    "proposal_xyxy_pixels": job["proposal_xyxy_pixels"],
                    "status": "ok",
                    "error": None,
                    "predicted_natural_label": top1,
                    "predicted_label": job["detector_label"],
                    "candidate_scores": scores,
                    "top1_natural_label": top1,
                    "top2_natural_label": top2,
                    "margin": 3.0,
                    "latency_ms": 1.0,
                }
            )
        return records

    def test_identity_cascade_fails_strict_gain_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "identity.jsonl"
            write_jsonl(path, self._identity_results())
            report = evaluate(path)
            self.assertEqual(
                report["rf_only"]["three_target_class_macro_f1"],
                report["primary_safety_preserving_cascade"]["three_target_class_macro_f1"],
            )
            self.assertFalse(
                report["continuation_gates"]["three_target_macro_f1_strictly_increased"]
            )
            self.assertFalse(report["continuation_gates"]["all_passed"])
            self.assertEqual(report["decision"], "stop_cascade_and_recommend_rfdetr_only")

            forged = self._identity_results()
            forged[0]["prediction_index"] = 999
            forged_path = Path(temporary) / "forged.jsonl"
            write_jsonl(forged_path, forged)
            with self.assertRaises(BaselineContractError):
                evaluate(forged_path)

            metadata_mutations = {
                "variant": "forged",
                "input_variant": "offline_annotation_oracle",
                "proposal_source": "offline_annotation_oracle",
                "predicted_natural_label": "none",
                "error": "forged success error",
            }
            for field, value in metadata_mutations.items():
                with self.subTest(field=field):
                    forged = self._identity_results()
                    forged[0][field] = value
                    forged_path = Path(temporary) / f"forged_{field}.jsonl"
                    write_jsonl(forged_path, forged)
                    with self.assertRaises(BaselineContractError):
                        evaluate(forged_path)

    def test_runner_dry_run_validates_real_proposal_crops(self):
        local = json.loads(
            (PROJECT_ROOT / "configs" / "data_sources.local.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "unused.jsonl"
            status = run(
                PROJECT_ROOT / "experiments" / "pilot_v1" / "cascade_jobs" / "development.jsonl",
                output,
                Path(local["image_root"]),
                PROTOCOL_PATH,
                PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json",
                PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json",
                CONFIG_PATH,
                None,
                True,
            )
            self.assertEqual(status, 0)
            self.assertFalse(output.exists())

    def test_output_reservation_is_atomic_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "reserved.jsonl"
            descriptor = reserve_output(output)
            self.assertTrue(output.exists())
            with self.assertRaises(BaselineContractError):
                reserve_output(output)
            write_reserved_jsonl(descriptor, [{"ok": True}])
            self.assertEqual(load_jsonl(output), [{"ok": True}])

    def test_sealed_stopped_run_recomputes_and_rejects_manifest_tampering(self):
        report = run_seal.validate()
        self.assertFalse(report["continuation_gates"]["all_passed"])
        self.assertEqual(report["decision"], "stop_cascade_and_recommend_rfdetr_only")
        manifest = json.loads(run_seal.CANONICAL_MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            for label, mutate in {
                "raw hash": lambda value: value["raw_results"].update(sha256="0" * 64),
                "pre-run commit": lambda value: value.update(pre_run_code_commit="0" * 40),
                "stop decision": lambda value: value["decision"].update(
                    locked_evaluation_authorized=True
                ),
            }.items():
                with self.subTest(label=label):
                    forged = copy.deepcopy(manifest)
                    mutate(forged)
                    path = Path(temporary) / f"{label.replace(' ', '_')}.json"
                    path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
                    with self.assertRaises(BaselineContractError):
                        run_seal.validate(path)

    def test_cascade_git_anchor_rejects_coherent_artifact_replacement(self):
        relative = "experiments/pilot_v1/results/cascade_b2_development.jsonl"
        canonical = PROJECT_ROOT / relative
        with tempfile.TemporaryDirectory() as temporary:
            exact = Path(temporary) / "exact.jsonl"
            exact.write_bytes(canonical.read_bytes())
            run_seal._require_git_sealed_artifact(exact, relative, "exact cascade copy")
            replaced = Path(temporary) / "replaced.jsonl"
            replaced.write_bytes(canonical.read_bytes() + b"\n")
            with self.assertRaises(BaselineContractError):
                run_seal._require_git_sealed_artifact(
                    replaced, relative, "coherent cascade replacement"
                )


if __name__ == "__main__":
    unittest.main()
