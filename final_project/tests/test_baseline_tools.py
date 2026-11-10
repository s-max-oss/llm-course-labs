import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


job_builder = load_module(
    "prepare_baseline_jobs", PROJECT_ROOT / "scripts" / "prepare_baseline_jobs.py"
)
job_validator = load_module(
    "validate_baseline_jobs", PROJECT_ROOT / "scripts" / "validate_baseline_jobs.py"
)
smol_runner = load_module(
    "run_smolvlm_zero_shot", PROJECT_ROOT / "scripts" / "run_smolvlm_zero_shot.py"
)
rfdetr_runner = load_module(
    "run_rfdetr_baseline", PROJECT_ROOT / "scripts" / "run_rfdetr_baseline.py"
)
smol_evaluator = load_module(
    "evaluate_smolvlm_baseline",
    PROJECT_ROOT / "scripts" / "evaluate_smolvlm_baseline.py",
)

from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    find_forbidden_fields,
    load_jsonl,
    parse_strict_label,
    sha256_file,
    validate_inference_jobs,
)
from robot_vlm_baselines.inference import convert_rfdetr_detections  # noqa: E402
from robot_vlm_baselines.metrics import classification_metrics  # noqa: E402


BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baselines.json"
PILOT_SPLITS = PROJECT_ROOT / "data" / "splits" / "pilot_v1"
BASELINE_JOBS = PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs"


def tree_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class FakeDetections:
    def __init__(self):
        self.xyxy = np.asarray([[10.0, 20.0, 30.0, 40.0]])
        self.confidence = np.asarray([0.91])
        self.class_id = np.asarray([0])
        self.mask = np.zeros((1, 64, 64), dtype=bool)
        self.mask[0, 20:40, 10:30] = True
        self.data = {"class_name": np.asarray(["Red_ball"])}


class FakeUnsortedDetections:
    def __init__(self):
        self.xyxy = np.asarray(
            [[10.0, 20.0, 30.0, 40.0], [32.0, 20.0, 52.0, 40.0]]
        )
        self.confidence = np.asarray([0.60, 0.90])
        self.class_id = np.asarray([0, 1])
        self.mask = np.zeros((2, 64, 64), dtype=bool)
        self.mask[0, 20:40, 10:30] = True
        self.mask[1, 20:40, 32:52] = True
        self.data = {"class_name": np.asarray(["Red_ball", "Green_ball"])}


class BaselineToolTests(unittest.TestCase):
    def test_strict_label_parser(self):
        self.assertEqual(parse_strict_label("danger_red_sphere"), "danger_red_sphere")
        self.assertEqual(
            parse_strict_label("  distractor_red_box. \n"), "distractor_red_box"
        )
        self.assertIsNone(parse_strict_label("The answer is danger_red_sphere"))
        self.assertIsNone(parse_strict_label("danger_red_sphere\nbackground_or_unknown"))
        self.assertIsNone(parse_strict_label("red_ball"))

    def test_rfdetr_conversion_preserves_mask_evidence(self):
        converted = convert_rfdetr_detections(
            FakeDetections(),
            width=64,
            height=64,
            class_names={"0": "Red_ball"},
            aliases={"Red_ball": "danger_red_sphere"},
        )
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["canonical_label"], "danger_red_sphere")
        self.assertAlmostEqual(converted[0]["mask_coverage_in_bbox"], 1.0)
        self.assertAlmostEqual(converted[0]["mask_area_fraction"], 400 / 4096)

    def test_rfdetr_conversion_rejects_unknown_class(self):
        with self.assertRaises(BaselineContractError):
            convert_rfdetr_detections(
                FakeDetections(),
                width=64,
                height=64,
                class_names={"0": "Red_ball"},
                aliases={},
            )

    def test_rfdetr_conversion_renumbers_after_confidence_sort(self):
        converted = convert_rfdetr_detections(
            FakeUnsortedDetections(),
            width=64,
            height=64,
            class_names={"0": "Red_ball", "1": "Green_ball"},
            aliases={
                "Red_ball": "danger_red_sphere",
                "Green_ball": "distractor_green_sphere",
            },
        )
        self.assertEqual([item["confidence"] for item in converted], [0.9, 0.6])
        self.assertEqual([item["prediction_index"] for item in converted], [0, 1])

    def test_crop_is_clipped_and_background_uses_full_frame(self):
        image = Image.new("RGB", (20, 10), "white")
        crop = smol_runner.crop_for_job(
            image,
            {
                "job_id": "crop",
                "target_region_xyxy_pixels": [-5.2, 2.1, 30.9, 8.2],
            },
        )
        self.assertEqual(crop.size, (20, 7))
        full = smol_runner.crop_for_job(
            image, {"job_id": "full", "target_region_xyxy_pixels": None}
        )
        self.assertEqual(full.size, image.size)

    def test_local_smolvlm_model_files_are_hash_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_file = root / "model.safetensors"
            config_file = root / "config.json"
            model_file.write_bytes(b"model")
            config_file.write_text("{}", encoding="utf-8")
            baseline = {
                "model_files": {
                    "model.safetensors": sha256_file(model_file),
                    "config.json": sha256_file(config_file),
                }
            }
            self.assertEqual(smol_runner.verify_local_model(str(root), baseline), root.resolve())
            unexpected = root / "tokenizer_config.json"
            unexpected.write_text("{}", encoding="utf-8")
            with self.assertRaises(BaselineContractError):
                smol_runner.verify_local_model(str(root), baseline)
            unexpected.unlink()
            model_file.write_bytes(b"changed")
            with self.assertRaises(BaselineContractError):
                smol_runner.verify_local_model(str(root), baseline)

    def test_metrics_include_invalid_predictions_as_errors(self):
        metrics = classification_metrics(
            [
                ("danger_red_sphere", "danger_red_sphere"),
                ("danger_red_sphere", None),
                ("distractor_red_box", "danger_red_sphere"),
            ],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(metrics["invalid_or_error_count"], 1)
        self.assertAlmostEqual(metrics["prediction_coverage"], 2 / 3)
        self.assertEqual(metrics["danger_binary"]["true_positive"], 1)
        self.assertEqual(metrics["danger_binary"]["false_positive"], 1)
        self.assertEqual(metrics["danger_binary"]["false_negative"], 1)
        self.assertIsNone(metrics["macro_f1"])
        self.assertEqual(metrics["latency_ms"]["p50"], 2.0)

    def test_generated_jobs_are_complete_and_label_free(self):
        protocol = json.loads((BASELINE_JOBS / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["total_detector_jobs"], 401)
        self.assertEqual(protocol["total_smolvlm_jobs"], 444)
        self.assertFalse(protocol["ground_truth_available_to_inference"])
        config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
        forbidden = set(config["inference_forbidden_fields"])
        for path in BASELINE_JOBS.glob("fold_*/*.jsonl"):
            for job in load_jsonl(path):
                self.assertEqual(find_forbidden_fields(job, forbidden), [])
        errors, warnings = job_validator.validate(
            BASELINE_CONFIG, PILOT_SPLITS, BASELINE_JOBS
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)

    def test_runner_trust_boundary_rejects_corrupt_jobs(self):
        config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
        source = load_jsonl(BASELINE_JOBS / "fold_01" / "smolvlm_candidates.jsonl")
        mutations = {
            "target label": lambda jobs: jobs[0].update(
                target_label="background_or_unknown"
            ),
            "wrong baseline": lambda jobs: jobs[0].update(baseline_id="other"),
            "duplicate ID": lambda jobs: jobs[1].update(job_id=jobs[0]["job_id"]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                jobs = [dict(source[0]), dict(source[1])]
                mutate(jobs)
                with self.assertRaises(BaselineContractError):
                    validate_inference_jobs(jobs, config, "smolvlm")

    def test_rfdetr_runner_preserves_per_sample_image_failure(self):
        config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
        jobs = [
            dict(item)
            for item in load_jsonl(BASELINE_JOBS / "fold_01" / "rfdetr_images.jsonl")[:2]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good_image = root / "good.png"
            Image.new("RGB", (64, 64), "white").save(good_image)
            for index, job in enumerate(jobs):
                job["image"] = "good.png" if index == 0 else "missing.png"
                job["image_sha256"] = sha256_file(good_image) if index == 0 else "0" * 64
                job["width"] = 64
                job["height"] = 64
            jobs_path = root / "jobs.jsonl"
            jobs_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in jobs),
                encoding="utf-8",
            )
            checkpoint = root / "weights.pth"
            checkpoint.write_bytes(b"test")
            output = root / "results.jsonl"
            fake_model = mock.Mock()
            fake_model.predict.return_value = FakeDetections()
            with mock.patch.object(
                rfdetr_runner,
                "sha256_file",
                return_value=config["rfdetr"]["checkpoint_sha256"],
            ), mock.patch.object(rfdetr_runner, "load_model", return_value=fake_model):
                self.assertEqual(
                    rfdetr_runner.run(
                        jobs_path,
                        output,
                        root,
                        BASELINE_CONFIG,
                        checkpoint,
                        "cpu",
                        None,
                        False,
                    ),
                    0,
                )
            records = load_jsonl(output)
            self.assertEqual([item["status"] for item in records], ["ok", "error"])
            self.assertIn("image does not exist", records[1]["error"])

    def test_smolvlm_runner_preserves_per_sample_image_failure(self):
        jobs = [
            dict(item)
            for item in load_jsonl(BASELINE_JOBS / "fold_01" / "smolvlm_candidates.jsonl")[:2]
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good_image = root / "good.png"
            Image.new("RGB", (64, 64), "white").save(good_image)
            for index, job in enumerate(jobs):
                job["image"] = "good.png" if index == 0 else "missing.png"
                job["image_sha256"] = sha256_file(good_image) if index == 0 else "0" * 64
                job["width"] = 64
                job["height"] = 64
                job["target_region_xyxy_pixels"] = [0, 0, 10, 10]
                job["proposal_source"] = "offline_annotation_oracle"
            jobs_path = root / "jobs.jsonl"
            jobs_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in jobs),
                encoding="utf-8",
            )
            output = root / "results.jsonl"
            with (
                mock.patch.object(
                    smol_runner, "verify_local_model", return_value=root
                ),
                mock.patch.object(
                    smol_runner,
                    "load_model",
                    return_value=(object(), object(), object()),
                ),
                mock.patch.object(
                    smol_runner,
                    "generate_label",
                    return_value="danger_red_sphere",
                ),
            ):
                self.assertEqual(
                    smol_runner.run(
                        jobs_path, output, root, BASELINE_CONFIG, str(root), "cpu", None, False
                    ),
                    0,
                )
            records = load_jsonl(output)
            self.assertEqual([item["status"] for item in records], ["ok", "error"])
            self.assertIn("image does not exist", records[1]["error"])

    def test_job_generation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            job_builder.build(BASELINE_CONFIG, PILOT_SPLITS, first)
            job_builder.build(BASELINE_CONFIG, PILOT_SPLITS, second)
            self.assertEqual(tree_bytes(first), tree_bytes(second))
            self.assertEqual(len(tree_bytes(first)), 11)

    def test_mandatory_baseline_policies_are_fail_closed(self):
        mutations = {
            "detector threshold": lambda config: config["rfdetr"].update(
                confidence_threshold=0.4
            ),
            "remote model loading": lambda config: config["smolvlm"].update(
                local_files_only=False
            ),
            "oracle deployment": lambda config: config["smolvlm"].update(
                deployment_eligible=True
            ),
            "missing model hashes": lambda config: config["smolvlm"].update(
                model_files={}
            ),
            "changed model hash": lambda config: config["smolvlm"]["model_files"].update(
                {"model.safetensors": "0" * 64}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
                mutate(config)
                config_path = Path(temporary) / "invalid.json"
                config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    job_builder.build(config_path, PILOT_SPLITS, Path(temporary) / "output")
                errors, _ = job_validator.validate(config_path, PILOT_SPLITS, BASELINE_JOBS)
                self.assertTrue(any("mandatory" in error for error in errors))

    def test_validator_rejects_job_corruption(self):
        def add_label(root: Path):
            path = root / "fold_01" / "smolvlm_candidates.jsonl"
            records = load_jsonl(path)
            records[0]["target_label"] = "background_or_unknown"
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )

        def remove_detector_job(root: Path):
            path = root / "fold_01" / "rfdetr_images.jsonl"
            records = load_jsonl(path)[1:]
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )

        def claim_ground_truth(root: Path):
            path = root / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            protocol["ground_truth_available_to_inference"] = True
            path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

        def duplicate_fold_summary(root: Path):
            path = root / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            protocol["folds"].append(dict(protocol["folds"][0]))
            path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

        for label, mutate in {
            "label leak": add_label,
            "missing job": remove_detector_job,
            "ground-truth claim": claim_ground_truth,
            "duplicate fold summary": duplicate_fold_summary,
        }.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "jobs"
                shutil.copytree(BASELINE_JOBS, copied)
                mutate(copied)
                errors, _ = job_validator.validate(BASELINE_CONFIG, PILOT_SPLITS, copied)
                self.assertTrue(errors, f"validator accepted corruption: {label}")

    def test_offline_evaluator_requires_complete_oof_results(self):
        predictions = []
        for path in sorted(PILOT_SPLITS.glob("fold_*/test_candidates.jsonl")):
            for candidate in load_jsonl(path):
                predictions.append(
                    {
                        "schema_version": 1,
                        "job_id": (
                            "b1_smolvlm_256m_zero_shot_oracle_crop::"
                            f"{candidate['fold_id']}::{candidate['candidate_id']}"
                        ),
                        "baseline_id": "b1_smolvlm_256m_zero_shot_oracle_crop",
                        "fold_id": candidate["fold_id"],
                        "candidate_id": candidate["candidate_id"],
                        "source_sample_id": candidate["source_sample_id"],
                        "image_sha256": candidate["image_sha256"],
                        "input_variant": (
                            "offline_oracle_padded_crop_or_background_full_frame"
                        ),
                        "proposal_source": (
                            "offline_annotation_oracle"
                            if candidate["padded_crop_xyxy_pixels"] is not None
                            else "offline_background_convention"
                        ),
                        "deployment_eligible": False,
                        "status": "ok",
                        "error": None,
                        "latency_ms": 1.0,
                        "raw_response": candidate["target_label"],
                        "predicted_label": candidate["target_label"],
                    }
                )
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "perfect.jsonl"
            result_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions),
                encoding="utf-8",
            )
            report = smol_evaluator.evaluate(
                [result_path], PILOT_SPLITS, BASELINE_CONFIG
            )
            self.assertEqual(report["pooled_out_of_fold"]["macro_f1"], 1.0)
            self.assertEqual(
                len(report["source_files"]["ground_truth_candidate_files"]), 5
            )
            fold_02 = next(
                fold for fold in report["folds"] if fold["fold_id"] == "fold_02"
            )
            self.assertIsNone(fold_02["metrics"]["macro_f1"])
            incomplete_path = Path(temporary) / "incomplete.jsonl"
            incomplete_path.write_text(
                "".join(
                    json.dumps(item, sort_keys=True) + "\n" for item in predictions[:-1]
                ),
                encoding="utf-8",
            )
            with self.assertRaises(BaselineContractError):
                smol_evaluator.evaluate(
                    [incomplete_path], PILOT_SPLITS, BASELINE_CONFIG
                )
            for label, mutate in {
                "error with prediction": lambda record: record.update(
                    status="error", error="failed", raw_response=""
                ),
                "unknown status": lambda record: record.update(status="cached"),
                "wrong proposal source": lambda record: record.update(
                    proposal_source="online_detector"
                ),
            }.items():
                with self.subTest(label=label):
                    invalid = [dict(item) for item in predictions]
                    mutate(invalid[0])
                    invalid_path = Path(temporary) / f"invalid_{label.replace(' ', '_')}.jsonl"
                    invalid_path.write_text(
                        "".join(
                            json.dumps(item, sort_keys=True) + "\n" for item in invalid
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaises(BaselineContractError):
                        smol_evaluator.evaluate(
                            [invalid_path], PILOT_SPLITS, BASELINE_CONFIG
                        )
            copied_splits = Path(temporary) / "corrupt_splits"
            shutil.copytree(PILOT_SPLITS, copied_splits)
            truth_path = copied_splits / "fold_01" / "test_candidates.jsonl"
            truth = load_jsonl(truth_path)
            truth[0]["target_label"] = "background_or_unknown"
            truth_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in truth),
                encoding="utf-8",
            )
            with self.assertRaises(BaselineContractError):
                smol_evaluator.evaluate(
                    [result_path], copied_splits, BASELINE_CONFIG
                )


if __name__ == "__main__":
    unittest.main()
