import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


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


evaluator = load_module(
    "evaluate_rfdetr_baseline", PROJECT_ROOT / "scripts" / "evaluate_rfdetr_baseline.py"
)
run_validator = load_module(
    "validate_rfdetr_run", PROJECT_ROOT / "scripts" / "validate_rfdetr_run.py"
)

from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    sha256_file,
)


PROTOCOL = PROJECT_ROOT / "configs" / "rfdetr_evaluation_protocol.json"
LOCK = PROJECT_ROOT / "experiments" / "pilot_v1" / "rfdetr_evaluation_protocol.lock.json"
CONFIG = PROJECT_ROOT / "configs" / "baselines.json"
JOBS = PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs"
SPLITS = PROJECT_ROOT / "data" / "splits" / "pilot_v1"


class RfDetrEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _perfect_results():
        truths = {}
        for path in sorted(SPLITS.glob("fold_*/test_candidates.jsonl")):
            for candidate in load_jsonl(path):
                if candidate["bbox_xyxy_pixels"] is not None:
                    truths.setdefault(candidate["source_sample_id"], []).append(candidate)
        class_details = {
            "danger_red_sphere": (0, "Red_ball"),
            "distractor_green_sphere": (1, "Green_ball"),
            "distractor_red_box": (3, "red_cube"),
        }
        records = []
        for path in sorted(JOBS.glob("fold_*/rfdetr_images.jsonl")):
            for job in load_jsonl(path):
                predictions = []
                for index, candidate in enumerate(truths.get(job["sample_id"], [])):
                    label = candidate["target_label"]
                    class_id, raw_name = class_details[label]
                    bbox = candidate["bbox_xyxy_pixels"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    fraction = area / (job["width"] * job["height"])
                    predictions.append(
                        {
                            "prediction_index": index,
                            "class_id": class_id,
                            "raw_class_name": raw_name,
                            "canonical_label": label,
                            "confidence": 0.99,
                            "bbox_xyxy_pixels": bbox,
                            "bbox_area_fraction": fraction,
                            "mask_area_fraction": fraction,
                            "mask_coverage_in_bbox": 1.0,
                        }
                    )
                records.append(
                    {
                        "schema_version": 1,
                        "job_id": job["job_id"],
                        "baseline_id": job["baseline_id"],
                        "fold_id": job["fold_id"],
                        "sample_id": job["sample_id"],
                        "image_sha256": job["image_sha256"],
                        "status": "ok",
                        "error": None,
                        "latency_ms": 1.0,
                        "predictions": predictions,
                    }
                )
        return records

    def test_protocol_lock_binds_every_runtime_source(self):
        protocol, lock = evaluator._load_frozen_protocol(
            PROTOCOL, LOCK, CONFIG, JOBS / "protocol.json", SPLITS / "protocol.json"
        )
        self.assertEqual(protocol["protocol_id"], "rfdetr_pilot_evaluation_v1")
        self.assertEqual(
            lock["sealed_files"]["evaluation_protocol"]["sha256"], sha256_file(PROTOCOL)
        )
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "baselines.json"
            changed.write_bytes(CONFIG.read_bytes() + b"\n")
            with self.assertRaises(BaselineContractError):
                evaluator._load_frozen_protocol(
                    PROTOCOL, LOCK, changed, JOBS / "protocol.json", SPLITS / "protocol.json"
                )

    def test_bbox_iou(self):
        self.assertEqual(evaluator.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(evaluator.bbox_iou([0, 0, 2, 2], [3, 3, 4, 4]), 0.0)
        self.assertAlmostEqual(
            evaluator.bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3
        )

    def test_matching_is_confidence_ordered_and_one_to_one(self):
        predictions = [
            {"prediction_index": 0, "confidence": 0.7, "bbox_xyxy_pixels": [0, 0, 10, 10]},
            {"prediction_index": 1, "confidence": 0.9, "bbox_xyxy_pixels": [0, 0, 10, 10]},
        ]
        report = evaluator.match_class([[0, 0, 10, 10]], predictions, 0.5)
        self.assertEqual(report["true_positive"], 1)
        self.assertEqual(report["false_positive"], 1)
        self.assertEqual(report["false_negative"], 0)
        self.assertEqual(report["matches"][0]["prediction_index"], 1)
        self.assertEqual(report["false_positive_prediction_indices"], [0])

    def test_protocol_lock_rejects_mutually_changed_jobs_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "jobs"
            shutil.copytree(JOBS, copied)
            path = copied / "protocol.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["total_detector_jobs"] = 400
            path.write_text(json.dumps(data) + "\n", encoding="utf-8")
            with self.assertRaises(BaselineContractError):
                evaluator._load_frozen_protocol(
                    PROTOCOL, LOCK, CONFIG, path, SPLITS / "protocol.json"
                )

    def test_bool_is_not_a_numeric_measurement(self):
        self.assertFalse(evaluator._is_finite_number(True))
        self.assertTrue(evaluator._is_finite_number(0))
        self.assertTrue(evaluator._is_finite_number(0.5))

    def test_full_evaluator_requires_exact_schema_and_scores_perfect_boxes(self):
        records = self._perfect_results()
        self.assertEqual(len(records), 401)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_by_fold(items, prefix):
                paths = []
                for fold_number in range(1, 6):
                    fold_id = f"fold_{fold_number:02d}"
                    path = root / f"{prefix}_{fold_id}.jsonl"
                    path.write_text(
                        "".join(
                            json.dumps(record, sort_keys=True) + "\n"
                            for record in items
                            if record["fold_id"] == fold_id
                        ),
                        encoding="utf-8",
                    )
                    paths.append(path)
                return paths

            paths = write_by_fold(records, "perfect")
            report = evaluator.evaluate(paths, PROTOCOL, LOCK, CONFIG, JOBS, SPLITS)
            self.assertEqual(report["metrics"]["image_count"], 401)
            self.assertEqual(report["metrics"]["three_target_class_macro_f1"], 1.0)
            self.assertEqual(report["metrics"]["danger_image_presence"]["recall"], 1.0)
            self.assertEqual(
                report["metrics"]["target_predictions_on_background_convention_images"], 0
            )

            invalid = [dict(record) for record in records]
            invalid[0]["latency_ms"] = True
            invalid_paths = write_by_fold(invalid, "invalid")
            with self.assertRaises(BaselineContractError):
                evaluator.evaluate(invalid_paths, PROTOCOL, LOCK, CONFIG, JOBS, SPLITS)

            with self.assertRaises(BaselineContractError):
                evaluator.evaluate(paths[:4], PROTOCOL, LOCK, CONFIG, JOBS, SPLITS)

    def test_sealed_full_run_recomputes_and_rejects_forged_hash(self):
        report = run_validator.validate()
        self.assertEqual(report["metrics"]["image_count"], 401)
        self.assertAlmostEqual(report["metrics"]["three_target_class_macro_f1"], 0.9635021423084074)
        manifest = json.loads(run_validator.CANONICAL_MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mutations = {
                "raw result hash": lambda value: value["raw_results"][0].update(sha256="0" * 64),
                "run id": lambda value: value.update(run_id="forged"),
                "pre-run anchor": lambda value: value.update(pre_run_anchor_commit="0" * 40),
                "interpretation boundary": lambda value: value.update(interpretation_boundary=[]),
                "checkpoint hash": lambda value: value["checkpoint"].update(sha256="0" * 64),
                "preflight hash": lambda value: value["preflight_manifest"].update(sha256="0" * 64),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    forged_manifest = json.loads(json.dumps(manifest))
                    mutate(forged_manifest)
                    forged = root / f"{label.replace(' ', '_')}.json"
                    forged.write_text(json.dumps(forged_manifest) + "\n", encoding="utf-8")
                    with self.assertRaises(BaselineContractError):
                        run_validator.validate(forged)

    def test_git_evidence_anchor_rejects_coherent_artifact_replacement(self):
        relative = "experiments/pilot_v1/results/rfdetr_fold_01.jsonl"
        canonical = PROJECT_ROOT / relative
        with tempfile.TemporaryDirectory() as temporary:
            exact_copy = Path(temporary) / "exact.jsonl"
            exact_copy.write_bytes(canonical.read_bytes())
            run_validator._require_git_sealed_artifact(exact_copy, relative, "exact copy")

            coherent_replacement = Path(temporary) / "replacement.jsonl"
            coherent_replacement.write_bytes(canonical.read_bytes() + b"\n")
            with self.assertRaises(BaselineContractError):
                run_validator._require_git_sealed_artifact(
                    coherent_replacement, relative, "coherent replacement"
                )


if __name__ == "__main__":
    unittest.main()
