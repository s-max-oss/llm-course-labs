import importlib.util
import json
import math
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


builder = load_module("prepare_smolvlm_b2_jobs", PROJECT_ROOT / "scripts" / "prepare_smolvlm_b2_jobs.py")
validator = load_module("validate_smolvlm_b2_jobs", PROJECT_ROOT / "scripts" / "validate_smolvlm_b2_jobs.py")
runner = load_module("run_smolvlm_b2", PROJECT_ROOT / "scripts" / "run_smolvlm_b2.py")
evaluator = load_module("evaluate_smolvlm_b2", PROJECT_ROOT / "scripts" / "evaluate_smolvlm_b2.py")
b1_evaluator = load_module(
    "evaluate_smolvlm_baseline_for_b2_tests",
    PROJECT_ROOT / "scripts" / "evaluate_smolvlm_baseline.py",
)
comparator = load_module(
    "compare_smolvlm_b1_b2_locked",
    PROJECT_ROOT / "scripts" / "compare_smolvlm_b1_b2_locked.py",
)

from robot_vlm_baselines.b2 import (  # noqa: E402
    B2_TRUSTED_DEVELOPMENT_GATE_SHA256,
    load_frozen_b2_protocol,
    parse_b2_a_label,
    rank_candidate_scores,
    validate_development_gate_report,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    sha256_file,
)


PROTOCOL = PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json"
LOCK = PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json"
CONFIG = PROJECT_ROOT / "configs" / "baselines.json"
SPLITS = PROJECT_ROOT / "data" / "splits" / "pilot_v1"
JOBS = PROJECT_ROOT / "experiments" / "pilot_v1" / "b2_jobs"


def tree_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def truth_by_candidate(partition: str):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seeds = set(protocol["partition"][partition]["scene_seeds"])
    result = {}
    for path in sorted(SPLITS.glob("fold_*/test_candidates.jsonl")):
        for candidate in load_jsonl(path):
            if candidate["scene_seed"] in seeds:
                result[candidate["candidate_id"]] = candidate
    return result


def make_b2_b_results(jobs, predictions):
    reverse = {
        "danger_red_sphere": "red ball",
        "distractor_red_box": "red box",
        "distractor_green_sphere": "green ball",
        "background_or_unknown": "none",
    }
    order = ["red ball", "red box", "green ball", "none"]
    records = []
    for job in jobs:
        natural = reverse[predictions[job["candidate_id"]]]
        scores = {label: (-1.0 if label != natural else 0.0) for label in order}
        top2 = next(label for label in order if label != natural)
        records.append(
            {
                "schema_version": 1,
                "b2_protocol_id": "smolvlm_b2_exploratory_v1",
                "variant": "b2_b_fixed_candidate_scoring",
                "job_id": job["job_id"],
                "partition": job["partition"],
                "candidate_id": job["candidate_id"],
                "source_sample_id": job["source_sample_id"],
                "image_sha256": job["image_sha256"],
                "input_variant": job["input_variant"],
                "proposal_source": job["proposal_source"],
                "deployment_eligible": False,
                "status": "ok",
                "error": None,
                "latency_ms": 1.0,
                "predicted_natural_label": natural,
                "predicted_label": predictions[job["candidate_id"]],
                "candidate_scores": scores,
                "top1_natural_label": natural,
                "top2_natural_label": top2,
                "margin": 1.0,
            }
        )
    return records


class B2ToolTests(unittest.TestCase):
    def test_locked_run_manifest_seals_all_result_artifacts(self):
        root = PROJECT_ROOT / "experiments" / "pilot_v1"
        manifest = json.loads((root / "smolvlm_b2_locked_run.json").read_text(encoding="utf-8"))
        self.assertIs(manifest["locked_evaluation_executed"], True)
        self.assertIs(manifest["b2_a_locked_evaluation_executed"], False)
        for artifact in manifest["artifacts"].values():
            self.assertEqual(sha256_file(root / artifact["path"]), artifact["sha256"])
        for artifact in manifest["authorization_chain"].values():
            self.assertEqual(sha256_file(root / artifact["path"]), artifact["sha256"])
        metrics = json.loads(
            (root / manifest["artifacts"]["metrics"]["path"]).read_text(encoding="utf-8")
        )["metrics"]
        expected = {
            "sample_count": metrics["sample_count"],
            "valid_prediction_count": metrics["valid_prediction_count"],
            "prediction_coverage": metrics["prediction_coverage"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "danger_red_sphere_recall": metrics["danger_binary"]["recall"],
            "distractor_red_box_recall": metrics["per_class"]["distractor_red_box"]["recall"],
            "distractor_green_sphere_recall": metrics["per_class"]["distractor_green_sphere"]["recall"],
            "background_or_unknown_recall": metrics["per_class"]["background_or_unknown"]["recall"],
            "latency_ms": metrics["latency_ms"],
        }
        self.assertEqual(manifest["locked_metrics"], expected)

    def test_locked_b1_b2_comparison_recomputes_same_candidate_subset(self):
        report = comparator.compare(
            PROTOCOL,
            LOCK,
            SPLITS,
            PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_zero_shot_run.json",
            PROJECT_ROOT / "experiments" / "pilot_v1" / "results" / "b2_b_locked_evaluation_metrics.json",
        )
        self.assertEqual(report["b1_strict_generation"]["sample_count"], 251)
        self.assertEqual(report["b1_strict_generation"]["prediction_coverage"], 0.0)
        self.assertEqual(report["b2_fixed_candidate_scoring"]["sample_count"], 251)
        self.assertEqual(report["b2_fixed_candidate_scoring"]["prediction_coverage"], 1.0)
        self.assertEqual(
            report["b2_fixed_candidate_scoring"]["per_class"]["background_or_unknown"]["recall"],
            0.0,
        )
        self.assertEqual(report["delta_b2_minus_b1"]["danger_red_sphere_recall"], 1.0)

    def test_locked_comparison_rejects_unpinned_manifests_and_reports(self):
        canonical_b1 = PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_zero_shot_run.json"
        canonical_b2 = (
            PROJECT_ROOT
            / "experiments"
            / "pilot_v1"
            / "results"
            / "b2_b_locked_evaluation_metrics.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied_b1 = root / "b1.json"
            copied_b2 = root / "b2.json"
            shutil.copyfile(canonical_b1, copied_b1)
            shutil.copyfile(canonical_b2, copied_b2)
            with self.assertRaises(BaselineContractError):
                comparator.compare(PROTOCOL, LOCK, SPLITS, copied_b1, canonical_b2)
            with self.assertRaises(BaselineContractError):
                comparator.compare(PROTOCOL, LOCK, SPLITS, canonical_b1, copied_b2)

    def test_locked_seal_rejects_joint_prediction_and_metrics_replacement(self):
        source_root = PROJECT_ROOT / "experiments" / "pilot_v1"
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary) / "project"
            run_root = project_root / "experiments" / "pilot_v1"
            result_root = run_root / "results"
            result_root.mkdir(parents=True)
            manifest_path = run_root / "smolvlm_b2_locked_run.json"
            metrics_path = result_root / "b2_b_locked_evaluation_metrics.json"
            predictions_path = result_root / "b2_b_locked_evaluation.jsonl"
            shutil.copyfile(source_root / "smolvlm_b2_locked_run.json", manifest_path)
            shutil.copyfile(source_root / "results" / metrics_path.name, metrics_path)
            shutil.copyfile(source_root / "results" / predictions_path.name, predictions_path)
            self.assertEqual(
                comparator.validate_locked_b2_artifacts(project_root, manifest_path, metrics_path),
                predictions_path,
            )

            predictions_path.write_text(
                predictions_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            forged_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            forged_metrics["source_files"]["results"]["sha256"] = sha256_file(predictions_path)
            metrics_path.write_text(json.dumps(forged_metrics, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(BaselineContractError):
                comparator.validate_locked_b2_artifacts(project_root, manifest_path, metrics_path)

    def test_full_evaluators_reject_prediction_contract_corruption(self):
        b1_run = json.loads(
            (PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_zero_shot_run.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            b1_paths = []
            for artifact in b1_run["prediction_files"]:
                source = PROJECT_ROOT / artifact["path"]
                target = root / source.name
                shutil.copyfile(source, target)
                b1_paths.append(target)
            records = load_jsonl(b1_paths[0])
            records[0]["target_label"] = "danger_red_sphere"
            b1_paths[0].write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )
            with self.assertRaises(BaselineContractError):
                b1_evaluator.evaluate(
                    b1_paths,
                    SPLITS,
                    CONFIG,
                    PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs" / "protocol.json",
                )

        with tempfile.TemporaryDirectory() as temporary:
            corrupted = Path(temporary) / "b2.jsonl"
            records = load_jsonl(
                PROJECT_ROOT
                / "experiments"
                / "pilot_v1"
                / "results"
                / "b2_b_locked_evaluation.jsonl"
            )
            records[0]["top1_natural_label"] = "none"
            corrupted.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )
            with self.assertRaises(BaselineContractError):
                evaluator.evaluate(
                    JOBS / "locked_evaluation.jsonl",
                    corrupted,
                    "b2_b",
                    "locked_evaluation",
                    PROTOCOL,
                    LOCK,
                    CONFIG,
                    SPLITS,
                    PROJECT_ROOT
                    / "experiments"
                    / "pilot_v1"
                    / "results"
                    / "b2_b_development_gate.json",
                )

    def test_locked_evaluation_is_bound_to_real_development_gate(self):
        self.assertEqual(
            B2_TRUSTED_DEVELOPMENT_GATE_SHA256,
            "e752b851b3f849a7aa42f52625bd96ce3542e904e0a67f00b798f5b93aaf91e4",
        )
        protocol, lock = load_frozen_b2_protocol(PROJECT_ROOT, PROTOCOL, LOCK)
        report = validate_development_gate_report(
            PROJECT_ROOT / "experiments" / "pilot_v1" / "results" / "b2_b_development_gate.json",
            protocol,
            lock["sealed_files"]["machine_protocol"]["sha256"],
            PROJECT_ROOT,
            B2_TRUSTED_DEVELOPMENT_GATE_SHA256,
        )
        self.assertIs(report["development_gates"]["passed"], True)

    def test_frozen_protocol_and_strict_parser(self):
        protocol, lock = load_frozen_b2_protocol(PROJECT_ROOT, PROTOCOL, LOCK)
        self.assertEqual(protocol["protocol_id"], lock["protocol_id"])
        self.assertEqual(
            parse_b2_a_label(" RED BALL. \n", protocol),
            ("red ball", "danger_red_sphere"),
        )
        self.assertIsNone(parse_b2_a_label("The answer is red ball", protocol))
        self.assertIsNone(parse_b2_a_label("red sphere", protocol))

    def test_ranker_is_finite_and_uses_frozen_tie_order(self):
        order = ["red ball", "red box", "green ball", "none"]
        scores = {label: -1.0 for label in order}
        self.assertEqual(rank_candidate_scores(scores, order), ("red ball", "red box", 0.0))
        with self.assertRaises(BaselineContractError):
            rank_candidate_scores({**scores, "none": float("nan")}, order)
        with self.assertRaises(BaselineContractError):
            rank_candidate_scores({"red ball": 0.0}, order)

    def test_job_generation_counts_label_boundary_and_determinism(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            builder.build(PROTOCOL, LOCK, CONFIG, SPLITS, first)
            builder.build(PROTOCOL, LOCK, CONFIG, SPLITS, second)
            self.assertEqual(tree_bytes(first), tree_bytes(second))
            self.assertEqual(len(load_jsonl(first / "development.jsonl")), 193)
            self.assertEqual(len(load_jsonl(first / "development_smoke.jsonl")), 8)
            self.assertEqual(len(load_jsonl(first / "locked_evaluation.jsonl")), 251)
            for path in first.glob("*.jsonl"):
                for record in load_jsonl(path):
                    self.assertNotIn("target_label", record)
                    self.assertNotIn("scene_seed", record)

    def test_independent_validator_rejects_corruption(self):
        def leak_label(root: Path):
            path = root / "development.jsonl"
            records = load_jsonl(path)
            records[0]["target_label"] = "danger_red_sphere"
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )

        def remove_locked(root: Path):
            path = root / "locked_evaluation.jsonl"
            records = load_jsonl(path)[1:]
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )

        def stale_hash(root: Path):
            path = root / "protocol.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["queues"]["development"]["sha256"] = "0" * 64
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

        for label, mutate in {
            "label leak": leak_label,
            "missing locked job": remove_locked,
            "stale queue hash": stale_hash,
        }.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "jobs"
                shutil.copytree(JOBS, copied)
                mutate(copied)
                errors, _ = validator.validate(PROTOCOL, LOCK, CONFIG, SPLITS, copied)
                self.assertTrue(errors)

    def test_development_evaluator_gates_perfect_and_poor_results(self):
        jobs = load_jsonl(JOBS / "development.jsonl")
        truth = truth_by_candidate("development")
        perfect_predictions = {key: value["target_label"] for key, value in truth.items()}
        poor_predictions = {key: "background_or_unknown" for key in truth}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, predictions, expected_pass in (
                ("perfect", perfect_predictions, True),
                ("poor", poor_predictions, False),
            ):
                path = root / f"{name}.jsonl"
                records = make_b2_b_results(jobs, predictions)
                path.write_text(
                    "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                    encoding="utf-8",
                )
                report = evaluator.evaluate(
                    JOBS / "development.jsonl",
                    path,
                    "b2_b",
                    "development",
                    PROTOCOL,
                    LOCK,
                    CONFIG,
                    SPLITS,
                )
                self.assertIs(report["development_gates"]["passed"], expected_pass)
                if expected_pass:
                    self.assertEqual(report["metrics"]["macro_f1"], 1.0)

    def test_locked_execution_requires_passing_development_report(self):
        with self.assertRaises(BaselineContractError):
            runner.run(
                JOBS / "locked_evaluation.jsonl",
                PROJECT_ROOT / "unused.jsonl",
                PROJECT_ROOT,
                PROTOCOL,
                LOCK,
                CONFIG,
                "b2_b",
                None,
                None,
                None,
                True,
            )

    def test_forged_gate_report_and_rehashed_queue_are_rejected(self):
        protocol, lock = load_frozen_b2_protocol(PROJECT_ROOT, PROTOCOL, LOCK)
        jobs = load_jsonl(JOBS / "development.jsonl")
        truth = truth_by_candidate("development")
        predictions = {key: value["target_label"] for key, value in truth.items()}
        temp_parent = PROJECT_ROOT / "experiments" / "pilot_v1"
        with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
            root = Path(temporary)
            results_path = root / "perfect.jsonl"
            results_path.write_text(
                "".join(
                    json.dumps(item, sort_keys=True) + "\n"
                    for item in make_b2_b_results(jobs, predictions)
                ),
                encoding="utf-8",
            )
            report = evaluator.evaluate(
                JOBS / "development.jsonl",
                results_path,
                "b2_b",
                "development",
                PROTOCOL,
                LOCK,
                CONFIG,
                SPLITS,
            )
            gate_path = root / "gate.json"
            gate_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            trusted_digest = sha256_file(gate_path)
            validate_development_gate_report(
                gate_path,
                protocol,
                lock["sealed_files"]["machine_protocol"]["sha256"],
                PROJECT_ROOT,
                trusted_digest,
            )
            report["metrics"]["macro_f1"] = 0.0
            report["development_gates"]["checks"]["macro_f1"].update(
                value=0.0, passed=False
            )
            gate_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(BaselineContractError):
                validate_development_gate_report(
                    gate_path,
                    protocol,
                    lock["sealed_files"]["machine_protocol"]["sha256"],
                    PROJECT_ROOT,
                    trusted_digest,
                )

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "jobs"
            shutil.copytree(JOBS, copied)
            queue_path = copied / "development.jsonl"
            records = load_jsonl(queue_path)
            records[0]["target_region_xyxy_pixels"] = None
            records[0]["proposal_source"] = "offline_background_convention"
            queue_path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )
            manifest_path = copied / "protocol.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["queues"]["development"]["sha256"] = sha256_file(queue_path)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            with self.assertRaises(BaselineContractError):
                runner._validate_jobs_manifest(
                    queue_path,
                    records,
                    protocol,
                    PROTOCOL,
                    LOCK,
                    CONFIG,
                )

    def test_fixed_candidate_scoring_returns_all_finite_candidates(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        class Tokenizer:
            eos_token_id = 0

            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                return {"input_ids": torch.tensor([[3]])}

        class Processor:
            tokenizer = Tokenizer()

            def apply_chat_template(self, *args, **kwargs):
                return "prompt"

            def __call__(self, text, images, return_tensors):
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

        class Model:
            def __call__(self, **inputs):
                shape = (1, inputs["input_ids"].shape[1], 8)
                return type("Output", (), {"logits": torch.zeros(shape)})()

        completions = [" red ball", " red box", " green ball", " none"]
        scores = runner.fixed_candidate_scores(
            Processor(), Model(), torch.device("cpu"), object(), "prompt", completions
        )
        self.assertEqual(set(scores), {item.strip() for item in completions})
        self.assertTrue(all(math.isfinite(value) for value in scores.values()))


if __name__ == "__main__":
    unittest.main()
