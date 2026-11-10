import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load_module("build_manifest", PROJECT_ROOT / "scripts" / "build_manifest.py")
candidate_builder = load_module(
    "build_candidate_manifest", PROJECT_ROOT / "scripts" / "build_candidate_manifest.py"
)
split_validator = load_module(
    "validate_pilot_splits", PROJECT_ROOT / "scripts" / "validate_pilot_splits.py"
)
split_builder = load_module(
    "build_pilot_splits", PROJECT_ROOT / "scripts" / "build_pilot_splits.py"
)


PILOT_CONFIG = PROJECT_ROOT / "configs" / "pilot_cv.json"
IMAGE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "batch_500.jsonl"
CANDIDATE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "batch_500.candidates.jsonl"
NEAR_DUPLICATES = PROJECT_ROOT / "data" / "manifests" / "batch_500.near_duplicates.json"
PILOT_SPLITS = PROJECT_ROOT / "data" / "splits" / "pilot_v1"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def tree_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ManifestToolTests(unittest.TestCase):
    def test_parse_image_name(self):
        self.assertEqual(
            builder.parse_image_name("seed101_016_R0284.jpg"),
            {"scene_seed": 101, "view_index": 16, "frame_index": 284},
        )

    def test_parse_yolo_line(self):
        class_id, bbox = builder.parse_yolo_line(
            "2 0.429843 0.663333 0.191940 0.241212", 1
        )
        self.assertEqual(class_id, 2)
        self.assertEqual(len(bbox), 4)

    def test_reject_out_of_bounds_yolo_line(self):
        with self.assertRaises(ValueError):
            builder.parse_yolo_line("0 1.2 0.5 0.1 0.1", 1)

    def test_generated_manifest_invariants(self):
        manifest = PROJECT_ROOT / "data" / "manifests" / "batch_500.jsonl"
        self.assertTrue(manifest.is_file(), "run build_manifest.py first")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 500)
        self.assertEqual({record["scene_seed"] for record in records}, {101, 202, 303, 404, 505})
        self.assertTrue(all(record["split"] == "unassigned" for record in records))
        self.assertEqual(len({record["sample_id"] for record in records}), 500)

    def test_padded_crop_is_clipped(self):
        self.assertEqual(
            candidate_builder.padded_crop_xyxy([0.0, 10.0, 20.0, 30.0], 640, 480, 0.15),
            [0.0, 7.0, 23.0, 33.0],
        )

    def test_generated_candidate_manifest_invariants(self):
        manifest = PROJECT_ROOT / "data" / "manifests" / "batch_500.candidates.jsonl"
        self.assertTrue(manifest.is_file(), "run build_candidate_manifest.py first")
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 543)
        counts = {}
        for record in records:
            counts[record["target_label"]] = counts.get(record["target_label"], 0) + 1
            self.assertEqual(record["split"], "unassigned")
            crop = record["padded_crop_xyxy_pixels"]
            if crop is not None:
                self.assertGreaterEqual(crop[0], 0.0)
                self.assertGreaterEqual(crop[1], 0.0)
                self.assertLessEqual(crop[2], record["width"])
                self.assertLessEqual(crop[3], record["height"])
        self.assertEqual(
            counts,
            {
                "danger_red_sphere": 200,
                "distractor_red_box": 112,
                "distractor_green_sphere": 114,
                "background_or_unknown": 117,
            },
        )

    def test_generated_pilot_protocol_invariants(self):
        protocol = json.loads((PILOT_SPLITS / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["total_image_count"], 500)
        self.assertEqual(protocol["eligible_image_count"], 401)
        self.assertEqual(protocol["quarantined_image_count"], 99)
        self.assertEqual(len(protocol["folds"]), 5)
        self.assertTrue(protocol["pooled_out_of_fold_all_expected_classes_present"])
        fold_02 = next(fold for fold in protocol["folds"] if fold["fold_id"] == "fold_02")
        self.assertEqual(fold_02["missing_test_classes"], ["distractor_red_box"])
        errors, warnings = split_validator.validate(PILOT_SPLITS)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(warnings), 2)

    def test_pilot_generation_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            for output in (first, second):
                split_builder.build(
                    PILOT_CONFIG,
                    IMAGE_MANIFEST,
                    CANDIDATE_MANIFEST,
                    NEAR_DUPLICATES,
                    output,
                )
            self.assertEqual(tree_bytes(first), tree_bytes(second))
            self.assertEqual(len(tree_bytes(first)), 37)

    def test_invalid_fold_rotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "invalid.json"
            config = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
            config["folds"][1]["test_seed"] = config["folds"][0]["test_seed"]
            write_json(config_path, config)
            with self.assertRaises(ValueError):
                split_builder.build(
                    config_path,
                    IMAGE_MANIFEST,
                    CANDIDATE_MANIFEST,
                    NEAR_DUPLICATES,
                    Path(temporary) / "output",
                )

    def test_mandatory_policies_are_fail_closed(self):
        mutations = {
            "exact quarantine": lambda config: config["quarantine_policy"].update(
                exclude_all_images_in_exact_duplicate_groups=False
            ),
            "cross-seed dHash quarantine": lambda config: config["quarantine_policy"].update(
                exclude_all_images_in_cross_seed_dhash_pairs=False
            ),
            "near-duplicate interpretation": lambda config: config["quarantine_policy"].update(
                near_duplicates_are_candidates_not_proven_duplicates=False
            ),
            "dHash threshold": lambda config: config["quarantine_policy"].update(
                dhash_hamming_threshold=5
            ),
            "per-fold class support": lambda config: config["reporting_policy"].update(
                per_fold_metrics_require_nonzero_class_support=False
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                config_path = Path(temporary) / "invalid.json"
                config = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
                mutate(config)
                write_json(config_path, config)
                with self.assertRaises(ValueError):
                    split_builder.build(
                        config_path,
                        IMAGE_MANIFEST,
                        CANDIDATE_MANIFEST,
                        NEAR_DUPLICATES,
                        Path(temporary) / "output",
                    )
                errors, _ = split_validator.validate(PILOT_SPLITS, config_path=config_path)
                self.assertTrue(
                    any("mandatory" in error for error in errors),
                    f"validator accepted disabled mandatory policy: {label}",
                )

    def test_validator_rejects_corruption_matrix(self):
        def remove_quarantine(root: Path):
            path = root / "quarantine.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            write_jsonl(path, records[1:])

        def corrupt_candidate_image(root: Path):
            path = root / "fold_01" / "test_candidates.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            records[0]["image_sha256"] = "0" * 64
            write_jsonl(path, records)

        def remove_candidate(root: Path):
            path = root / "fold_01" / "train_candidates.jsonl"
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            write_jsonl(path, records[1:])

        def stale_source_hash(root: Path):
            path = root / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            protocol["source_files"]["near_duplicates"]["sha256"] = "0" * 64
            write_json(path, protocol)

        def falsify_fold_support(root: Path):
            metadata_path = root / "fold_02" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["missing_test_classes"] = []
            metadata["per_fold_four_class_macro_f1_valid"] = True
            write_json(metadata_path, metadata)

        def allow_generalization_claim(root: Path):
            path = root / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            protocol["reporting_policy"]["formal_generalization_claim_allowed"] = True
            write_json(path, protocol)

        def duplicate_protocol_fold(root: Path):
            path = root / "protocol.json"
            protocol = json.loads(path.read_text(encoding="utf-8"))
            protocol["folds"].append(dict(protocol["folds"][0]))
            write_json(path, protocol)

        mutations = {
            "incomplete quarantine": remove_quarantine,
            "candidate/image mismatch": corrupt_candidate_image,
            "missing candidate": remove_candidate,
            "stale source hash": stale_source_hash,
            "falsified fold support": falsify_fold_support,
            "generalization flag": allow_generalization_claim,
            "duplicate protocol fold": duplicate_protocol_fold,
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                copied = Path(temporary) / "pilot_v1"
                shutil.copytree(PILOT_SPLITS, copied)
                mutate(copied)
                errors, _ = split_validator.validate(copied)
                self.assertTrue(errors, f"validator accepted corruption: {label}")


if __name__ == "__main__":
    unittest.main()
