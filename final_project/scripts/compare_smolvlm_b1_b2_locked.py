#!/usr/bin/env python3
"""Recompute B1 on the B2 locked subset and compare it with frozen B2 output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_vlm_baselines.b2 import expected_partition, load_frozen_b2_protocol  # noqa: E402
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    sha256_file,
    write_json,
)
from robot_vlm_baselines.metrics import classification_metrics  # noqa: E402


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BaselineContractError(f"cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_locked_b2_artifacts(
    project_root: Path,
    run_manifest_path: Path,
    b2_metrics_path: Path,
) -> Path:
    canonical_manifest = project_root / "experiments" / "pilot_v1" / "smolvlm_b2_locked_run.json"
    if run_manifest_path.resolve() != canonical_manifest.resolve():
        raise BaselineContractError("B2 locked run manifest is not the canonical artifact")
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != "smolvlm_b2_exploratory_v1"
        or manifest.get("locked_evaluation_executed") is not True
        or manifest.get("b2_a_locked_evaluation_executed") is not False
    ):
        raise BaselineContractError("invalid B2 locked run manifest state")
    run_root = run_manifest_path.parent
    expected_paths = {
        "predictions": "results/b2_b_locked_evaluation.jsonl",
        "metrics": "results/b2_b_locked_evaluation_metrics.json",
    }
    resolved: dict[str, Path] = {}
    for name, expected_path in expected_paths.items():
        artifact = manifest.get("artifacts", {}).get(name, {})
        if artifact.get("path") != expected_path:
            raise BaselineContractError(f"B2 locked {name} path differs from the sealed manifest")
        path = run_root / expected_path
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise BaselineContractError(f"B2 locked {name} differs from the sealed run")
        resolved[name] = path
    if b2_metrics_path.resolve() != resolved["metrics"].resolve():
        raise BaselineContractError("B2 locked metrics are not the sealed artifact")
    return resolved["predictions"]


def compare(
    protocol_path: Path,
    lock_path: Path,
    split_root: Path,
    b1_run_path: Path,
    b2_metrics_path: Path,
) -> dict[str, Any]:
    protocol, _ = load_frozen_b2_protocol(PROJECT_ROOT, protocol_path, lock_path)
    parent_jobs_path = PROJECT_ROOT / protocol["parent_artifacts"]["baseline_jobs_protocol"]["path"]
    parent_jobs = json.loads(parent_jobs_path.read_text(encoding="utf-8"))
    truth: dict[str, dict[str, Any]] = {}
    truth_sources = []
    for fold_id in [f"fold_{index:02d}" for index in range(1, 6)]:
        path = split_root / fold_id / "test_candidates.jsonl"
        digest = sha256_file(path)
        source_key = f"{fold_id}_test_candidates"
        if parent_jobs["source_files"][source_key]["sha256"] != digest:
            raise BaselineContractError(f"sealed truth hash mismatch: {source_key}")
        truth_sources.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest})
        for candidate in load_jsonl(path):
            if expected_partition(protocol, int(candidate["scene_seed"])) == "locked_evaluation":
                candidate_id = candidate["candidate_id"]
                if candidate_id in truth:
                    raise BaselineContractError(f"duplicate locked truth candidate: {candidate_id}")
                truth[candidate_id] = candidate
    if len(truth) != protocol["partition"]["locked_evaluation"]["candidate_count"]:
        raise BaselineContractError("locked truth count differs from frozen protocol")

    canonical_b1_run = PROJECT_ROOT / protocol["parent_artifacts"]["b1_run_manifest"]["path"]
    if b1_run_path.resolve() != canonical_b1_run.resolve():
        raise BaselineContractError("B1 run manifest is not the frozen canonical artifact")
    if sha256_file(b1_run_path) != protocol["parent_artifacts"]["b1_run_manifest"]["sha256"]:
        raise BaselineContractError("B1 run manifest differs from the frozen protocol")
    b1_run = json.loads(b1_run_path.read_text(encoding="utf-8"))
    if b1_run.get("baseline_id") != "b1_smolvlm_256m_zero_shot_oracle_crop":
        raise BaselineContractError("unexpected B1 run manifest")
    b1_results: dict[str, dict[str, Any]] = {}
    b1_sources = []
    for artifact in b1_run["prediction_files"]:
        path = PROJECT_ROOT / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise BaselineContractError(f"B1 result hash mismatch: {path}")
        b1_sources.append({"path": artifact["path"], "sha256": artifact["sha256"]})
        for result in load_jsonl(path):
            candidate_id = result["candidate_id"]
            if candidate_id in b1_results:
                raise BaselineContractError(f"duplicate B1 candidate: {candidate_id}")
            b1_results[candidate_id] = result
    if not set(truth).issubset(b1_results):
        raise BaselineContractError("B1 results do not cover the B2 locked subset")
    b1_evaluator = _load_script_module(
        "_b1_full_evaluator", PROJECT_ROOT / "scripts" / "evaluate_smolvlm_baseline.py"
    )
    b1_evaluator.evaluate(
        [PROJECT_ROOT / artifact["path"] for artifact in b1_run["prediction_files"]],
        split_root,
        PROJECT_ROOT / "configs" / "baselines.json",
        parent_jobs_path,
    )

    pairs = []
    latencies = []
    for candidate_id, candidate in truth.items():
        result = b1_results[candidate_id]
        if result.get("source_sample_id") != candidate["source_sample_id"]:
            raise BaselineContractError(f"{candidate_id}: B1 identity differs from locked truth")
        if result.get("image_sha256") != candidate["image_sha256"]:
            raise BaselineContractError(f"{candidate_id}: B1 image differs from locked truth")
        if result.get("status") not in {"ok", "invalid_output", "error"}:
            raise BaselineContractError(f"{candidate_id}: invalid B1 status")
        prediction = result.get("predicted_label") if result["status"] == "ok" else None
        latency = result.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or latency < 0
        ):
            raise BaselineContractError(f"{candidate_id}: invalid B1 latency")
        pairs.append((candidate["target_label"], prediction))
        latencies.append(float(latency))
    b1_metrics = classification_metrics(pairs, latencies)

    b2_results_path = validate_locked_b2_artifacts(
        PROJECT_ROOT,
        PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_locked_run.json",
        b2_metrics_path,
    )
    b2_report = json.loads(b2_metrics_path.read_text(encoding="utf-8"))
    if (
        b2_report.get("protocol_id") != protocol["protocol_id"]
        or b2_report.get("variant") != "b2_b_fixed_candidate_scoring"
        or b2_report.get("partition") != "locked_evaluation"
    ):
        raise BaselineContractError("unexpected B2 locked report")
    b2_metrics = b2_report["metrics"]
    if b2_metrics.get("sample_count") != len(truth):
        raise BaselineContractError("B2 metric count differs from locked truth")
    b2_results_source = b2_report.get("source_files", {}).get("results", {})
    if (
        (PROJECT_ROOT / str(b2_results_source.get("path", ""))).resolve()
        != b2_results_path.resolve()
        or sha256_file(b2_results_path) != b2_results_source.get("sha256")
    ):
        raise BaselineContractError("B2 locked result hash mismatch")
    b2_evaluator = _load_script_module(
        "_b2_full_evaluator", PROJECT_ROOT / "scripts" / "evaluate_smolvlm_b2.py"
    )
    recomputed_b2_report = b2_evaluator.evaluate(
        PROJECT_ROOT / "experiments" / "pilot_v1" / "b2_jobs" / "locked_evaluation.jsonl",
        b2_results_path,
        "b2_b",
        "locked_evaluation",
        protocol_path,
        lock_path,
        PROJECT_ROOT / "configs" / "baselines.json",
        split_root,
        PROJECT_ROOT / "experiments" / "pilot_v1" / "results" / "b2_b_development_gate.json",
    )
    if recomputed_b2_report != b2_report:
        raise BaselineContractError("B2 locked report differs from full evaluator recomputation")

    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "partition": "locked_evaluation",
        "scope_label": "same-251-candidate exploratory comparison",
        "confirmatory_claim_allowed": False,
        "deployment_eligible": False,
        "source_files": {
            "machine_protocol": {
                "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "b1_run_manifest": {
                "path": b1_run_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(b1_run_path),
            },
            "b1_predictions": b1_sources,
            "b2_locked_metrics": {
                "path": b2_metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(b2_metrics_path),
            },
            "truth_candidates": truth_sources,
        },
        "b1_strict_generation": b1_metrics,
        "b2_fixed_candidate_scoring": b2_metrics,
        "delta_b2_minus_b1": {
            "prediction_coverage": b2_metrics["prediction_coverage"] - b1_metrics["prediction_coverage"],
            "accuracy": b2_metrics["accuracy"] - b1_metrics["accuracy"],
            "macro_f1": b2_metrics["macro_f1"] - b1_metrics["macro_f1"],
            "danger_red_sphere_recall": (
                b2_metrics["danger_binary"]["recall"] - b1_metrics["danger_binary"]["recall"]
            ),
        },
        "interpretation": [
            "B2 removes B1's strict-output coverage failure and separates all three object classes on this subset.",
            "B2 still fails every background example, so it is not a standalone presence detector or deployable four-class system.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json")
    parser.add_argument(
        "--lock", type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json",
    )
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data" / "splits" / "pilot_v1")
    parser.add_argument(
        "--b1-run", type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_zero_shot_run.json",
    )
    parser.add_argument(
        "--b2-metrics", type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "results" / "b2_b_locked_evaluation_metrics.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"ERROR: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        report = compare(
            args.protocol.resolve(), args.lock.resolve(), args.split_root.resolve(),
            args.b1_run.resolve(), args.b2_metrics.resolve(),
        )
    except (BaselineContractError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_json(args.output.resolve(), report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
