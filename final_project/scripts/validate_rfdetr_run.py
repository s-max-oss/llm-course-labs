#!/usr/bin/env python3
"""Validate and fully recompute the sealed RF-DETR pilot evidence bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from evaluate_rfdetr_baseline import evaluate  # noqa: E402
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    sha256_file,
)


CANONICAL_MANIFEST = PROJECT_ROOT / "experiments" / "pilot_v1" / "rfdetr_run.json"
EVIDENCE_SEAL_COMMIT = "9ce17d0a5124852e14536b882b8c88d2c8635f34"
EVIDENCE_SEAL_PATHS = {
    "experiments/pilot_v1/rfdetr_run.json",
    "experiments/pilot_v1/rfdetr_checkpoint_provenance.json",
    "experiments/pilot_v1/results/rfdetr_fold_01.jsonl",
    "experiments/pilot_v1/results/rfdetr_fold_02.jsonl",
    "experiments/pilot_v1/results/rfdetr_fold_03.jsonl",
    "experiments/pilot_v1/results/rfdetr_fold_04.jsonl",
    "experiments/pilot_v1/results/rfdetr_fold_05.jsonl",
    "experiments/pilot_v1/results/rfdetr_pooled_metrics.json",
}
EXPECTED_RUN_ID = "rfdetr_pilot_cpu_full_v1"
EXPECTED_PRE_RUN_ANCHOR = "976f1effe1cb57bf5c43a39486b8ffd3f9b8f85e"
EXPECTED_PREFLIGHT_CODE_COMMIT = "a23baa0d343cdc5bb5496a1921e9d6722f8345d0"
EXPECTED_PREFLIGHT = {
    "path": "experiments/pilot_v1/rfdetr_preflight.json",
    "sha256": "96ac33686874aec3c2ee6aef38e2c08256657c00265549121dadc84ad25a05a3",
}
EXPECTED_SCOPE = (
    "fixed-checkpoint in-domain exploratory detector pilot with a duplicate-controlled "
    "evaluation corpus"
)
EXPECTED_BOUNDARY = [
    "results describe fixed-checkpoint in-domain exploratory performance and do not establish held-out detector generalization",
    "checkpoint training-data provenance is unavailable, so overlap with the 401 pilot images or near duplicates cannot be excluded",
    "the evaluation corpus controls exact and cross-seed perceptual duplicates internally, but this does not prove separation from checkpoint training data",
    "missing-label frames are an uncertain background convention rather than audited negatives",
    "green-wall predictions are auxiliary and unscored because no sealed green-wall boxes exist",
    "no ground-truth masks exist, so mask IoU is not reported",
    "routing thresholds must not be tuned on these pooled out-of-fold test results",
]


def _artifact(entry: dict[str, Any], label: str) -> Path:
    path = (PROJECT_ROOT / entry["path"]).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise BaselineContractError(f"{label} escapes the project root") from exc
    if not path.is_file():
        raise BaselineContractError(f"{label} does not exist: {path}")
    if entry.get("sha256") != sha256_file(path):
        raise BaselineContractError(f"{label} SHA-256 differs from sealed manifest")
    if "size_bytes" in entry and entry["size_bytes"] != path.stat().st_size:
        raise BaselineContractError(f"{label} size differs from sealed manifest")
    return path


def _require_git_ancestor(ancestor: str, descendant: str, label: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BaselineContractError(f"{label} is not a valid Git ancestor relation")


def _git_blob(commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise BaselineContractError(
            f"cannot read trusted Git evidence object {commit}:{relative_path}"
        )
    return completed.stdout


def _require_git_sealed_artifact(path: Path, relative_path: str, label: str) -> None:
    if relative_path not in EVIDENCE_SEAL_PATHS:
        raise BaselineContractError(f"{label} is not registered in the Git evidence seal")
    if path.read_bytes() != _git_blob(EVIDENCE_SEAL_COMMIT, relative_path):
        raise BaselineContractError(
            f"{label} differs from trusted evidence commit {EVIDENCE_SEAL_COMMIT}"
        )


def validate(manifest_path: Path = CANONICAL_MANIFEST) -> dict[str, Any]:
    _require_git_ancestor(EVIDENCE_SEAL_COMMIT, "HEAD", "RF-DETR evidence seal")
    _require_git_sealed_artifact(
        manifest_path,
        "experiments/pilot_v1/rfdetr_run.json",
        "RF-DETR run manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete_and_sealed":
        raise BaselineContractError("RF-DETR run manifest is not complete and sealed")
    if manifest.get("formal_generalization_claim_allowed") is not False:
        raise BaselineContractError("RF-DETR run manifest overclaims generalization")
    if manifest.get("run_id") != EXPECTED_RUN_ID:
        raise BaselineContractError("RF-DETR run ID differs from the sealed run")
    if manifest.get("pre_run_anchor_commit") != EXPECTED_PRE_RUN_ANCHOR:
        raise BaselineContractError("RF-DETR pre-run Git anchor differs from the sealed run")
    if manifest.get("scope_label") != EXPECTED_SCOPE:
        raise BaselineContractError("RF-DETR scope label differs from the sealed boundary")
    if manifest.get("interpretation_boundary") != EXPECTED_BOUNDARY:
        raise BaselineContractError("RF-DETR interpretation boundary differs from the sealed boundary")
    if manifest.get("preflight_manifest") != EXPECTED_PREFLIGHT:
        raise BaselineContractError("RF-DETR preflight entry differs from the trusted pre-run manifest")

    preflight_path = _artifact(manifest["preflight_manifest"], "RF-DETR preflight manifest")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed_before_full_inference":
        raise BaselineContractError("RF-DETR preflight was not passed before full inference")
    if preflight.get("working_tree_tracked_changes") is not False:
        raise BaselineContractError("RF-DETR preflight did not seal a clean tracked tree")
    if preflight.get("code_commit") != EXPECTED_PREFLIGHT_CODE_COMMIT:
        raise BaselineContractError("RF-DETR preflight code commit differs from the trusted commit")
    _require_git_ancestor(EXPECTED_PREFLIGHT_CODE_COMMIT, EXPECTED_PRE_RUN_ANCHOR, "preflight code")
    _require_git_ancestor(EXPECTED_PRE_RUN_ANCHOR, "HEAD", "pre-run anchor")
    sealed_inputs = preflight.get("sealed_inputs")
    if not isinstance(sealed_inputs, dict):
        raise BaselineContractError("RF-DETR preflight has no sealed inputs")
    for name, entry in sealed_inputs.items():
        _artifact(entry, f"RF-DETR preflight input {name}")

    checkpoint_path = _artifact(manifest["checkpoint"], "RF-DETR checkpoint")
    if manifest["checkpoint"] != sealed_inputs.get("checkpoint"):
        raise BaselineContractError("RF-DETR run checkpoint differs from the preflight checkpoint")
    baseline_config = json.loads((PROJECT_ROOT / "configs" / "baselines.json").read_text(encoding="utf-8"))
    if baseline_config["rfdetr"]["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]:
        raise BaselineContractError("RF-DETR run checkpoint differs from the frozen production config")
    provenance_path = _artifact(manifest["checkpoint_provenance"], "RF-DETR provenance audit")
    _require_git_sealed_artifact(
        provenance_path,
        "experiments/pilot_v1/rfdetr_checkpoint_provenance.json",
        "RF-DETR provenance audit",
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("checkpoint_sha256") != manifest["checkpoint"]["sha256"]
        or provenance.get("workspace_search", {}).get("overlap_status") != "unknown"
        or provenance.get("reporting_decision", {}).get("prohibited_label")
        != "held-out detector generalization or leakage-free detector evaluation"
    ):
        raise BaselineContractError("RF-DETR provenance audit differs from the reporting boundary")
    if manifest["execution"] != {
        "started_at": "2026-11-08T15:01:22+08:00",
        "completed_at": "2026-11-08T15:06:57+08:00",
        "device": "cpu",
        "expected_image_count": 401,
        "completed_image_count": 401,
        "error_count": 0,
        "result_paths_were_empty_before_run": True,
        "ground_truth_available_to_inference": False,
        "threshold": 0.5,
        "resolution": 432,
    }:
        raise BaselineContractError("RF-DETR execution summary differs from sealed contract")

    raw_entries = manifest.get("raw_results")
    expected_folds = [f"fold_{number:02d}" for number in range(1, 6)]
    if not isinstance(raw_entries, list) or [entry.get("fold_id") for entry in raw_entries] != expected_folds:
        raise BaselineContractError("RF-DETR manifest must contain five ordered fold results")
    result_paths = []
    total_records = 0
    for entry in raw_entries:
        fold_id = entry["fold_id"]
        path = _artifact(entry, f"RF-DETR {fold_id} result")
        _require_git_sealed_artifact(path, entry["path"], f"RF-DETR {fold_id} result")
        records = load_jsonl(path)
        if len(records) != entry.get("record_count"):
            raise BaselineContractError(f"RF-DETR {fold_id} record count differs")
        if not records or {record.get("fold_id") for record in records} != {fold_id}:
            raise BaselineContractError(f"RF-DETR {fold_id} result mixes or omits folds")
        result_paths.append(path)
        total_records += len(records)
    if total_records != manifest["execution"]["completed_image_count"]:
        raise BaselineContractError("RF-DETR raw result total differs from execution summary")

    metrics_path = _artifact(manifest["evaluation"], "RF-DETR pooled metrics")
    _require_git_sealed_artifact(
        metrics_path,
        manifest["evaluation"]["path"],
        "RF-DETR pooled metrics",
    )
    stored_report = json.loads(metrics_path.read_text(encoding="utf-8"))
    recomputed_report = evaluate(
        result_paths,
        PROJECT_ROOT / "configs" / "rfdetr_evaluation_protocol.json",
        PROJECT_ROOT / "experiments" / "pilot_v1" / "rfdetr_evaluation_protocol.lock.json",
        PROJECT_ROOT / "configs" / "baselines.json",
        PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs",
        PROJECT_ROOT / "data" / "splits" / "pilot_v1",
    )
    if recomputed_report != stored_report:
        raise BaselineContractError("RF-DETR stored metrics differ from full evaluator recomputation")

    metrics = stored_report["metrics"]
    danger = metrics["danger_object_detection"]
    expected_summary = {
        "bbox_iou_threshold": metrics["bbox_iou_threshold"],
        "three_target_class_macro_f1": metrics["three_target_class_macro_f1"],
        "danger_object_precision": danger["precision"],
        "danger_object_recall": danger["recall"],
        "danger_object_f1": danger["f1"],
        "danger_object_false_positive": danger["false_positive"],
        "danger_object_false_negative": danger["false_negative"],
        "danger_image_presence_recall": metrics["danger_image_presence"]["recall"],
        "background_convention_image_count": metrics["background_convention_image_count"],
        "target_predictions_on_background_convention_images": metrics[
            "target_predictions_on_background_convention_images"
        ],
        "latency_ms_p50": metrics["latency_ms"]["p50"],
        "latency_ms_p95": metrics["latency_ms"]["p95"],
    }
    for key, expected in expected_summary.items():
        if manifest["evaluation"].get(key) != expected:
            raise BaselineContractError(f"RF-DETR run summary differs from metrics: {key}")
    if manifest["evaluation"].get("byte_recompute_equal") is not True:
        raise BaselineContractError("RF-DETR run does not record byte-identical recomputation")
    if manifest["evaluation"].get("byte_recompute_sha256") != sha256_file(metrics_path):
        raise BaselineContractError("RF-DETR byte-recompute hash differs from stored metrics")
    if checkpoint_path.stat().st_size != 143429329:
        raise BaselineContractError("RF-DETR checkpoint size differs from production evidence")
    return stored_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    args = parser.parse_args()
    try:
        report = validate(args.manifest.resolve())
    except (BaselineContractError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "PASSED: sealed RF-DETR run recomputed exactly; "
        f"images={report['metrics']['image_count']} "
        f"macro_f1={report['metrics']['three_target_class_macro_f1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
