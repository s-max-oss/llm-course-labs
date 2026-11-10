#!/usr/bin/env python3
"""Validate and fully recompute the stopped cascade development run."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from evaluate_cascade_feasibility import evaluate  # noqa: E402
from robot_vlm_baselines.cascade import CASCADE_RESULT_FIELDS  # noqa: E402
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    sha256_file,
)


CANONICAL_MANIFEST = PROJECT_ROOT / "experiments" / "pilot_v1" / "cascade_development_run.json"
EXPECTED_PRE_RUN_COMMIT = "15c0fa63761d6d1952933cba27bd0f568b890d0b"
EVIDENCE_SEAL_COMMIT = "636f828f2b75bcecede6aff2bffcf45c50835310"
EVIDENCE_SEAL_PATHS = {
    "experiments/pilot_v1/cascade_development_run.json",
    "experiments/pilot_v1/results/cascade_b2_development.jsonl",
    "experiments/pilot_v1/results/cascade_feasibility_development_metrics.json",
}


def _artifact(entry: dict[str, Any], label: str) -> Path:
    path = (PROJECT_ROOT / entry["path"]).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise BaselineContractError(f"{label} escapes project root") from exc
    if not path.is_file() or sha256_file(path) != entry["sha256"]:
        raise BaselineContractError(f"{label} differs from the sealed run manifest")
    if "size_bytes" in entry and path.stat().st_size != entry["size_bytes"]:
        raise BaselineContractError(f"{label} size differs from the sealed run manifest")
    return path


def _require_git_sealed_artifact(path: Path, relative_path: str, label: str) -> None:
    if relative_path not in EVIDENCE_SEAL_PATHS:
        raise BaselineContractError(f"{label} is not registered in the cascade Git evidence seal")
    completed = subprocess.run(
        ["git", "show", f"{EVIDENCE_SEAL_COMMIT}:{relative_path}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or path.read_bytes() != completed.stdout:
        raise BaselineContractError(
            f"{label} differs from trusted cascade evidence commit {EVIDENCE_SEAL_COMMIT}"
        )


def validate(manifest_path: Path = CANONICAL_MANIFEST) -> dict[str, Any]:
    seal_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EVIDENCE_SEAL_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if seal_ancestor.returncode != 0:
        raise BaselineContractError("cascade evidence seal is not an ancestor of HEAD")
    _require_git_sealed_artifact(
        manifest_path,
        "experiments/pilot_v1/cascade_development_run.json",
        "cascade run manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != "rfdetr_smolvlm_cascade_development_v1"
        or manifest.get("status") != "complete_and_stopped_by_frozen_gate"
        or manifest.get("confirmatory_claim_allowed") is not False
        or manifest.get("deployment_claim_allowed") is not False
        or manifest.get("pre_run_code_commit") != EXPECTED_PRE_RUN_COMMIT
    ):
        raise BaselineContractError("cascade run identity or reporting boundary differs")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_PRE_RUN_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise BaselineContractError("cascade pre-run commit is not an ancestor of HEAD")
    protocol_path = _artifact(manifest["protocol"], "cascade protocol")
    jobs_manifest_path = _artifact(manifest["jobs_manifest"], "cascade jobs manifest")
    jobs_path = _artifact(manifest["jobs"], "cascade jobs")
    inference_path = _artifact(manifest["inference_code"], "cascade inference code")
    evaluation_path = _artifact(manifest["evaluation_code"], "cascade evaluation code")
    raw_path = _artifact(manifest["raw_results"], "cascade raw results")
    metrics_path = _artifact(manifest["evaluation"], "cascade metrics")
    _require_git_sealed_artifact(
        raw_path,
        "experiments/pilot_v1/results/cascade_b2_development.jsonl",
        "cascade raw results",
    )
    _require_git_sealed_artifact(
        metrics_path,
        "experiments/pilot_v1/results/cascade_feasibility_development_metrics.json",
        "cascade metrics",
    )
    del protocol_path, jobs_manifest_path, inference_path, evaluation_path

    jobs = load_jsonl(jobs_path)
    results = load_jsonl(raw_path)
    if len(jobs) != 24 or len(results) != 24:
        raise BaselineContractError("cascade sealed job/result count differs from 24")
    if any(set(item) != CASCADE_RESULT_FIELDS for item in results):
        raise BaselineContractError("cascade sealed result fields differ from contract")
    if Counter(item["status"] for item in results) != Counter({"ok": 24}):
        raise BaselineContractError("cascade sealed results are not 24/24 successful")
    if manifest["execution"] != {
        "started_at": "2026-11-08T20:13:28+08:00",
        "completed_at": "2026-11-08T20:15:44+08:00",
        "expected_job_count": 24,
        "completed_job_count": 24,
        "error_count": 0,
        "output_atomically_reserved_before_model_load": True,
        "ground_truth_available_to_job_generation": False,
        "ground_truth_available_to_inference": False,
        "oracle_boxes_used": False,
        "locked_seeds_executed": False,
    }:
        raise BaselineContractError("cascade execution boundary differs from sealed run")

    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    recomputed = evaluate(raw_path)
    if recomputed != stored:
        raise BaselineContractError("cascade stored metrics differ from full recomputation")
    expected_summary = {
        "rf_only_macro_f1": stored["rf_only"]["three_target_class_macro_f1"],
        "primary_cascade_macro_f1": stored["primary_safety_preserving_cascade"][
            "three_target_class_macro_f1"
        ],
        "rf_only_danger_recall": stored["rf_only"]["danger_recall"],
        "primary_cascade_danger_recall": stored["primary_safety_preserving_cascade"][
            "danger_recall"
        ],
        "rf_only_target_false_positive_total": stored["rf_only"][
            "target_false_positive_total"
        ],
        "primary_cascade_target_false_positive_total": stored[
            "primary_safety_preserving_cascade"
        ]["target_false_positive_total"],
        "routed_target_proposal_fraction": stored["routing"]["routed_target_proposal_fraction"],
        "b2_latency_ms_p50": stored["b2_latency_ms"]["p50"],
        "b2_latency_ms_p95": stored["b2_latency_ms"]["p95"],
        "all_continuation_gates_passed": stored["continuation_gates"]["all_passed"],
    }
    if manifest["evaluation"] != {**manifest["evaluation"], **expected_summary}:
        raise BaselineContractError("cascade run summary differs from recomputed metrics")
    if stored["decision"] != "stop_cascade_and_recommend_rfdetr_only":
        raise BaselineContractError("cascade evaluator did not stop the cascade")
    if stored["continuation_gates"] != {
        "danger_recall_not_decreased": True,
        "three_target_macro_f1_strictly_increased": False,
        "target_false_positive_total_not_increased": True,
        "routed_target_proposal_fraction_at_most_0_15": True,
        "all_passed": False,
    }:
        raise BaselineContractError("cascade continuation gates differ from sealed outcome")
    if manifest["decision"].get("outcome") != stored["decision"] or manifest["decision"].get(
        "locked_evaluation_authorized"
    ) is not False:
        raise BaselineContractError("cascade run manifest does not enforce the stop decision")

    changes = sum(item["detector_label"] != item["predicted_label"] for item in results)
    pairs = Counter((item["detector_label"], item["predicted_label"]) for item in results)
    margins = [float(item["margin"]) for item in results]
    observed = manifest["observed_routing_behavior"]
    if observed != {
        "detector_to_b2_label_changes": changes,
        "danger_red_sphere_to_same_label": pairs[("danger_red_sphere", "danger_red_sphere")],
        "distractor_green_sphere_to_same_label": pairs[
            ("distractor_green_sphere", "distractor_green_sphere")
        ],
        "red_box_proposals_routed": sum(item["detector_label"] == "distractor_red_box" for item in results),
        "background_predictions": sum(item["predicted_label"] == "background_or_unknown" for item in results),
        "margin_min": min(margins),
        "margin_p50": statistics.median(margins),
        "margin_max": max(margins),
    }:
        raise BaselineContractError("cascade observed routing behavior differs from raw results")
    return stored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST)
    args = parser.parse_args()
    try:
        report = validate(args.manifest.resolve())
    except (BaselineContractError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "PASSED: stopped cascade development run recomputed exactly; "
        f"jobs={report['routing']['job_count']} "
        f"macro_f1={report['primary_safety_preserving_cascade']['three_target_class_macro_f1']:.6f} "
        f"continue={report['continuation_gates']['all_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
