from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_replay_storyboard import validate_storyboard  # noqa: E402


EXPECTED_ARTIFACT_PATHS = {
    "configs/replay_protocol_v1.json",
    "configs/replay_event_contract_v1.json",
    "experiments/replay_v1/outputs/replay_manifest.jsonl",
    "experiments/replay_v1/outputs/danger_detection_events.jsonl",
    "experiments/replay_v1/outputs/decision_trace.jsonl",
    "experiments/replay_v1/outputs/replay_summary.json",
    "demo/replay_v1_level_a.gif",
    "demo/replay_v1_storyboard_manifest.json",
    "replay-report-supplement.md",
    "deliverables/2024311281_孙驰_机器人感知LevelA回放增补报告.docx",
}
EXPECTED_ROBOT_SOURCE = {
    "branch": "WindySky-Final",
    "commit": "e26b48ebd6871eb5753ac4b7750726c36b05a65a",
    "access": "committed_tree_read_only",
}
EXPECTED_RESULTS_KEYS = {
    "frame_count", "detection_count", "state_counts", "confirmed_count",
    "reported_count", "world_coordinate_count", "truth_joined",
    "official_result_generated", "storyboard_cases",
}


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run_manifest(run_path: Path | None = None) -> None:
    if run_path is None:
        run_path = PROJECT_ROOT / "experiments" / "replay_v1" / "replay_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    expected_top_keys = {
        "schema_version", "run_id", "status", "scope", "robot_source",
        "artifacts", "results", "preserved_baseline", "limitations",
    }
    if set(run) != expected_top_keys:
        raise ValueError("replay run top-level schema changed")
    if run["schema_version"] != 1 or run["run_id"] != "robot_replay_level_a_v1":
        raise ValueError("unexpected replay run identity")
    if run["status"] != "complete_pending_git_anchor":
        raise ValueError("unexpected replay run status")
    if run["scope"] != "interface-compatible non-temporal Level-A development storyboard":
        raise ValueError("replay run scope changed")
    if run["robot_source"] != EXPECTED_ROBOT_SOURCE:
        raise ValueError("robot source binding changed")

    artifacts = run["artifacts"]
    if not isinstance(artifacts, list) or any(set(item) != {"path", "sha256", "size_bytes"} for item in artifacts):
        raise ValueError("replay artifact schema changed")
    paths = [item["path"] for item in artifacts]
    if len(paths) != len(set(paths)) or set(paths) != EXPECTED_ARTIFACT_PATHS:
        raise ValueError("replay artifact set changed or contains duplicates")
    for item in artifacts:
        path = PROJECT_ROOT / item["path"]
        content = path.read_bytes()
        if len(content) != item["size_bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"replay run artifact mismatch: {item['path']}")

    results = run["results"]
    if set(results) != EXPECTED_RESULTS_KEYS:
        raise ValueError("replay results schema changed")
    if set(results["state_counts"]) != {"HELD_UNLOCALIZED", "REJECTED", "SEARCHING"}:
        raise ValueError("replay state-count schema changed")
    summary = json.loads((PROJECT_ROOT / "experiments" / "replay_v1" / "outputs" / "replay_summary.json").read_text(encoding="utf-8"))
    for key in EXPECTED_RESULTS_KEYS - {"storyboard_cases"}:
        if summary[key] != results[key]:
            raise ValueError(f"replay summary mismatch: {key}")
    if any(results[key] != 0 for key in ("confirmed_count", "reported_count", "world_coordinate_count")):
        raise ValueError("Level A cannot claim confirmation, reporting, or world coordinates")
    if results["truth_joined"] is not False or results["official_result_generated"] is not False:
        raise ValueError("Level A truth or official-output boundary changed")

    validate_storyboard()
    storyboard = json.loads((PROJECT_ROOT / "demo" / "replay_v1_storyboard_manifest.json").read_text(encoding="utf-8"))
    if storyboard["frame_count"] != results["storyboard_cases"]:
        raise ValueError("storyboard case count mismatch")

    preserved = run["preserved_baseline"]
    if set(preserved) != {"report_path", "report_sha256"}:
        raise ValueError("preserved baseline schema changed")
    if preserved != {
        "report_path": "deliverables/2024311281_孙驰_期末报告_RF-DETR_SmolVLM.docx",
        "report_sha256": "464d267fffd8d7ba5fd442631dd7ffddd81fbe84a85edbc75a464a636a8d1bfb",
    }:
        raise ValueError("preserved baseline binding changed")
    report = PROJECT_ROOT / preserved["report_path"]
    if sha256_path(report) != preserved["report_sha256"]:
        raise ValueError("preserved report hash changed")
    if (PROJECT_ROOT / "results" / "detected_danger.json").exists():
        raise ValueError("Level A must not generate detected_danger.json")
    print("PASSED: exact replay artifact set, schemas, hashes, counts, storyboard, and fail-closed boundary verified")


if __name__ == "__main__":
    validate_run_manifest()
