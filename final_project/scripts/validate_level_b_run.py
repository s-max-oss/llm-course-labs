#!/usr/bin/env python3
"""Validate the reclassified Level-B v2 Gazebo-pose-oracle debug run."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROBOT_BRANCH = "WindySky-Final"
EXPECTED_ROBOT_COMMIT = "e26b48ebd6871eb5753ac4b7750726c36b05a65a"
EXPECTED_CORRECTION_SHA256 = "6275bc4ba8c4f223aeebba7ef9d7de219d677d31b99d4953754ca9079beaf770"
EXPECTED_DEPLOYMENT_MANIFEST_SHA256 = "bde69a9eed9ca85255d48e57d0389fc523417ea7694da05a58cb97a09a81bc73"
EXPECTED_GIF_SHA256 = "f4ecd0b0ec382265e042a7b0dfa670f0607848f4465a07aafdfbb3d37af6f192"
EXPECTED_REPORT_SHA256 = "8ae0d5edc0e61762b7822139b1a2102809071a4d653277e198eedc70b9f17190"
EXPECTED_GATES = {
    "G0_data_sufficiency": "passed",
    "G1_contract": "passed",
    "G2_truth_isolation": "failed_pose_ground_truth_present_reclassified_debug_only",
    "G3_spatial_quality": "passed",
    "G4_determinism": "passed",
    "G5_system_value": "passed",
}
POSE_TRUTH_TOPIC = "/Odometry_gazebo"
EXPECTED_SUPPORTED_CLAIMS = [
    "The pinned detector ran on synchronized RGB frames without danger labels or danger coordinates.",
    "RGB-D back-projection, depth-quality filtering, temporal confirmation, and official-format result writing completed deterministically.",
    "The reported world coordinate and 0.1238 m matched error were obtained with Gazebo pose ground truth and are debug-chain results, not autonomous-localization performance.",
]
EXPECTED_PROHIBITED_CLAIMS = [
    "truth-isolated spatial replay",
    "autonomous localization or autonomous spatial closure",
    "competition-compliant world-coordinate output",
    "online closed-loop navigation or full-match completion",
]
EXPECTED_VISUALIZATION_BOUNDARY = (
    "Gazebo-pose-oracle development debug replay; not autonomous localization, "
    "online control, competition compliance, or full-building coverage"
)
EXPECTED_VISIBLE_BANNER = "DEBUG REPLAY | TARGET-TRUTH-FREE | GAZEBO POSE ORACLE"
EXPECTED_REGISTRY_LINE = (
    "`/Odometry_gazebo` 和 `/ground_truth/*` 仅属于测试侧评分，导航运行禁止订阅。"
)
EXPECTED_ALGORITHM_LINE = (
    "`/Odometry_gazebo`、`/ground_truth/base_w`、`/ground_truth/base_trunk` 和 "
    "`/ground_truth/*_foot` 是 Gazebo 真值通道，不作为正式比赛算法输入。"
    "即使本地调试时能够看到这些话题，参赛算法也不得订阅或读取。"
)
REPLAY_FILES = (
    "danger_detection_events.jsonl",
    "decision_trace.jsonl",
    "detected_danger.json",
    "replay_summary.json",
)
EXPECTED_CLAIM_BOUNDARY = {
    "result_type": "single-seed single-view Gazebo-pose-oracle development debug replay",
    "formal_benchmark": False,
    "online_closed_loop_navigation": False,
    "production_competition_entry_executed": False,
    "danger_target_truth_available_to_inference": False,
    "pose_ground_truth_used": True,
    "pose_source": POSE_TRUTH_TOPIC,
    "autonomous_odometry_validated": False,
    "full_truth_isolation": False,
    "truth_ordering_cryptographically_proven": False,
    "production_competition_entry_blocker": (
        "the VMware guest has no NVIDIA/CUDA device, while the WindySky-Final "
        "production environment gate requires CUDA"
    ),
    "supported_claim": (
        "the pinned detector, synchronized RGB-D back-projection, depth filtering, "
        "temporal confirmation, and official-format result writer completed "
        "deterministically with Gazebo pose ground truth"
    ),
}


@lru_cache(maxsize=1)
def read_authoritative_contract(
    bundle_path: str,
    commit: str,
    interface_registry_path: str,
    algorithm_interfaces_path: str,
) -> tuple[str, str]:
    """Read the pinned robot contract from the sealed Git history bundle."""
    with tempfile.TemporaryDirectory() as directory:
        bare_repo = Path(directory) / "robot.git"
        clone = subprocess.run(
            ["git", "clone", "--bare", "--quiet", bundle_path, str(bare_repo)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if clone.returncode:
            raise RuntimeError(clone.stderr.strip() or "cannot open robot history bundle")

        def show(path: str) -> str:
            process = subprocess.run(
                ["git", f"--git-dir={bare_repo}", "show", f"{commit}:{path}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if process.returncode:
                raise RuntimeError(process.stderr.strip() or f"missing robot contract: {path}")
            return process.stdout

        return show(interface_registry_path), show(algorithm_interfaces_path)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(raw: str) -> Path:
    path = (PROJECT_ROOT / raw).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact escapes project root: {raw}") from exc
    return path


def extract_docx_text(path: Path) -> str:
    """Extract visible text from the main OOXML document without trusting report metadata."""
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(document_xml)
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "".join(node.text or "" for node in root.iter(f"{namespace}t"))


def validate_run(run_path: Path) -> list[str]:
    errors: list[str] = []
    run = json.loads(run_path.read_text(encoding="utf-8"))

    if run.get("status") != "completed_pose_oracle_debug_replay":
        errors.append("unexpected run status")
    if run.get("official_format_detected_danger_generated") is not True:
        errors.append("official-format result is not declared generated")
    claim = run.get("claim_boundary", {})
    if claim != EXPECTED_CLAIM_BOUNDARY:
        errors.append("claim boundary is missing or overstates the development replay")
    if run.get("gates") != EXPECTED_GATES:
        errors.append("Level-B gates do not preserve the failed truth-isolation finding")
    if run.get("research_integrity_correction", {}).get("sha256") != EXPECTED_CORRECTION_SHA256:
        errors.append("research-integrity correction hash is not the reviewed correction")
    if run.get("deployment", {}).get("manifest_sha256") != EXPECTED_DEPLOYMENT_MANIFEST_SHA256:
        errors.append("deployment manifest hash is not the reviewed deployment")
    if run.get("visualization", {}).get("sha256") != EXPECTED_GIF_SHA256:
        errors.append("visualization is not the reviewed disclosure-bearing GIF")
    if run.get("report", {}).get("sha256") != EXPECTED_REPORT_SHA256:
        errors.append("Level-B supplement is not the reviewed corrected report")

    correction = json.loads(
        project_path(run["research_integrity_correction"]["path"]).read_text(encoding="utf-8")
    )
    truth_boundaries = correction.get("truth_boundaries", {})
    authoritative_contract = correction.get("authoritative_contract", {})
    artifact_preservation = correction.get("artifact_preservation", {})
    if (
        correction.get("affected_run_id") != run.get("run_id")
        or correction.get("reclassified_status") != run.get("status")
        or correction.get("original_run_manifest_sha256")
        != "8b14cd69e5ac063978228f8f2b7d66685ca9ffe1bf2d2b9ad182feaa0ec632c9"
        or truth_boundaries.get("danger_target_labels_or_coordinates_available_during_inference")
        is not False
        or truth_boundaries.get("gazebo_pose_ground_truth_available_during_replay") is not True
        or truth_boundaries.get("pose_ground_truth_topic") != POSE_TRUTH_TOPIC
        or truth_boundaries.get("autonomous_odometry_validated") is not False
        or truth_boundaries.get("full_truth_isolation_achieved") is not False
        or correction.get("supported_claims") != EXPECTED_SUPPORTED_CLAIMS
        or correction.get("prohibited_claims") != EXPECTED_PROHIBITED_CLAIMS
    ):
        errors.append("research-integrity correction is incomplete or inconsistent")
    if (
        artifact_preservation.get("bag_predictions_and_replay_outputs_modified") is not False
        or artifact_preservation.get("interpretive_metadata_validator_visualization_and_report_corrected")
        is not True
        or artifact_preservation.get("original_run_manifest_hash_recorded_before_correction")
        is not True
        or artifact_preservation.get("original_run_manifest_bytes_separately_preserved") is not False
    ):
        errors.append("research-integrity correction misstates artifact preservation")
    if (
        authoritative_contract.get("repository_commit") != EXPECTED_ROBOT_COMMIT
        or run["deployment"].get("robot_branch") != EXPECTED_ROBOT_BRANCH
        or run["deployment"].get("robot_commit") != EXPECTED_ROBOT_COMMIT
        or authoritative_contract.get("interface_registry_path") != "docs/INTERFACE_REGISTRY.md"
        or authoritative_contract.get("interface_registry_line") != 20
        or authoritative_contract.get("algorithm_interfaces_path")
        != "docs/official/algorithm-interfaces.md"
        or authoritative_contract.get("algorithm_interfaces_line") != 60
    ):
        errors.append("research-integrity correction is not bound to the robot contract")

    deployment_manifest = json.loads(
        project_path(run["deployment"]["manifest_path"]).read_text(encoding="utf-8")
    )
    deployment_source = deployment_manifest.get("source", {})
    if (
        deployment_source.get("branch") != EXPECTED_ROBOT_BRANCH
        or deployment_source.get("commit") != EXPECTED_ROBOT_COMMIT
        or deployment_source.get("export_method") != "git archive of committed tree paths only"
        or deployment_source.get("external_worktree_modified") is not False
    ):
        errors.append("deployment source is not the reviewed committed WindySky-Final tree")
    bindings = [
        (
            run["research_integrity_correction"]["path"],
            run["research_integrity_correction"]["sha256"],
        ),
        (run["protocol"]["path"], run["protocol"]["sha256"]),
        (run["deployment"]["manifest_path"], run["deployment"]["manifest_sha256"]),
        (run["deployment"]["sensor_patch_path"], run["deployment"]["sensor_patch_sha256"]),
        (run["input"]["bag_path"], run["input"]["bag_sha256"]),
        (run["input"]["manifest_path"], run["input"]["manifest_sha256"]),
        (run["inference"]["path"], run["inference"]["sha256"]),
        (run["visualization"]["path"], run["visualization"]["sha256"]),
        (run["visualization"]["manifest_path"], run["visualization"]["manifest_sha256"]),
        (run["posthoc_evaluation"]["truth_path"], run["posthoc_evaluation"]["truth_sha256"]),
        (run["posthoc_evaluation"]["result_path"], run["posthoc_evaluation"]["result_sha256"]),
        (
            run["posthoc_evaluation"]["matched_position_audit_path"],
            run["posthoc_evaluation"]["matched_position_audit_sha256"],
        ),
        (run["report"]["path"], run["report"]["sha256"]),
        (run["report"]["original_report_path"], run["report"]["original_report_sha256"]),
        (deployment_manifest["archive"]["path"], deployment_manifest["archive"]["sha256"]),
        (
            deployment_manifest["history_bundle"]["path"],
            deployment_manifest["history_bundle"]["sha256"],
        ),
    ]
    bindings.extend((item["path"], item["sha256"]) for item in run["replay"]["artifacts"])
    for raw_path, expected_hash in bindings:
        try:
            path = project_path(raw_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"missing bound artifact: {raw_path}")
        elif sha256_path(path) != expected_hash:
            errors.append(f"artifact hash mismatch: {raw_path}")
    for section in ("archive", "history_bundle"):
        nested = deployment_manifest[section]
        path = project_path(nested["path"])
        if path.is_file() and path.stat().st_size != nested["size_bytes"]:
            errors.append(f"deployment size mismatch: {nested['path']}")

    history_bundle_path = project_path(deployment_manifest["history_bundle"]["path"])
    if history_bundle_path.is_file():
        try:
            registry_text, interfaces_text = read_authoritative_contract(
                str(history_bundle_path),
                authoritative_contract["repository_commit"],
                authoritative_contract["interface_registry_path"],
                authoritative_contract["algorithm_interfaces_path"],
            )
        except (KeyError, RuntimeError) as exc:
            errors.append(f"cannot verify authoritative robot contract: {exc}")
        else:
            registry_lines = registry_text.splitlines()
            interface_lines = interfaces_text.splitlines()
            if len(registry_lines) < 20 or registry_lines[19] != EXPECTED_REGISTRY_LINE:
                errors.append("interface registry line 20 does not match the pinned pose-truth rule")
            if len(interface_lines) < 60 or interface_lines[59] != EXPECTED_ALGORITHM_LINE:
                errors.append("algorithm interface line 60 does not match the pinned Gazebo-truth rule")

    archive_path = project_path(deployment_manifest["archive"]["path"])
    if archive_path.is_file():
        with tarfile.open(archive_path, "r") as archive:
            for path_key, hash_key in (
                ("evaluator_source_path", "evaluator_source_sha256"),
                ("evaluator_wrapper_path", "evaluator_wrapper_sha256"),
            ):
                member_path = run["posthoc_evaluation"][path_key]
                try:
                    member = archive.getmember(member_path)
                    extracted = archive.extractfile(member)
                    payload = extracted.read() if extracted is not None else b""
                except KeyError:
                    errors.append(f"evaluator is absent from committed archive: {member_path}")
                    continue
                canonical_payload = payload.replace(b"\r\n", b"\n")
                if hashlib.sha256(canonical_payload).hexdigest() != run["posthoc_evaluation"][hash_key]:
                    errors.append(f"evaluator source hash mismatch: {member_path}")

    protocol = json.loads(project_path(run["protocol"]["path"]).read_text(encoding="utf-8"))
    manifest = json.loads(project_path(run["input"]["manifest_path"]).read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "level_b_replay_v2":
        errors.append("protocol ID is not Level-B v2")
    if protocol.get("status") != "frozen_before_v2_acquisition_and_inference":
        errors.append("protocol was not frozen before v2 acquisition/inference")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        errors.append("input manifest/protocol mismatch")
    source_bag = manifest.get("source_bag", {})
    topics = source_bag.get("topics", [])
    if source_bag.get("ground_truth_topics_present") is not False:
        errors.append("legacy manifest target-truth flag changed unexpectedly")
    if any(name.startswith("/ground_truth") for name in topics):
        errors.append("danger-target ground-truth topic leaked into the bag")
    if POSE_TRUTH_TOPIC not in topics:
        errors.append("historical pose-ground-truth topic is missing from the bag")
    if protocol.get("acquisition", {}).get("runtime", {}).get("referee_odometry") is not True:
        errors.append("protocol no longer records referee odometry")
    if POSE_TRUTH_TOPIC not in protocol.get("acquisition", {}).get("required_topics", []):
        errors.append("protocol no longer binds the pose-ground-truth topic")
    run_input = run["input"]
    if (
        run_input.get("danger_target_truth_available_to_inference") is not False
        or run_input.get("pose_ground_truth_available_to_replay") is not True
        or run_input.get("pose_ground_truth_topic") != POSE_TRUTH_TOPIC
    ):
        errors.append("run does not declare both target-truth and pose-truth boundaries")

    visualization = json.loads(
        project_path(run["visualization"]["manifest_path"]).read_text(encoding="utf-8")
    )
    if visualization.get("artifact") != run["visualization"]["path"]:
        errors.append("visualization manifest artifact path mismatch")
    if visualization.get("artifact_sha256") != run["visualization"]["sha256"]:
        errors.append("visualization manifest artifact hash mismatch")
    if visualization.get("frame_count") != run["visualization"]["frames"] or len(
        visualization.get("frames", [])
    ) != run["visualization"]["frames"]:
        errors.append("visualization frame count mismatch")
    if (
        visualization.get("schema_version") != 2
        or visualization.get("danger_target_truth_used") is not False
        or visualization.get("pose_ground_truth_used") is not True
        or visualization.get("pose_source_topic") != POSE_TRUTH_TOPIC
        or visualization.get("visible_banner") != EXPECTED_VISIBLE_BANNER
        or visualization.get("claim_boundary") != EXPECTED_VISUALIZATION_BOUNDARY
        or run["visualization"].get("danger_target_truth_used") is not False
        or run["visualization"].get("pose_ground_truth_coordinates_displayed") is not True
    ):
        errors.append("visualization does not disclose its target-truth and pose-truth boundaries")
    if visualization.get("source_bindings") != {
        "input_manifest_sha256": run["input"]["manifest_sha256"],
        "events_sha256": next(
            item["sha256"]
            for item in run["replay"]["artifacts"]
            if item["path"].endswith("danger_detection_events.jsonl")
        ),
        "trace_sha256": next(
            item["sha256"]
            for item in run["replay"]["artifacts"]
            if item["path"].endswith("decision_trace.jsonl")
        ),
    }:
        errors.append("visualization source bindings mismatch")

    prediction_rows = [
        json.loads(line)
        for line in project_path(run["inference"]["path"]).read_text(encoding="utf-8").splitlines()
    ]
    if len(prediction_rows) != run["inference"]["frames"]:
        errors.append("inference frame count mismatch")
    if any(row.get("protocol_id") != protocol["protocol_id"] for row in prediction_rows):
        errors.append("prediction/protocol mismatch")

    artifact_by_name = {Path(item["path"]).name: item for item in run["replay"]["artifacts"]}
    if set(artifact_by_name) != set(REPLAY_FILES):
        errors.append("unexpected canonical replay artifact set")
    repeat_root = project_path(run["replay"]["determinism"]["repeat_path"])
    for name in REPLAY_FILES:
        first = project_path(artifact_by_name[name]["path"])
        repeat = repeat_root / name
        if not repeat.is_file() or sha256_path(first) != sha256_path(repeat):
            errors.append(f"non-deterministic replay artifact: {name}")

    with tempfile.TemporaryDirectory() as directory:
        rebuilt_roots = [Path(directory) / "rebuild_1", Path(directory) / "rebuild_2"]
        for rebuilt in rebuilt_roots:
            process = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "build_level_b_replay.py"),
                    "--manifest",
                    str(project_path(run["input"]["manifest_path"])),
                    "--predictions",
                    str(project_path(run["inference"]["path"])),
                    "--protocol",
                    str(project_path(run["protocol"]["path"])),
                    "--output-dir",
                    str(rebuilt),
                ],
                capture_output=True,
                text=True,
            )
            if process.returncode:
                errors.append(f"replay reconstruction failed: {process.stderr.strip() or process.stdout.strip()}")
                break
        else:
            for name in REPLAY_FILES:
                stored = project_path(artifact_by_name[name]["path"])
                first = rebuilt_roots[0] / name
                second = rebuilt_roots[1] / name
                if first.read_bytes() != second.read_bytes() or first.read_bytes() != stored.read_bytes():
                    errors.append(f"reconstructed replay differs: {name}")

    summary = json.loads(project_path(artifact_by_name["replay_summary.json"]["path"]).read_text(encoding="utf-8"))
    detected = json.loads(project_path(artifact_by_name["detected_danger.json"]["path"]).read_text(encoding="utf-8"))
    if summary.get("official_result_generated") is not True or summary.get("reported_danger_count", 0) < 1:
        errors.append("replay summary does not support an official-format danger result")
    if not detected.get("detected_danger_sources"):
        errors.append("official-format detected_danger.json is empty")

    evaluation_detected = project_path("experiments/level_b_v2/evaluation/detected_danger.json")
    if not evaluation_detected.is_file() or evaluation_detected.read_bytes() != project_path(
        artifact_by_name["detected_danger.json"]["path"]
    ).read_bytes():
        errors.append("post-hoc evaluator did not consume the finalized official result bytes")
    evaluation = json.loads(project_path(run["posthoc_evaluation"]["result_path"]).read_text(encoding="utf-8"))
    expected_metrics = {"truth_count": 2, "detected_count": 1, "correct": 1, "missed": 1, "false_alarms": 0}
    if any(evaluation.get("metrics", {}).get(key) != value for key, value in expected_metrics.items()):
        errors.append("post-hoc evaluation metrics mismatch")
    if run["posthoc_evaluation"].get("truth_attached_only_after_result_finalized") is not True:
        errors.append("truth attachment ordering is not declared")
    if run["posthoc_evaluation"].get("matched_position_error_source") != (
        "independent post-hoc spatial audit, not an official evaluator output field"
    ):
        errors.append("matched position error is misattributed")
    if run["posthoc_evaluation"].get("evaluator_commit") != EXPECTED_ROBOT_COMMIT:
        errors.append("post-hoc evaluator is not bound to the reviewed robot commit")

    truth = json.loads(project_path(run["posthoc_evaluation"]["truth_path"]).read_text(encoding="utf-8"))
    match_audit = json.loads(
        project_path(run["posthoc_evaluation"]["matched_position_audit_path"]).read_text(encoding="utf-8")
    )
    detected_position = detected["detected_danger_sources"][0]["position"]
    distances = [
        (
            math.dist(detected_position, source["position"]),
            source,
        )
        for source in truth["danger_sources"]
    ]
    distance, matched = min(distances, key=lambda item: item[0])
    if (
        match_audit.get("official_evaluator_metric") is not False
        or match_audit.get("matched_truth_id") != matched["id"]
        or match_audit.get("detected_position") != detected_position
        or round(distance, 4) != match_audit.get("distance_m")
        or round(distance, 4) != run["posthoc_evaluation"].get("matched_position_error_m")
    ):
        errors.append("independent matched-position audit mismatch")

    if archive_path.is_file():
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            evaluator_path = temp_root / "evaulate_danger.py"
            with tarfile.open(archive_path, "r") as archive:
                extracted = archive.extractfile(run["posthoc_evaluation"]["evaluator_source_path"])
                evaluator_payload = extracted.read() if extracted is not None else b""
                evaluator_path.write_bytes(evaluator_payload.replace(b"\r\n", b"\n"))
            recomputed_path = temp_root / "evaluation_result.json"
            process = subprocess.run(
                [
                    sys.executable,
                    str(evaluator_path),
                    "--truth-file",
                    str(project_path(run["posthoc_evaluation"]["truth_path"])),
                    "--detected-file",
                    str(evaluation_detected),
                    "--output-file",
                    str(recomputed_path),
                    "--threshold",
                    str(run["posthoc_evaluation"]["threshold_m"]),
                ],
                capture_output=True,
                text=True,
            )
            if process.returncode:
                errors.append(f"official evaluator reconstruction failed: {process.stderr.strip()}")
            elif json.loads(recomputed_path.read_text(encoding="utf-8")) != evaluation:
                errors.append("official evaluator reconstruction differs")
    if run["report"].get("pages") != 4:
        errors.append("unexpected Level-B supplement page count")
    if run["report"].get("word_open_and_pdf_export") != "passed":
        errors.append("Level-B supplement did not pass Word/PDF verification")
    if run["report"].get("page_by_page_visual_qa") != "passed":
        errors.append("Level-B supplement did not pass page-by-page visual QA")
    if run["report"].get("accessibility_findings") != 0:
        errors.append("Level-B supplement has unresolved accessibility findings")
    report_path = project_path(run["report"]["path"])
    if report_path.is_file():
        try:
            report_text = extract_docx_text(report_path)
        except (KeyError, OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            errors.append(f"cannot inspect corrected Level-B supplement text: {exc}")
        else:
            required_report_phrases = (
                "/Odometry_gazebo 是 Gazebo 位姿真值",
                "不满足完整真值隔离",
                "不能解释为自主定位精度",
                "G2 真值隔离保持失败状态",
                "位姿真值辅助的开发调试回放",
            )
            for phrase in required_report_phrases:
                if phrase not in report_text:
                    errors.append(f"corrected Level-B supplement lacks mandatory disclosure: {phrase}")
            for phrase in ("真值隔离的同步 RGB D 空间回放", "自主定位精度为"):
                if phrase in report_text:
                    errors.append(f"corrected Level-B supplement retains prohibited claim: {phrase}")
    if run.get("regression") != "87/87 passed":
        errors.append("Level-B run manifest does not record the current regression count")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "level_b_v2" / "level_b_run.json",
    )
    args = parser.parse_args()
    errors = validate_run(args.run_manifest)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        "PASSED: Level-B v2 artifacts and deterministic replay verified; "
        "pose-ground-truth use is disclosed and G2 remains failed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
