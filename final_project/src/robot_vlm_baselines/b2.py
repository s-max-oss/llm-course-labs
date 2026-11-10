"""Frozen contracts and dependency-light helpers for SmolVLM B2."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .contracts import BaselineContractError, find_forbidden_fields, sha256_file


B2_JOB_FIELDS = {
    "schema_version",
    "job_id",
    "b2_protocol_id",
    "partition",
    "candidate_id",
    "source_sample_id",
    "image_root_id",
    "image",
    "image_sha256",
    "width",
    "height",
    "target_region_xyxy_pixels",
    "input_variant",
    "proposal_source",
    "deployment_eligible",
}

B2_COMMON_RESULT_FIELDS = {
    "schema_version",
    "b2_protocol_id",
    "variant",
    "job_id",
    "partition",
    "candidate_id",
    "source_sample_id",
    "image_sha256",
    "input_variant",
    "proposal_source",
    "deployment_eligible",
    "status",
    "error",
    "latency_ms",
    "predicted_natural_label",
    "predicted_label",
}

B2_A_RESULT_FIELDS = B2_COMMON_RESULT_FIELDS | {"raw_response"}
B2_B_RESULT_FIELDS = B2_COMMON_RESULT_FIELDS | {
    "candidate_scores",
    "top1_natural_label",
    "top2_natural_label",
    "margin",
}

B2_QUEUE_ARTIFACTS = {
    "development": {
        "path": "development.jsonl",
        "sha256": "69b255969a12acd0eefe79cf8faa32cce7880eb6445cad0cb882fa9c4d84712b",
        "count": 193,
    },
    "development_smoke": {
        "path": "development_smoke.jsonl",
        "sha256": "b6b692c7d5a9b45cbec20f1c854571ed719d6596d6ef1b002fa0c0e8d6eb3ab7",
        "count": 8,
    },
    "locked_evaluation": {
        "path": "locked_evaluation.jsonl",
        "sha256": "3c05b734bb92890d0f8fc30d97740bfa9e73ae31f9adb42d9023f3e603e70abc",
        "count": 251,
    },
}
B2_JOBS_MANIFEST_SHA256 = (
    "a60f77abe5af205ec52fdcbc9feafdb3c323ca8dcdb077ee069cd2a7a737c8e9"
)
# User authorized opening locked exploratory evaluation on 2026-11-08 after the
# real 193-candidate development report passed all frozen gates.
B2_TRUSTED_DEVELOPMENT_GATE_SHA256: str | None = (
    "e752b851b3f849a7aa42f52625bd96ce3542e904e0a67f00b798f5b93aaf91e4"
)


def load_frozen_b2_protocol(
    project_root: Path,
    protocol_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the machine protocol only when its pre-execution lock still matches."""

    root = project_root.resolve()
    protocol_path = protocol_path.resolve()
    lock_path = lock_path.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_id = "smolvlm_b2_exploratory_v1"
    if protocol.get("protocol_id") != expected_id or lock.get("protocol_id") != expected_id:
        raise BaselineContractError("B2 protocol ID differs from the frozen v1 contract")
    if protocol.get("status") != "frozen_before_execution":
        raise BaselineContractError("B2 machine protocol is not frozen before execution")
    if lock.get("status") != "frozen_before_execution":
        raise BaselineContractError("B2 protocol lock has an invalid status")
    sealed = lock.get("sealed_files", {})
    machine = sealed.get("machine_protocol", {})
    machine_path = (root / str(machine.get("path", ""))).resolve()
    if machine_path != protocol_path:
        raise BaselineContractError("runner protocol path differs from the sealed path")
    for label, item in sealed.items():
        path = (root / str(item.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise BaselineContractError(f"sealed {label} path escapes project root") from exc
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise BaselineContractError(f"sealed B2 protocol hash mismatch: {label}")
    for label, item in protocol.get("parent_artifacts", {}).items():
        path = (root / str(item.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise BaselineContractError(f"parent {label} path escapes project root") from exc
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise BaselineContractError(f"B2 parent artifact hash mismatch: {label}")
    if protocol.get("implementation_started") is not False:
        raise BaselineContractError("frozen protocol must record pre-implementation state")
    if protocol.get("execution_started") is not False:
        raise BaselineContractError("frozen protocol must record pre-execution state")
    return protocol, lock


def expected_partition(protocol: dict[str, Any], scene_seed: int) -> str | None:
    for partition in ("development", "locked_evaluation"):
        seeds = protocol["partition"][partition]["scene_seeds"]
        if scene_seed in seeds:
            return partition
    return None


def b2_job(candidate: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    partition = expected_partition(protocol, int(candidate["scene_seed"]))
    if partition is None:
        raise BaselineContractError(f"unassigned B2 scene seed: {candidate['scene_seed']}")
    region = candidate["padded_crop_xyxy_pixels"]
    return {
        "schema_version": 1,
        "job_id": f"{protocol['protocol_id']}::{partition}::{candidate['candidate_id']}",
        "b2_protocol_id": protocol["protocol_id"],
        "partition": partition,
        "candidate_id": candidate["candidate_id"],
        "source_sample_id": candidate["source_sample_id"],
        "image_root_id": candidate["image_root_id"],
        "image": candidate["image"],
        "image_sha256": candidate["image_sha256"],
        "width": candidate["width"],
        "height": candidate["height"],
        "target_region_xyxy_pixels": region,
        "input_variant": "offline_oracle_padded_crop_or_background_full_frame",
        "proposal_source": (
            "offline_annotation_oracle"
            if region is not None
            else "offline_background_convention"
        ),
        "deployment_eligible": False,
    }


def validate_b2_jobs(
    jobs: list[dict[str, Any]],
    protocol: dict[str, Any],
    forbidden_fields: set[str],
    expected_partition_name: str | None = None,
) -> None:
    seen: set[str] = set()
    for index, job in enumerate(jobs):
        leaked = find_forbidden_fields(job, forbidden_fields)
        if leaked:
            raise BaselineContractError(f"B2 job {index} contains forbidden fields: {leaked}")
        if set(job) != B2_JOB_FIELDS:
            raise BaselineContractError(f"B2 job {index} fields differ from the frozen contract")
        job_id = job["job_id"]
        if not isinstance(job_id, str) or job_id in seen:
            raise BaselineContractError(f"B2 job {index} has a missing or duplicate job_id")
        seen.add(job_id)
        partition = job["partition"]
        if partition not in {"development", "locked_evaluation"}:
            raise BaselineContractError(f"{job_id}: invalid B2 partition")
        if expected_partition_name is not None and partition != expected_partition_name:
            raise BaselineContractError(f"{job_id}: job is in the wrong B2 partition")
        expected_id = f"{protocol['protocol_id']}::{partition}::{job['candidate_id']}"
        if job_id != expected_id or job["b2_protocol_id"] != protocol["protocol_id"]:
            raise BaselineContractError(f"{job_id}: B2 job identity differs from protocol")
        if job["deployment_eligible"] is not False:
            raise BaselineContractError(f"{job_id}: oracle input cannot be deployable")
        if job["input_variant"] != "offline_oracle_padded_crop_or_background_full_frame":
            raise BaselineContractError(f"{job_id}: invalid B2 input variant")
        if not isinstance(job["image"], str) or not job["image"]:
            raise BaselineContractError(f"{job_id}: invalid image name")
        if not re.fullmatch(r"[0-9a-f]{64}", str(job["image_sha256"])):
            raise BaselineContractError(f"{job_id}: invalid image hash")
        if type(job["width"]) is not int or type(job["height"]) is not int:
            raise BaselineContractError(f"{job_id}: image dimensions must be integers")
        if job["width"] <= 0 or job["height"] <= 0:
            raise BaselineContractError(f"{job_id}: image dimensions must be positive")
        region = job["target_region_xyxy_pixels"]
        if region is None:
            if job["proposal_source"] != "offline_background_convention":
                raise BaselineContractError(f"{job_id}: invalid background proposal source")
        else:
            if job["proposal_source"] != "offline_annotation_oracle":
                raise BaselineContractError(f"{job_id}: invalid object proposal source")
            if not isinstance(region, list) or len(region) != 4:
                raise BaselineContractError(f"{job_id}: invalid crop region")
            try:
                values = [float(value) for value in region]
            except (TypeError, ValueError) as exc:
                raise BaselineContractError(f"{job_id}: non-numeric crop region") from exc
            if any(not math.isfinite(value) for value in values):
                raise BaselineContractError(f"{job_id}: non-finite crop region")


def parse_b2_a_label(raw_response: str, protocol: dict[str, Any]) -> tuple[str, str] | None:
    if not isinstance(raw_response, str):
        return None
    expression = protocol["variants"]["b2_a_strict_natural_generation"]["parser_regex"]
    match = re.fullmatch(expression, raw_response, flags=re.IGNORECASE)
    if not match:
        return None
    natural = match.group(1).lower()
    return natural, protocol["label_mapping"][natural]


def rank_candidate_scores(
    scores: dict[str, float], ordered_natural_labels: list[str]
) -> tuple[str, str, float]:
    if set(scores) != set(ordered_natural_labels):
        raise BaselineContractError("candidate score labels differ from the frozen protocol")
    if any(not math.isfinite(float(score)) for score in scores.values()):
        raise BaselineContractError("candidate scores must all be finite")
    order = {label: index for index, label in enumerate(ordered_natural_labels)}
    ranked = sorted(ordered_natural_labels, key=lambda label: (-scores[label], order[label]))
    return ranked[0], ranked[1], float(scores[ranked[0]] - scores[ranked[1]])


def validate_development_gate_report(
    path: Path,
    protocol: dict[str, Any],
    machine_protocol_sha256: str,
    project_root: Path,
    trusted_report_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineContractError(f"missing B2 development gate report: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", trusted_report_sha256):
        raise BaselineContractError("trusted development gate digest is not configured")
    if sha256_file(path) != trusted_report_sha256:
        raise BaselineContractError("development gate report differs from trusted digest")
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": protocol["protocol_id"],
        "variant": "b2_b_fixed_candidate_scoring",
        "partition": "development",
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise BaselineContractError(f"development gate report has invalid {field}")
    sources = report.get("source_files", {})
    if sources.get("machine_protocol", {}).get("sha256") != machine_protocol_sha256:
        raise BaselineContractError("development gate report uses another B2 protocol")
    root = project_root.resolve()
    for name in ("jobs", "jobs_manifest", "results"):
        source = sources.get(name, {})
        source_path = (root / str(source.get("path", ""))).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise BaselineContractError(f"development gate {name} path escapes project root") from exc
        if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
            raise BaselineContractError(f"development gate {name} artifact hash mismatch")
    development_artifact = B2_QUEUE_ARTIFACTS["development"]
    if sources["jobs"].get("sha256") != development_artifact["sha256"]:
        raise BaselineContractError("development gate report is not bound to frozen development jobs")
    if sources["jobs_manifest"].get("sha256") != B2_JOBS_MANIFEST_SHA256:
        raise BaselineContractError("development gate report uses another B2 jobs manifest")

    metrics = report.get("metrics", {})
    metric_values = {
        "prediction_coverage": metrics.get("prediction_coverage"),
        "macro_f1": metrics.get("macro_f1"),
        "danger_red_sphere_recall": metrics.get("danger_binary", {}).get("recall"),
        "distractor_red_box_recall": metrics.get("per_class", {})
        .get("distractor_red_box", {})
        .get("recall"),
    }
    frozen = protocol["development_gates_for_b2_b"]
    minimums = {
        "prediction_coverage": frozen["prediction_coverage_min"],
        "macro_f1": frozen["macro_f1_min"],
        "danger_red_sphere_recall": frozen["danger_red_sphere_recall_min"],
        "distractor_red_box_recall": frozen["distractor_red_box_recall_min"],
    }
    gates = report.get("development_gates", {})
    checks = gates.get("checks", {})
    if set(checks) != set(minimums):
        raise BaselineContractError("development gate report has missing or extra checks")
    computed_passes = []
    for name, minimum in minimums.items():
        check = checks[name]
        value = metric_values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise BaselineContractError(f"development gate metric is invalid: {name}")
        if check.get("value") != value or check.get("minimum") != minimum:
            raise BaselineContractError(f"development gate check differs from metrics/protocol: {name}")
        passed = float(value) >= float(minimum)
        if check.get("passed") is not passed:
            raise BaselineContractError(f"development gate pass flag is inconsistent: {name}")
        computed_passes.append(passed)
    if gates.get("passed") is not all(computed_passes):
        raise BaselineContractError("development gate aggregate pass flag is inconsistent")
    if gates.get("passed") is not True:
        raise BaselineContractError("B2 development gates did not pass")
    return report
