"""Contracts for the frozen RF-DETR to SmolVLM development feasibility test."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .contracts import (
    BaselineContractError,
    find_forbidden_fields,
    load_jsonl,
    sha256_file,
)


CASCADE_PROTOCOL_SHA256 = "8982a79c5fb857ad01173e8ad774af277a34313179cc842fcab4d27e52b55240"
CASCADE_JOB_FIELDS = {
    "schema_version",
    "protocol_id",
    "partition",
    "job_id",
    "fold_id",
    "sample_id",
    "image_root_id",
    "image",
    "image_sha256",
    "width",
    "height",
    "prediction_index",
    "detector_label",
    "detector_confidence",
    "proposal_xyxy_pixels",
    "target_region_xyxy_pixels",
    "input_variant",
    "proposal_source",
    "deployment_eligible",
}
CASCADE_RESULT_FIELDS = {
    "schema_version",
    "protocol_id",
    "b2_protocol_id",
    "variant",
    "job_id",
    "fold_id",
    "sample_id",
    "prediction_index",
    "image_sha256",
    "input_variant",
    "proposal_source",
    "detector_label",
    "detector_confidence",
    "proposal_xyxy_pixels",
    "status",
    "error",
    "predicted_natural_label",
    "predicted_label",
    "candidate_scores",
    "top1_natural_label",
    "top2_natural_label",
    "margin",
    "latency_ms",
}


def load_cascade_protocol(project_root: Path, protocol_path: Path) -> dict[str, Any]:
    if sha256_file(protocol_path) != CASCADE_PROTOCOL_SHA256:
        raise BaselineContractError("cascade protocol differs from the preimplementation-reviewed v1")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol_id") != "rfdetr_smolvlm_cascade_feasibility_v1"
        or protocol.get("status") != "frozen_before_implementation"
        or protocol.get("implementation_started") is not False
        or protocol.get("execution_started") is not False
        or protocol.get("continuation_gates", {}).get("locked_evaluation_authorized") is not False
    ):
        raise BaselineContractError("cascade protocol is not the frozen development-only contract")
    for label, entry in protocol["parent_artifacts"].items():
        path = (project_root / entry["path"]).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise BaselineContractError(f"cascade parent escapes project root: {label}") from exc
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise BaselineContractError(f"cascade parent artifact hash mismatch: {label}")
    anchor = protocol["pre_protocol_code_commit"]
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise BaselineContractError("cascade pre-protocol commit is not an ancestor of HEAD")
    return protocol


def padded_region(box: list[float], width: int, height: int, fraction: float) -> list[float]:
    if len(box) != 4 or any(not math.isfinite(float(value)) for value in box):
        raise BaselineContractError("invalid RF-DETR proposal box")
    left, top, right, bottom = map(float, box)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise BaselineContractError("RF-DETR proposal box is outside the image")
    pad_x = (right - left) * fraction
    pad_y = (bottom - top) * fraction
    return [
        round(max(0.0, left - pad_x), 6),
        round(max(0.0, top - pad_y), 6),
        round(min(float(width), right + pad_x), 6),
        round(min(float(height), bottom + pad_y), 6),
    ]


def build_cascade_jobs(project_root: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    parents = protocol["parent_artifacts"]
    selection = protocol["proposal_selection"]
    allowed = set(selection["allowed_detector_labels"])
    jobs: list[dict[str, Any]] = []
    for fold_id in protocol["development_partition"]["fold_ids"]:
        suffix = fold_id[-2:]
        image_jobs_path = project_root / parents[f"fold_{suffix}_rfdetr_jobs"]["path"]
        results_path = project_root / parents[f"fold_{suffix}_rfdetr_results"]["path"]
        image_jobs = {item["sample_id"]: item for item in load_jsonl(image_jobs_path)}
        for result in load_jsonl(results_path):
            sample_id = result["sample_id"]
            if result.get("status") != "ok" or sample_id not in image_jobs:
                raise BaselineContractError(f"cascade source result is incomplete: {fold_id}/{sample_id}")
            source = image_jobs[sample_id]
            if (
                result.get("fold_id") != fold_id
                or result.get("job_id") != source["job_id"]
                or result.get("image_sha256") != source["image_sha256"]
            ):
                raise BaselineContractError(f"cascade source identity mismatch: {fold_id}/{sample_id}")
            for prediction in result["predictions"]:
                confidence = float(prediction["confidence"])
                if prediction["canonical_label"] not in allowed:
                    continue
                if not (
                    selection["confidence_min_inclusive"]
                    <= confidence
                    < selection["confidence_max_exclusive"]
                ):
                    continue
                index = prediction["prediction_index"]
                proposal = [round(float(value), 6) for value in prediction["bbox_xyxy_pixels"]]
                jobs.append(
                    {
                        "schema_version": 1,
                        "protocol_id": protocol["protocol_id"],
                        "partition": "development",
                        "job_id": f"{protocol['protocol_id']}::{fold_id}::{sample_id}::{index}",
                        "fold_id": fold_id,
                        "sample_id": sample_id,
                        "image_root_id": source["image_root_id"],
                        "image": source["image"],
                        "image_sha256": source["image_sha256"],
                        "width": source["width"],
                        "height": source["height"],
                        "prediction_index": index,
                        "detector_label": prediction["canonical_label"],
                        "detector_confidence": round(confidence, 8),
                        "proposal_xyxy_pixels": proposal,
                        "target_region_xyxy_pixels": padded_region(
                            proposal,
                            source["width"],
                            source["height"],
                            selection["bbox_padding_fraction"],
                        ),
                        "input_variant": protocol["vlm_contract"]["input_variant"],
                        "proposal_source": protocol["vlm_contract"]["proposal_source"],
                        "deployment_eligible": False,
                    }
                )
    jobs.sort(key=lambda item: (item["fold_id"], item["sample_id"], item["prediction_index"]))
    validate_cascade_jobs(jobs, protocol, set())
    if len(jobs) != selection["expected_job_count"]:
        raise BaselineContractError("cascade job count differs from frozen protocol")
    if len({item["sample_id"] for item in jobs}) != selection["expected_routed_image_count"]:
        raise BaselineContractError("cascade routed image count differs from frozen protocol")
    return jobs


def validate_cascade_jobs(
    jobs: list[dict[str, Any]], protocol: dict[str, Any], forbidden_fields: set[str]
) -> None:
    seen: set[str] = set()
    allowed = set(protocol["proposal_selection"]["allowed_detector_labels"])
    for index, job in enumerate(jobs):
        leaked = find_forbidden_fields(job, forbidden_fields)
        if leaked:
            raise BaselineContractError(f"cascade job {index} contains forbidden fields: {leaked}")
        if set(job) != CASCADE_JOB_FIELDS:
            raise BaselineContractError(f"cascade job {index} fields differ from the frozen contract")
        if job["job_id"] in seen:
            raise BaselineContractError("duplicate cascade job ID")
        seen.add(job["job_id"])
        expected_id = (
            f"{protocol['protocol_id']}::{job['fold_id']}::{job['sample_id']}::"
            f"{job['prediction_index']}"
        )
        if job["job_id"] != expected_id or job["protocol_id"] != protocol["protocol_id"]:
            raise BaselineContractError("cascade job identity differs from protocol")
        if job["partition"] != "development" or job["fold_id"] not in {"fold_01", "fold_02"}:
            raise BaselineContractError("cascade job escapes the development partition")
        if job["detector_label"] not in allowed:
            raise BaselineContractError("cascade job uses an unrouteable detector label")
        confidence = job["detector_confidence"]
        selection = protocol["proposal_selection"]
        if type(confidence) not in {int, float} or not (
            selection["confidence_min_inclusive"] <= confidence < selection["confidence_max_exclusive"]
        ):
            raise BaselineContractError("cascade job confidence is outside the frozen interval")
        if (
            job["input_variant"] != protocol["vlm_contract"]["input_variant"]
            or job["proposal_source"] != protocol["vlm_contract"]["proposal_source"]
            or job["deployment_eligible"] is not False
            or not isinstance(job["target_region_xyxy_pixels"], list)
        ):
            raise BaselineContractError("cascade job violates the real-proposal input boundary")
