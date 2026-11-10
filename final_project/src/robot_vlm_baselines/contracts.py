"""Shared, dependency-light contracts for offline baseline inference."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


EXPECTED_LABELS = (
    "danger_red_sphere",
    "distractor_red_box",
    "distractor_green_sphere",
    "background_or_unknown",
)

REQUIRED_RFDETR_POLICY = {
    "baseline_id": "b0_rfdetr_seg_medium",
    "model_class": "RFDETRSegMedium",
    "checkpoint_sha256": "ec2de357f402901cf624835f11d5c54bb0eab8980d1268d7ee8a9122a446fa50",
    "confidence_threshold": 0.5,
    "resolution": 432,
    "num_classes": 4,
    "class_names": {
        "0": "Red_ball",
        "1": "Green_ball",
        "2": "Green_wall",
        "3": "red_cube",
    },
}

REQUIRED_SMOLVLM_POLICY = {
    "baseline_id": "b1_smolvlm_256m_zero_shot_oracle_crop",
    "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "local_files_only": True,
    "input_variant": "offline_oracle_padded_crop_or_background_full_frame",
    "deployment_eligible": False,
    "image_longest_edge": 512,
    "max_new_tokens": 24,
    "model_files": {
        "added_tokens.json": "74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a",
        "chat_template.json": "a68ad1a42681ae44eacd109ff8dd56a840f761c03d72f4ff4c515d092f882168",
        "model.safetensors": "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
        "config.json": "b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52",
        "generation_config.json": (
            "067a2a54e5f87162ecac6e0e911cc4665fc8f7f3324794ecbac0f76badb56636"
        ),
        "processor_config.json": (
            "e7bff42da73ae9eec9042ef20e066e11f1ee20f025358ff79131e3c0fb549b46"
        ),
        "preprocessor_config.json": (
            "6cb6e36d6fcb88ca1502c4a26750715dc3e7dedddc9a8f17b27d8d167d1457e7"
        ),
        "special_tokens_map.json": (
            "aa0ff906077086dfa9734a7f97f68c825877a48f9468807be65504495cdeef09"
        ),
        "tokenizer_config.json": (
            "36c6fd44d07d10fd8180ee6b46dcccf69fb7c06753968ff0d7e17b8bfe17b777"
        ),
        "tokenizer.json": "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
    },
    "prompt": (
        "Classify the image into exactly one label and output only that label: "
        "danger_red_sphere, distractor_red_box, distractor_green_sphere, "
        "background_or_unknown."
    ),
}


class BaselineContractError(RuntimeError):
    """Raised when an inference artifact violates the sealed protocol."""


def validate_baseline_policy(config: dict[str, Any]) -> None:
    """Fail closed if a production or diagnostic baseline contract was changed."""

    if config.get("protocol_id") != "pilot_v1":
        raise BaselineContractError("mandatory baseline policy must use protocol_id='pilot_v1'")
    if tuple(config.get("expected_labels", ())) != EXPECTED_LABELS:
        raise BaselineContractError("mandatory baseline label order differs from contract")
    for section, required in (
        ("rfdetr", REQUIRED_RFDETR_POLICY),
        ("smolvlm", REQUIRED_SMOLVLM_POLICY),
    ):
        actual = config.get(section, {})
        for field, required_value in required.items():
            if actual.get(field) != required_value:
                raise BaselineContractError(
                    f"mandatory {section} policy must remain {field}={required_value!r}"
                )
    aliases = config.get("rfdetr", {}).get("class_aliases", {})
    required_aliases = {
        "Red_ball": "danger_red_sphere",
        "red_ball": "danger_red_sphere",
        "red_cube": "distractor_red_box",
        "Green_ball": "distractor_green_sphere",
        "green_ball": "distractor_green_sphere",
        "Green_wall": "distractor_green_wall",
        "green_wall": "distractor_green_wall",
    }
    if aliases != required_aliases:
        raise BaselineContractError("mandatory RF-DETR class aliases differ from contract")
    model_files = config.get("smolvlm", {}).get("model_files")
    if not isinstance(model_files, dict) or not model_files:
        raise BaselineContractError("mandatory SmolVLM model file hashes are missing")
    for name, digest in model_files.items():
        if not isinstance(name, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise BaselineContractError("invalid SmolVLM model file hash manifest")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaselineContractError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise BaselineContractError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def parse_strict_label(raw_response: str) -> str | None:
    """Accept one class label with optional surrounding whitespace/punctuation."""

    if not isinstance(raw_response, str):
        return None
    alternatives = "|".join(re.escape(label) for label in EXPECTED_LABELS)
    match = re.fullmatch(rf"\s*({alternatives})\s*[.!]?\s*", raw_response)
    return match.group(1) if match else None


def canonical_detector_label(raw_name: str, aliases: dict[str, str]) -> str:
    if raw_name not in aliases:
        raise BaselineContractError(f"unknown detector class name: {raw_name!r}")
    return aliases[raw_name]


def find_forbidden_fields(value: Any, forbidden: set[str], prefix: str = "") -> list[str]:
    """Return dotted paths for forbidden keys at any nesting level."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden:
                found.append(path)
            found.extend(find_forbidden_fields(child, forbidden, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_fields(child, forbidden, f"{prefix}[{index}]"))
    return found


RFDETR_JOB_FIELDS = {
    "schema_version",
    "job_id",
    "baseline_id",
    "pilot_protocol_id",
    "fold_id",
    "split",
    "task",
    "sample_id",
    "image_root_id",
    "image",
    "image_sha256",
    "width",
    "height",
}

SMOLVLM_JOB_FIELDS = {
    "schema_version",
    "job_id",
    "baseline_id",
    "pilot_protocol_id",
    "fold_id",
    "split",
    "task",
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


def validate_inference_jobs(
    jobs: list[dict[str, Any]], config: dict[str, Any], kind: str
) -> None:
    """Validate a runner input at the trust boundary, not only at generation."""

    if kind not in {"rfdetr", "smolvlm"}:
        raise ValueError(f"unknown inference job kind: {kind}")
    forbidden = set(config["inference_forbidden_fields"])
    baseline = config[kind]
    allowed = RFDETR_JOB_FIELDS if kind == "rfdetr" else SMOLVLM_JOB_FIELDS
    expected_task = (
        "instance_segmentation" if kind == "rfdetr" else "strict_four_class_classification"
    )
    identifier_field = "sample_id" if kind == "rfdetr" else "candidate_id"
    seen_job_ids: set[str] = set()
    for index, job in enumerate(jobs):
        leaked = find_forbidden_fields(job, forbidden)
        if leaked:
            raise BaselineContractError(
                f"job {index} contains forbidden inference fields: {leaked}"
            )
        fields = set(job)
        if fields != allowed:
            missing = sorted(allowed - fields)
            unexpected = sorted(fields - allowed)
            raise BaselineContractError(
                f"job {index} fields differ from contract: "
                f"missing={missing} unexpected={unexpected}"
            )
        job_id = job["job_id"]
        if not isinstance(job_id, str) or job_id in seen_job_ids:
            raise BaselineContractError(f"job {index} has missing or duplicate job_id")
        seen_job_ids.add(job_id)
        common = {
            "schema_version": 1,
            "baseline_id": baseline["baseline_id"],
            "pilot_protocol_id": config["protocol_id"],
            "split": "test",
            "task": expected_task,
        }
        for field, expected in common.items():
            if job[field] != expected:
                raise BaselineContractError(
                    f"{job_id}: {field} must equal {expected!r}"
                )
        if not re.fullmatch(r"fold_0[1-5]", str(job["fold_id"])):
            raise BaselineContractError(f"{job_id}: invalid fold_id")
        identifier = job[identifier_field]
        expected_job_id = f"{baseline['baseline_id']}::{job['fold_id']}::{identifier}"
        if job_id != expected_job_id:
            raise BaselineContractError(f"{job_id}: job_id does not match record identity")
        if not isinstance(job["image"], str) or not job["image"]:
            raise BaselineContractError(f"{job_id}: invalid image name")
        if not re.fullmatch(r"[0-9a-f]{64}", str(job["image_sha256"])):
            raise BaselineContractError(f"{job_id}: invalid image SHA-256")
        if type(job["width"]) is not int or type(job["height"]) is not int:
            raise BaselineContractError(f"{job_id}: image dimensions must be integers")
        if job["width"] <= 0 or job["height"] <= 0:
            raise BaselineContractError(f"{job_id}: image dimensions must be positive")
        if kind == "smolvlm":
            if job["deployment_eligible"] is not False:
                raise BaselineContractError(f"{job_id}: oracle crop cannot be deployable")
            if job["input_variant"] != baseline["input_variant"]:
                raise BaselineContractError(
                    f"{job_id}: SmolVLM input variant differs from config"
                )
            region = job["target_region_xyxy_pixels"]
            proposal_source = job["proposal_source"]
            if region is None:
                if proposal_source != "offline_background_convention":
                    raise BaselineContractError(f"{job_id}: invalid full-frame proposal source")
            else:
                if proposal_source != "offline_annotation_oracle":
                    raise BaselineContractError(f"{job_id}: invalid oracle-crop proposal source")
                if not isinstance(region, list) or len(region) != 4:
                    raise BaselineContractError(f"{job_id}: invalid target region")
                try:
                    values = [float(value) for value in region]
                except (TypeError, ValueError) as exc:
                    raise BaselineContractError(f"{job_id}: non-numeric target region") from exc
                if any(not (value == value and abs(value) != float("inf")) for value in values):
                    raise BaselineContractError(f"{job_id}: non-finite target region")


def resolve_image_path(image_root: Path, job: dict[str, Any]) -> Path:
    root = image_root.resolve()
    candidate = (root / str(job["image"])).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise BaselineContractError("image path escapes configured image root") from exc
    if not candidate.is_file():
        raise BaselineContractError(f"image does not exist: {candidate}")
    if sha256_file(candidate) != job["image_sha256"]:
        raise BaselineContractError(f"image hash mismatch: {candidate}")
    return candidate


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
