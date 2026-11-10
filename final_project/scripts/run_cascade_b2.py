#!/usr/bin/env python3
"""Run frozen B2-B scoring on label-free RF-DETR proposal crops."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from robot_vlm_baselines.b2 import load_frozen_b2_protocol, rank_candidate_scores  # noqa: E402
from robot_vlm_baselines.cascade import (  # noqa: E402
    build_cascade_jobs,
    load_cascade_protocol,
    validate_cascade_jobs,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    resolve_image_path,
    sha256_file,
    validate_baseline_policy,
)
from run_smolvlm_b2 import (  # noqa: E402
    crop_for_job,
    fixed_candidate_scores,
    load_model,
    verify_local_model,
)


def validate_queue(jobs_path: Path, jobs: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    expected = build_cascade_jobs(PROJECT_ROOT, protocol)
    if jobs != expected:
        raise BaselineContractError("cascade runner queue differs from independent reconstruction")
    manifest_path = jobs_path.parent / "protocol.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("queue") != {
        "path": jobs_path.name,
        "sha256": sha256_file(jobs_path),
        "count": 24,
        "routed_image_count": 22,
    }:
        raise BaselineContractError("cascade runner queue is not bound by its manifest")


def reserve_output(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BaselineContractError(f"refusing to overwrite existing output: {path}") from exc


def write_reserved_jsonl(descriptor: int, records: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def run(
    jobs_path: Path,
    output_path: Path,
    image_root: Path,
    cascade_protocol_path: Path,
    b2_protocol_path: Path,
    b2_lock_path: Path,
    config_path: Path,
    model_ref: str | None,
    dry_run: bool,
) -> int:
    cascade_protocol = load_cascade_protocol(PROJECT_ROOT, cascade_protocol_path)
    b2_protocol, _ = load_frozen_b2_protocol(PROJECT_ROOT, b2_protocol_path, b2_lock_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    jobs = load_jsonl(jobs_path)
    validate_cascade_jobs(jobs, cascade_protocol, set(config["inference_forbidden_fields"]))
    validate_queue(jobs_path, jobs, cascade_protocol)
    for job in jobs:
        image_path = resolve_image_path(image_root, job)
        with Image.open(image_path) as source:
            crop_for_job(source.convert("RGB"), job)
    if dry_run:
        print(f"READY: validated {len(jobs)} real-proposal cascade jobs; model was not loaded")
        return 0
    output_descriptor = reserve_output(output_path)
    baseline = config["smolvlm"]
    resolved_model = model_ref or os.environ.get(baseline["model_path_env"], "")
    if not resolved_model:
        raise BaselineContractError(
            f"set {baseline['model_path_env']} or pass --model with a local SmolVLM directory"
        )
    try:
        model_root = verify_local_model(resolved_model, baseline)
        processor, model, target = load_model(
            str(model_root), int(b2_protocol["model_contract"]["image_longest_edge"])
        )
        settings = b2_protocol["variants"]["b2_b_fixed_candidate_scoring"]
        completions = settings["candidate_completions_in_tie_order"]
        order = [item.strip() for item in completions]
        records: list[dict[str, Any]] = []
        for job in jobs:
            started = time.perf_counter()
            common = {
            "schema_version": 1,
            "protocol_id": cascade_protocol["protocol_id"],
            "b2_protocol_id": b2_protocol["protocol_id"],
            "variant": "b2_b_fixed_candidate_scoring",
            "job_id": job["job_id"],
            "fold_id": job["fold_id"],
            "sample_id": job["sample_id"],
            "prediction_index": job["prediction_index"],
            "image_sha256": job["image_sha256"],
            "input_variant": job["input_variant"],
            "proposal_source": job["proposal_source"],
            "detector_label": job["detector_label"],
            "detector_confidence": job["detector_confidence"],
            "proposal_xyxy_pixels": job["proposal_xyxy_pixels"],
            }
            try:
                image_path = resolve_image_path(image_root, job)
                with Image.open(image_path) as source:
                    crop = crop_for_job(source.convert("RGB"), job)
                scores = fixed_candidate_scores(
                    processor,
                    model,
                    target,
                    crop,
                    settings["prompt"],
                    completions,
                )
                top1, top2, margin = rank_candidate_scores(scores, order)
                record = {
                    **common,
                    "status": "ok",
                    "error": None,
                    "predicted_natural_label": top1,
                    "predicted_label": b2_protocol["label_mapping"][top1],
                    "candidate_scores": scores,
                    "top1_natural_label": top1,
                    "top2_natural_label": top2,
                    "margin": margin,
                }
            except Exception as exc:
                record = {
                    **common,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "predicted_natural_label": None,
                    "predicted_label": None,
                    "candidate_scores": {},
                    "top1_natural_label": None,
                    "top2_natural_label": None,
                    "margin": None,
                }
            record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            records.append(record)
        write_reserved_jsonl(output_descriptor, records)
        output_descriptor = -1
    finally:
        if output_descriptor >= 0:
            os.close(output_descriptor)
    print(output_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "cascade_jobs" / "development.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--cascade-protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "cascade_feasibility_v1.json",
    )
    parser.add_argument(
        "--b2-protocol", type=Path, default=PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json"
    )
    parser.add_argument(
        "--b2-lock",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json")
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(
            args.jobs.resolve(),
            args.output.resolve(),
            args.image_root.resolve(),
            args.cascade_protocol.resolve(),
            args.b2_protocol.resolve(),
            args.b2_lock.resolve(),
            args.config.resolve(),
            args.model,
            args.dry_run,
        )
    except (BaselineContractError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
