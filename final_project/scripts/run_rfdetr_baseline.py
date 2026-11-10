#!/usr/bin/env python3
"""Run production RF-DETR-Seg weights over a label-free image queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    resolve_image_path,
    sha256_file,
    validate_baseline_policy,
    validate_inference_jobs,
    write_jsonl,
)
from robot_vlm_baselines.inference import convert_rfdetr_detections  # noqa: E402


def load_model(checkpoint: Path, baseline: dict, device: str):
    try:
        from rfdetr import RFDETRSegMedium
    except ImportError as exc:
        raise BaselineContractError(
            "rfdetr is not installed; use a separate environment with requirements-rfdetr.txt"
        ) from exc
    kwargs = {
        "pretrain_weights": str(checkpoint),
        "num_classes": int(baseline["num_classes"]),
    }
    if device:
        kwargs["device"] = device
    try:
        return RFDETRSegMedium(resolution=int(baseline["resolution"]), **kwargs)
    except TypeError:
        return RFDETRSegMedium(**kwargs)


def run(
    jobs_path: Path,
    output_path: Path,
    image_root: Path,
    config_path: Path,
    checkpoint_path: Path | None,
    device: str,
    limit: int | None,
    dry_run: bool,
) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    baseline = config["rfdetr"]
    jobs = load_jsonl(jobs_path)
    if limit is not None:
        jobs = jobs[:limit]
    validate_inference_jobs(jobs, config, "rfdetr")
    if dry_run:
        for job in jobs:
            resolve_image_path(image_root, job)
        print(f"READY: validated {len(jobs)} RF-DETR jobs; model was not loaded")
        return 0
    if output_path.exists():
        raise BaselineContractError(f"refusing to overwrite existing output: {output_path}")
    checkpoint = checkpoint_path
    if checkpoint is None:
        configured = os.environ.get(baseline["checkpoint_env"], "")
        checkpoint = Path(configured) if configured else None
    if checkpoint is None or not checkpoint.is_file():
        raise BaselineContractError(
            f"set {baseline['checkpoint_env']} or pass --checkpoint with the production weights"
        )
    if sha256_file(checkpoint) != baseline["checkpoint_sha256"]:
        raise BaselineContractError("RF-DETR checkpoint SHA-256 does not match production manifest")
    model = load_model(checkpoint, baseline, device)
    records = []
    for job in jobs:
        started = time.perf_counter()
        try:
            image_path = resolve_image_path(image_root, job)
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            try:
                detections = model.predict(
                    image,
                    threshold=float(baseline["confidence_threshold"]),
                    include_source_image=False,
                )
            except TypeError:
                detections = model.predict(
                    image, threshold=float(baseline["confidence_threshold"])
                )
            predictions = convert_rfdetr_detections(
                detections,
                job["width"],
                job["height"],
                baseline["class_names"],
                baseline["class_aliases"],
            )
            status = "ok"
            error = None
        except Exception as exc:  # preserve every failed sample
            predictions = []
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "schema_version": 1,
                "job_id": job["job_id"],
                "baseline_id": job["baseline_id"],
                "fold_id": job["fold_id"],
                "sample_id": job["sample_id"],
                "image_sha256": job["image_sha256"],
                "status": status,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "predictions": predictions,
            }
        )
    write_jsonl(output_path, records)
    print(output_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return run(
            args.jobs.resolve(),
            args.output.resolve(),
            args.image_root.resolve(),
            args.config.resolve(),
            args.checkpoint.resolve() if args.checkpoint else None,
            args.device,
            args.limit,
            args.dry_run,
        )
    except BaselineContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
