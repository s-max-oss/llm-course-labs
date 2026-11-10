#!/usr/bin/env python3
"""Run the pinned RF-DETR checkpoint over a Level-B input manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_vlm_baselines.contracts import sha256_file  # noqa: E402
from robot_vlm_baselines.inference import convert_rfdetr_detections  # noqa: E402
from build_level_b_replay import validate_input_manifest  # noqa: E402


EXPECTED_PROTOCOL_STATUSES = {
    "level_b_replay_v1": "frozen_before_accepted_inference",
    "level_b_replay_v2": "frozen_before_v2_acquisition_and_inference",
}


def validate_protocol_bindings(protocol: dict, manifest: dict, manifest_root: Path) -> None:
    protocol_id = protocol.get("protocol_id")
    expected_status = EXPECTED_PROTOCOL_STATUSES.get(protocol_id)
    if expected_status is None:
        raise ValueError("unexpected Level-B protocol ID")
    if protocol.get("status") != expected_status:
        raise ValueError("Level-B protocol is not frozen for accepted inference")
    validate_input_manifest(manifest, protocol, manifest_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    detector = protocol["detector"]
    validate_protocol_bindings(protocol, manifest, args.manifest.parent)
    if sha256_file(args.checkpoint) != detector["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 mismatch")

    baseline = json.loads(
        (PROJECT_ROOT / "configs" / "baselines.json").read_text(encoding="utf-8")
    )["rfdetr"]
    if (
        baseline["checkpoint_sha256"] != detector["checkpoint_sha256"]
        or float(baseline["confidence_threshold"])
        != float(detector["confidence_threshold"])
    ):
        raise ValueError("Level-B detector policy differs from the pinned baseline")

    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(
        pretrain_weights=str(args.checkpoint),
        num_classes=int(baseline["num_classes"]),
        device=args.device,
    )
    manifest_root = args.manifest.parent
    records = []
    for frame in manifest["records"]:
        rgb = frame["rgb"]
        image_path = manifest_root / rgb["path"]
        if sha256_file(image_path) != rgb["sha256"]:
            raise ValueError(f"RGB hash mismatch: {frame['frame_id']}")
        started = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        try:
            detections = model.predict(
                image,
                threshold=float(detector["confidence_threshold"]),
                include_source_image=False,
            )
        except TypeError:
            detections = model.predict(
                image, threshold=float(detector["confidence_threshold"])
            )
        predictions = convert_rfdetr_detections(
            detections,
            rgb["width"],
            rgb["height"],
            baseline["class_names"],
            baseline["class_aliases"],
        )
        unexpected = sorted(
            {item["canonical_label"] for item in predictions}
            - set(detector["allowed_classes"])
        )
        if unexpected:
            raise ValueError(f"unexpected canonical classes: {unexpected}")
        records.append(
            {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "frame_id": frame["frame_id"],
                "sequence_index": frame["sequence_index"],
                "rgb_sha256": rgb["sha256"],
                "rgb_stamp_ns": rgb["stamp_ns"],
                "source_model": detector["name"],
                "source_model_sha256": detector["checkpoint_sha256"],
                "confidence_threshold": detector["confidence_threshold"],
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "predictions": predictions,
            }
        )
    write_jsonl_atomic(args.output, records)
    print(args.output)
    print(
        json.dumps(
            {
                "frames": len(records),
                "predictions": sum(len(item["predictions"]) for item in records),
                "danger_predictions": sum(
                    prediction["canonical_label"] == "danger_red_sphere"
                    for item in records
                    for prediction in item["predictions"]
                ),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
