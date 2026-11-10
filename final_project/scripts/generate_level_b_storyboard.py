#!/usr/bin/env python3
"""Generate a deterministic target-truth-free, pose-oracle debug replay GIF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "experiments" / "level_b_v2"
INPUT_ROOT = RUN_ROOT / "inputs"
REPLAY_ROOT = RUN_ROOT / "outputs" / "replay_v1"
GIF_PATH = PROJECT_ROOT / "demo" / "level_b_v2_spatial_replay.gif"
MANIFEST_PATH = RUN_ROOT / "visualization_manifest.json"
VISIBLE_BANNER = "DEBUG REPLAY | TARGET-TRUTH-FREE | GAZEBO POSE ORACLE"
CLAIM_BOUNDARY = (
    "Gazebo-pose-oracle development debug replay; not autonomous localization, "
    "online control, competition compliance, or full-building coverage"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    input_manifest_path = INPUT_ROOT / "input_manifest.json"
    event_path = REPLAY_ROOT / "danger_detection_events.jsonl"
    trace_path = REPLAY_ROOT / "decision_trace.jsonl"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    source_bag = input_manifest.get("source_bag", {})
    topics = source_bag.get("topics", [])
    if source_bag.get("ground_truth_topics_present") is not False:
        raise ValueError("storyboard input contains declared danger-target truth")
    if "/Odometry_gazebo" not in topics:
        raise ValueError("historical pose-oracle replay is missing /Odometry_gazebo")

    records = {record["frame_id"]: record for record in input_manifest["records"]}
    events = load_jsonl(event_path)
    traces = {trace["event_record_id"]: trace for trace in load_jsonl(trace_path)}
    if len(events) != 14 or set(traces) != {event["record_id"] for event in events}:
        raise ValueError("expected exactly 14 paired Level-B events and traces")

    colours = {
        "danger_red_sphere": "#ff4d4d",
        "distractor_green_sphere": "#65e38a",
    }
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    frame_records: list[dict] = []
    for event in sorted(events, key=lambda item: item["sequence_index"]):
        trace = traces[event["record_id"]]
        record = records[event["frame_id"]]
        image_path = INPUT_ROOT / record["rgb"]["path"]
        if sha256_path(image_path) != record["rgb"]["sha256"]:
            raise ValueError(f"RGB hash mismatch: {image_path}")

        source = Image.open(image_path).convert("RGB")
        if source.size != (640, 480):
            raise ValueError(f"unexpected frame size: {source.size}")
        canvas = Image.new("RGB", (640, 574), "#071525")
        canvas.paste(source, (0, 42))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 9),
            VISIBLE_BANNER,
            fill="white",
            font=font,
        )
        draw.text(
            (435, 9),
            f"FRAME {event['sequence_index'] + 1:02d}/14",
            fill="#ffd166",
            font=font,
        )

        legend_y = 48
        for detection in event["detections"]:
            colour = colours[detection["class_name"]]
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            draw.rectangle((x1, y1 + 42, x2, y2 + 42), outline=colour, width=3)
            position = detection["position"]
            label = (
                f"{detection['track_id']} {detection['class_name']} "
                f"p={detection['confidence']:.2f} xyz=({position[0]:.2f},"
                f"{position[1]:.2f},{position[2]:.2f})m"
            )
            draw.rectangle((5, legend_y - 2, 474, legend_y + 11), fill="#071525")
            draw.text((9, legend_y), label, fill=colour, font=font)
            legend_y += 15

        draw.rectangle((0, 522, 639, 573), fill="#071525")
        draw.text(
            (10, 530),
            f"STATE={trace['final_state']} | {trace['reason']}",
            fill="#ffd166",
            font=font,
        )
        draw.text(
            (10, 547),
            f"latency={event['inference_latency_ms']:.1f} ms | world={event['world_frame']} | "
            f"localized={sum(1 for item in event['detections'] if item['localized'])}/"
            f"{len(event['detections'])}",
            fill="white",
            font=font,
        )
        draw.text(
            (10, 562),
            "Official-format output begins at frame 03 after 3-hit confirmation",
            fill="#a7c7e7",
            font=font,
        )
        frames.append(canvas)
        frame_records.append(
            {
                "sequence_index": event["sequence_index"],
                "frame_id": event["frame_id"],
                "rgb_sha256": record["rgb"]["sha256"],
                "event_record_id": event["record_id"],
                "final_state": trace["final_state"],
                "localized_detections": sum(1 for item in event["detections"] if item["localized"]),
            }
        )

    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=750,
        loop=0,
        optimize=False,
        disposal=2,
    )
    manifest = {
        "schema_version": 2,
        "protocol_id": "level_b_replay_v2",
        "artifact": str(GIF_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256_path(GIF_PATH),
        "frame_count": len(frames),
        "selection_rule": "all exported frames in ascending sequence_index",
        "danger_target_truth_used": False,
        "pose_ground_truth_used": True,
        "pose_source_topic": "/Odometry_gazebo",
        "visible_banner": VISIBLE_BANNER,
        "source_bindings": {
            "input_manifest_sha256": sha256_path(input_manifest_path),
            "events_sha256": sha256_path(event_path),
            "trace_sha256": sha256_path(trace_path),
        },
        "frames": frame_records,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASSED: wrote {GIF_PATH} with {len(frames)} target-truth-free, "
        "Gazebo-pose-oracle debug frames"
    )


if __name__ == "__main__":
    main()
