from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_replay.replay import load_replay_protocol  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "replay_v1" / "outputs"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_cases(traces, states, per_state):
    selected = []
    for state in states:
        candidates = sorted(
            (item for item in traces if item["final_state"] == state),
            key=lambda item: item["sequence_index"],
        )
        if len(candidates) < per_state:
            raise ValueError(f"insufficient {state} cases")
        selected.extend(candidates[:per_state])
    return selected


def main() -> None:
    protocol = load_replay_protocol(PROJECT_ROOT, PROJECT_ROOT / "configs" / "replay_protocol_v1.json")
    policy = protocol["storyboard_policy"]
    states = tuple(policy["states_in_order"])
    per_state = int(policy["cases_per_state"])
    gif_path = PROJECT_ROOT / policy["artifact"]
    manifest_path = PROJECT_ROOT / policy["manifest"]
    data_config = json.loads((PROJECT_ROOT / "configs" / "data_sources.local.json").read_text(encoding="utf-8"))
    image_root = Path(data_config["image_root"])
    manifests = {item["record_id"]: item for item in load_jsonl(OUTPUT_ROOT / "replay_manifest.jsonl")}
    events = {item["record_id"]: item for item in load_jsonl(OUTPUT_ROOT / "danger_detection_events.jsonl")}
    traces = load_jsonl(OUTPUT_ROOT / "decision_trace.jsonl")
    cases = select_cases(traces, states, per_state)
    frames = []
    case_records = []
    colours = {
        "danger_red_sphere": "#ff4d4d",
        "distractor_red_box": "#ffb347",
        "distractor_green_sphere": "#66ff99",
        "distractor_green_wall": "#66ccff",
    }
    font = ImageFont.load_default()
    for trace in cases:
        event = events[trace["event_record_id"]]
        manifest = manifests[event["record_id"]]
        image_path = image_root / manifest["image"]
        if sha256_path(image_path) != manifest["input_sha256"]:
            raise ValueError(f"image hash mismatch: {image_path}")
        source = Image.open(image_path).convert("RGB").resize((640, 480))
        canvas = Image.new("RGB", (640, 560), "#071525")
        canvas.paste(source, (0, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "OFFLINE LEVEL A STORYBOARD | NON-TEMPORAL | NO 3D", fill="white", font=font)
        legend_rows = []
        for detection in event["detections"]:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
            box = (x1, y1 + 40, x2, y2 + 40)
            colour = colours[detection["class_name"]]
            draw.rectangle(box, outline=colour, width=3)
            short_id = f"#{detection['prediction_index']}"
            draw.rectangle((x1, y1 + 40, x1 + 18, y1 + 52), fill="#071525")
            draw.text((x1 + 2, y1 + 41), short_id, fill=colour, font=font)
            legend_rows.append(
                (colour, f"{short_id} {detection['class_name']} {detection['confidence']:.2f}")
            )
        if legend_rows:
            legend_width = min(360, max(190, max(len(text) for _, text in legend_rows) * 7))
            draw.rectangle((4, 44, legend_width, 48 + 15 * len(legend_rows)), fill="#071525")
            for row_index, (colour, text) in enumerate(legend_rows):
                draw.text((9, 47 + 15 * row_index), text, fill=colour, font=font)
        draw.rectangle((0, 520, 639, 559), fill="#071525")
        draw.text((10, 528), f"{trace['sample_id']} | STATE={trace['final_state']}", fill="#ffd166", font=font)
        draw.text((10, 544), trace["reason"], fill="white", font=font)
        frames.append(canvas)
        case_records.append({
            "sample_id": trace["sample_id"],
            "sequence_index": trace["sequence_index"],
            "final_state": trace["final_state"],
            "selection_rank_within_state": 1 + sum(
                1 for item in case_records if item["final_state"] == trace["final_state"]
            ),
            "input_sha256": manifest["input_sha256"],
            "event_record_id": event["record_id"],
        })
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=900, loop=0, optimize=False)
    payload = {
        "schema_version": 1,
        "protocol_id": "robot_replay_level_a_v1",
        "artifact": policy["artifact"],
        "artifact_sha256": sha256_path(gif_path),
        "selection_rule": policy["selection_rule"],
        "truth_used_for_selection": policy["truth_used_for_selection"],
        "frame_count": len(frames),
        "states": list(states),
        "cases": case_records,
        "limitations": [
            "non-temporal seed-grouped storyboard",
            "no RGB-D localization or world coordinates",
            "HELD_UNLOCALIZED is not a confirmed or reported danger",
        ],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASSED: wrote {gif_path} with {len(frames)} deterministic cases")


if __name__ == "__main__":
    main()
