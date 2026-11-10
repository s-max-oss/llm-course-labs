from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_replay.replay import load_replay_protocol  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_storyboard() -> None:
    protocol = load_replay_protocol(PROJECT_ROOT, PROJECT_ROOT / "configs" / "replay_protocol_v1.json")
    policy = protocol["storyboard_policy"]
    gif_path = PROJECT_ROOT / policy["artifact"]
    manifest_path = PROJECT_ROOT / policy["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version", "protocol_id", "artifact", "artifact_sha256",
        "selection_rule", "truth_used_for_selection", "frame_count", "states",
        "cases", "limitations",
    }
    if set(manifest) != expected_keys:
        raise ValueError("storyboard manifest schema changed")
    if manifest["schema_version"] != 1 or manifest["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("storyboard identity changed")
    if manifest["artifact"] != policy["artifact"]:
        raise ValueError("storyboard artifact path changed")
    digest = hashlib.sha256(gif_path.read_bytes()).hexdigest()
    if digest != manifest["artifact_sha256"]:
        raise ValueError("storyboard GIF hash mismatch")
    if manifest["truth_used_for_selection"] != policy["truth_used_for_selection"]:
        raise ValueError("storyboard selection truth policy changed")
    if manifest["selection_rule"] != policy["selection_rule"]:
        raise ValueError("storyboard selection rule changed")
    if manifest["states"] != policy["states_in_order"]:
        raise ValueError("storyboard state order changed")

    traces = load_jsonl(PROJECT_ROOT / protocol["outputs"]["decision_trace"])
    events = {item["record_id"]: item for item in load_jsonl(PROJECT_ROOT / protocol["outputs"]["events"])}
    replay_manifest = {
        item["record_id"]: item for item in load_jsonl(PROJECT_ROOT / protocol["outputs"]["manifest"])
    }
    expected_cases = []
    per_state = policy["cases_per_state"]
    for state in policy["states_in_order"]:
        selected = sorted(
            (item for item in traces if item["final_state"] == state),
            key=lambda item: item["sequence_index"],
        )[:per_state]
        if len(selected) != per_state:
            raise ValueError(f"insufficient {state} cases in decision trace")
        for rank, trace in enumerate(selected, 1):
            event = events[trace["event_record_id"]]
            record = replay_manifest[event["record_id"]]
            expected_cases.append({
                "sample_id": trace["sample_id"],
                "sequence_index": trace["sequence_index"],
                "final_state": trace["final_state"],
                "selection_rank_within_state": rank,
                "input_sha256": record["input_sha256"],
                "event_record_id": event["record_id"],
            })
    if manifest["cases"] != expected_cases:
        raise ValueError("storyboard cases are not the frozen first-by-sequence selection")
    expected_count = len(policy["states_in_order"]) * per_state
    if manifest["frame_count"] != expected_count:
        raise ValueError("storyboard frame count changed")
    expected_counts = Counter({state: per_state for state in policy["states_in_order"]})
    if Counter(item["final_state"] for item in manifest["cases"]) != expected_counts:
        raise ValueError("unexpected storyboard state coverage")
    with Image.open(gif_path) as image:
        if image.n_frames != expected_count or image.size != (640, 560):
            raise ValueError("unexpected storyboard dimensions or frame count")
    print(f"PASSED: Level-A storyboard has {expected_count} independently verified label-free cases")


if __name__ == "__main__":
    validate_storyboard()
