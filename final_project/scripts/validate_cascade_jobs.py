#!/usr/bin/env python3
"""Independently reconstruct and validate the frozen cascade development queue."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_vlm_baselines.cascade import (  # noqa: E402
    build_cascade_jobs,
    load_cascade_protocol,
    validate_cascade_jobs,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    sha256_file,
)


def validate() -> None:
    protocol_path = PROJECT_ROOT / "configs" / "cascade_feasibility_v1.json"
    config = json.loads((PROJECT_ROOT / "configs" / "baselines.json").read_text(encoding="utf-8"))
    output_root = PROJECT_ROOT / "experiments" / "pilot_v1" / "cascade_jobs"
    queue_path = output_root / "development.jsonl"
    manifest_path = output_root / "protocol.json"
    protocol = load_cascade_protocol(PROJECT_ROOT, protocol_path)
    actual = load_jsonl(queue_path)
    expected = build_cascade_jobs(PROJECT_ROOT, protocol)
    validate_cascade_jobs(actual, protocol, set(config["inference_forbidden_fields"]))
    if actual != expected:
        raise BaselineContractError("cascade jobs differ from independent reconstruction")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != protocol["protocol_id"]
        or manifest.get("ground_truth_available_to_generation") is not False
        or manifest.get("ground_truth_available_to_inference") is not False
        or manifest.get("queue")
        != {
            "path": "development.jsonl",
            "sha256": sha256_file(queue_path),
            "count": 24,
            "routed_image_count": 22,
        }
    ):
        raise BaselineContractError("cascade jobs manifest differs from the frozen queue")
    if manifest.get("source_files", {}).get("cascade_protocol", {}).get("sha256") != sha256_file(
        protocol_path
    ):
        raise BaselineContractError("cascade manifest does not bind the frozen protocol")


def main() -> int:
    try:
        validate()
    except (BaselineContractError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("PASSED: 24 label-free RF-DETR proposal jobs across 22 development images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
