#!/usr/bin/env python3
"""Generate the frozen, label-free development proposal queue for cascade feasibility."""

from __future__ import annotations

import argparse
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
    sha256_file,
    validate_baseline_policy,
    write_json,
    write_jsonl,
)


def build(protocol_path: Path, config_path: Path, output_root: Path) -> Path:
    protocol = load_cascade_protocol(PROJECT_ROOT, protocol_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    jobs = build_cascade_jobs(PROJECT_ROOT, protocol)
    validate_cascade_jobs(jobs, protocol, set(config["inference_forbidden_fields"]))
    output_root.mkdir(parents=True, exist_ok=True)
    queue_path = output_root / "development.jsonl"
    write_jsonl(queue_path, jobs)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "artifact": "label_free_rfdetr_proposal_smolvlm_jobs",
        "partition": "development",
        "ground_truth_available_to_generation": False,
        "ground_truth_available_to_inference": False,
        "deployment_eligible": False,
        "source_files": {
            "cascade_protocol": {
                "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(protocol_path),
            },
            "baseline_config": {
                "path": config_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(config_path),
            },
            **protocol["parent_artifacts"],
        },
        "queue": {
            "path": queue_path.name,
            "sha256": sha256_file(queue_path),
            "count": len(jobs),
            "routed_image_count": len({item["sample_id"] for item in jobs}),
        },
    }
    manifest_path = output_root / "protocol.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs" / "cascade_feasibility_v1.json"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "cascade_jobs",
    )
    args = parser.parse_args()
    try:
        path = build(args.protocol.resolve(), args.config.resolve(), args.output_root.resolve())
    except (BaselineContractError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
