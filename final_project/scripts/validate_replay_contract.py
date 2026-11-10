from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_replay.contracts import load_replay_contract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen replay contract")
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "configs" / "replay_event_contract_v1.json",
    )
    args = parser.parse_args()
    contract = load_replay_contract(args.contract)
    print(
        "PASSED: frozen {contract_id}; replay_level={level}; "
        "world_coordinate_output_allowed={world}".format(
            contract_id=contract["contract_id"],
            level=contract["replay_level"],
            world=contract["g0_decision"]["world_coordinate_output_allowed"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
