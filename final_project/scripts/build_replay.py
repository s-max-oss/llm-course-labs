from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_replay.replay import load_replay_protocol, write_outputs  # noqa: E402

protocol_path = PROJECT_ROOT / "configs" / "replay_protocol_v1.json"
protocol = load_replay_protocol(PROJECT_ROOT, protocol_path)
write_outputs(PROJECT_ROOT, protocol)
print("PASSED: wrote deterministic Level-A replay outputs")
