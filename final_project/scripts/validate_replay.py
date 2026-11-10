from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_replay.contracts import ReplayContractError  # noqa: E402
from robot_replay.replay import load_replay_protocol, render_outputs  # noqa: E402

protocol = load_replay_protocol(PROJECT_ROOT, PROJECT_ROOT / "configs" / "replay_protocol_v1.json")
rendered = render_outputs(PROJECT_ROOT, protocol)
for relative_path, expected in rendered.items():
    path = PROJECT_ROOT / relative_path
    if not path.exists() or path.read_bytes() != expected:
        raise ReplayContractError(f"replay output differs from deterministic rebuild: {relative_path}")
summary_path = PROJECT_ROOT / protocol["outputs"]["summary"]
print(f"PASSED: deterministic replay validated ({summary_path})")
