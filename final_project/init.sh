#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -r requirements.txt
python3 scripts/build_manifest.py --config configs/data_sources.local.json
python3 scripts/build_candidate_manifest.py
python3 scripts/build_pilot_splits.py
python3 scripts/prepare_baseline_jobs.py
python3 scripts/prepare_smolvlm_b2_jobs.py
python3 scripts/prepare_cascade_jobs.py
python3 scripts/validate_manifest.py --config configs/data_sources.local.json --verify-hashes
python3 scripts/validate_pilot_splits.py
python3 scripts/validate_baseline_jobs.py
python3 scripts/validate_smolvlm_b2_jobs.py
python3 scripts/validate_cascade_jobs.py
python3 scripts/validate_rfdetr_run.py
python3 scripts/validate_cascade_run.py
python3 scripts/validate_replay_contract.py
python3 scripts/build_replay.py
python3 scripts/validate_replay.py
python3 scripts/generate_replay_storyboard.py
python3 scripts/validate_replay_storyboard.py
python3 scripts/validate_replay_run.py
python3 scripts/generate_level_b_storyboard.py
python3 scripts/validate_level_b_run.py
python3 -m unittest discover -s tests -v
