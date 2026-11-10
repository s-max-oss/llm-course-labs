$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Invoke-Python {
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: python $($Args -join ' ')"
    }
}

Invoke-Python scripts/build_manifest.py --config configs/data_sources.local.json
Invoke-Python scripts/build_candidate_manifest.py
Invoke-Python scripts/build_pilot_splits.py
Invoke-Python scripts/prepare_baseline_jobs.py
Invoke-Python scripts/prepare_smolvlm_b2_jobs.py
Invoke-Python scripts/prepare_cascade_jobs.py
Invoke-Python scripts/validate_manifest.py --config configs/data_sources.local.json --verify-hashes
Invoke-Python scripts/validate_pilot_splits.py
Invoke-Python scripts/validate_baseline_jobs.py
Invoke-Python scripts/validate_smolvlm_b2_jobs.py
Invoke-Python scripts/validate_rfdetr_run.py
Invoke-Python scripts/validate_cascade_jobs.py
Invoke-Python scripts/validate_cascade_run.py
Invoke-Python scripts/validate_replay_contract.py
Invoke-Python scripts/build_replay.py
Invoke-Python scripts/validate_replay.py
Invoke-Python scripts/generate_replay_storyboard.py
Invoke-Python scripts/validate_replay_storyboard.py
Invoke-Python scripts/validate_replay_run.py
Invoke-Python scripts/generate_level_b_storyboard.py
Invoke-Python scripts/validate_level_b_run.py
Invoke-Python -m unittest discover -s tests -v
