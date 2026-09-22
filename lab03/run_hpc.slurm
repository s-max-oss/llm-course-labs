#!/bin/bash
#SBATCH --job-name=lab03-cpu
#SBATCH --partition=gpu-a30
#SBATCH --nodelist=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log

# CPU-only allocation: intentionally no --gres GPU request.
# Pin the CPU model for timing comparisons; change nodelist on other clusters.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit through sbatch}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
LAB03_PYTHON="${LAB03_PYTHON:-$HOME/.conda/envs/yolo/bin/python}"
LAB03_OUTPUT="${LAB03_OUTPUT:-results}"
hostname
"$LAB03_PYTHON" -m unittest test_min_llm -v
"$LAB03_PYTHON" run_suite.py --workers 3 --threads "$OMP_NUM_THREADS" --output "$LAB03_OUTPUT"
