#!/bin/bash
#SBATCH --job-name=lab06-jointdev
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=jointdev-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
"${HOME}/llm_course_lab05/.venv/bin/python" finish_grounding.py --model "${HOME}/llm_course_lab05/model" --adapter improvement/vision_compact_v1/step1200/adapter --offline --output improvement/joint_development_v1 --development-only
