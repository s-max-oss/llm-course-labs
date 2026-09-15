#!/bin/bash
#SBATCH --job-name=lab02-convlstm
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log

# Submit from the lab02 directory: sbatch run_hpc.slurm
# Set LAB02_PYTHON to a CUDA-enabled Python with torch/numpy/matplotlib installed.
# This cluster reports RealMemory=1 in Slurm; omit --mem as in its official examples.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit through sbatch from the experiment directory}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
LAB02_PYTHON="${LAB02_PYTHON:-$HOME/.conda/envs/yolo/bin/python}"
LAB02_OUTPUT="${LAB02_OUTPUT:-results}"
if [ -f fonts/msyh.ttc ]; then
    export LAB02_FONT="$PWD/fonts/msyh.ttc"
fi
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$LAB02_PYTHON" -m unittest test_convlstm -v
"$LAB02_PYTHON" convlstm_bounce.py --smoke --suite all --device cuda --micro-batch 64 --threads "$OMP_NUM_THREADS"
"$LAB02_PYTHON" convlstm_bounce.py --suite all --device cuda --micro-batch 64 --threads "$OMP_NUM_THREADS" --output "$LAB02_OUTPUT"
