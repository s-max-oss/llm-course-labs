#!/bin/bash
#SBATCH --job-name=lab08-kd
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

set -euo pipefail

LAB_DIR="${LAB08_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$LAB_DIR" || ! -f "$LAB_DIR/distill_lm.py" ]]; then
  echo "LAB08_DIR（或 SLURM_SUBMIT_DIR）必须指向含 distill_lm.py 的实验八目录" >&2
  exit 2
fi
PYTHON_BIN="${LAB08_PYTHON:-$LAB_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "找不到可执行 Python：$PYTHON_BIN；请设置 LAB08_PYTHON" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c "import torch; print(torch.__version__)"; then
  echo "所选 Python 无法导入 PyTorch：$PYTHON_BIN" >&2
  exit 2
fi

cd "$LAB_DIR"
mkdir -p results runs_distill_hpc
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONIOENCODING=utf-8

"$PYTHON_BIN" -u distill_lm.py \
  --mode "${LAB08_MODE:-full}" \
  --retrain \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out-dir "${LAB08_OUT_DIR:-runs_distill_hpc}" \
  2>&1 | tee results/hpc_full.log
