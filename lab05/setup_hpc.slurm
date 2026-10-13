#!/bin/bash
#SBATCH --job-name=lab05-setup
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=setup-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=45
export HF_HUB_ETAG_TIMEOUT=15
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if [ ! -x .venv/bin/python ]; then
    "$HOME/.conda/envs/yolo/bin/python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --retries 1 --timeout 25
.venv/bin/python bootstrap_model.py
