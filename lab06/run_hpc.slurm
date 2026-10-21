#!/bin/bash
#SBATCH --job-name=lab06-rsi
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
PYTHON_BIN="${HOME}/llm_course_lab05/.venv/bin/python"
MODEL_DIR="${HOME}/llm_course_lab05/model"
test -x "$PYTHON_BIN"
test -f "$MODEL_DIR/config.json"
hostname
nvidia-smi
"$PYTHON_BIN" -c 'import sys,torch,transformers,peft,bitsandbytes; print("versions",sys.version.split()[0],torch.__version__,transformers.__version__,peft.__version__,bitsandbytes.__version__)'
"$PYTHON_BIN" hw6_rsi_localize.py --mode make_data
"$PYTHON_BIN" hw6_rsi_localize.py --mode make_sft
"$PYTHON_BIN" hw6_rsi_localize.py --mode infer --model "$MODEL_DIR" --device cuda --offline
"$PYTHON_BIN" hw6_rsi_localize.py --mode sft --steps 10 --model "$MODEL_DIR" --device cuda --offline
"$PYTHON_BIN" hw6_rsi_localize.py --mode eval --model "$MODEL_DIR" --device cuda --offline
"$PYTHON_BIN" hw6_rsi_localize.py --mode rsi --rounds 2 --steps 10 --model "$MODEL_DIR" --device cuda --offline
echo '[complete] lab06'
