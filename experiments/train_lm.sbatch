#!/bin/bash --login
#SBATCH -J lm-train
#SBATCH -c 32
#SBATCH -t 4-00:00
#SBATCH -o lm-train.%j.out

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
if [ ! -f "$REPO_ROOT/experiments/train_lm.sh" ]; then
  echo "Submit from the repo root"
  exit 1
fi

if [ -n "${VENV:-}" ]; then
  source "$VENV/bin/activate"
fi
if [ -n "${4:-}" ]; then
  export WANDB_PROJECT="$4"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.triton_cache}"

NGPU=$(nvidia-smi -L | wc -l)
bash "$REPO_ROOT/experiments/train_lm.sh" --model "$1" --size "$2" --data-dir "$3" --ngpu "$NGPU"
