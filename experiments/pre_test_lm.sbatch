#!/bin/bash --login
#SBATCH -J lm-pre-test
#SBATCH -G 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -o lm-pre-test.%j.out

set -euo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
if [ -n "${VENV:-}" ]; then
  source "$VENV/bin/activate"
fi

python "$REPO_ROOT/experiments/pre_test_lm.py" --out "$REPO_ROOT/pre_test_results.$SLURM_JOB_ID.jsonl" "$@"
