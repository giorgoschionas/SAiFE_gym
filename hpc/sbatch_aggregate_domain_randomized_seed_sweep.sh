#!/bin/bash -l

# Aggregate a completed domain-randomized PPO training-seed sweep.

#SBATCH -J dr_ppo_aggregate
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$PROJECT_DIR/experiments/train_robust_lp_agent.py" ] || [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "ERROR: no SAiFE_gym checkout at $PROJECT_DIR. Set SAIFE_PROJECT_DIR or submit from the project root." >&2
  exit 1
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)
VENV_DIR=${SAIFE_VENV_DIR:-$PROJECT_DIR/venv}
if [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$PROJECT_DIR/$VENV_DIR"
fi
INPUT_DIR=${INPUT_DIR:-experiments/results/domain_randomized_ppo/seed_sweep}
OUTPUT_DIR=${OUTPUT_DIR:-$INPUT_DIR/aggregate}
BOOTSTRAP_RESAMPLES=${BOOTSTRAP_RESAMPLES:-10000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-42}
CONFIDENCE_LEVEL=${CONFIDENCE_LEVEL:-0.95}

if [ -n "${SAIFE_PYTHON_MODULE:-}" ]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: SAIFE_PYTHON_MODULE is set, but 'module' is unavailable. Initialize your cluster's module system or unset SAIFE_PYTHON_MODULE." >&2
    exit 1
  fi
  module purge
  module load "$SAIFE_PYTHON_MODULE"
fi
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: no virtualenv found at $VENV_DIR" >&2
  echo "Create it with 'bash hpc/setup_env.sh', or point SAIFE_VENV_DIR at an existing env." >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export NUMEXPR_NUM_THREADS=$SLURM_NTASKS
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
mkdir -p logs

echo "Aggregating training-seed runs under: $INPUT_DIR"
echo "Bootstrap: resamples=$BOOTSTRAP_RESAMPLES seed=$BOOTSTRAP_SEED confidence=$CONFIDENCE_LEVEL"
echo "Output dir: $OUTPUT_DIR"

python -u experiments/aggregate_domain_randomized_seed_sweep.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --bootstrap-resamples "$BOOTSTRAP_RESAMPLES" \
  --bootstrap-seed "$BOOTSTRAP_SEED" \
  --confidence-level "$CONFIDENCE_LEVEL"

echo "Aggregation finished at: $(date)"
