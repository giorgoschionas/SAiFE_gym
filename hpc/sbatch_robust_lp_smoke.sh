#!/bin/bash -l

# Slurm smoke-test job for experiments/train_robust_lp_agent.py.

#SBATCH -J dr_ppo_smoke
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 02:00:00
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
TRAIN_DOMAINS_PER_RESET=${TRAIN_DOMAINS_PER_RESET:-2}
DECISION_STRIDE=${DECISION_STRIDE:-100}
TAU=${TAU:-50}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/results/domain_randomized_ppo/smoke}

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
if [ -z "${MPLCONFIGDIR:-}" ]; then
  MPLCONFIGDIR=$(mktemp -d "${TMPDIR:-/tmp}/saife_matplotlib.XXXXXX")
fi
export MPLCONFIGDIR

mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "SLURM job id: $SLURM_JOB_ID"
echo "Training domains per reset: $TRAIN_DOMAINS_PER_RESET"
echo "Decision stride: $DECISION_STRIDE"
echo "Tau: $TAU"

python -u experiments/train_robust_lp_agent.py \
  --smoke-test \
  --decision-stride "$DECISION_STRIDE" \
  --tau "$TAU" \
  --convergence-eval-every-rollouts 1 \
  --train-domains-per-reset "$TRAIN_DOMAINS_PER_RESET" \
  --output-dir "$OUTPUT_DIR"

echo "Job finished at: $(date)"
