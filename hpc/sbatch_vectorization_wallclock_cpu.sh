#!/bin/bash -l

# Train for one hour, then evaluate fixed-size held-out batches offline.
#SBATCH -J lp_vectorization_wallclock
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$PROJECT_DIR/experiments/vectorization_wallclock.py" ] || [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "ERROR: no SAiFE_gym checkout at $PROJECT_DIR. Set SAIFE_PROJECT_DIR or submit from the project root." >&2
  exit 1
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)
VENV_DIR=${SAIFE_VENV_DIR:-$PROJECT_DIR/venv}
if [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$PROJECT_DIR/$VENV_DIR"
fi

case "${SLURM_ARRAY_TASK_ID:-}" in
  [0-2]) ;;
  *) echo "ERROR: SLURM_ARRAY_TASK_ID must be 0, 1, or 2." >&2; exit 1 ;;
esac
if [ -z "${SLURM_ARRAY_JOB_ID:-}" ]; then
  echo "ERROR: SLURM_ARRAY_JOB_ID is required to identify this sweep." >&2
  exit 1
fi
TRAJECTORY_COUNTS=(10 100 1000)
NUM_TRAJECTORIES=${TRAJECTORY_COUNTS[$SLURM_ARRAY_TASK_ID]}
TRAINING_SECONDS=${TRAINING_SECONDS:-3600}
CHECKPOINT_SECONDS=${CHECKPOINT_SECONDS:-300}
EVALUATION_EPISODES=${EVALUATION_EPISODES:-1000}
EVALUATION_SEED=${EVALUATION_SEED:-100042}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/results/vectorization_wallclock}
SWEEP_DIR="$OUTPUT_DIR/sweep_$SLURM_ARRAY_JOB_ID"

if [ -n "${SAIFE_PYTHON_MODULE:-}" ]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: SAIFE_PYTHON_MODULE is set, but 'module' is unavailable. Initialize your cluster's module system or unset SAIFE_PYTHON_MODULE." >&2
    exit 1
  fi
  module purge
  module load "$SAIFE_PYTHON_MODULE"
fi
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: no virtualenv found at $VENV_DIR. Run 'bash hpc/setup_env.sh' or set SAIFE_VENV_DIR." >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
if [ -z "${MPLCONFIGDIR:-}" ]; then
  MPLCONFIGDIR=$(mktemp -d "${TMPDIR:-/tmp}/saife_matplotlib.XXXXXX")
fi
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR"
mkdir -p logs

echo "Job started at: $(date) on $(hostname)"
echo "Sweep: $SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID cpus=$OMP_NUM_THREADS"
echo "Nominal training: trajectories=$NUM_TRAJECTORIES training_seconds=$TRAINING_SECONDS seed=$SEED"
echo "Sweep directory: $SWEEP_DIR"

python -u experiments/vectorization_wallclock.py train \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --training-seconds "$TRAINING_SECONDS" \
  --checkpoint-seconds "$CHECKPOINT_SECONDS" \
  --evaluation-episodes "$EVALUATION_EPISODES" \
  --evaluation-seed "$EVALUATION_SEED" \
  --seed "$SEED" \
  --output-dir "$SWEEP_DIR"

python -u experiments/vectorization_wallclock.py evaluate \
  --run-dir "$SWEEP_DIR/trajectories_$NUM_TRAJECTORIES"

echo "Job finished at: $(date)"
