#!/bin/bash -l

# Submit with --dependency=afterok:<array-job-id> and INPUT_DIR set to its sweep.
#SBATCH -J lp_vectorization_wallclock_plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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
if [ -z "${INPUT_DIR:-}" ]; then
  echo "ERROR: set INPUT_DIR to the completed sweep_<array-job-id> directory." >&2
  exit 1
fi

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
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

echo "Plotting completed vectorization sweep: $INPUT_DIR"
python -u experiments/vectorization_wallclock.py plot \
  --input-dir "$INPUT_DIR" \
  --output-dir "${PLOT_OUTPUT_DIR:-$INPUT_DIR/plots}"

echo "Plotting finished at: $(date)"
