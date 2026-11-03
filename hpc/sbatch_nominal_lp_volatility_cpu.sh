#!/bin/bash -l

# One fixed-volatility PPO policy per array task (5 volatilities x 3 seeds).
#SBATCH -J nominal_lp_vol
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2-00:00:00
#SBATCH --array=0-14
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$PROJECT_DIR/experiments/sweep_nominal_lp_volatility.py" ] || [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "ERROR: no SAiFE_gym checkout at $PROJECT_DIR. Set SAIFE_PROJECT_DIR or submit from the project root." >&2
  exit 1
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)
VENV_DIR=${SAIFE_VENV_DIR:-$PROJECT_DIR/venv}
if [[ "$VENV_DIR" != /* ]]; then
  VENV_DIR="$PROJECT_DIR/$VENV_DIR"
fi

if [ -n "${SAIFE_PYTHON_MODULE:-}" ]; then
  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: SAIFE_PYTHON_MODULE is set, but 'module' is unavailable." >&2
    exit 1
  fi
  module purge
  module load "$SAIFE_PYTHON_MODULE"
fi
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: no virtualenv found at $VENV_DIR. Set SAIFE_VENV_DIR or run hpc/setup_env.sh." >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
if [ -z "${MPLCONFIGDIR:-}" ]; then
  MPLCONFIGDIR=$(mktemp -d "${TMPDIR:-/tmp}/saife_matplotlib.XXXXXX")
fi
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR"

read -r -a SWEEP_SIGMAS <<< "${SIGMA_VALUES:-0.03 0.035 0.04 0.045 0.05}"
read -r -a SWEEP_SEEDS <<< "${TRAINING_SEEDS:-43 44 45}"
read -r -a SWEEP_WIDTHS <<< "${HALF_WIDTHS:-2 5 10 15 20 25 30 35 40 45 50}"
SWEEP_JOBS=$((${#SWEEP_SIGMAS[@]} * ${#SWEEP_SEEDS[@]}))
SWEEP_TASK=${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}
if [[ ! "$SWEEP_TASK" =~ ^[0-9]+$ ]] || (( SWEEP_TASK >= SWEEP_JOBS )); then
  echo "ERROR: array task index is outside the configured volatility/seed grid." >&2
  exit 1
fi

python -u experiments/sweep_nominal_lp_volatility.py train \
  --task-index "$SWEEP_TASK" \
  --sigma-values "${SWEEP_SIGMAS[@]}" \
  --seeds "${SWEEP_SEEDS[@]}" \
  --half-widths "${SWEEP_WIDTHS[@]}" \
  --output-dir "${OUTPUT_DIR:-experiments/results/nominal_lp_volatility}" \
  --total-timesteps "${TOTAL_TIMESTEPS:-10000000}" \
  --num-trajectories "${NUM_TRAJECTORIES:-100}" \
  --n-steps "${N_STEPS:-10000}" \
  --decision-stride "${DECISION_STRIDE:-100}" \
  --tau "${TAU:-50}" \
  --initial-wealth "${INITIAL_WEALTH:-1000}" \
  --background-liquidity "${BACKGROUND_LIQUIDITY:-30000}" \
  --trade-size-notional "${TRADE_SIZE_NOTIONAL:-250}" \
  --price-impact-depth-window "${PRICE_IMPACT_DEPTH_WINDOW:-15}" \
  --nominal-arrival-rate "${NOMINAL_ARRIVAL_RATE:-300}" \
  --nominal-gas-cost "${NOMINAL_GAS_COST:-2}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --n-eval-episodes "${N_EVAL_EPISODES:-10}" \
  --validation-seed "${VALIDATION_SEED:-100042}" \
  --evaluation-seed "${EVALUATION_SEED:-200042}" \
  --convergence-eval-every-rollouts "${CONVERGENCE_EVAL_EVERY_ROLLOUTS:-5}" \
  --convergence-n-eval-episodes "${CONVERGENCE_N_EVAL_EPISODES:-1}"
