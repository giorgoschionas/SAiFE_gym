#!/bin/bash -l

# Run the complete liquidity sweep sequentially, then save its combined summary.
# Submit from the checkout root after creating logs/ and sourcing hpc/.env.
# Arguments after the script name are forwarded to the Python sweep.
#SBATCH -J rl_background_sweep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$PROJECT_DIR/experiments/sweep_rl_background_liquidity.py" ] || [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
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
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
if [ -z "${MPLCONFIGDIR:-}" ]; then
  MPLCONFIGDIR=$(mktemp -d "${TMPDIR:-/tmp}/saife_matplotlib.XXXXXX")
fi
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR"

read -r -a SWEEP_BACKGROUNDS <<< "${BACKGROUND_VALUES:-10000 15000 20000 25000 30000}"
read -r -a SWEEP_SEEDS <<< "${TRAINING_SEEDS:-42}"

echo "Job started at: $(date)"
echo "Node: $(hostname); job: ${SLURM_JOB_ID:-local}; CPUs: $OMP_NUM_THREADS"
echo "Project: $PROJECT_DIR"
echo "Python: $(command -v python)"

python -u "$PROJECT_DIR/experiments/sweep_rl_background_liquidity.py" \
  --background-values "${SWEEP_BACKGROUNDS[@]}" \
  --seeds "${SWEEP_SEEDS[@]}" \
  --output-dir "${OUTPUT_DIR:-experiments/results/rl_background_liquidity}" \
  --total-timesteps "${TOTAL_TIMESTEPS:-100000000}" \
  --num-trajectories "${NUM_TRAJECTORIES:-100}" \
  --n-steps "${N_STEPS:-1000}" \
  --decision-stride "${DECISION_STRIDE:-100}" \
  --tau "${TAU:-50}" \
  --tick-stride "${TICK_STRIDE:-5}" \
  --initial-wealth "${INITIAL_WEALTH:-1000}" \
  --inventory-phi "${INVENTORY_PHI:-0.4}" \
  --nominal-sigma "${NOMINAL_SIGMA:-0.03}" \
  --nominal-arrival-rate "${NOMINAL_ARRIVAL_RATE:-300}" \
  --nominal-gas-cost "${NOMINAL_GAS_COST:-2}" \
  --alpha3 "${ALPHA3:-4000}" \
  --trade-size-notional "${TRADE_SIZE_NOTIONAL:-250}" \
  --price-impact-depth-window "${PRICE_IMPACT_DEPTH_WINDOW:-15}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --evaluation-seed "${EVALUATION_SEED:-100042}" \
  --n-eval-episodes "${N_EVAL_EPISODES:-10}" \
  --convergence-eval-every-rollouts "${CONVERGENCE_EVAL_EVERY_ROLLOUTS:-10}" \
  --convergence-n-eval-episodes "${CONVERGENCE_N_EVAL_EPISODES:-1}" \
  "$@"

echo "Job finished at: $(date)"
