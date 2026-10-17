#!/bin/bash -l

# Two nominal PPO agents: endpoint pairs (0) and center/half-width pairs (1).

#SBATCH -J nominal_action_grids
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 2-00:00:00
#SBATCH --array=0-1
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

case "${SLURM_ARRAY_TASK_ID:-}" in
  0)
    CONFIGURATION=endpoint_pairs
    ACTION_COUNT=211
    ACTION_ARGS=(--tau 50 --tick-stride 5)
    ;;
  1)
    CONFIGURATION=center_half_width
    ACTION_COUNT=55
    ACTION_ARGS=(--tau 500 --tick-stride 5
      --center-magnitudes 50 100 150 200
      --half-widths 50 100 150 200 250 300)
    ;;
  *)
    echo "ERROR: SLURM_ARRAY_TASK_ID must be 0 (endpoint pairs) or 1 (center/half-width)." >&2
    exit 1
    ;;
esac

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

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-100}
N_STEPS=${N_STEPS:-1000}
DECISION_STRIDE=${DECISION_STRIDE:-100}
SEED=${SEED:-42}
EVALUATION_SEED=${EVALUATION_SEED:-100042}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-10}
CONVERGENCE_EVAL_EVERY_ROLLOUTS=${CONVERGENCE_EVAL_EVERY_ROLLOUTS:-10}
CONVERGENCE_N_EVAL_EPISODES=${CONVERGENCE_N_EVAL_EPISODES:-1}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
ALPHA3=${ALPHA3:-4000.0}
INITIAL_WEALTH=${INITIAL_WEALTH:-1000.0}
INVENTORY_PHI=${INVENTORY_PHI:-0.4}
NOMINAL_SIGMA=${NOMINAL_SIGMA:-0.03}
NOMINAL_ARRIVAL_RATE=${NOMINAL_ARRIVAL_RATE:-300.0}
NOMINAL_GAS_COST=${NOMINAL_GAS_COST:-2.0}
PERIODIC_REBALANCE_EVERY=${PERIODIC_REBALANCE_EVERY:-100}
PERIODIC_WIDTH=${PERIODIC_WIDTH:-50}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/results/nominal_action_grids}
TASK_OUTPUT_DIR="$OUTPUT_DIR/$CONFIGURATION/seed_$SEED"

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
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
mkdir -p logs "$TASK_OUTPUT_DIR"

echo "Job started at: $(date)"
echo "Working directory: $PROJECT_DIR"
echo "Slurm array=${SLURM_ARRAY_JOB_ID:-unknown} task=$SLURM_ARRAY_TASK_ID CPUs=$OMP_NUM_THREADS"
echo "Nominal PPO: configuration=$CONFIGURATION actions=$ACTION_COUNT seed=$SEED"
echo "Training: simulator_transitions=$TOTAL_TIMESTEPS trajectories=$NUM_TRAJECTORIES n_steps=$N_STEPS decision_stride=$DECISION_STRIDE"
echo "Nominal market: sigma=$NOMINAL_SIGMA arrival_rate=$NOMINAL_ARRIVAL_RATE gas=$NOMINAL_GAS_COST"
echo "Nominal evaluation: seed=$EVALUATION_SEED episodes=$N_EVAL_EPISODES"
echo "Output directory: $TASK_OUTPUT_DIR"

python -u experiments/train_robust_lp_agent.py \
  --training-mode nominal \
  --evaluation-scope nominal \
  --output-dir "$TASK_OUTPUT_DIR" \
  "${ACTION_ARGS[@]}" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --terminal-time 1.0 \
  --n-steps "$N_STEPS" \
  --decision-stride "$DECISION_STRIDE" \
  --seed "$SEED" \
  --evaluation-seed "$EVALUATION_SEED" \
  --n-eval-episodes "$N_EVAL_EPISODES" \
  --convergence-eval-every-rollouts "$CONVERGENCE_EVAL_EVERY_ROLLOUTS" \
  --convergence-n-eval-episodes "$CONVERGENCE_N_EVAL_EPISODES" \
  --learning-rate "$LEARNING_RATE" \
  --alpha3 "$ALPHA3" \
  --initial-wealth "$INITIAL_WEALTH" \
  --inventory-phi "$INVENTORY_PHI" \
  --nominal-sigma "$NOMINAL_SIGMA" \
  --nominal-arrival-rate "$NOMINAL_ARRIVAL_RATE" \
  --nominal-gas-cost "$NOMINAL_GAS_COST" \
  --arrival-alpha2 0.0 \
  --kernel-beta 0.5 \
  --kernel-window 10 \
  --liquidity-scale 1000000 \
  --trade-size-notional 250 \
  --price-impact-depth-window 10 \
  --price-impact-min-depth 1e-12 \
  --periodic-rebalance-every "$PERIODIC_REBALANCE_EVERY" \
  --periodic-width "$PERIODIC_WIDTH"

echo "Job finished at: $(date)"
