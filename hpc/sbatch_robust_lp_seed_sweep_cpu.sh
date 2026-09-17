#!/bin/bash -l

# Slurm array job for domain-randomized LP PPO seed replicates.

#SBATCH -J dr_ppo_seed_sweep
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
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

SEEDS=(43 44 45 46 47 48 49 50 51 52)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EVALUATION_SEED=${EVALUATION_SEED:-100042}

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-10000000}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-100}
TRAIN_DOMAINS_PER_RESET=${TRAIN_DOMAINS_PER_RESET:-$NUM_TRAJECTORIES}
N_STEPS=${N_STEPS:-1000}
DECISION_STRIDE=${DECISION_STRIDE:-100}
TAU=${TAU:-50}
ALPHA3=${ALPHA3:-4000.0}
INITIAL_WEALTH=${INITIAL_WEALTH:-1000.0}
INVENTORY_PHI=${INVENTORY_PHI:-0.4}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-10}
CONVERGENCE_EVAL_EVERY_ROLLOUTS=${CONVERGENCE_EVAL_EVERY_ROLLOUTS:-10}
CONVERGENCE_N_EVAL_EPISODES=${CONVERGENCE_N_EVAL_EPISODES:-1}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
PERIODIC_REBALANCE_EVERY=${PERIODIC_REBALANCE_EVERY:-100}
PERIODIC_WIDTH=${PERIODIC_WIDTH:-50}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/results/domain_randomized_ppo/seed_sweep}/seed_${SEED}

NOMINAL_SIGMA=${NOMINAL_SIGMA:-0.03}
NOMINAL_ARRIVAL_RATE=${NOMINAL_ARRIVAL_RATE:-300.0}
NOMINAL_GAS_COST=${NOMINAL_GAS_COST:-2.0}

TRAIN_SIGMA_MIN=${TRAIN_SIGMA_MIN:-0.01}
TRAIN_SIGMA_MAX=${TRAIN_SIGMA_MAX:-0.05}
TRAIN_ARRIVAL_RATE_MIN=${TRAIN_ARRIVAL_RATE_MIN:-200.0}
TRAIN_ARRIVAL_RATE_MAX=${TRAIN_ARRIVAL_RATE_MAX:-400.0}

EVAL_IN_DISTRIBUTION_SIGMA_VALUES=${EVAL_IN_DISTRIBUTION_SIGMA_VALUES:-"0.015 0.030 0.045"}
EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES=${EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES:-"250.0 300.0 350.0"}
EVAL_STRESS_SIGMA_VALUES=${EVAL_STRESS_SIGMA_VALUES:-"0.065 0.08"}
EVAL_STRESS_ARRIVAL_RATE_VALUES=${EVAL_STRESS_ARRIVAL_RATE_VALUES:-"150.0 450.0"}

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
echo "SLURM job id: $SLURM_JOB_ID array=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "Resources: partition=$SLURM_JOB_PARTITION nodes=$SLURM_JOB_NUM_NODES tasks=$SLURM_NTASKS"
echo "Training: timesteps=$TOTAL_TIMESTEPS trajectories=$NUM_TRAJECTORIES n_steps=$N_STEPS decision_stride=$DECISION_STRIDE tau=$TAU alpha3=$ALPHA3 initial_wealth=$INITIAL_WEALTH inventory_phi=$INVENTORY_PHI seed=$SEED"
echo "Fixed evaluation seed: $EVALUATION_SEED"
echo "Periodic baseline: rebalance_every=$PERIODIC_REBALANCE_EVERY width=$PERIODIC_WIDTH"
echo "Fixed gas cost: $NOMINAL_GAS_COST"
echo "Training domain: sigma=[$TRAIN_SIGMA_MIN, $TRAIN_SIGMA_MAX] arrival=[$TRAIN_ARRIVAL_RATE_MIN, $TRAIN_ARRIVAL_RATE_MAX] domains_per_reset=$TRAIN_DOMAINS_PER_RESET"
echo "Evaluation: full Cartesian grid including mixed regimes; episodes=$N_EVAL_EPISODES"
echo "In-distribution evaluation axes: sigma=[$EVAL_IN_DISTRIBUTION_SIGMA_VALUES] arrival=[$EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES]"
echo "Stress evaluation axes: sigma=[$EVAL_STRESS_SIGMA_VALUES] arrival=[$EVAL_STRESS_ARRIVAL_RATE_VALUES]"
echo "Evaluation groups: in_distribution, sigma_only_stress, arrival_only_stress, stress"
echo "Convergence evaluation: every_rollouts=$CONVERGENCE_EVAL_EVERY_ROLLOUTS episodes=$CONVERGENCE_N_EVAL_EPISODES"
echo "Output dir: $OUTPUT_DIR"

python -u experiments/train_robust_lp_agent.py \
  --output-dir "$OUTPUT_DIR" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --n-steps "$N_STEPS" \
  --decision-stride "$DECISION_STRIDE" \
  --tau "$TAU" \
  --alpha3 "$ALPHA3" \
  --initial-wealth "$INITIAL_WEALTH" \
  --inventory-phi "$INVENTORY_PHI" \
  --seed "$SEED" \
  --evaluation-seed "$EVALUATION_SEED" \
  --n-eval-episodes "$N_EVAL_EPISODES" \
  --convergence-eval-every-rollouts "$CONVERGENCE_EVAL_EVERY_ROLLOUTS" \
  --convergence-n-eval-episodes "$CONVERGENCE_N_EVAL_EPISODES" \
  --learning-rate "$LEARNING_RATE" \
  --periodic-rebalance-every "$PERIODIC_REBALANCE_EVERY" \
  --periodic-width "$PERIODIC_WIDTH" \
  --nominal-sigma "$NOMINAL_SIGMA" \
  --nominal-arrival-rate "$NOMINAL_ARRIVAL_RATE" \
  --nominal-gas-cost "$NOMINAL_GAS_COST" \
  --train-sigma-range "$TRAIN_SIGMA_MIN" "$TRAIN_SIGMA_MAX" \
  --train-arrival-rate-range "$TRAIN_ARRIVAL_RATE_MIN" "$TRAIN_ARRIVAL_RATE_MAX" \
  --train-domains-per-reset "$TRAIN_DOMAINS_PER_RESET" \
  --eval-in-distribution-sigma-values $EVAL_IN_DISTRIBUTION_SIGMA_VALUES \
  --eval-in-distribution-arrival-rate-values $EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES \
  --eval-stress-sigma-values $EVAL_STRESS_SIGMA_VALUES \
  --eval-stress-arrival-rate-values $EVAL_STRESS_ARRIVAL_RATE_VALUES

echo "Job finished at: $(date)"
