#!/bin/bash -l

# Example Barkla2 single-node CPU Slurm job for domain-randomized LP PPO.

#SBATCH -J dr_ppo_train
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-/mnt/scratch/users/$USER/rl_experiments/SAiFE_gym}
VENV_DIR=${SAIFE_VENV_DIR:-/mnt/fastscratch/users/$USER/venvs/rl_venv}

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-100}
TRAIN_DOMAINS_PER_RESET=${TRAIN_DOMAINS_PER_RESET:-$NUM_TRAJECTORIES}
N_STEPS=${N_STEPS:-1000}
TAU=${TAU:-500}
ALPHA3=${ALPHA3:-4000.0}
INITIAL_WEALTH=${INITIAL_WEALTH:-5000.0}
INVENTORY_PHI=${INVENTORY_PHI:-0.02}
SEED=${SEED:-42}
EVALUATION_SEED=${EVALUATION_SEED:-100042}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-10}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
PERIODIC_REBALANCE_EVERY=${PERIODIC_REBALANCE_EVERY:-5}
PERIODIC_WIDTH=${PERIODIC_WIDTH:-2}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/results/domain_randomized_ppo/barkla2_full}

NOMINAL_SIGMA=${NOMINAL_SIGMA:-0.10}
NOMINAL_ARRIVAL_RATE=${NOMINAL_ARRIVAL_RATE:-100.0}
NOMINAL_GAS_COST=${NOMINAL_GAS_COST:-0.0}

TRAIN_SIGMA_MIN=${TRAIN_SIGMA_MIN:-0.01}
TRAIN_SIGMA_MAX=${TRAIN_SIGMA_MAX:-0.10}
TRAIN_ARRIVAL_RATE_MIN=${TRAIN_ARRIVAL_RATE_MIN:-50.0}
TRAIN_ARRIVAL_RATE_MAX=${TRAIN_ARRIVAL_RATE_MAX:-200.0}
TRAIN_GAS_COST_MIN=${TRAIN_GAS_COST_MIN:-1.0}
TRAIN_GAS_COST_MAX=${TRAIN_GAS_COST_MAX:-6.0}

EVAL_IN_DISTRIBUTION_SIGMA_VALUES=${EVAL_IN_DISTRIBUTION_SIGMA_VALUES:-"0.025 0.055 0.085"}
EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES=${EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES:-"75.0 125.0 175.0"}
EVAL_IN_DISTRIBUTION_GAS_COST_VALUES=${EVAL_IN_DISTRIBUTION_GAS_COST_VALUES:-"2.0 3.5 5.0"}
EVAL_STRESS_SIGMA_VALUES=${EVAL_STRESS_SIGMA_VALUES:-"0.025 0.10 0.30"}
EVAL_STRESS_ARRIVAL_RATE_VALUES=${EVAL_STRESS_ARRIVAL_RATE_VALUES:-"25.0 100.0 300.0"}
EVAL_STRESS_GAS_COST_VALUES=${EVAL_STRESS_GAS_COST_VALUES:-"0.0 10.0 40.0"}

module purge
module load miniforge3/25.3.0-python3.12.10
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "ERROR: no virtualenv found at $VENV_DIR" >&2
  echo "Create it with 'bash hpc/setup_barkla2_env.sh', or point SAIFE_VENV_DIR at an existing env." >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=$SLURM_NTASKS
export MKL_NUM_THREADS=$SLURM_NTASKS
export OPENBLAS_NUM_THREADS=$SLURM_NTASKS
export NUMEXPR_NUM_THREADS=$SLURM_NTASKS
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/users/$USER/saife_matplotlib_${SLURM_JOB_ID}}

mkdir -p "$MPLCONFIGDIR"
cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "SLURM job id: $SLURM_JOB_ID"
echo "Resources: partition=$SLURM_JOB_PARTITION nodes=$SLURM_JOB_NUM_NODES tasks=$SLURM_NTASKS"
echo "Training: timesteps=$TOTAL_TIMESTEPS trajectories=$NUM_TRAJECTORIES n_steps=$N_STEPS tau=$TAU alpha3=$ALPHA3 initial_wealth=$INITIAL_WEALTH inventory_phi=$INVENTORY_PHI seed=$SEED"
echo "Fixed evaluation seed: $EVALUATION_SEED"
echo "Periodic baseline: rebalance_every=$PERIODIC_REBALANCE_EVERY width=$PERIODIC_WIDTH"
echo "Training domain: sigma=[$TRAIN_SIGMA_MIN, $TRAIN_SIGMA_MAX] arrival=[$TRAIN_ARRIVAL_RATE_MIN, $TRAIN_ARRIVAL_RATE_MAX] gas=[$TRAIN_GAS_COST_MIN, $TRAIN_GAS_COST_MAX] domains_per_reset=$TRAIN_DOMAINS_PER_RESET"
echo "In-distribution evaluation: episodes=$N_EVAL_EPISODES sigma=[$EVAL_IN_DISTRIBUTION_SIGMA_VALUES] arrival=[$EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES] gas=[$EVAL_IN_DISTRIBUTION_GAS_COST_VALUES]"
echo "Stress evaluation: episodes=$N_EVAL_EPISODES sigma=[$EVAL_STRESS_SIGMA_VALUES] arrival=[$EVAL_STRESS_ARRIVAL_RATE_VALUES] gas=[$EVAL_STRESS_GAS_COST_VALUES]"
echo "Output dir: $OUTPUT_DIR"

python -u experiments/train_robust_lp_agent.py \
  --output-dir "$OUTPUT_DIR" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --n-steps "$N_STEPS" \
  --tau "$TAU" \
  --alpha3 "$ALPHA3" \
  --initial-wealth "$INITIAL_WEALTH" \
  --inventory-phi "$INVENTORY_PHI" \
  --seed "$SEED" \
  --evaluation-seed "$EVALUATION_SEED" \
  --n-eval-episodes "$N_EVAL_EPISODES" \
  --learning-rate "$LEARNING_RATE" \
  --periodic-rebalance-every "$PERIODIC_REBALANCE_EVERY" \
  --periodic-width "$PERIODIC_WIDTH" \
  --nominal-sigma "$NOMINAL_SIGMA" \
  --nominal-arrival-rate "$NOMINAL_ARRIVAL_RATE" \
  --nominal-gas-cost "$NOMINAL_GAS_COST" \
  --train-sigma-range "$TRAIN_SIGMA_MIN" "$TRAIN_SIGMA_MAX" \
  --train-arrival-rate-range "$TRAIN_ARRIVAL_RATE_MIN" "$TRAIN_ARRIVAL_RATE_MAX" \
  --train-gas-cost-range "$TRAIN_GAS_COST_MIN" "$TRAIN_GAS_COST_MAX" \
  --train-domains-per-reset "$TRAIN_DOMAINS_PER_RESET" \
  --eval-in-distribution-sigma-values $EVAL_IN_DISTRIBUTION_SIGMA_VALUES \
  --eval-in-distribution-arrival-rate-values $EVAL_IN_DISTRIBUTION_ARRIVAL_RATE_VALUES \
  --eval-in-distribution-gas-cost-values $EVAL_IN_DISTRIBUTION_GAS_COST_VALUES \
  --eval-stress-sigma-values $EVAL_STRESS_SIGMA_VALUES \
  --eval-stress-arrival-rate-values $EVAL_STRESS_ARRIVAL_RATE_VALUES \
  --eval-stress-gas-cost-values $EVAL_STRESS_GAS_COST_VALUES

echo "Job finished at: $(date)"
