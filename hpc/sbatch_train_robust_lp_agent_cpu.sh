#!/bin/bash -l

# Example Barkla2 single-node CPU Slurm job for robust LP PPO training.

#SBATCH -J robust_lp_train
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-/mnt/scratch/users/$USER/rl_experiments/SAiFE_gym}
VENV_DIR=${SAIFE_VENV_DIR:-/mnt/fastscratch/users/$USER/venvs/saife_gym_py312}

TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
NUM_TRAJECTORIES=${NUM_TRAJECTORIES:-100}
N_STEPS=${N_STEPS:-200}
TAU=${TAU:-5}
ALPHA3=${ALPHA3:-0.0}
SEED=${SEED:-42}
N_EVAL_EPISODES=${N_EVAL_EPISODES:-10}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/results/robust_rl/barkla2_full}

NOMINAL_SIGMA=${NOMINAL_SIGMA:-2.0}
NOMINAL_ARRIVAL_RATE=${NOMINAL_ARRIVAL_RATE:-100.0}
NOMINAL_GAS_COST=${NOMINAL_GAS_COST:-0.0}

TRAIN_SIGMA_MIN=${TRAIN_SIGMA_MIN:-1.0}
TRAIN_SIGMA_MAX=${TRAIN_SIGMA_MAX:-4.0}
TRAIN_ARRIVAL_RATE_MIN=${TRAIN_ARRIVAL_RATE_MIN:-50.0}
TRAIN_ARRIVAL_RATE_MAX=${TRAIN_ARRIVAL_RATE_MAX:-200.0}
TRAIN_GAS_COST_MIN=${TRAIN_GAS_COST_MIN:-0.0}
TRAIN_GAS_COST_MAX=${TRAIN_GAS_COST_MAX:-20.0}

EVAL_SIGMA_VALUES=${EVAL_SIGMA_VALUES:-"0.5 2.0 6.0"}
EVAL_ARRIVAL_RATE_VALUES=${EVAL_ARRIVAL_RATE_VALUES:-"25.0 100.0 300.0"}
EVAL_GAS_COST_VALUES=${EVAL_GAS_COST_VALUES:-"0.0 10.0 40.0"}

module purge
module load miniforge3/25.3.0-python3.12.10
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
echo "Training: timesteps=$TOTAL_TIMESTEPS trajectories=$NUM_TRAJECTORIES n_steps=$N_STEPS tau=$TAU alpha3=$ALPHA3 seed=$SEED"
echo "Evaluation: episodes=$N_EVAL_EPISODES sigma=[$EVAL_SIGMA_VALUES] arrival=[$EVAL_ARRIVAL_RATE_VALUES] gas=[$EVAL_GAS_COST_VALUES]"
echo "Output dir: $OUTPUT_DIR"

python -u experiments/train_robust_lp_agent.py \
  --output-dir "$OUTPUT_DIR" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --num-trajectories "$NUM_TRAJECTORIES" \
  --n-steps "$N_STEPS" \
  --tau "$TAU" \
  --alpha3 "$ALPHA3" \
  --seed "$SEED" \
  --n-eval-episodes "$N_EVAL_EPISODES" \
  --learning-rate "$LEARNING_RATE" \
  --nominal-sigma "$NOMINAL_SIGMA" \
  --nominal-arrival-rate "$NOMINAL_ARRIVAL_RATE" \
  --nominal-gas-cost "$NOMINAL_GAS_COST" \
  --train-sigma-range "$TRAIN_SIGMA_MIN" "$TRAIN_SIGMA_MAX" \
  --train-arrival-rate-range "$TRAIN_ARRIVAL_RATE_MIN" "$TRAIN_ARRIVAL_RATE_MAX" \
  --train-gas-cost-range "$TRAIN_GAS_COST_MIN" "$TRAIN_GAS_COST_MAX" \
  --eval-sigma-values $EVAL_SIGMA_VALUES \
  --eval-arrival-rate-values $EVAL_ARRIVAL_RATE_VALUES \
  --eval-gas-cost-values $EVAL_GAS_COST_VALUES

echo "Job finished at: $(date)"
