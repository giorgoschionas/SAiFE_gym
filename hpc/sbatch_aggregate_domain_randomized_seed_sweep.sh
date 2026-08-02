#!/bin/bash -l

# Aggregate a completed domain-randomized PPO training-seed sweep.

#SBATCH -J dr_ppo_aggregate
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-/mnt/scratch/users/$USER/rl_experiments/SAiFE_gym}
VENV_DIR=${SAIFE_VENV_DIR:-/mnt/fastscratch/users/$USER/venvs/rl_venv}
INPUT_DIR=${INPUT_DIR:-experiments/results/domain_randomized_ppo/barkla2_seed_sweep}
OUTPUT_DIR=${OUTPUT_DIR:-$INPUT_DIR/aggregate}
BOOTSTRAP_RESAMPLES=${BOOTSTRAP_RESAMPLES:-10000}
BOOTSTRAP_SEED=${BOOTSTRAP_SEED:-42}
CONFIDENCE_LEVEL=${CONFIDENCE_LEVEL:-0.95}

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
