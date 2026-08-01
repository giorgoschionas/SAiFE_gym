#!/bin/bash -l

# Example Barkla2 Slurm smoke-test job for experiments/train_robust_lp_agent.py.

#SBATCH -J robust_lp_smoke
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

PROJECT_DIR=${SAIFE_PROJECT_DIR:-/mnt/scratch/users/$USER/rl_experiments/SAiFE_gym}
VENV_DIR=${SAIFE_VENV_DIR:-/mnt/fastscratch/users/$USER/venvs/rl_venv}

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
mkdir -p logs experiments/results/robust_rl

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "SLURM job id: $SLURM_JOB_ID"

python -u experiments/train_robust_lp_agent.py \
  --smoke-test \
  --output-dir experiments/results/robust_rl/barkla2_smoke

echo "Job finished at: $(date)"
