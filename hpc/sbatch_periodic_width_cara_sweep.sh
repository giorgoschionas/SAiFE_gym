#!/bin/bash -l
#SBATCH --job-name=periodic_width_cara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

cd "${SAIFE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ -n "${SAIFE_PYTHON_MODULE:-}" ]]; then
  module purge
  module load "$SAIFE_PYTHON_MODULE"
fi
venv_dir=${SAIFE_VENV_DIR:-venv}
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  echo "Set SAIFE_VENV_DIR or run bash hpc/setup_env.sh in the checkout." >&2
  exit 1
fi
source "$venv_dir/bin/activate"

# Each worker runs an independent episode batch on one CPU.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

python -u experiments/sweep_dense_exponential_width.py \
  --half-widths 25 50 100 150 200 250 --tau 250 \
  --risk-aversion-values 0 0.0003 0.001 0.003 0.01 \
  --periodic-rebalance-every 100 --initial-wealth 1000 \
  --n-steps 1000 --evaluation-seed 100042 \
  --n-eval-episodes "${N_EVAL_EPISODES:-10}" --num-trajectories "${NUM_TRAJECTORIES:-100}" \
  --workers "${SLURM_CPUS_PER_TASK:-4}" \
  --output-dir "${OUTPUT_DIR:-experiments/results/periodic_width_cara}"
