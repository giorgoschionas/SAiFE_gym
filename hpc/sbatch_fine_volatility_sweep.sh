#!/bin/bash -l
#SBATCH -J lp_fine_volatility
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail
cd "${SAIFE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
runner=experiments/wider-positions-drivers/fine_volatility_sweep.py
if [[ ! -f "$runner" ]]; then
  echo "Missing $runner in $PWD. Set SAIFE_PROJECT_DIR to the checkout containing the fine sweep." >&2
  exit 1
fi
if [[ -n "${SAIFE_PYTHON_MODULE:-}" ]]; then
  module purge
  module load "$SAIFE_PYTHON_MODULE"
fi
venv_dir=${SAIFE_VENV_DIR:-venv}
if [[ ! -f "$venv_dir/bin/activate" ]]; then
  echo "Set SAIFE_VENV_DIR or run bash hpc/setup_env.sh to create the project environment." >&2
  exit 1
fi
source "$venv_dir/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
  MPLCONFIGDIR=$(mktemp -d "${TMPDIR:-/tmp}/saife_matplotlib.XXXXXX")
fi
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"
output_dir=${OUTPUT_DIR:-experiments/results/periodic_widths/fine_volatility/${SLURM_JOB_ID}}
echo "Fine volatility sweep output directory: $output_dir"
python -u "$runner" --workers "${SLURM_CPUS_PER_TASK:-1}" --output-dir "$output_dir"
